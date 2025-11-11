import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.init as init
from Structure.Model import *
version = 'COND'

#
# Channel Attention
#

class ChannelAttentionCompress(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(ChannelAttentionCompress, self).__init__()
        # Step 1: squeeze (global average pooling)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Step 2: excitation (MLP for channel attention)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

        # Step 3: 通道压缩
        self.compress = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        b, c, h, w = x.size()
        # Channel attention weights
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        # Apply attention
        out = x * y
        
        # Compress channels
        out = self.compress(out)
        return out


#
# Teacher
# Asymmetric VAE
# Img -> Feature | Feature -> z -> Img
#

class ImageEncoder(nn.Module):
    name = 'imgen'
    
    def __init__(self):
        super(ImageEncoder, self).__init__()
        
        block = [[1, 128, 3, 2, 1],
                [128, 256, 3, 2, 1],
                [256, 512, 3, 2, 1],
                [512, 512, 3, 2, 1],
                # [512, 128, 1, 1, 1], # Squeeze
                ]
        
        cnn = []

        for [in_ch, out_ch, ks, st, pd] in block:
            cnn.extend([nn.Conv2d(in_ch, out_ch, ks, st, pd),
                        nn.LeakyReLU(inplace=True)])

        cnn.extend(ChannelAttentionCompress(512, 128))
            
        self.cnn = nn.Sequential(*cnn)

        # 1 * 128 * 128
        # 128 * 64 * 64
        # Re
        # 256 * 32 * 32
        # Re
        # 512 * 16 * 16
        # Re
        # 512 * 8 * 8
        # Re
        # 128 * 8 * 8

    def __str__(self):
        return f"IMGEN{version}"

    def forward(self, x):
        fea = self.cnn(x)
        fea = fea.view(fea.shape[0], -1)

        return fea


class ImageDecoder(nn.Module):
    name = 'imgde'
    
    def __init__(self, active_func=nn.Sigmoid(), *args, **kwargs):
        super(ImageDecoder, self).__init__(*args, **kwargs)

        self.latent_dim = 128
        self.active_func = active_func
        
        block = [
                [128, 256, 4, 2, 1],
                [256, 128, 4, 2, 1],
                [128, 64, 4, 2, 1],
                [64, 32, 4, 2, 1],
                [32, 1, 3, 1, 1]]
        
        cnn = []
        # cnn.extend([nn.Conv2d(6, 512, 1, 1, 0)])
        
        for [in_ch, out_ch, ks, st, pd] in block:
            if ks == 3:
                cnn.extend([
                    nn.Conv2d(in_ch, out_ch, ks, st, pd)
                    ])
            else:
                cnn.extend([nn.ConvTranspose2d(in_ch, out_ch, ks, st, pd),
                            nn.LeakyReLU(inplace=True)])
        
        self.cnn = nn.Sequential(*cnn, self.active_func)

        # 128 * 8 * 8
        # 256 * 16 * 16
        # Re
        # 128 * 32 * 32
        # Re
        # 64 * 64 * 64
        # Re
        # 32 * 128 * 128
        # Re
        # 1 * 128 * 128
        # Sigmoid

        self.sampling_fc = nn.Linear(128 * 8 * 8, 2 * self.latent_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 8 * 8 * 128),
            nn.ReLU()
        )

        self.cnn = init_weights(self.cnn)
        self.fc = init_weights(self.fc)

    def __str__(self):
        return f"IMGDE{version}"

    def forward(self, feature):
        gaussian_params = self.sampling_fc(feature)
        mu, logvar = gaussian_params.chunk(2, dim=-1)
        z = reparameterize(mu, logvar)

        out = self.fc(z)
        out = self.cnn(out.view(-1, 512, 8, 8))
        return out.view(-1, 1, 128, 128), mu, logvar


class Teacher(nn.Module):

    def __init__(self, device=None):
        super(Teacher, self).__init__()

        self.imgen = ImageEncoder(latent_dim=128)
        self.imgde = ImageDecoder(latent_dim=128)

        if device is not None:
            self.to(device)

    def forward(self, rimg):
        feature = self.imgen(rimg)
        recon, mu, logvar = self.imgde(feature)

        ret = {
        'feature': feature,
        'rimage' : recon,
        'mu': mu,
        'logvar': logvar
                }

        return ret


#
# Student: 3V Hierarchical Pool
#

class CSIEncoderHPool(nn.Module):
    name = 'csien'

    def __init__(self, *args, **kwargs):
        super(CSIEncoderHPool, self).__init__(*args, **kwargs)

        self.latent_dim = 256
        self.lstm_step_length = 512 * 7

        self.lstm_steps = 75
        self.csi_feature_length = 128
        self.pd_feature_length = 128
        self.feature_length = 8192
        self.pd_length = 62

        # 6 * 30 * 100
        # 128 * 28 * 98
        # 256 * 14 * 49
        # 512 * 7 * 25

        self.cnn = nn.Sequential(
            nn.Conv2d(6, 128, 5, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.LeakyReLU(inplace=True)
        )

        # 1536 = 6 * 16 * 16
        # 32768 = 512 * 8 * 8
        # 8192 = 128 * 8 * 8
        # ALTER: No LSTM

        self.lstm = nn.LSTM(self.lstm_step_length, self.csi_feature_length, 2, batch_first=True, dropout=0.1)
        
    def __str__(self):
        return f"CSIENHP"

    def forward(self, csi, pd):
        fea_csi = self.cnn(csi)
        fea_pd = self.fc_pd(pd)
        lstm_out, (final_hidden_state, final_cell_state) = self.lstm.forward(
            fea_csi.view(-1, 512*7, 75).transpose(1, 2))

        chunks = torch.chunk(lstm_out, 3, dim=1)
        summaries = [chunk.mean(dim=1) for chunk in chunks]
        features = torch.cat(summaries, dim=1)

        out = torch.cat((features.view(-1, self.csi_feature_length * 3), fea_pd.view(-1, self.pd_feature_length)), -1)
        out = self.fc_feature(out)

        return out


#
# Student: 2D+1D Convolution
#

class CSIEncoder2D1D(nn.Module):
    name = 'csien'

    def __init__(self, *args, **kwargs):
        super(CSIEncoder2D1D, self).__init__(*args, **kwargs)


        # 6 * 30 * 300
        # 2D
        # 64 * 30 * 150
        # 128 * 15 * 75
        # 256 * 8 * 38
        #
        # 1D
        # (256 * 8 = 4096) * 19
        # 8192 * 10
        # 1024 * 8 * 8
        #
        # Channel
        # 128 * 8 * 8

        self.cnn2d = nn.Sequential(
            nn.Conv2d(6, 64, 3, (1, 2), 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(inplace=True)
        )

        self.cnn1d = nn.Sequential(
            nn.Conv1d(4096, 8192, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8192, 8192, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
        )

        self.cnnch = ChannelAttentionCompress(1024, 128)
        )

    def __str__(self):
        return f"CSIEN2D1D"

    def forward(self, csi):
        csi_fea = self.cnn2d(csi)
        csi_fea = self.cnn1d(csi_fea.reshape(csi_fea.shape[0], 4096, 38))
        csi_fea = self.cnnch(csi_fea.reshape(csi_fea.shape[0], 1024, 8, 8))

        return csi_fea


# -----------------------------------------------------
# 子网络：生成 s 和 t，用于仿射变换
# -----------------------------------------------------

class STNet(nn.Module):
    def __init__(self, in_dim, cond_dim, hid_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, in_dim * 2)  # 输出 s 和 t
        )

    def forward(self, x, cond):
        h = torch.cat([x, cond], dim=1)
        st = self.net(h)
        s, t = torch.chunk(st, 2, dim=1)
        return s, t

# -----------------------------------------------------
# 仿射耦合层（Affine Coupling）
# -----------------------------------------------------
class AffineCoupling(nn.Module):
    def __init__(self, dim, cond_dim, swap=False):
        super().__init__()
        self.swap = swap
        self.net = STNet(dim // 2, cond_dim)

    def forward(self, x, cond, reverse=False):
        # 可能交换输入
        if self.swap:
            x1, x2 = x.chunk(2, dim=1)
            x1, x2 = x2, x1
        else:
            x1, x2 = x.chunk(2, dim=1)

        s, t = self.net(x1, cond)

        if not reverse:
            # 正向：用于训练
            y2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=1)
        else:
            # 反向：用于生成
            y2 = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=1)

        y = torch.cat([x1, y2], dim=1)

        # 如果交换过，再还原顺序
        if self.swap:
            y1, y2 = y.chunk(2, dim=1)
            y = torch.cat([y2, y1], dim=1)

        return y, log_det


# -----------------------------------------------------
# 条件 Flow 模型：多层耦合结构
# -----------------------------------------------------

class ConditionalFlow(nn.Module):
    def __init__(self, dim, cond_dim, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            AffineCoupling(dim, cond_dim, swap=(i % 2 == 1))  # 每层交替交换
            for i in range(n_layers)
        ])

    def forward(self, x, cond, reverse=False):

        log_det_total = 0
        if not reverse:
            for layer in self.layers:
                x, log_det = layer(x, cond, reverse=False)
                log_det_total += log_det
        else:
            for layer in reversed(self.layers):
                x, log_det = layer(x, cond, reverse=True)
                log_det_total += log_det
        return x, log_det_total


class Student(nn.Module):
    def __init__(self, device=None, teacher=None):
        super(Student, self).__init__()

        self.imgen = ImageEncoder()
        self.imgde = ImageDecoder(latent_dim=128)
        self.csien = CSIEncoder2D1D()
        self.flow = ConditionalFlow(8192, 8192)

        if device is not None:
            self.to(device)
            
        if isinstance(teacher, Teacher):
            # Share with teacher
            self.imgen = teacher.imgen
            self.imgde = teacher.rimgde

    def forward(self, csi, rimg):
    
        # Forward Flow (to z)
        s_fea = self.csien(csi=csi)
        z, log_det_total = self.flow(t_fea, s_fea, False)

        with torch.no_grad():
            t_fea = self.imgen(rimg)
            t_image, mu, logvar = self.imgde(t_fea)

        # Reverse Flow (to t_fea_hat and s_rimage)
        z = torch.randn(s_fea.shape[0], self.flow.layers[0].dim).to(s_fea.device)
        t_fea_hat, log_det_total = self.flow(z, s_fea, True)
        s_rimage, mu, logvar = self.imgde(t_fea_hat)

        ret = {
        's_fea'   : s_fea,
        's_rimage': s_rimage,
        't_fea'   : t_fea,
        't_rimage': t_rimage,
        'z': z,
        'log_det': log_det_total
        }

        return ret

