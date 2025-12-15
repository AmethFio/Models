import torch
import torch.nn as nn
from WiDepthLite.TrainerLite import ModelTrainer

"""
Teacher = CNN AutoEncoder
loss = RECON

Student = CNN+LSTM+HPOOL + CONDFLOW
loss = FLOW + MMD + RECON

StudentPretrain = CNN+LSTM+HPOOL + CNN
loss = RECON + FFT
"""

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

    def forward(self, x):
        fea = self.cnn(x)
        fea = fea.view(fea.shape[0], -1)

        return fea

    
class ImageDecoder(nn.Module):
    name = 'imgde'
    
    def __init__(self, *args, **kwargs):
        super(ImageDecoder, self).__init__(*args, **kwargs)
        
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

        self.cnn = init_weights(self.cnn)

    def forward(self, feature):
        out = self.cnn(out.view(-1, 512, 8, 8))
        return out.view(-1, 1, 128, 128)


class Teacher(nn.Module):

    def __init__(self, device=None):
        super(Teacher, self).__init__()

        # Named children
        self.imgen = ImageEncoder(latent_dim=128)
        self.imgde = ImageDecoder(latent_dim=128)

        if device is not None:
            self.to(device)

        self.img_loss = nn.BCEWithLogitsLoss(reduction='sum')

        self.img_weight = 1.

    def forward(self, data):
        rimg = data['depthimage']
        feature = self.imgen(rimg)
        r_recon = self.imgde(feature)
        img_loss = self.img_loss(r_recon, rimg) / r_recon.shape[0] * self.img_weight

        ret = {
        'IMG': r_recon,
        'GT': rimg
                }

        loss = {
            'LOSS': img_loss
        }

        return ret, loss



class CSIEncoderHPool(nn.Module):
    name = 'csien'
    
    def __init__(self,latent_dim=128, lstm_step_length=512 * 7, lstm_steps=75):
        
        super(CSIEncoderHPool, self).__init__()
        self.latent_dim = latent_dim
        self.lstm_step_length = lstm_step_length

        self.lstm_steps = lstm_steps
        self.csi_feature_length = 128
        self.pd_feature_length = 128
        self.feature_length = 1536
        self.pd_length = 62

        # 6 * 30 * 300
        # 128 * 28 * 298
        # 256 * 14 * 149
        # 512 * 7 * 75

        self.cnn = nn.Sequential(
            nn.Conv2d(6, 128, 5, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.LeakyReLU(inplace=True)
        )

        self.lstm = nn.LSTM(self.lstm_step_length, self.csi_feature_length, 2, batch_first=True, dropout=0.1)
        
        self.fc_feature = GEGLU_proj(self.csi_feature_length * 3 + self.pd_feature_length, 
                       self.feature_length)
        
        self.fc_pd = nn.Sequential(
            nn.Linear(self.pd_length, self.pd_feature_length),
            nn.ReLU()
        )

    def forward(self, csi, pd):
        fea_csi = self.cnn(csi)
        fea_pd = self.fc_pd(pd)
        lstm_out, (final_hidden_state, final_cell_state) = self.lstm.forward(
            fea_csi.view(-1, 512*7, 75).transpose(1, 2))

        chunks = torch.chunk(lstm_out, 3, dim=1)
        summaries = [chunk.mean(dim=1) for chunk in chunks]
        features = torch.cat(summaries, dim=1)

        fea = torch.cat((features.view(-1, self.csi_feature_length * 3), fea_pd.view(-1, self.pd_feature_length)), -1)
        fea = self.fc_feature(fea)

        return fea.view(fea.shape[0], -1)


class CSIDecoder(nn.Module):
    def __init__(self):
        suter(CSIDecoder, self).__init__()

        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=(2,2), padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=(2,2), padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 6, kernel_size=3, padding=1)  # 输出形状近似 (B,1,30,300)
            )

    def forward(self, csi_fea):
        recon = self.decoder(csi_fea)

        return recon


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


class MMDLoss(nn.Module):
	def __init__(self, kernel_mul = 2.0, kernel_num = 5):
		super(MMDLoss, self).__init__()
		self.kernel_num = kernel_num
		self.kernel_mul = kernel_mul
		self.fix_sigma = None
		return
	def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
		n_samples = int(source.size()[0])+int(target.size()[0])
    	total = torch.cat([source, target], dim=0)

    	total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    	total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    	L2_distance = ((total0-total1)**2).sum(2) 
    	if fix_sigma:
    		bandwidth = fix_sigma
    	else:
    		bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    	bandwidth /= kernel_mul ** (kernel_num // 2)
    	bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    	kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    	return sum(kernel_val)

    def forward(self, source, target):
    	batch_size = int(source.size()[0])
    	kernels = guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
    	XX = kernels[:batch_size, :batch_size]
    	YY = kernels[batch_size:, batch_size:]
    	XY = kernels[:batch_size, batch_size:]
    	YX = kernels[batch_size:, :batch_size]
    	loss = torch.mean(XX + YY - XY -YX)
    	return loss


class Student(nn.Module):
    def __init__(self, device=None, teacher=None):
        super(Student, self).__init__()

        self.imgen = ImageEncoder()
        self.imgde = ImageDecoder()
        self.csien = CSIEncoderHPool()
        self.flow = ConditionalFlow(8192, 8192)

        if device is not None:
            self.to(device)
            
        if isinstance(teacher, Teacher):
            # Share with teacher
            self.imgen = teacher.imgen
            self.imgde = teacher.rimgde

        self.feature_loss = MMDLoss()
        self.recon_loss = nn.MSELoss()

        self.flow_weight = 1.
        self.feature_weight = 1.
        self.recon_weight = 1.

    def forward(self, data):
        
        # Forward Flow (to z)
        s_fea = self.csien(csi=data['csi'], pd=data['pd'])
        z, log_det_total = self.flow(t_fea, s_fea, False)

        with torch.no_grad():
            t_fea = self.imgen(rimg)
            t_image = self.imgde(t_fea)

        # Reverse Flow (to t_fea_hat and s_rimage)
        z = torch.randn(s_fea.shape[0], self.flow.layers[0].dim).to(s_fea.device)
        t_fea_hat, log_det_total = self.flow(z, s_fea, True)
        s_image = self.imgde(t_fea_hat)

        log_prob_z = -0.5 * (z ** 2).sum(dim=1) - 0.5 * z.size(1) * torch.log(torch.tensor(2 * torch.pi))
        flow_loss = -(log_prob_z + log_det_total).mean() * self.flow_weight
        feature_loss = self.feature_loss(s_fea, t_fea) * self.feature_weight
        recon_loss = self.recon_loss(s_image, t_image) * self.recon_weight

        loss = flow_loss + feature_loss + recon_loss

        ret = {
        'S_FEA'   : s_fea,
        'S_PRED': s_image,
        'T_FEA'   : t_fea,
        'T_PRED': t_image,
        'Z': z,
        'DET': log_det_total
        }

        loss = {
            'LOSS': loss,
            'FLOW': flow_loss,
            'FEATURE': feature_loss,
            'RECON': recon_loss
        }

        return ret, loss


class StudentPretrain(nn.Module):
    def __init__(self, device=None):
        super(StudentPretrain, self).__init__()

        self.csien = CSIEncoderHPool()
        self.cside = CSIDecoder()

        if device is not None:
            self.to(device)

        self.recon_loss = nn.L1Loss()
        self.recon_weight = 0.3
        self.spec_weight = 0.7

    def spec_loss(self, x_hat, x):
        # 频域 L1（使用复频谱模值）
        X = torch.abs(fft.rfft(x, dim=-1))
        X_hat = torch.abs(fft.rfft(x_hat, dim=-1))
        loss_f = self.recon_loss(X_hat, X)

        return loss_f, X_hat, X

    def forward(self, data):
        csi_fea = self.encoder(data['csi'])
        recon_csi = self.decoder(csi_fea)

        # Recover complex CSI: (64, 6, 30, 300) -> (64, 3, 30, 300)
        gt_csi = torch.chunk(data['csi'], 2, dim=1)
        gt_csi = gt_csi[0] + 1.j * gt_csi[1]
        recon_csi = torch.chunk(recon_csi, 2, dim=1)
        recon_csi = recon_csi[0] + 1.j * recon_csi[1]

        recon_loss = self.recon_loss(recon_csi, gt_csi) * self.recon_weight
        spec_loss, *specs = self.spec_loss(recon_csi, gt_csi) * self.spec_weight
        loss = recon_loss + spec_loss

        ret = {
        'RSPEC': specs[0],
        'GTSPEC': specs[1]
        }

        loss = {
            'LOSS': loss,
            'RECON': recon_loss,
            'FFT': spec_loss,
        }

        return ret, loss


class StudentPreTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super(StudentTrainer, self).__init__(model=StudentPretrain(), *args, **kwargs)
        self.trainer = Trainer(self.device, f"{self.name}_{self.notion}_TRAIN", 
                        train_module=['csien', 'cside'])

class StudentTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super(StudentTrainer, self).__init__(model=Student(), *args, **kwargs)
        self.trainer = Trainer(self.device, f"{self.name}_{self.notion}_TRAIN", 
                        train_module=['csien', 'cside'])
        self.pred_terms = ('GT', 'T_PRED', 'S_PRED')

class TeacherTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super(TeacherTrainer, self).__init__(model=Teacher(), *args, **kwargs)
        self.pred_terms = ('GT', 'IMG')