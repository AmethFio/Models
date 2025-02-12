import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.init as init

##############################################################################
# -------------------------------------------------------------------------- #
# Models named with 'b' are AEs
# Models named with 'c' are VAEs
# Models named with 'i' have interpolation layers
# Numbers after 'V' are versions
# Numbers after 'b' or 'c' are variations
# eg: ModelV03b1 means Ver3 AE Var1
# -------------------------------------------------------------------------- #
##############################################################################

version = 'v08F3'

# For torchinfo.summary
IMG = (1, 1, 128, 128)
CSI = (1, 6, 30, 100)
CSI2 = (1, 6, 30, 30)
LAT = (1, 128)
RIMG = (1, 1, 128, 226)
PD = (1, 62)


def batchnorm_layer(channels, batchnorm='identity'):
    """
    Definition of optional batchnorm layer.
    :param channels: input channels
    :param batchnorm: False or 'batch' or 'instance'
    :return: batchnorm layer or Identity layer (no batchnorm)
    """

    assert batchnorm in {'identity', 'batch', 'instance'}

    if batchnorm == 'identity':
        return nn.Identity(channels)
    elif batchnorm == 'batch':
        return nn.BatchNorm2d(channels)
    elif batchnorm == 'instance':
        return nn.InstanceNorm2d(channels)


def reparameterize(mu, logvar):
    """
    Reparameterization trick in VAE.
    :param mu: mu vector
    :param logvar: logvar vector
    :return: reparameterized vector
    """
    eps = torch.randn_like(mu)
    return mu + eps * torch.exp(logvar / 2)


class Interpolate(nn.Module):
    def __init__(self, size, mode='bilinear'):
        """
        Definition of interpolate layer.
        :param size: (height, width)
        :param mode: default is 'bilinear'
        """
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        out = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return out


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)

        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=2)

        proj_value = self.value_conv(x).view(batch_size, -1, height * width)

        out = torch.bmm(attention, proj_value.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        out = self.gamma * out + x
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            batchnorm_layer(out_channels, batchnorm),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            batchnorm_layer(out_channels, batchnorm)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out += x
        out = self.relu(out)
        return out

    
class ImageEncoder(nn.Module):
    name = 'imgen'
    
    def __init__(self, batchnorm='identity', latent_dim=16, *args, **kwargs):
        super(ImageEncoder, self).__init__(*args, **kwargs)

        self.batchnorm = batchnorm
        self.latent_dim = latent_dim
        
        block = [[1, 128, 3, 2, 1],
                [128, 128, 3, 1, 1],
                [128, 128, 3, 2, 1],
                [128, 128, 3, 1, 1],
                [128, 256, 3, 2, 1],
                [256, 256, 3, 1, 1],
                [256, 512, 3, 1, 1],
                [512, 512, 1, 1, 0],
                [512, 6, 1, 1, 0]]
        
        cnn = []

        for [in_ch, out_ch, ks, st, pd] in block:
            if in_ch != 512:
                cnn.extend([nn.Conv2d(in_ch, out_ch, ks, st, pd),
                            batchnorm_layer(out_ch, self.batchnorm),
                            nn.LeakyReLU(inplace=True)])
            else:
                cnn.extend([nn.Conv2d(in_ch, out_ch, ks, st, pd)])
            
        self.cnn = nn.Sequential(*cnn)

        # 1 * 128 * 128
        # 128 * 64 * 64
        # Re
        # 128 * 32 * 32
        # Re
        # 256 * 16 * 16
        # Re
        # 512 * 16* 16
        # 6 * 16 * 16

        self.fc_mu = nn.Sequential(
            nn.Linear(6 * 16 * 16, self.latent_dim)
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(6 * 16 * 16, self.latent_dim)
        )

    def __str__(self):
        return f"IMGEN{version}"

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(-1, 6 * 16 * 16)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        z = reparameterize(mu, logvar)

        return z, mu, logvar, out


class ImageDecoder(nn.Module):
    name = 'imgde'
    
    def __init__(self, batchnorm='identity', latent_dim=16, active_func=nn.Sigmoid(), *args, **kwargs):
        super(ImageDecoder, self).__init__(*args, **kwargs)

        self.batchnorm = batchnorm
        self.latent_dim = latent_dim
        self.active_func = active_func
        
        block = [
                [512, 256, 3, 1, 1],
                [256, 256, 4, 2, 1],
                [256, 128, 3, 1, 1],
                [128, 128, 4, 2, 1],
                [128, 128, 4, 2, 1],
                [128, 1, 3, 1, 1]]
        
        cnn = []
        # cnn.extend([nn.Conv2d(6, 512, 1, 1, 0)])
        
        for [in_ch, out_ch, ks, st, pd] in block:
            if ks == 3:
                cnn.extend([nn.Conv2d(in_ch, out_ch, ks, st, pd),
                            batchnorm_layer(out_ch, self.batchnorm)
                            ])
            else:
                cnn.extend([nn.ConvTranspose2d(in_ch, out_ch, ks, st, pd),
                            batchnorm_layer(out_ch, self.batchnorm),
                            nn.LeakyReLU(inplace=True)])
        
        self.cnn = nn.Sequential(*cnn, self.active_func)

        # 6 * 16 * 16
        # 512 * 16 * 16
        # 256 * 16 * 16
        # 256 * 32 * 32
        # 128 * 32 * 32
        # 128 * 64 * 64
        # 128 * 128 * 128
        # 1 * 128 * 128

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 512 * 16 * 16),
        )

    def __str__(self):
        return f"IMGDE{version}"

    def forward(self, x):
        out = self.fclayers(x)
        out = self.cnn(out.view(-1, 512, 16, 16))
        return out.view(-1, 1, 128, 128)


class CenterDecoder(nn.Module):
    name = 'ctrde'

    def __init__(self):
        super(CenterDecoder, self).__init__()
        self.feature_length = 1536

        self.fc = nn.Sequential(
            nn.Linear(self.feature_length, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

        init.xavier_normal_(self.fc[-2].weight)

    def __str__(self):
        return f"CTRDE{version}"

    def forward(self, x):
        out = self.fc(x.view(-1, self.feature_length))
        center = out[..., :2]
        depth = out[..., -1]
        return center, depth


class CSIEncoder(nn.Module):
    name = 'csien'
    
    def __init__(self, batchnorm='identity', latent_dim=16, lstm_step_length=512 * 7, lstm_steps=75, *args, **kwargs):
        super(CSIEncoder, self).__init__(*args, **kwargs)

        self.batchnorm = batchnorm
        self.latent_dim = latent_dim
        self.lstm_step_length = lstm_step_length

        self.lstm_steps = lstm_steps
        self.csi_feature_length = 128
        self.pd_feature_length = 128
        self.feature_length = 1536
        self.pd_length = 62

        # 6 * 30 * 100
        # 128 * 28 * 98
        # 256 * 14 * 49
        # 512 * 7 * 25

        self.cnn = nn.Sequential(
            nn.Conv2d(6, 128, 5, 1, 1),
            batchnorm_layer(128, self.batchnorm),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            batchnorm_layer(256, self.batchnorm),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1),
            batchnorm_layer(512, self.batchnorm),
            nn.LeakyReLU(inplace=True)
        )

        self.lstm = nn.LSTM(self.lstm_step_length, self.csi_feature_length, 2, batch_first=True, dropout=0.1)
        
        self.fc_feature = nn.Sequential(
            nn.Linear(self.csi_feature_length + self.pd_feature_length, 
                      self.feature_length),
            nn.ReLU()
        )
        
        self.fc_pd = nn.Sequential(
            nn.Linear(self.pd_length, self.pd_feature_length),
            nn.ReLU()
        )

        self.fc_mu = nn.Sequential(
            nn.Linear(self.feature_length, self.latent_dim)
            # nn.ReLU()
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(self.feature_length, self.latent_dim)
            # nn.ReLU()
        )

    def __str__(self):
        return f"CSIEN{version}"

    def forward(self, csi, pd):
        fea_csi = self.cnn(csi)
        fea_pd = self.fc_pd(pd)
        features, (final_hidden_state, final_cell_state) = self.lstm.forward(
            fea_csi.view(-1, self.lstm_step_length, self.lstm_steps).transpose(1, 2))
        # 256-dim output
        out = torch.cat((features[:, -1, :].view(-1, self.csi_feature_length), fea_pd.view(-1, self.pd_feature_length)), -1)
        out = self.fc_feature(out)
        
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        z = reparameterize(mu, logvar)
        return out, z, mu, logvar