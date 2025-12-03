import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary
import torch.nn.init as init

from Trainers import *
from TrainerLite import ModelTrainer

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    return m

def reparameterize(mu, logvar):
    """
    Reparameterization trick in VAE.
    :param mu: mu vector
    :param logvar: logvar vector
    :return: reparameterized vector
    """
    eps = torch.randn_like(mu)
    return mu + eps * torch.exp(logvar / 2)

class GEGLU_proj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GEGLU_proj, self).__init__()
        self.proj = nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x):
        x = self.proj(x)
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class ImageEncoder(nn.Module):
    name = 'imgen'
    
    def __init__(self, latent_dim=128):
        super(ImageEncoder, self).__init__()

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
        return f"IMGEN"

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(-1, 6 * 16 * 16)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        z = reparameterize(mu, logvar)

        return z, mu, logvar, out


class ImageDecoder(nn.Module):
    name = 'imgde'
    
    def __init__(self, latent_dim=128):
        super(ImageDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.activate_func = nn.Sigmoid()
        
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
                            ])
            else:
                cnn.extend([nn.ConvTranspose2d(in_ch, out_ch, ks, st, pd),
                            nn.LeakyReLU(inplace=True)])
        
        self.cnn = nn.Sequential(*cnn)

        # 512 * 16 * 16
        # 256 * 16 * 16
        # Re
        # 256 * 32 * 32
        # 128 * 32 * 32
        # Re
        # 128 * 64 * 64
        # Re
        # 128 * 128 * 128
        # Re
        # 1 * 128 * 128
        # Re

        self.fclayers = nn.Linear(self.latent_dim, 512 * 16 * 16)

        self.cnn = init_weights(self.cnn)
        self.fclayers = init_weights(self.fclayers)

    def __str__(self):
        return f"IMGDE"

    def forward(self, x):
        out = self.fclayers(x)
        out = self.cnn(out.view(-1, 512, 16, 16))
        # DO NOT use sigmoid with BCEWithLogitsLoss
        # if not self.training:
        #    out = torch.sigmoid(out)
        return out.view(-1, 1, 128, 128)



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

        self.fc_mu = nn.Sequential(
            nn.Linear(self.feature_length, self.latent_dim)
            # nn.ReLU()
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(self.feature_length, self.latent_dim)
            # nn.ReLU()
        )

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
        
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        z = reparameterize(mu, logvar)

        return z, mu, logvar, out

class Student(nn.Module):

    def __init__(self, device=None, teacher=None):
        super(Student, self).__init__()

        # Named children
        self.imgen = ImageEncoder(latent_dim=128)
        self.imgde = ImageDecoder(latent_dim=128)
        self.csien = CSIEncoderHPool(latent_dim=128)

        if device is not None:
            self.to(device)
            
        if isinstance(teacher, Teacher):
            # Share with teacher
            self.imgen = teacher.imgen
            self.imgde = teacher.rimgde

        self.latent_loss = nn.MSELoss(reduction='sum')
        self.img_loss = nn.MSELoss()
        self.feature_loss = nn.MSELoss()

        self.alpha = 0.8
        self.latent_weight = 0.1
        self.rimg_weight = 1.e-5
        self.feature_weight = 10

    def kd_loss(self, mu_s, logvar_s, mu_t, logvar_t):
        mu_loss = self.latent_loss(mu_s, mu_t) / mu_s.shape[0]
        logvar_loss = self.latent_loss(logvar_s, logvar_t) / logvar_s.shape[0]
        # latent_loss = self.alpha * mu_loss + (1 - self.alpha) * logvar_loss
        return mu_loss, logvar_loss

    def forward(self, data):
        csi, pd, rimg = data['csi'], data['pd'], data['shape']
        s_z, s_mu, s_logvar, s_fea = self.csien(csi=csi, pd=pd)
        s_rimage = self.imgde(s_z)

        with torch.no_grad():
            t_z, t_mu, t_logvar, t_fea = self.imgen(rimg)
            t_rimage = self.imgde(t_z)

        mu_loss, logvar_loss = self.kd_loss(s_mu, s_logvar, t_mu, t_logvar)
        mu_loss, logvar_loss = self.alpha * mu_loss * self.latent_weight, (1 - self.alpha) * logvar_loss * self.latent_weight
        feature_loss = self.feature_loss(s_fea, t_fea) * self.feature_weight
        image_loss = self.img_loss(s_rimage, rimg)

        ret = {
        'S_LAT'     : s_z,
        'S_PRED': s_rimage,
        'T_LAT'     : t_z,
        'T_PRED': t_rimage,
        'GT': rimg
        }

        loss = {
            'LOSS': mu_loss + logvar_loss + feature_loss + image_loss,
            'MU': mu_loss,
            'LOGVAR': logvar_loss,
            'FEA': feature_loss,
            'IMG': image_loss
        }

        return ret, loss


class Teacher(nn.Module):

    def __init__(self, device=None):
        super(Teacher, self).__init__()

        # Named children
        self.imgen = ImageEncoder(latent_dim=128)
        self.imgde = ImageDecoder(latent_dim=128)

        if device is not None:
            self.to(device)

        self.img_loss = nn.BCEWithLogitsLoss(reduction='sum')

        self.beta = 0.5
        self.img_weight = 1.

    def kl_loss(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss

    def forward(self, data):
        rimg = data['shape']
        z, mu, logvar, feature = self.imgen(rimg)
        r_recon = self.imgde(z)

        kl_loss = self.kl_loss(mu, logvar) * self.beta
        img_loss = self.img_loss(r_recon, rimg) / r_recon.shape[0] * self.img_weight

        vae_loss = kl_loss + img_loss

        ret = {
        'LAT'      : z,
        'IMG': r_recon,
        'GT': rimg,
                }

        loss = {
            'LOSS': vae_loss,
            'KL': kl_loss,
            'IMG': img_loss
        }

        return ret, loss


class StudentTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super(StudentTrainer, self).__init__(model=Student(), *args, **kwargs)
        self.trainer = Trainer(self.device, f"{self.name}_{self.notion}_TRAIN", 
                        train_module=['csien'])
        self.pred_terms = ('GT', 'T_PRED', 'S_PRED')

class TeacherTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super(TeacherTrainer, self).__init__(model=Teacher(), *args, **kwargs)
        self.pred_terms = ('GT', 'IMG')