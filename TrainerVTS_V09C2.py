import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.ops import complete_box_iou_loss
import numpy as np
import matplotlib.pyplot as plt
import os
from Trainer import BasicTrainer
from Models.Structure.Model import *
from Loss import MyLoss, MyLossCTR

version = 'V09C2'

##############################################################################
# -------------------------------------------------------------------------- #
# Version V09C2
# Teacher1 learns raw images and estimates raw images
# Teacher2 learns cropped images and estimates binary masks
# Teacher3 learns raw images and estimates bbx and depth
# Features of Teacher1 are aligned with Teacher2 and Teacher3
# 
# Student learns (6, 30, m) CSIs and (2, 30, m) PhaseDiffs and estimates latent
#
# rawImageEncoder: in = 128 * 128,
#               out = [z:latent_dim, mu:latent_dim, logvar:latent_dim, features: 544 * 4 * 4]
# rawImageDecoder: in = 1 * latent_dim,
#               out = 128 * 128
# croppedImageEncoder: in = 128 * 128,
#               out = [z:latent_dim, mu:latent_dim, logvar:latent_dim, features: 512 * 4 * 4]
# croppedmageDecoder: in = 1 * latent_dim,
#               out = 128 * 128
# CSIEncoder: in = [6 * 30 * m], [2 * 30 * m]
#               out = [z:latent_dim, mu:latent_dim, logvar:latent_dim]
# BBXDecoder: in = 32 * 4 * 4,
#               out = [BBX:4, depth:1]
# -------------------------------------------------------------------------- #
##############################################################################

feature_length = 512 * 15
steps = 25


class ImageEncoder(BasicImageEncoder):
    def __init__(self, mode='raw', *args, **kwargs):
        super(ImageEncoder, self).__init__(*args, **kwargs)
        self.mode = mode
        self.last_nodes = 544 if mode=='raw' else 512
        
        channels = [1, 128, 128, 256, 256, self.last_nodes]
        block = []
        for i in range(len(channels) - 1):
            block.extend([nn.Conv2d(channels[i], channels[i+1], 3, 2, 1),
                          batchnorm_layer(channels[i+1], self.batchnorm),
                          nn.LeakyReLU(inplace=True)])
        self.cnn = nn.Sequential(*block)

        # 1 * 128 * 128
        # 128 * 64 * 64
        # 128 * 32 * 32
        # 256 * 16 * 16
        # 256 * 8 * 8
        # 544 * 4 * 4

        self.fc_mu = nn.Sequential(
            nn.Linear(4 * 4 * self.last_nodes, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.latent_dim),
            # self.active_func
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(4 * 4 * self.last_nodes, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.latent_dim),
            # self.active_func
        )

    def __str__(self):
        return f"IMGEN_{self.mode}_{version}"

    def forward(self, x):
        features = self.cnn(x)
        mu = self.fc_mu(features.view(-1, 4 * 4 * self.last_nodes))
        logvar = self.fc_logvar(features.view(-1, 4 * 4 * self.last_nodes))
        z = reparameterize(mu, logvar)

        #return z, mu, logvar, features
        return features


class ImageDecoder(BasicImageDecoder):
    def __init__(self, mode='raw', *args, **kwargs):
        super(ImageDecoder, self).__init__(*args, **kwargs)
        self.mode = mode
        channels = [512, 256, 256, 128, 128, 1]
        block = []
        for i in range(len(channels) - 1):
            block.extend([nn.ConvTranspose2d(channels[i], channels[i+1], 4, 2, 1),
                          batchnorm_layer(channels[i+1], self.batchnorm),
                          nn.LeakyReLU(inplace=True)])
        # Replace the last LeakyReLU
        block.pop()
        self.cnn = nn.Sequential(*block, self.active_func)

        # 512 * 4 * 4
        # 256 * 8 * 8
        # 256 * 16 * 16
        # 128 * 32 * 32
        # 128 * 64 * 64
        # 1 * 128 * 128

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8192),
            nn.ReLU()
        )

    def __str__(self):
        return f"IMGDE_{self.mode}_{version}"

    def forward(self, x):
        out = self.fclayers(x)
        out = self.cnn(out.view(-1, 512, 4, 4))
        return out.view(-1, 1, 128, 128)


class BBXDecoder(nn.Module):
    name = 'bbxde'

    def __init__(self):
        super(BBXDecoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            # self.active_func
        )

        init.xavier_normal_(self.fc[-2].weight)

    def __str__(self):
        return f"BBXDE{version}"

    def forward(self, x):
        out = self.fc(x.view(-1, 4 * 4 * 32))
        center = out[..., :4]
        depth = out[..., -1]
        return center, depth


class CSIEncoder(BasicCSIEncoder):
    def __init__(self, lstm_steps=steps, lstm_feature_length=feature_length, *args, **kwargs):
        super(CSIEncoder, self).__init__(lstm_feature_length=lstm_feature_length, *args, **kwargs)

        self.lstm_steps = lstm_steps

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

        self.cnn_aoa = nn.Sequential(
            nn.Conv1d(1, 128, 5, 1, 1),
            batchnorm_layer(128, self.batchnorm),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(128, 256, 3, 2, 1),
            batchnorm_layer(256, self.batchnorm),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(256, 512, 3, 2, 1),
            batchnorm_layer(512, self.batchnorm),
            nn.LeakyReLU(inplace=True)
        )

        self.cnn_tof = nn.Sequential(
            nn.Conv2d(1, 128, 5, 1, 1),
            batchnorm_layer(128, self.batchnorm),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            batchnorm_layer(256, self.batchnorm),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1),
            batchnorm_layer(512, self.batchnorm),
            nn.LeakyReLU(inplace=True)
        )

        self.lstm = nn.LSTM(self.lstm_feature_length, 128, 2, batch_first=True, dropout=0.1)

        self.fc_mu = nn.Sequential(
            nn.Linear(128, self.latent_dim)
            # nn.ReLU()
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(128, self.latent_dim)
            # nn.ReLU()
        )

    def __str__(self):
        return f"CSIEN{version}"

    def forward(self, csi, pd):
        fea_csi = self.cnn(csi)
        aoa = pd[..., 0, :]
        tof = pd[..., 1:, :]
        fea_aoa = self.cnn_aoa(aoa)
        fea_tof = self.cnn_tof(tof)

        features = torch.cat((fea_csi.view(-1, 512 * 7, self.lstm_steps),
                              fea_aoa.view(-1, 512 * 1, self.lstm_steps),
                              fea_tof.view(-1, 512 * 7, self.lstm_steps)), -2)
        features, (final_hidden_state, final_cell_state) = self.lstm.forward(
            features.transpose(1, 2))
        # 128-dim output
        out = features[:, -1, :]
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        z = reparameterize(mu, logvar)
        return z, mu, logvar, out


class TeacherTrainer(BasicTrainer):
    def __init__(self,
                 beta=0.5,
                 *args, **kwargs):
        super(TeacherTrainer, self).__init__(*args, **kwargs)

        self.modality = {'cimg', 'rimg', 'bbx', 'depth', 'tag', 'ind'}

        self.beta = beta
        self.rimg_loss = nn.MSELoss(reduction='sum')
        self.cimg_loss = nn.BCELoss(reduction='sum')
        self.feature_loss = nn.MSELoss(reduction='sum')

        self.loss_terms = ('LOSS', 'FEAT', 'LOSS_R', 'KL_R', 'RECON_R', 'LOSS_C', 'KL_C', 'RECON_C')
        self.pred_terms = ('GT_R', 'PRED_R', 'LAT_R', 'GT_C', 'PRED_C', 'LAT_C', 'TAG')
        self.loss = MyLoss(name=self.name,
                           loss_terms=self.loss_terms,
                           pred_terms=self.pred_terms)
        self.raw_weight = 1.
        self.cropped_weight = 1.
        self.feature_weight = 1.

    def vae_loss(self, pred, gt, mu, logvar):
        recon_loss = self.recon_lossfunc(pred, gt) / pred.shape[0]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss * self.beta
        return loss, kl_loss, recon_loss

    def calculate_loss(self, data):
        cimg_mask = torch.where(data['cimg'] > 0, 1., 0.)
        z_r, mu_r, logvar_r, feature_r = self.models['rimgen'](data['rimg'])
        z_c, mu_c, logvar_c, feature_c = self.models['cimgen'](cimg_mask)
        rimg = self.models['rimgde'](z_r)
        cimg = self.models['cimgde'](z_c)
        loss_r, kl_loss_r, recon_loss_r = self.vae_loss(rimg, data['rimg'], mu_r, logvar_r)
        loss_c, kl_loss_c, recon_loss_c = self.vae_loss(cimg, cimg_mask, mu_c, logvar_c)
        
        feature_loss = self.feature_loss(feature_r[:, :512, ...], feature_c)
        loss = loss_r * self.raw_weight + loss_c * self.cropped_weight + feature_loss * self.feature_weight

        self.temp_loss = {
            'LOSS'   : loss,
            'LOSS_R' : loss_r,
            'KL_R'   : kl_loss_r,
            'RECON_R': recon_loss_r,
            'LOSS_C' : loss_c,
            'KL_C'   : kl_loss_c,
            'RECON_C': recon_loss_c,
            'FEAT'   : feature_loss
                          }
        return {
            'GT_R'  : data['rimg'],
            'PRED_R': rimg,
            'LAT_R' : torch.cat((mu_r, logvar_r), -1),
            'GT_C'  : cimg_mask,
            'PRED_C': cimg,
            'LAT_C' : torch.cat((mu_c, logvar_c), -1),
            'TAG'   : data['tag']
                }

    def plot_test(self, select_ind=None, select_num=8, autosave=False, notion='', **kwargs):
        save_path = f'../saved/{notion}/'
        figs = []
        self.loss.generate_indices(select_ind, select_num)

        figs.append(self.loss.plot_predict(plot_terms=('GT_R', 'PRED_R')))
        figs.append(self.loss.plot_predict(plot_terms=('GT_C', 'PRED_C')))
        figs.append(self.loss.plot_latent(plot_terms={'LAT_R'}))
        figs.append(self.loss.plot_latent(plot_terms={'LAT_C'}))
        # figs.append(self.loss.plot_test(plot_terms='all'))
        # figs.append(self.loss.plot_tsne(plot_terms=('GT', 'LAT', 'PRED')))

        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for fig, filename in figs:
                fig.savefig(f"{save_path}{filename}")


# Student learns the latent
class StudentTrainer(BasicTrainer):
    def __init__(self,
                 alpha=0.8,
                 *args, **kwargs):
        super(StudentTrainer, self).__init__(*args, **kwargs)

        self.modality = {'rimg', 'csi', 'pd', 'tag'}

        self.alpha = alpha
        self.recon_lossfunc = nn.MSELoss()

        self.loss_terms = ('LOSS', 'MU', 'LOGVAR', 'IMG')
        self.pred_terms = ('GT', 'T_PRED', 'S_PRED',
                           'T_LATENT', 'S_LATENT',
                           'TAG')
        self.loss = MyLossCTR(name=self.name,
                              loss_terms=self.loss_terms,
                              pred_terms=self.pred_terms,
                              depth=True)

    def kd_loss(self, mu_s, logvar_s, mu_t, logvar_t):
        mu_loss = self.recon_lossfunc(mu_s, mu_t) / mu_s.shape[0]
        logvar_loss = self.recon_lossfunc(logvar_s, logvar_t) / logvar_s.shape[0]
        loss = self.alpha * mu_loss + (1 - self.alpha) * logvar_loss
        return loss, mu_loss, logvar_loss

    def calculate_loss(self, data):
        s_z, s_mu, s_logvar = self.models['csien'](csi=data['csi'], pd=data['pd'])

        with torch.no_grad():
            s_image = self.models['imgde'](s_z)
            t_z, t_mu, t_logvar = self.models['rimgen'](data['rimg'])
            t_image = self.models['imgde'](t_z)
            img_loss = self.recon_lossfunc(s_image, t_image)

        latent_loss, mu_loss, logvar_loss = self.kd_loss(s_mu, s_logvar, t_mu, t_logvar)
        loss = latent_loss

        self.temp_loss = {
            'LOSS'  : loss,
            'MU'    : mu_loss,
            'LOGVAR': logvar_loss,
            'IMG'   : img_loss
            }
        
        return {
            'GT'      : data['rimg'],
            'T_LATENT': torch.cat((t_mu, t_logvar), -1),
            'S_LATENT': torch.cat((s_mu, s_logvar), -1),
            'T_PRED'  : t_image,
            'S_PRED'  : s_image,
            'TAG'     : data['tag']
                }

    def plot_test(self, select_ind=None, select_num=8, autosave=False, notion='', **kwargs):
        save_path = f'../saved/{notion}/'
        figs = []
        self.loss.generate_indices(select_ind, select_num)

        figs.append(self.loss.plot_predict(plot_terms=('GT', 'T_PRED', 'S_PRED')))
        figs.append(self.loss.plot_latent(plot_terms=('T_LATENT', 'S_LATENT')))
        figs.append(self.loss.plot_center())
        figs.append(self.loss.plot_test(plot_terms='all'))
        figs.append(self.loss.plot_test_cdf(plot_terms='all'))
        #figs.append(self.loss.plot_tsne(plot_terms=('GT', 'T_LATENT', 'S_LATENT')))

        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for fig, filename in figs:
                fig.savefig(f"{save_path}{filename}")


if __name__ == '__main__':
    cc = ImageEncoder().to(torch.device('cuda:5'))
    summary(cc, input_size=(1, 1, 128, 128))
