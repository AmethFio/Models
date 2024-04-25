import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.ops import complete_box_iou_loss
import numpy as np
import matplotlib.pyplot as plt
import os
from Trainer import BasicTrainer
from Model import *
from Loss import MyLoss, MyLossCTR

version = 'V07C1'

##############################################################################
# -------------------------------------------------------------------------- #
# Version V07C1
# Teacher learns and estimates binary masks and the depth value of the center
# pixel of the cropped depth image
# Student learns (6, 30, m) CSIs and (2, 30, m) PhaseDiffs

# ImageEncoder: in = 128 * 128,
#               out = [z:latent_dim, mu:latent_dim, logvar:latent_dim]
# ImageDecoder: in = 1 * latent_dim,
#               out = 128 * 128
# CSIEncoder: in = [6 * 30 * m], [2 * 30 * m]
#               out = [z:latent_dim, mu:latent_dim, logvar:latent_dim]
# DepthDecoder: in = 128,
#               out = [center:2, depth:1]
# -------------------------------------------------------------------------- #
##############################################################################

feature_length = 512 * 15
steps = 25


class ImageEncoder(BasicImageEncoder):
    def __init__(self, *args, **kwargs):
        super(ImageEncoder, self).__init__(*args, **kwargs)

        channels = [1, 128, 128, 256, 256, 512]
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
        # 512 * 4 * 4

        self.fc_mu = nn.Sequential(
            nn.Linear(4 * 4 * 512, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.latent_dim),
            # self.active_func
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(4 * 4 * 512, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.latent_dim),
            # self.active_func
        )

    def __str__(self):
        return f"IMGEN{version}"

    def forward(self, x):
        out = self.cnn(x)
        mu = self.fc_mu(out.view(-1, 4 * 4 * 512))
        logvar = self.fc_logvar(out.view(-1, 4 * 4 * 512))
        z = reparameterize(mu, logvar)

        return z, mu, logvar


class ImageDecoder(BasicImageDecoder):
    def __init__(self, *args, **kwargs):
        super(ImageDecoder, self).__init__(*args, **kwargs)

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
        return f"IMGDE{version}"

    def forward(self, x):
        out = self.fclayers(x)
        out = self.cnn(out.view(-1, 512, 4, 4))
        return out.view(-1, 1, 128, 128)


class CenterDecoder(nn.Module):
    name = 'ctrde'

    def __init__(self):
        super(CenterDecoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )

        init.xavier_normal_(self.fc[-2].weight)

    def __str__(self):
        return f"CTRDE{version}"

    def forward(self, x):
        out = self.fc(x.view(-1, 128))
        center = out[..., :2]
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
        return out, z, mu, logvar


class TeacherTrainer(BasicTrainer):
    def __init__(self,
                 beta=0.5,
                 mask=False,
                 recon_lossfunc=nn.BCELoss(reduction='sum'),
                 *args, **kwargs):
        super(TeacherTrainer, self).__init__(*args, **kwargs)

        self.modality = {'img'}

        self.beta = beta
        self.recon_lossfunc = recon_lossfunc
        self.mask = mask

        self.loss_terms = ('LOSS', 'KL', 'RECON')
        self.pred_terms = ('GT', 'PRED', 'LAT', 'IND')
        self.loss = MyLoss(name=self.name,
                           loss_terms=self.loss_terms,
                           pred_terms=self.pred_terms)

    def vae_loss(self, pred, gt, mu, logvar):
        recon_loss = self.recon_lossfunc(pred, gt) / pred.shape[0]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss * self.beta
        return loss, kl_loss, recon_loss

    def calculate_loss(self, data):
        img = torch.where(data['img'] > 0, 1., 0.) if self.mask else data['img']
        z, mu, logvar = self.models['imgen'](img)
        output = self.models['imgde'](z)
        loss, kl_loss, recon_loss = self.vae_loss(output, img, mu, logvar)

        self.temp_loss = {'LOSS': loss,
                          'KL': kl_loss,
                          'RECON': recon_loss
                          }
        return {'GT': img,
                'PRED': output,
                'LAT': torch.cat((mu, logvar), -1),
                'IND': data['ind']
                }

    def plot_test(self, select_ind=None, select_num=8, autosave=False, notion='', **kwargs):
        save_path = f'../saved/{notion}/'
        figs = []
        self.loss.generate_indices(select_ind, select_num)

        figs.append(self.loss.plot_predict(plot_terms=('GT', 'PRED')))
        figs.append(self.loss.plot_latent(plot_terms={'LAT'}))
        # figs.append(self.loss.plot_test(plot_terms='all'))
        # figs.append(self.loss.plot_tsne(plot_terms=('GT', 'LAT', 'PRED')))

        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for fig, filename in figs:
                fig.savefig(f"{save_path}{notion}_{filename}")


# Student is Mask + Center + Depth
class StudentTrainer(BasicTrainer):
    def __init__(self,
                 alpha=0.8,
                 mask=True,
                 recon_lossfunc=nn.MSELoss(),
                 *args, **kwargs):
        super(StudentTrainer, self).__init__(*args, **kwargs)

        self.modality = {'img', 'csi', 'ctr', 'dpt', 'pd'}

        self.alpha = alpha
        self.recon_lossfunc = recon_lossfunc
        self.depth_loss = nn.MSELoss()
        self.center_loss = nn.MSELoss()
        self.mask = mask

        self.loss_terms = ('LOSS', 'MU', 'LOGVAR', 'CTR', 'DPT', 'IMG')
        self.pred_terms = ('GT', 'T_PRED', 'S_PRED',
                           'T_LATENT', 'S_LATENT',
                           'GT_CTR', 'S_CTR',
                           'GT_DPT', 'S_DPT',
                           'IND')
        self.loss = MyLossCTR(name=self.name,
                              loss_terms=self.loss_terms,
                              pred_terms=self.pred_terms,
                              depth=True)

        self.latent_weight = 1.
        self.img_weight = 0
        self.center_weight = 100.
        self.depth_weight = 1000.

    def kd_loss(self, mu_s, logvar_s, mu_t, logvar_t):
        mu_loss = self.recon_lossfunc(mu_s, mu_t) / mu_s.shape[0]
        logvar_loss = self.recon_lossfunc(logvar_s, logvar_t) / logvar_s.shape[0]
        loss = self.alpha * mu_loss + (1 - self.alpha) * logvar_loss
        return loss, mu_loss, logvar_loss

    def calculate_loss(self, data):
        img = torch.where(data['img'] > 0, 1., 0.) if self.mask else data['img']
        features, s_z, s_mu, s_logvar = self.models['csien'](csi=data['csi'], pd=data['pd'])
        s_image = self.models['imgde'](s_z)
        s_ctr, s_depth = self.models['ctrde'](features)

        with torch.no_grad():
            t_z, t_mu, t_logvar = self.models['imgen'](img)
            t_image = self.models['imgde'](t_z)
            image_loss = self.recon_lossfunc(s_image, img)

        center_loss = self.center_loss(s_ctr, torch.squeeze(data['ctr']))
        depth_loss = self.depth_loss(s_depth, torch.squeeze(data['dpt']))
        latent_loss, mu_loss, logvar_loss = self.kd_loss(s_mu, s_logvar, t_mu, t_logvar)

        loss = image_loss * self.img_weight + \
               center_loss * self.center_weight + \
               depth_loss * self.depth_weight + \
               latent_loss * self.latent_weight

        self.temp_loss = {'LOSS': loss,
                          'MU': mu_loss,
                          'LOGVAR': logvar_loss,
                          'IMG': image_loss,
                          'CTR': center_loss,
                          'DPT': depth_loss}
        return {'GT': img,
                'T_LATENT': torch.cat((t_mu, t_logvar), -1),
                'S_LATENT': torch.cat((s_mu, s_logvar), -1),
                'T_PRED': t_image,
                'S_PRED': s_image,
                'GT_CTR': data['ctr'],
                'S_CTR': s_ctr,
                'GT_DPT': data['dpt'],
                'S_DPT': s_depth,
                'IND': data['ind']}

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
                fig.savefig(f"{save_path}{notion}_{filename}")


if __name__ == '__main__':
    cc = CSIEncoder()
    summary(cc, input_size=[(6, 30, 100), (1, 30, 100)])
