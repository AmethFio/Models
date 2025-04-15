import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.ops import complete_box_iou_loss
import numpy as np
import matplotlib.pyplot as plt
import os
from Trainer import BasicTrainer
from Models.Structure.Model import *
from Loss import MyLossLog, MyLossBBX2

version = 'V09C1'

##############################################################################
# -------------------------------------------------------------------------- #
# Version V09C1
# Teacher1 learns raw images and estimates raw images and bbx and depth
# Teacher2 learns cropped images and estimates binary masks
# 
# Student learns (6, 30, m) CSIs and (2, 30, m) PhaseDiffs and estimates latent
#
# rawImageEncoder: in = 128 * 128,
#               out = [z:latent_dim, mu:latent_dim, logvar:latent_dim, features: 512 * 4 * 4]
# rawImageDecoder: in = 1 * latent_dim,
#               out = 128 * 128
# croppedImageEncoder: in = 128 * 128,
#               out = [z:latent_dim, mu:latent_dim, logvar:latent_dim, features: 512 * 4 * 4]
# croppedmageDecoder: in = 1 * latent_dim,
#               out = 128 * 128
# CSIEncoder: in = [6 * 30 * m], [2 * 30 * m]
#               out = [z:latent_dim, mu:latent_dim, logvar:latent_dim]
# BBXDecoder: in = 512 * 4 * 4,
#               out = [BBX:4, depth:1]
# -------------------------------------------------------------------------- #
##############################################################################

feature_length = 512 * 15
steps = 25


class ImageEncoder(BasicImageEncoder):
    def __init__(self, mode='raw', *args, **kwargs):
        super(ImageEncoder, self).__init__(*args, **kwargs)
        self.mode = mode
        self.last_nodes = 512
        
        channels = [1, 128, 128, 256, 256, self.last_nodes]
        block = []
        for i in range(len(channels) - 1):
            block.extend([nn.Conv2d(channels[i], channels[i+1], 3, 2, 1),
                          batchnorm_layer(channels[i+1], self.batchnorm),
                          nn.LeakyReLU(inplace=True)])
            block.extend([nn.Conv2d(512, 512, 3, 1, 1),
                          batchnorm_layer(512, self.batchnorm),
                          nn.LeakyReLU(inplace=True)])
        self.cnn = nn.Sequential(*block)

        # 1 * 128 * 128
        # 128 * 64 * 64
        # 128 * 32 * 32
        # 256 * 16 * 16
        # 256 * 8 * 8
        # 512 * 4 * 4
        # 512 * 4 * 4

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

        return z, mu, logvar, features


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
            nn.Linear(4 * 4 * 512, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 5),
            nn.Sigmoid()
        )

        init.xavier_normal_(self.fc[-2].weight)

    def __str__(self):
        return f"BBXDE{version}"

    def forward(self, x):
        out = self.fc(x.view(-1, 4 * 4 * 544))
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
        return z, mu, logvar


class TeacherTrainer(BasicTrainer):
    def __init__(self,
                 beta=0.5,
                 *args, **kwargs):
        super(TeacherTrainer, self).__init__(*args, **kwargs)

        self.modality = {'cimg', 'rimg', 'bbx', 'depth', 'tag', 'ind'}

        self.beta = beta
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCELoss(reduction='sum')
        self.feature_loss = nn.MSELoss(reduction='sum')
        self.depth_loss = nn.MSELoss(reduction='sum')

        self.loss_terms = ('LOSS', 'FEAT', 'LOSS_R', 'KL_R', 'RECON_R', 'LOSS_C', 'KL_C', 'RECON_C')
        self.pred_terms = ('GT_R', 'PRED_R', 'LAT_R', 'GT_C', 'PRED_C', 'LAT_C', 'TAG')
        self.losslog = MyLossBBX2(name=self.name,
                           loss_terms=self.loss_terms,
                           pred_terms=self.pred_terms)
        self.raw_weight = 1.
        self.cropped_weight = 1.
        self.feature_weight = 1.
        self.bbx_weight = 1.
        self.depth_weight = 1.

    @staticmethod
    def bbx_loss(bbx1, bbx2):
        # --- x, y, w, h to x1, y1, x2, y2 ---
        # Done in datasetting
        return complete_box_iou_loss(bbx1, bbx2, reduction='sum')

    def rimg_loss(self, pred, gt, mu, logvar):
        recon_loss = self.mse(pred, gt) / pred.shape[0]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss * self.beta
        return loss, kl_loss, recon_loss
    
    def cimg_loss(self, pred, gt, mu, logvar):
        recon_loss = self.bce(pred, gt) / pred.shape[0]
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
        
        feature_loss = self.feature_loss(feature_r, feature_c)

        t_bbx, t_depth = self.models['bbxde'](feature_r)
        bbx_loss = self.bbx_loss(t_bbx, torch.squeeze(data['bbx']))
        depth_loss = self.depth_loss(t_depth, torch.squeeze(data['depth']))

        loss = loss_r * self.raw_weight + \
            loss_c * self.cropped_weight + \
            bbx_loss * self.bbx_weight +\
            depth_loss * self.depth_weight +\
            feature_loss * self.feature_weight

        self.temp_loss = {
            'LOSS'   : loss,
            'LOSS_R' : loss_r,
            'KL_R'   : kl_loss_r,
            'RECON_R': recon_loss_r,
            'LOSS_C' : loss_c,
            'KL_C'   : kl_loss_c,
            'RECON_C': recon_loss_c,
            'FEAT'   : feature_loss,
            'BBX'    : bbx_loss,
            'DPT'    : depth_loss
                          }
        return {
            'GT_R'  : data['rimg'],
            'PRED_R': rimg,
            'LAT_R' : torch.cat((mu_r, logvar_r), -1),
            'GT_C'  : cimg_mask,
            'PRED_C': cimg,
            'LAT_C' : torch.cat((mu_c, logvar_c), -1),
            'GT_BBX': data['bbx'],
            'T_BBX' : t_bbx,
            'GT_DPT': data['dpt'],
            'T_DPT' : t_depth,
            'TAG'   : data['tag']
                }

    def plot_test(self, select_ind=None, select_num=8, autosave=False, notion='', **kwargs):
        save_path = f'../saved/{notion}/'
        figs: dict = {}
        self.loss.generate_indices(select_ind, select_num)

        figs.update(self.losslog.plot_predict(plot_terms=('GT_R', 'PRED_R')))
        figs.update(self.losslog.plot_predict(plot_terms=('GT_C', 'PRED_C')))
        figs.update(self.losslog.plot_bbx())
        figs.update(self.losslog.plot_latent(plot_terms={'LAT_R'}))
        figs.update(self.losslog.plot_latent(plot_terms={'LAT_C'}))
        # figs.update(self.loss.plot_test(plot_terms='all'))
        # figs.update(self.loss.plot_tsne(plot_terms=('GT', 'LAT', 'PRED')))

        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for filename, fig in figs.items():
                fig.savefig(f"{save_path}{filename}")


# Student learns the latent
class StudentTrainer(BasicTrainer):
    def __init__(self,
                 alpha=0.8,
                 with_img_loss=False,
                 *args, **kwargs):
        super(StudentTrainer, self).__init__(*args, **kwargs)

        self.modality = {'rimg', 'cimg', 'csi', 'pd', 'tag'}

        self.alpha = alpha
        self.recon_lossfunc = nn.MSELoss()
        self.with_img_loss = with_img_loss

        self.loss_terms = ('LOSS', 'MU', 'LOGVAR', 'IMG')
        self.pred_terms = ('GT', 'T_PRED_R', 'T_PRED_C', 'S_PRED',
                           'T_LATENT', 'S_LATENT',
                           'TAG')
        self.loss = MyLossLog(name=self.name,
                              loss_terms=self.loss_terms,
                              pred_terms=self.pred_terms,
                              depth=True)
        self.img_weight = 1.
        self.kd_weight = 1.

    def kd_loss(self, mu_s, logvar_s, mu_t, logvar_t):
        mu_loss = self.recon_lossfunc(mu_s, mu_t) / mu_s.shape[0]
        logvar_loss = self.recon_lossfunc(logvar_s, logvar_t) / logvar_s.shape[0]
        loss = self.alpha * mu_loss + (1 - self.alpha) * logvar_loss
        return loss, mu_loss, logvar_loss

    def calculate_loss(self, data):
        cimg_mask = torch.where(data['cimg'] > 0, 1., 0.)
        s_z, s_mu, s_logvar = self.models['csien'](csi=data['csi'], pd=data['pd'])

        with torch.no_grad():
            s_image = self.models['rimgde'](s_z)
            tr_z, tr_mu, tr_logvar = self.models['rimgen'](data['rimg'])
            tc_z, tc_mu, tc_logvar = self.models['cimgen'](cimg_mask)
            tr_image = self.models['rimgde'](tr_z)
            tc_image = self.models['cimgde'](tc_z)

        latent_loss, mu_loss, logvar_loss = self.kd_loss(s_mu, s_logvar, tr_mu, tr_logvar)
        
        if self.with_img_loss:
            img_loss = self.recon_lossfunc(s_image, tr_image)
            loss = self.kd_weight * latent_loss + self.img_weight * img_loss
        else:
            with torch.no_grad():
                img_loss = self.recon_lossfunc(s_image, tr_image)
            loss = latent_loss

        self.temp_loss = {
            'LOSS'  : loss,
            'MU'    : mu_loss,
            'LOGVAR': logvar_loss,
            'IMG'   : img_loss
            }
        
        return {
            'GT'        : data['rimg'],
            'T_LATENT_R': torch.cat((tr_mu, tr_logvar), -1),
            'S_LATENT_R': torch.cat((s_mu, s_logvar), -1),
            'T_PRED_R'  : tr_image,
            'T_PRED_C'  : tc_image,
            'S_PRED'    : s_image,
            'TAG'       : data['tag']
                }

    def plot_test(self, select_ind=None, select_num=8, autosave=False, notion='', **kwargs):
        save_path = f'../saved/{notion}/'
        figs: dict = {}
        self.loss.generate_indices(select_ind, select_num)

        figs.update(self.loss.plot_predict(plot_terms=('GT', 'T_PRED_R', 'T_PRED_C', 'S_PRED')))
        figs.update(self.loss.plot_latent(plot_terms=('T_LATENT_R', 'S_LATENT')))
        figs.update(self.loss.plot_test_cdf(plot_terms='all'))

        if autosave:
            for filename, fig in figs.items():
                fig.savefig(f"{save_path}{filename}")


if __name__ == '__main__':
    cc = ImageEncoder().to(torch.device('cuda:6'))
    summary(cc, input_size=(1, 1, 128, 128))
