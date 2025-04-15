import torch
import torch.nn as nn
import torch.nn.init as init
# from torchvision.ops import complete_box_iou_loss
import numpy as np
import matplotlib.pyplot as plt
import os
from Trainer import BasicTrainer
from Models.Structure.Model import *
from Loss import MyLossLog, MyLossCTR

version = 'V08F3_notx'

##############################################################################
# -------------------------------------------------------------------------- #
# Version V08F3
# Teacher learns and estimates cropped images
# Student learns (6, 30, m) CSIs and (62) filtered PhaseDiffs
# A new branch for learning median-filtered PhaseDiff
# Student adopts whole image loss
# Student adopts 1 / size as the weight of image loss
# Increased num of channels and latent dimensions
# Applied feature loss to CSIEncoder
#
# ImageEncoder: in = 128 * 128,
#               out = [z:latent_dim, mu:latent_dim, logvar:latent_dim]
# ImageDecoder: in = 1 * latent_dim,
#               out = 128 * 128
# CSIEncoder: in = [6 * 30 * m], [62]
#               out = [out:256, z:latent_dim, mu:latent_dim, logvar:latent_dim]
# CenterSDecoder: in = 256,
#               out = [center:2, depth:1]
# -------------------------------------------------------------------------- #
##############################################################################

feature_length = 512 * 7
steps = 75

class ImageDecoder(BasicImageDecoder):
    def __init__(self, *args, **kwargs):
        super(ImageDecoder, self).__init__(*args, **kwargs)

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


class CSIEncoder(BasicCSIEncoder):
    def __init__(self, lstm_steps=steps, lstm_feature_length=feature_length, *args, **kwargs):
        super(CSIEncoder, self).__init__(lstm_feature_length=lstm_feature_length, *args, **kwargs)

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

        self.lstm = nn.LSTM(self.lstm_feature_length, self.csi_feature_length, 2, batch_first=True, dropout=0.1)
        
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
            fea_csi.view(-1, 512 * 7, self.lstm_steps).transpose(1, 2))
        # 256-dim output
        out = torch.cat((features[:, -1, :].view(-1, self.csi_feature_length), fea_pd.view(-1, self.pd_feature_length)), -1)
        out = self.fc_feature(out)
        
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        z = reparameterize(mu, logvar)
        return z, mu, logvar, out


class AbTrainer(BasicTrainer):
    def __init__(self,
                 alpha=0.8,
                 recon_lossfunc=nn.MSELoss(),
                 lstm_steps=75,
                 *args, **kwargs):
        super(AbTrainer, self).__init__(*args, **kwargs)

        self.modality = {'cimg', 'rimg', 'csi', 'center', 'depth', 'pd', 'tag', 'ctr', 'dpt', 'ind'}

        self.alpha = alpha
        self.recon_lossfunc = recon_lossfunc
        self.mse = nn.MSELoss(reduction='sum')
        self.img_loss = nn.BCEWithLogitsLoss(reduction='sum')

        self.loss_terms = ('LOSS', 'KL', 'RIMG', 'CIMG', 'CTR', 'DPT')
        self.pred_terms = ('C_GT', 'R_GT',
                           'R_PRED',
                           'C_PRED',
                           'LATENT',
                           'GT_CTR', 'GT_DPT', 
                           'CTR', 'DPT',
                           'TAG', 'IND')
        self.losslog = MyLossCTR(name=self.name,
                              loss_terms=self.loss_terms,
                              pred_terms=self.pred_terms,
                              depth=True)
        
        self.losslog.ctr = ['GT_CTR', 'CTR']
        self.losslog.dpt = ['GT_DPT', 'DPT']
        
        self.models = {
            'cimgde': ImageDecoder(latent_dim=128).to(self.device),
            'rimgde': ImageDecoder(latent_dim=128).to(self.device),
            'csien' : CSIEncoder(latent_dim=128, lstm_steps=lstm_steps).to(self.device),
            'ctrde': CenterDecoder().to(self.device)
                }

        self.latent_weight = 0.1
        self.img_weight = 1.
        self.center_weight = 1.
        self.depth_weight = 1.
        self.feature_weight = 1.
        
    def kl_loss(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss

    def calculate_loss(self, data):
        cimg = torch.where(data['cimg'] > 0, 1., 0.)
        rimg = data['rimg']
        z, mu, logvar, feature = self.models['csien'](csi=data['csi'], pd=data['pd'])
        
        center, depth = self.models['ctrde'](feature)
        cimage = self.models['cimgde'](z)
        rimage = self.models['rimgde'](z)
        
        # Mix loss
        kl_loss = self.kl_loss(mu, logvar)
        center_loss = self.recon_lossfunc(center, torch.squeeze(data['center']))
        depth_loss = self.recon_lossfunc(depth, torch.squeeze(data['depth']))
        rimage_loss = self.mse(rimage, rimg) / rimage.shape[0]
        cimage_loss = self.img_loss(cimage, cimg) / cimage.shape[0]
        
        loss = kl_loss * self.latent_weight +\
            rimage_loss * self.img_weight +\
            cimage_loss * self.img_weight +\
            center_loss * self.center_weight +\
            depth_loss * self.depth_weight


        self.temp_loss = {'LOSS': loss,
                          'KL': kl_loss,
                          'RIMG': rimage_loss,
                          'CIMG': cimage_loss,
                          'CTR': center_loss,
                          'DPT': depth_loss
                          }
        return {'R_GT': rimg,
                'C_GT': cimg,
                'LATENT': z,
                'R_PRED': rimage,
                'C_PRED': cimage,
                'GT_CTR': data['center'],
                'CTR': center,
                'GT_DPT': data['depth'],
                'DPT': depth,
                'TAG': data['tag'],
                'IND': data['ind']}

    def plot_test(self, select_ind=None, select_num=8, autosave=False, **kwargs):
        figs: dict = {}
        self.losslog.generate_indices(select_ind, select_num)

        figs.update(self.losslog.plot_predict(plot_terms=('R_GT', 'R_PRED'), title='RIMG_PRED'))
        figs.update(self.losslog.plot_predict(plot_terms=('C_GT', 'C_PRED'), title='CIMG_PRED'))
        figs.update(self.losslog.plot_latent(plot_terms=({'LATENT'})))
        figs.update(self.losslog.plot_center())
        # figs.update(self.losslog.plot_test_cdf(plot_terms='all'))
        #figs.update(self.losslog.plot_tsne(plot_terms=('GT', 'T_LATENT', 'S_LATENT')))

        if autosave:
            for filename, fig in figs.items():
                fig.savefig(f"{self.save_path}{filename}")


if __name__ == '__main__':
    cc = CSIEncoder(latent_dim=128).to(torch.device('cuda:6'))
    summary(cc, input_size=((64, 6, 30, 300), (64, 62)))
    