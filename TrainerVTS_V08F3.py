import torch
import torch.nn as nn
import torch.nn.init as init
# from torchvision.ops import complete_box_iou_loss
import numpy as np
import matplotlib.pyplot as plt
import os
from Trainer import BasicTrainer, TrainingPhase, ValidationPhase
from Model import *
from Loss import MyLossLog, MyLossCTR

version = 'V08F3'

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
steps = 25


class ImageEncoder(BasicImageEncoder):
    def __init__(self, *args, **kwargs):
        super(ImageEncoder, self).__init__(*args, **kwargs)

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
        return out, z, mu, logvar


class TeacherTrainer(BasicTrainer):
    def __init__(self,
                 beta=0.5,
                 recon_lossfunc=nn.BCELoss(reduction='sum'),
                 *args, **kwargs):
        super(TeacherTrainer, self).__init__(*args, **kwargs)

        self.modality = {'rimg', 'cimg', 'center', 'depth', 'tag', 'ctr', 'dpt', 'ind'}

        self.beta = beta
        self.recon_lossfunc = recon_lossfunc

        self.loss_terms = ('LOSS', 'KL', 'R_RECON', 'C_RECON', 'CTR', 'DPT')
        self.pred_terms = ('R_GT', 'C_GT', 
                           'GT_DPT', 'GT_CTR', 
                           'R_PRED', 'C_PRED', 
                           'DPT_PRED', 'CTR_PRED', 
                           'LAT', 'TAG', 'IND')
        self.depth_loss = nn.MSELoss()
        self.center_loss = nn.MSELoss()
        
        self.losslog = MyLossCTR(name=self.name,
                           loss_terms=self.loss_terms,
                           pred_terms=self.pred_terms,
                           depth=True)
        self.losslog.ctr = ['GT_CTR', 'CTR_PRED']
        self.losslog.dpt = ['GT_DPT', 'DPT_PRED']
        
        self.models = {'imgen': ImageEncoder(latent_dim=128).to(self.device),
                       'cimgde': ImageDecoder(latent_dim=128).to(self.device),
                       'rimgde': ImageDecoder(latent_dim=128).to(self.device),
                       'ctrde': CenterDecoder().to(self.device)
                       }
        
    def kl_loss(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss

    def calculate_loss(self, data):
        cimg = torch.where(data['cimg'] > 0, 1., 0.)
        rimg = data['rimg']
        
        z, mu, logvar, feature = self.models['imgen'](rimg)
        rimg_re = self.models['rimgde'](z)
        cimg_re = self.models['cimgde'](z)
        kl_loss = self.kl_loss(mu, logvar)
        r_recon_loss = self.recon_lossfunc(rimg_re, rimg) / rimg_re.shape[0]
        c_recon_loss = self.recon_lossfunc(cimg_re, cimg) / cimg_re.shape[0]
        vae_loss = kl_loss * self.beta + r_recon_loss + c_recon_loss
        
        ctr, depth = self.models['ctrde'](feature)
        center_loss = self.center_loss(ctr, torch.squeeze(data['center']))
        depth_loss = self.depth_loss(depth, torch.squeeze(data['depth']))
        
        loss = vae_loss + center_loss + depth_loss

        self.temp_loss = {'LOSS': loss,
                          'KL': kl_loss,
                          'R_RECON': r_recon_loss,
                          'C_RECON': c_recon_loss,
                          'CTR': center_loss, 
                          'DPT': depth_loss
                          }
        
        return {'R_GT': rimg,
                'C_GT': cimg,
                'R_PRED': rimg_re,
                'C_PRED': cimg_re,
                'GT_CTR': data['center'],
                'CTR_PRED': ctr,
                'GT_DPT': data['depth'],
                'DPT_PRED': depth,
                'LAT': torch.cat((mu, logvar), -1),
                'TAG': data['tag'],
                'IND': data['ind']
                }

    def plot_test(self, select_ind=None, select_num=8, autosave=False, **kwargs):
        figs: dict = {}
        self.losslog.generate_indices(select_ind, select_num)

        figs.update(self.losslog.plot_predict(plot_terms=('R_GT', 'R_PRED', 'C_GT', 'C_PRED')))
        figs.update(self.losslog.plot_latent(plot_terms={'LAT'}))
        figs.update(self.losslog.plot_center())
        # figs.update(self.loss.plot_test(plot_terms='all'))
        # figs.update(self.loss.plot_tsne(plot_terms=('GT', 'LAT', 'PRED')))

        if autosave:
            for filename, fig in figs.items():
                fig.savefig(f"{self.save_path}{filename}")


class StudentTrainer(BasicTrainer):
    def __init__(self,
                 alpha=0.8,
                 recon_lossfunc=nn.MSELoss(),
                 with_cimg_loss=False,
                 lstm_steps=7,
                 *args, **kwargs):
        super(StudentTrainer, self).__init__(*args, **kwargs)

        self.modality = {'cimg', 'rimg', 'csi', 'center', 'depth', 'pd', 'tag', 'ctr', 'dpt', 'ind'}

        self.alpha = alpha
        self.recon_lossfunc = recon_lossfunc
        self.with_cimg_loss = with_cimg_loss

        self.loss_terms = ('LOSS', 'LATENT', 'MU', 'LOGVAR', 'FEATURE', 'IMG', 'CTR', 'DPT')
        self.pred_terms = ('C_GT', 'R_GT',
                           'TR_PRED', 'R_PRED',
                           'TC_PRED', 'SC_PRED',
                           'T_LATENT', 'S_LATENT',
                           'GT_CTR', 'GT_DPT', 
                           'T_CTR', 'T_DPT',
                           'S_CTR', 'S_DPT',
                           'TAG', 'IND')
        self.losslog = MyLossCTR(name=self.name,
                              loss_terms=self.loss_terms,
                              pred_terms=self.pred_terms,
                              depth=True)
        
        self.losslog.ctr = ['GT_CTR', 'T_CTR', 'S_CTR']
        self.losslog.dpt = ['GT_DPT', 'T_DPT', 'S_DPT']
        
        self.models = {
            'imgen' : ImageEncoder(latent_dim=128).to(self.device),
            'cimgde': ImageDecoder(latent_dim=128).to(self.device),
            'rimgde': ImageDecoder(latent_dim=128).to(self.device),
            'csien' : CSIEncoder(latent_dim=128, lstm_steps=lstm_steps).to(self.device),
            'ctrde': CenterDecoder().to(self.device)
                }

        self.training_phases = {'Feature_extractor': TrainingPhase(name = 'main',
                                                            train_module = ['csien'],
                                                            eval_module = ['imgen', 'rimgde', 'cimgde', 'ctrde'],
                                                            verbose=False
                                                            )}
        
        self.valid_phases = {
            'source': ValidationPhase(name='source', loader='valid'),
            'target': ValidationPhase(name='target', loader='valid2')
        }
        
        self.early_stopping_trigger = 'target'
        
        self.latent_weight = 0.1
        self.img_weight = 1.
        self.center_weight = 1.
        self.depth_weight = 1.
        self.feature_weight = 1.

    def kd_loss(self, mu_s, logvar_s, mu_t, logvar_t):
        mu_loss = self.recon_lossfunc(mu_s, mu_t) / mu_s.shape[0]
        logvar_loss = self.recon_lossfunc(logvar_s, logvar_t) / logvar_s.shape[0]
        latent_loss = self.alpha * mu_loss + (1 - self.alpha) * logvar_loss

        return latent_loss, mu_loss, logvar_loss
    
    def feature_loss(self, feature_s, feature_t):
        feature_loss = self.recon_lossfunc(feature_s, feature_t)
        return feature_loss
    
    def img_loss(self, cimg, center, depth, rimg):
        #recon_img = torch.zeros_like(rimg).to(self.device)
        x = center[..., 0]
        y = center[..., 1]
        x = (x * 226).to(torch.int) - 113
        y = (y * 128).to(torch.int) - 64
        recon_img = nn.functional.pad(cimg, (49, 49, 0, 0), 'constant', 0)
        recon_img *= depth.view(-1, 1, 1, 1)
        # Not in GPU?
        weight = torch.zeros_like(depth, dtype=float)
        for i in range(recon_img.shape[0]):
            recon_img[i] = torch.roll(recon_img[i], (y[i].item(), x[i].item()), dims=(-2, -1))
            weight[i] = 1. / torch.nonzero(recon_img[i]).shape[0]
        weight = torch.sqrt(weight).view(-1, 1, 1, 1)
        # Apply weight before MSE (non-feasible after MSE)
        loss = self.recon_lossfunc(recon_img * weight, rimg * weight)
        return loss

    def calculate_loss(self, data):
        cimg = torch.where(data['cimg'] > 0, 1., 0.)
        rimg = data['rimg']
        s_feature, s_z, s_mu, s_logvar = self.models['csien'](csi=data['csi'], pd=data['pd'])
        s_center, s_depth = self.models['ctrde'](s_feature)
        s_cimage = self.models['cimgde'](s_z)
        s_rimage = self.models['rimgde'](s_z)

        # Enable / Disable grad from img_loss
        with torch.no_grad():
            t_z, t_mu, t_logvar, t_feature = self.models['imgen'](rimg)
            t_cimage = self.models['cimgde'](t_z)
            t_rimage = self.models['rimgde'](t_z)
            t_center, t_depth = self.models['ctrde'](t_feature)
        
        # 3-level loss
        feature_loss = self.feature_loss(s_feature, t_feature)
       
        latent_loss, mu_loss, logvar_loss = self.kd_loss(s_mu, s_logvar, t_mu, t_logvar)
       
        center_loss = self.recon_lossfunc(s_center, torch.squeeze(data['center']))
        depth_loss = self.recon_lossfunc(s_depth, torch.squeeze(data['depth']))
        image_loss = self.recon_lossfunc(s_rimage, rimg)
        if self.with_cimg_loss:
            image_loss += self.recon_lossfunc(s_cimage, cimg)
        
        loss = feature_loss * self.feature_weight +\
            latent_loss * self.latent_weight +\
            image_loss * self.img_weight +\
            center_loss * self.center_weight +\
            depth_loss * self.depth_weight
        
        TMP_LOSS = {'LOSS': loss,
                          'LATENT': latent_loss,
                          'MU': mu_loss,
                          'LOGVAR': logvar_loss,
                          'FEATURE': feature_loss,
                          'IMG': image_loss,
                          'CTR': center_loss,
                          'DPT': depth_loss
                          }
        
        PREDS = {'R_GT': rimg,
                'C_GT': cimg,
                'T_LATENT': torch.cat((t_mu, t_logvar), -1),
                'S_LATENT': torch.cat((s_mu, s_logvar), -1),
                'TR_PRED': t_rimage,
                'R_PRED': s_rimage,
                'TC_PRED': t_cimage,
                'SC_PRED': s_cimage,
                'GT_CTR': data['center'],
                'S_CTR': s_center,
                'T_CTR': t_center,
                'GT_DPT': data['depth'],
                'S_DPT': s_depth,
                'T_DPT': t_depth,
                'TAG': data['tag'],
                'IND': data['ind']}
        
        self.temp_loss = TMP_LOSS
        
        return PREDS, TMP_LOSS

    def plot_test(self, select_ind=None, select_num=8, autosave=False, **kwargs):
        figs: dict = {}
        self.losslog.generate_indices(select_ind, select_num)

        figs.update(self.losslog.plot_predict(plot_terms=('R_GT', 'TR_PRED', 'R_PRED'), title='RIMG_PRED'))
        figs.update(self.losslog.plot_predict(plot_terms=('C_GT', 'TC_PRED', 'SC_PRED'), title='CIMG_PRED'))
        figs.update(self.losslog.plot_latent(plot_terms=('T_LATENT', 'S_LATENT')))
        figs.update(self.losslog.plot_center())
        figs.update(self.losslog.plot_test_cdf(plot_terms='all'))
        #figs.update(self.losslog.plot_tsne(plot_terms=('GT', 'T_LATENT', 'S_LATENT')))

        if autosave:
            for filename, fig in figs.items():
                fig.savefig(f"{self.save_path}{filename}")


if __name__ == '__main__':
    cc = ImageEncoder(latent_dim=128).to(torch.device('cuda:7'))
    summary(cc, input_size=(1, 128, 128))
    
