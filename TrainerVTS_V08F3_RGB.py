import torch
import torch.nn as nn
import torch.nn.init as init
# from torchvision.ops import complete_box_iou_loss
import numpy as np
import matplotlib.pyplot as plt
import os
from Trainer import BasicTrainer, TrainingPhase, ValidationPhase
from ModelVTS_v08F3 import *
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

class Teacher(nn.Module):

    def __init__(self, device=None):
        super(Teacher, self).__init__()

        self.imgen = ImageEncoder(latent_dim=128)
        self.rimgde = ImageDecoder(latent_dim=128)
        self.cimgde = ImageDecoder(latent_dim=128)
        self.ctrde = CenterDecoder()

        if device is not None:
            self.imgen = self.imgen.to(device)
            self.rimgde = self.rimgde.to(device)
            self.cimgde = self.cimgde.to(device)
            self.ctrde = self.ctrde.to(device)

    def forward(self, rimg):
        z, mu, logvar, feature = self.imgen(rimg)
        r_recon = self.rimgde(z)
        c_recon = self.cimgde(z)
        center, depth = self.ctrde(feature)

        ret = {
        'z'      : z,
        'mu'     : mu,
        'logvar' : logvar,
        'feature': feature,
        'rimage' : r_recon,
        'cimage' : c_recon,
        'center' : center,
        'depth'  : depth
                }

        return ret


class TeacherTrainer(BasicTrainer):
    def __init__(self,
                 beta=0.5,
                 recon_lossfunc=nn.BCEWithLogitsLoss(reduction='sum'),
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
        
        self.teacher = Teacher(device=self.device)
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
        
        ret = self.teacher(rimg)

        kl_loss = self.kl_loss(ret['mu'], ret['logvar'])
        r_recon_loss = self.recon_lossfunc(ret['rimage'], rimg) / ret['rimage'].shape[0]
        c_recon_loss = self.recon_lossfunc(ret['cimage'], cimg) / ret['cimage'].shape[0]
        vae_loss = kl_loss * self.beta + r_recon_loss + c_recon_loss
        
        center_loss = self.center_loss(ret['center'], torch.squeeze(data['center']))
        depth_loss = self.depth_loss(ret['depth'], torch.squeeze(data['depth']))
        
        loss = vae_loss + center_loss + depth_loss

        TEMP_LOSS = {'LOSS': loss,
                    'KL': kl_loss,
                    'R_RECON': r_recon_loss,
                    'C_RECON': c_recon_loss,
                    'CTR': center_loss, 
                    'DPT': depth_loss
                    }
        
        PREDS = {'R_GT': rimg,
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
        
        return PREDS, TEMP_LOSS

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


class Student(nn.Module):

    def __init__(self, device=None, teacher=None):
        super(Student, self).__init__()

        self.imgen = ImageEncoder(latent_dim=128)
        self.rimgde = ImageDecoder(latent_dim=128)
        self.cimgde = ImageDecoder(latent_dim=128)
        self.ctrde = CenterDecoder()
        self.csien = CSIEncoder(latent_dim=128)

        if device is not None:
            self.imgen = self.imgen.to(device)
            self.rimgde = self.rimgde.to(device)
            self.cimgde = self.cimgde.to(device)
            self.ctrde = self.ctrde.to(device)
            self.csien = self.csien.to(device)
            
        if isinstance(teacher, Teacher):
            # Share with teacher
            self.imgen = teacher.imgen
            self.rimgde = teacher.rimgde
            self.cimgde = teacher.cimgde
            self.ctrde = teacher.ctrde

    def forward(self, csi, pd, rimg):
        s_fea, s_z, s_mu, s_logvar = self.csien(csi=csi, pd=pd)
        s_center, s_depth = self.ctrde(s_fea)
        s_cimage = self.cimgde(s_z)
        s_rimage = self.rimgde(s_z)

        with torch.no_grad():
            t_z, t_mu, t_logvar, t_fea = self.imgen(rimg)
            t_cimage = self.cimgde(t_z)
            t_rimage = self.rimgde(t_z)
            t_center, t_depth = self.ctrde(t_fea)

        ret = {
        's_z'     : s_z,
        's_mu'    : s_mu,
        's_logvar': s_logvar,
        's_fea'   : s_fea,
        's_cimage': s_cimage,
        's_rimage': s_rimage,
        's_center': s_center,
        's_depth' : s_depth,
        't_z'     : t_z,
        't_mu'    : t_mu,
        't_logvar': t_logvar,
        't_fea'   : t_fea,
        't_cimage': t_cimage,
        't_rimage': t_rimage,
        't_center': t_center,
        't_depth' : t_depth
        }

        return ret


class StudentTrainer(BasicTrainer):
    def __init__(self,
                 alpha=0.8,
                 recon_lossfunc=nn.MSELoss(),
                 *args, **kwargs):
        super(StudentTrainer, self).__init__(*args, **kwargs)

        self.modality = {'cimg', 'rimg', 'csi', 'center', 'depth', 'pd', 'tag', 'ctr', 'dpt', 'ind'}

        self.alpha = alpha
        self.recon_lossfunc = recon_lossfunc
        self.img_loss = nn.BCEWithLogitsLoss(reduction='sum')

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
        
        self.teacher = Teacher(device=self.device)
        self.student = Student(device=self.device, teacher=self.teacher)
        
        self.models = {
            'imgen' : self.teacher.imgen,
            'cimgde': self.teacher.cimgde,
            'rimgde': self.teacher.rimgde,
            'ctrde': self.teacher.ctrde,
            'csien' : self.student.csien
                }

        self.training_phases = {'main': TrainingPhase(name = 'main',
                                                        train_module = ['csien'],
                                                        eval_module = ['imgen', 'rimgde', 'cimgde', 'ctrde'],
                                                        verbose=False
                                                        )}
        
        # FOR ADAPTING
        # self.valid_phases = {
        #     'source': ValidationPhase(name='source', loader='valid'),
        #     'target': ValidationPhase(name='target', loader='valid2')
        # }
        # self.early_stopping_trigger = 'target'
        
        self.latent_weight = 0.1
        self.img_weight = 1.
        self.center_weight = 1.
        self.depth_weight = 1.
        self.feature_weight = 1.
        
    def data_preprocess(self, mode, data2):

        def to_device(data):
            if self.preprocess:
                data = self.preprocess(data, self.modality)
            data = {key: data[key].to(torch.float32).to(self.device) for key in self.modality if key in data}
            if 'tag' in data:
                data['tag'] = data['tag'].to(torch.int32).to(self.device)
                data['tag'][data['tag']==4] = 1  # Replace A308 with A308T
            return data
    
        if len(data2) == 2:
            source_data, target_data = data2
            
            source_data = to_device(source_data)
            target_data = to_device(target_data)
            
            return source_data, target_data
        
        else:
            return to_device(data2)

    def kd_loss(self, mu_s, logvar_s, mu_t, logvar_t):
        mu_loss = self.recon_lossfunc(mu_s, mu_t)
        logvar_loss = self.recon_lossfunc(logvar_s, logvar_t)
        latent_loss = self.alpha * mu_loss + (1 - self.alpha) * logvar_loss

        return latent_loss, mu_loss, logvar_loss
    
    def feature_loss(self, feature_s, feature_t):
        feature_loss = self.recon_lossfunc(feature_s, feature_t)
        return feature_loss

    def calculate_loss(self, data):
        if len(data) == 2:
            # Source + Target
            source_data, target_data = data2
            
            cimg = torch.where(source_data['cimg'] > 0, 1., 0.)
            rimg = target_data['rimg']
            ctr = target_data['center']
            dpt = target_data['depth']
            
            source_ret = self.student(source_data['csi'], source_data['pd'], rimg)
            target_ret = self.student(target_data['csi'], target_data['pd'], rimg)
            
            # 3-level loss
            feature_loss = torch.mean(self.feature_loss(source_ret['s_fea'], source_ret['t_fea']),
                                      self.feature_loss(target_ret['s_fea'], target_ret['t_fea']))
            
            latent_loss = torch.mean(self.kd_loss(source_ret['s_mu'], source_ret['s_logvar'], source_ret['t_mu'], source_ret['t_logvar']),
                                     self.kd_loss(target_ret['s_mu'], target_ret['s_logvar'], target_ret['t_mu'], target_ret['t_logvar']))
            
            center_loss = self.recon_lossfunc(target_ret['s_center'], torch.squeeze(ctr))
            depth_loss = self.recon_lossfunc(target_ret['s_depth'], torch.squeeze(dpt))
            image_loss = self.img_loss(source_ret['s_cimage'], cimg) / source_ret['s_cimage'].shape[0]
            
        else:
            # Single domain
            cimg = torch.where(data['cimg'] > 0, 1., 0.)
            rimg = data['rimg']
            ctr = data['center']
            dpt = data['depth']
            
            # 3-level loss
            feature_loss = self.feature_loss(ret['s_fea'], ret['t_fea'])
            latent_loss = self.kd_loss(ret['s_mu'], ret['s_logvar'], ret['t_mu'], ret['t_logvar'])
            center_loss = self.recon_lossfunc(ret['s_center'], torch.squeeze(data['center']))
            depth_loss = self.recon_lossfunc(ret['s_depth'], torch.squeeze(data['depth']))
            image_loss = self.img_loss(ret['s_rimage'], rimg) / ret['s_rimage'].shape[0]
        
        loss = feature_loss * self.feature_weight +\
            latent_loss * self.latent_weight +\
            image_loss * self.img_weight +\
            center_loss * self.center_weight +\
            depth_loss * self.depth_weight
        
        TMP_LOSS = {
            'LOSS'   : loss,
            'LATENT' : latent_loss * self.latent_weight,
            'MU'     : mu_loss * self.alpha,
            'LOGVAR' : logvar_loss * (1 - self.alpha),
            'FEATURE': feature_loss * self.feature_weight,
            'IMG'    : image_loss * self.img_weight,
            'CTR'    : center_loss * self.center_weight,
            'DPT'    : depth_loss * self.depth_weight
            }
        
        PREDS = {
            'R_GT'    : rimg,
            'C_GT'    : cimg,
            'T_LATENT': torch.cat((ret['t_mu'], ret['t_logvar']), -1),
            'S_LATENT': torch.cat((ret['s_mu'], ret['s_logvar']), -1),
            'TR_PRED' : ret['t_rimage'],
            'R_PRED'  : ret['s_rimage'],
            'TC_PRED' : ret['t_cimage'],
            'SC_PRED' : ret['s_cimage'],
            'GT_CTR'  : ctr,
            'S_CTR'   : ret['s_center'],
            'T_CTR'   : ret['t_center'],
            'GT_DPT'  : dpt,
            'S_DPT'   : ret['s_depth'],
            'T_DPT'   : ret['t_depth'],
            'TAG'     : data['tag'],
            'IND'     : data['ind']
                }
        
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
    
