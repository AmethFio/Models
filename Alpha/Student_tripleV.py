import torch
import torch.nn as nn
import torch.nn.init as init
# from torchvision.ops import complete_box_iou_loss
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.autograd import Variable

import sys
sys.path.append('..')
from Trainer import BasicTrainer, TrainingPhase, ValidationPhase
from Loss import MyLossLog, MyLossCTR

from Structure.StandardTeacher import Teacher
from Models.Structure.Model import *
from Models.Structure.ModModel import CSIEncoder3V
from Alpha.Losses import NCCMSELoss, MMDLoss

from Wasserstein import WGANCritic, WGANLoss

class Student(nn.Module):

    def __init__(self, device=None, teacher=None):
        super(Student, self).__init__()

        self.imgen = ImageEncoder(latent_dim=128)
        self.rimgde = ImageDecoder(latent_dim=128)
        self.cimgde = ImageDecoder(latent_dim=128)
        self.ctrde = CenterDecoder()
        self.csien = CSIEncoder3V(latent_dim=128)

        if device is not None:
            self.to(device)

        self.csien = initialize_weights(self.csien)
            
        if isinstance(teacher, Teacher):
            # Share with teacher
            self.imgen = teacher.imgen
            self.rimgde = teacher.rimgde
            self.cimgde = teacher.cimgde
            self.ctrde = teacher.ctrde

    def forward(self, csi, pd, rimg):
        s_fea, s_z, s_mu, s_logvar = self.csien(csi=csi, pd=pd)
        centerdepth = [self.ctrde(fea) for fea in s_fea]
        s_center = [centerdepth[i][0] for i in range(len(centerdepth))]
        s_depth = [centerdepth[i][1] for i in range(len(centerdepth))]
        s_cimage = [self.cimgde(z) for z in s_z]
        s_rimage = [self.rimgde(z) for z in s_z]

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
                 adapting=False,
                 *args, **kwargs):
        super(StudentTrainer, self).__init__(*args, **kwargs)

        self.modality = {'cimg', 'rimg', 'csi', 'center', 'depth', 'pd', 'tag', 'ctr', 'dpt', 'ind'}

        self.alpha = alpha
        self.mse = nn.MSELoss(reduction='sum')
        self.mmd = MMDLoss()
        self.rimg_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.cimg_loss = NCCMSELoss(reduction='sum')

        self.loss_terms = ('LOSS', 'LATENT', 'MU', 'LOGVAR', 'FEATURE', 'RIMG', 'CIMG', 'CTR', 'DPT')
        self.pred_terms = ('C_GT', 'R_GT',
                           'TR_PRED', 'R_PRED1', 'R_PRED2', 'R_PRED3',
                           'TC_PRED', 'SC_PRED',
                           'T_LATENT', 'S_LATENT1', 'S_LATENT2', 'S_LATENT3',
                           'GT_CTR', 'GT_DPT', 
                           'T_CTR', 'T_DPT',
                           'S_CTR', 'S_DPT',
                           'TAG', 'IND')

        # FOR ADAPTING
        if adapting:
            self.valid_phases = {
                'source': ValidationPhase(name='source', loader='valid', lossfunc=self.calculate_loss),
                'target': ValidationPhase(name='target', loader='valid2', lossfunc=self.calculate_loss)
            }
            self.early_stopping_trigger = 'target'

        self.training_phases = {'main': TrainingPhase(name = 'main',
                                                        lossfunc = self.calculate_loss,
                                                        train_module = ['csien'],
                                                        eval_module = ['imgen', 'rimgde', 'cimgde', 'ctrde'],
                                                        verbose=False,
                                                        lr=kwargs.get('lr', 1.e-4)
                                                        )}

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
            'csien': self.student.csien,
                }
        
        self.latent_weight = 4.e-2
        self.rimg_weight = 1.e-4
        self.cimg_weight = 4.e-1
        self.center_weight = 30
        self.depth_weight = 50.
        self.feature_weight = 1.e-2
        self.domain_weight = 1.
        
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
        mu_loss = self.mse(mu_s, mu_t)
        logvar_loss = self.mse(logvar_s, logvar_t)
        latent_loss = self.alpha * mu_loss + (1 - self.alpha) * logvar_loss
        return latent_loss, mu_loss, logvar_loss

    @staticmethod
    def aggregate_loss(preds, gt, lossfun):
        all_preds = torch.cat(preds, dim=0)
        # Repeat ground truth 3 times to match
        #repeated_target = gt.repeat(3, *([1] * (gt.dim() - 1)))
        repeated_target = torch.cat((gt, gt, gt), dim=0)
        return lossfun(all_preds, repeated_target)

    def calculate_loss(self, data):
        gt_cimg = torch.where(data['cimg'] > 0.1, 1., 0.)
        
        ret = self.student(data['csi'], data['pd'], data['rimg'])
        
        # 3-level loss
        # mu_loss = self.aggregate_loss(ret['s_mu'], ret['t_mu'], self.mse) / (3 * ret['t_mu'].shape[0])
        # logvar_loss = self.aggregate_loss(ret['s_logvar'], ret['t_logvar'], self.mse) / (3 * ret['t_logvar'].shape[0])
        # latent_loss =  self.alpha * mu_loss + (1 - self.alpha) * logvar_loss

        # feature_loss = self.aggregate_loss(ret['s_fea'], ret['t_fea'], self.mse) / (3 * ret['t_fea'].shape[0])
        # center_loss = self.aggregate_loss(ret['s_center'], torch.squeeze(data['center']), self.mse) / (3 * data['center'].shape[0])
        # depth_loss = self.aggregate_loss(ret['s_depth'], torch.squeeze(data['depth']), self.mse) / (3 * data['depth'].shape[0])
        # cimage_loss = self.aggregate_loss(ret['s_cimage'], gt_cimg, self.cimg_loss) / (3 * gt_cimg.shape[0])
        # rimage_loss = self.aggregate_loss(ret['s_rimage'], data['rimg'], self.rimg_loss) / (3 * data['rimg'].shape[0])

        mu_loss = self.mse(ret['s_mu'][-1], ret['t_mu']) / ret['t_mu'].shape[0]
        logvar_loss = self.mse(ret['s_logvar'][-1], ret['t_logvar']) / ret['t_logvar'].shape[0]
        latent_loss =  self.alpha * mu_loss + (1 - self.alpha) * logvar_loss

        feature_loss = self.mse(ret['s_fea'][-1], ret['t_fea']) / ret['t_fea'].shape[0]
        center_loss = self.mse(ret['s_center'][-1], torch.squeeze(data['center']),) / data['center'].shape[0]
        depth_loss = self.mse(ret['s_depth'][-1], torch.squeeze(data['depth'])) / data['depth'].shape[0]
        cimage_loss = self.cimg_loss(ret['s_cimage'][-1], gt_cimg) / gt_cimg.shape[0]
        rimage_loss = self.rimg_loss(ret['s_rimage'][-1], data['rimg']) / data['rimg'].shape[0]
        
        LOSS = feature_loss * self.feature_weight
        LOSS += latent_loss * self.latent_weight
        LOSS += rimage_loss * self.rimg_weight
        LOSS += cimage_loss * self.cimg_weight
        LOSS += center_loss * self.center_weight
        LOSS += depth_loss * self.depth_weight
        
        TMP_LOSS = {'LOSS': LOSS,
                    'LATENT': latent_loss * self.alpha * self.latent_weight,
                    'MU': mu_loss * self.latent_weight,
                    'LOGVAR': logvar_loss * (1 - self.alpha) * self.latent_weight,
                    'FEATURE': feature_loss * self.feature_weight,
                    'RIMG': rimage_loss * self.rimg_weight,
                    'CIMG': cimage_loss * self.cimg_weight,
                    'CTR': center_loss * self.center_weight,
                    'DPT': depth_loss * self.depth_weight
                    }
        
        PREDS = {'R_GT': data['rimg'],  
                'C_GT': gt_cimg,
                'T_LATENT': torch.cat((ret['t_mu'], ret['t_logvar']), -1),
                'S_LATENT1': torch.cat((ret['s_mu'][0], ret['s_logvar'][0]), -1),
                'S_LATENT2': torch.cat((ret['s_mu'][1], ret['s_logvar'][1]), -1),
                'S_LATENT3': torch.cat((ret['s_mu'][2], ret['s_logvar'][2]), -1),
                'TR_PRED': ret['t_rimage'],
                'R_PRED1': ret['s_rimage'][0],
                'R_PRED2': ret['s_rimage'][1],
                'R_PRED3': ret['s_rimage'][2],
                'TC_PRED': ret['t_cimage'],
                'SC_PRED': ret['s_cimage'][-1],
                'GT_CTR': data['center'],
                'S_CTR': ret['s_center'][-1],
                'T_CTR': ret['t_center'],
                'GT_DPT': data['depth'],
                'S_DPT': ret['s_depth'][-1],
                'T_DPT': ret['t_depth'],
                'TAG': data['tag'],
                'IND': data['ind']}
        
        return PREDS, TMP_LOSS

    def plot_test(self, select_ind=None, select_num=8, autosave=False, **kwargs):
        figs: dict = {}
        self.losslog.generate_indices(select_ind, select_num)

        figs.update(self.losslog.plot_predict(plot_terms=('R_GT', 'TR_PRED', 'R_PRED1', 'R_PRED2', 'R_PRED3'), title='RIMG_PRED'))
        figs.update(self.losslog.plot_predict(plot_terms=('C_GT', 'TC_PRED', 'SC_PRED'), title='CIMG_PRED'))
        figs.update(self.losslog.plot_latent(plot_terms=('T_LATENT', 'S_LATENT1'), title='1st latent'))
        figs.update(self.losslog.plot_latent(plot_terms=('T_LATENT', 'S_LATENT2'), title='2nd latent'))
        figs.update(self.losslog.plot_latent(plot_terms=('T_LATENT', 'S_LATENT3'), title='3rd latent'))
        figs.update(self.losslog.plot_center())
        figs.update(self.losslog.plot_test_cdf(plot_terms='all'))
        #figs.update(self.losslog.plot_tsne(plot_terms=('GT', 'T_LATENT', 'S_LATENT')))

        if autosave:
            for filename, fig in figs.items():
                print(f"Saving {filename}...")
                fig.savefig(f"{self.save_path}{filename}")


if __name__ == '__main__':
    cc = ImageEncoder(latent_dim=128).to(torch.device('cuda:7'))
    summary(cc, input_size=(1, 128, 128))
    
