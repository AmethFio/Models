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

from Structure.StandardStudent import Student
from Structure.StandardTeacher import Teacher
from Models.Structure.Model import *
from Models.Structure.ModModel import CSIEncoderConcat as CSIen
from Alpha.Losses import NCCMSELoss, MMDLoss

from Wasserstein import WGANCritic, WGANLoss

def re_init(self, device=None, teacher=None):
    super(Student, self).__init__()

    self.imgen = ImageEncoder(latent_dim=128)
    self.rimgde = ImageDecoder(latent_dim=128)
    self.cimgde = ImageDecoder(latent_dim=128)
    self.ctrde = CenterDecoder()
    self.csien = CSIen(latent_dim=128)

    if device is not None:
        self.to(device)

    self.csien = initialize_weights(self.csien)
        
    if isinstance(teacher, Teacher):
        # Share with teacher
        self.imgen = teacher.imgen
        self.rimgde = teacher.rimgde
        self.cimgde = teacher.cimgde
        self.ctrde = teacher.ctrde

Student.__init__ = re_init

class StudentTrainer(BasicTrainer):
    def __init__(self,
                 alpha=0.8,
                 adapting=False,
                 *args, **kwargs):
        super(StudentTrainer, self).__init__(*args, **kwargs)

        self.modality = {'cimg', 'rimg', 'csi', 'center', 'depth', 'pd', 'tag', 'ctr', 'dpt', 'ind'}

        self.alpha = alpha

        self.loss_terms = ('LOSS', 'LATENT', 'MU', 'LOGVAR', 'FEATURE', 'RIMAGE', 'CIMAGE', 'CENTER', 'DEPTH')
        self.pred_terms = ('C_GT', 'R_GT',
                           'TR_PRED', 'R_PRED',
                           'TC_PRED', 'SC_PRED',
                           'T_LATENT', 'S_LATENT',
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
        
        self.lossfuncs = {
            'LATENT': nn.MSELoss(reduction='sum'),
            'FEATURE': nn.MSELoss(reduction='sum'),
            'CENTER': nn.MSELoss(reduction='sum'),
            'DEPTH': nn.MSELoss(reduction='sum'),
            'CIMAGE': NCCMSELoss(reduction='sum'),
            'RIMAGE': nn.BCEWithLogitsLoss(reduction='sum')
        }

        self.weights = {
            'LATENT': 4.e-2,
            'MU': self.alpha * 4.e-2,
            'LOGVAR': (1 - self.alpha) * 4.e-2,
            'RIMAGE': 1.e-4,
            'CIMAGE': 4.e-1,
            'CENTER': 30.,
            'DEPTH': 50.,
            'FEATURE': 1.e-2
        }
        
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
        all_preds = torch.mean(preds, dim=1)
        return lossfun(all_preds, gt)

    @BasicTrainer.loss_weighting
    def calculate_loss(self, data):
        gt_cimg = torch.where(data['cimg'] > 0.1, 1., 0.)
        
        ret = self.student(data['csi'], data['pd'], data['rimg'])
        
        # 3-level loss
        mu_loss = self.lossfuncs['LATENT'](ret['s_mu'], ret['t_mu']) / ret['t_mu'].shape[0]
        logvar_loss = self.lossfuncs['LATENT'](ret['s_logvar'], ret['t_logvar']) / ret['t_logvar'].shape[0]
        latent_loss =  mu_loss * self.alpha + logvar_loss * (1 - self.alpha)

        feature_loss = self.lossfuncs['FEATURE'](ret['s_fea'], ret['t_fea']) / ret['t_fea'].shape[0]
        center_loss = self.lossfuncs['CENTER'](ret['s_center'], torch.squeeze(data['center']),) / data['center'].shape[0]
        depth_loss = self.lossfuncs['DEPTH'](ret['s_depth'], torch.squeeze(data['depth'])) / data['depth'].shape[0]
        cimage_loss = self.lossfuncs['CIMAGE'](ret['s_cimage'], gt_cimg) / gt_cimg.shape[0]
        rimage_loss = self.lossfuncs['RIMAGE'](ret['s_rimage'], data['rimg']) / data['rimg'].shape[0]
        
        LOSS = latent_loss * self.weights['LATENT']
        LOSS += feature_loss * self.weights['FEATURE']
        # LOSS += rimage_loss * self.weights['RIMAGE']
        # LOSS += cimage_loss * self.weights['CIMAGE']
        # LOSS += center_loss * self.weights['CENTER']
        # LOSS += depth_loss * self.weights['DEPTH']
        
        TMP_LOSS = {'LOSS': LOSS,
                    'LATENT': latent_loss,
                    'MU': mu_loss,
                    'LOGVAR': logvar_loss,
                    'FEATURE': feature_loss,
                    'RIMAGE': rimage_loss,
                    'CIMAGE': cimage_loss,
                    'CENTER': center_loss,
                    'DEPTH': depth_loss
                    }
        
        PREDS = {'R_GT': data['rimg'],  
                'C_GT': gt_cimg,
                'T_LATENT': torch.cat((ret['t_mu'], ret['t_logvar']), -1),
                'S_LATENT': torch.cat((ret['s_mu'], ret['s_logvar']), -1),
                'TR_PRED': ret['t_rimage'],
                'R_PRED': ret['s_rimage'],
                'TC_PRED': ret['t_cimage'],
                'SC_PRED': ret['s_cimage'],
                'GT_CTR': data['center'],
                'S_CTR': ret['s_center'],
                'T_CTR': ret['t_center'],
                'GT_DPT': data['depth'],
                'S_DPT': ret['s_depth'],
                'T_DPT': ret['t_depth'],
                'TAG': data['tag'],
                'IND': data['ind']}
        
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
                print(f"Saving {filename}...")
                fig.savefig(f"{self.save_path}{filename}")


if __name__ == '__main__':
    cc = ImageEncoder(latent_dim=128).to(torch.device('cuda:7'))
    summary(cc, input_size=(1, 128, 128))
    
