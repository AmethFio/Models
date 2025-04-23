import torch
import torch.nn as nn
import torch.nn.init as init
# from torchvision.ops import complete_box_iou_loss
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append('..')
from Trainer import BasicTrainer, TrainingPhase, ValidationPhase
from Loss import MyLossLog, MyLossCTR

from Structure.StandardTeacher import Teacher
from Structure.StandardStudent import Student
from Wasserstein import WGANCritic, WGANLoss

feature_length = 512 * 7
steps = 25

Feature_extractor_train = ['csien']
Feature_extractor_eval = ['imgen', 'cimgde', 'rimgde', 'ctrde', 'dmnde']
Domain_classifier_train = ['dmnde']
Domain_classifier_eval = ['imgen', 'cimgde', 'rimgde', 'ctrde', 'csien']


class StudentTrainer(BasicTrainer):
    def __init__(self,
                 alpha=0.8,
                 adapting=False,
                 *args, **kwargs):
        super(StudentTrainer, self).__init__(*args, **kwargs)

        self.modality = {'cimg', 'rimg', 'csi', 'center', 'depth', 'pd', 'tag', 'ctr', 'dpt', 'ind'}

        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.img_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.wganloss = WGANLoss()

        self.loss_terms = ('LOSS', 'WDIS', 'WGEN', 'LAT', 'IMG', 'CTR', 'DPT')
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
                'source': ValidationPhase(name='source', loader='valid'),
                'target': ValidationPhase(name='target', loader='valid2')
            }
            self.early_stopping_trigger = 'target'

            # self.loss_terms = ('LOSS', 'IMG', 'CTR', 'DPT')

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
            'dmnde': WGANCritic().to(self.device)
                }

        self.training_phases = {
                                'Discriminating': TrainingPhase(name='Discriminating',
                                                        lossfunc=self.calculate_loss_dis,
                                                       train_module=Domain_classifier_train,
                                                       eval_module=Domain_classifier_eval,
                                                       loss='WDIS',
                                                       tolerance=1,
                                                       plot_terms=(['WDIS']),
                                                       lr=1.e-4
                                                       ),
                                'Generating': TrainingPhase(name='Generating',
                                                        lossfunc=self.calculate_loss,
                                                        train_module=Feature_extractor_train,
                                                        eval_module=Feature_extractor_eval,
                                                        lr=1.e-4,
                                                        plot_terms=('LOSS', 'WGEN', 'LAT', 'IMG', 'CTR', 'DPT')
                                                        )
                                }
        
        self.latent_weight = 20
        self.img_weight = 1.e-4
        self.center_weight = 50.
        self.depth_weight = 100.
        self.feature_weight = 1.
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

    def phase_condition(self, name, epoch):
        if name == 'Generating' and epoch % 5 != 0:
            return False
        else:
            return True

    def kd_loss(self, mu_s, logvar_s, mu_t, logvar_t):
        mu_loss = self.mse(mu_s, mu_t)
        logvar_loss = self.mse(logvar_s, logvar_t)
        latent_loss = self.alpha * mu_loss + (1 - self.alpha) * logvar_loss

        return latent_loss, mu_loss, logvar_loss
    
    def feature_loss(self, feature_s, feature_t):
        feature_loss = self.recon_lossfunc(feature_s, feature_t)
        return feature_loss

    def calculate_loss(self, data):
        
        cimg = torch.where(data['cimg'] > 0, 1., 0.)
        rimg = data['rimg']
        ctr = data['center']
        dpt = data['depth']
        tag = data['tag']
        ind = data['ind']
        
        ret = self.student(data['csi'], data['pd'], rimg)
        wscore = self.models['dmnde'](ret['s_z'])
        # 3-level loss
        # feature_loss = self.feature_loss(ret['s_fea'], ret['t_fea'])
        latent_loss, mu_loss, logvar_loss = self.kd_loss(ret['s_mu'], ret['s_logvar'], ret['t_mu'], ret['t_logvar'])
        center_loss = self.mse(ret['s_center'], torch.squeeze(ctr))
        depth_loss = self.mse(ret['s_depth'], torch.squeeze(dpt))
        image_loss = self.img_loss(ret['s_rimage'], rimg) / ret['s_rimage'].shape[0]
        wgen_loss = self.wganloss(self.models['dmnde'], None, ret['s_z'], 'g')

        loss = image_loss * self.img_weight +\
            center_loss * self.center_weight +\
            depth_loss * self.depth_weight +\
            wgen_loss * self.domain_weight

        TMP_LOSS = {
        'LOSS'   : loss,
        'WGEN'   : wgen_loss * self.domain_weight,
        'LAT'    : latent_loss * self.latent_weight, 
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
            'TAG'     : tag,
            'IND'     : ind
                }
    
        return PREDS, TMP_LOSS

    def calculate_loss_dis(self, data):
        ret = self.student(data['csi'], data['pd'], data['rimg'])
        wdis_loss = self.wganloss(self.models['dmnde'], ret['t_z'], ret['s_z'], 'd')
        wdis_loss *= self.domain_weight

        TMP_LOSS = {
        'WDIS': wdis_loss
        }

        PREDS = {
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
    
