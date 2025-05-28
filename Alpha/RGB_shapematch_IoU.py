import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
# from torchvision.ops import complete_box_iou_loss
import numpy as np
import matplotlib.pyplot as plt
import os
from Trainer import BasicTrainer, TrainingPhase, ValidationPhase
from Structure.Model import *
from Loss import MyLossLog, MyLossCTR
from Alpha.Losses import PairwiseIoU, NCC

from torch.autograd import Function

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
steps = 75

Feature_extractor_train = ['csien']
Feature_extractor_eval = ['imgen', 'cimgde', 'rimgde', 'ctrde', 'dmnde']
Domain_classifier_train = ['dmnde']
Domain_classifier_eval = ['imgen', 'cimgde', 'rimgde', 'ctrde', 'csien']

class ShapeCoordLoss:
    def __init__(self, mode='c', device='gpu:0'):
        self.mode = mode
        self.device = device
        self.recon_lossfunc = nn.MSELoss()
        self.iou_loss = NCC()

    def weighted_loss(self, weight, est, gt):
        return (self.recon_lossfunc(est, gt) * weight.to(self.device)).sum() / weight.to(self.device).sum()
    
    def __call__(self, source_coord, target_coord,
                    source_shape, target_shape,
                    source_fea, target_fea, 
                    source_mu, source_logvar, 
                    target_mu, target_logvar):

        source_coord = F.normalize(source_coord, p=2, dim=1)  # Normalize source_coord
        target_coord = F.normalize(target_coord, p=2, dim=1)  # Normalize target_coord
        cos_sim = torch.matmul(target_coord, source_coord.T)
        cos_sim = torch.clamp(cos_sim, min=-1.0, max=1.0)
        _, indices = cos_sim.max(dim=1)

        if self.mode == 'c':
            match_fea_loss = self.recon_lossfunc(target_fea, source_fea[indices])
            match_mu_loss = self.recon_lossfunc(target_mu, source_mu[indices])
            match_logvar_loss = self.recon_lossfunc(target_logvar, source_logvar[indices])

        elif self.mode == 's':
            iou = self.iou_loss(source_shape, target_shape)
            max_sim_values, indices = iou.max(dim=1)
            sim_weight = max_sim_values / 2 + 0.5  # Rearrange into (0, 1)

            match_fea_loss = self.weighted_loss(sim_weight, target_fea, source_fea[indices])
            match_mu_loss = self.weighted_loss(sim_weight, target_mu, source_mu[indices])
            match_logvar_loss = self.weighted_loss(sim_weight, target_logvar, source_logvar[indices])

        elif self.mode == 'cs':
            iou = self.iou_loss(source_shape[indices], target_shape)
            max_sim_values, shp_indices = iou.max(dim=1)
            sim_weight = max_sim_values / 2 + 0.5  # Rearrange into (0, 1)

            match_fea_loss = self.weighted_loss(sim_weight, target_fea, source_fea[indices])
            match_mu_loss = self.weighted_loss(sim_weight, target_mu, source_mu[indices])
            match_logvar_loss = self.weighted_loss(sim_weight, target_logvar, source_logvar[indices])

        return match_fea_loss, match_mu_loss, match_logvar_loss
                    

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


class CSIEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CSIEncoder, self).__init__()

        self.batchnorm = 'identity'
        self.latent_dim = 128

        self.lstm_feature_length = 512 * 7
        self.lstm_steps = 75

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

        # self.fc_coord = nn.Linear(2, 16) # x and d

    def __str__(self):
        return f"CSIEN{version}"

    def forward(self, csi, pd):
        fea_csi = self.cnn(csi)
        fea_pd = self.fc_pd(pd)
        # fea_ctr = self.fc_coord(torch.cat((ctr[..., 0], dpt)), -1)
        csi_features, (final_hidden_state, final_cell_state) = self.lstm.forward(
            fea_csi.view(-1, 512 * 7, self.lstm_steps).transpose(1, 2))
        # 256-dim output
        features = torch.cat((csi_features[:, -1, :].view(-1, self.csi_feature_length), fea_pd.view(-1, self.pd_feature_length)), -1)
        out = self.fc_feature(features)
        
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        z = reparameterize(mu, logvar)

        return out, features.reshape(-1, 256), z, mu, logvar


class DomainClassifier(nn.Module):
    name = 'DmnDe'
    
    def __init__(self):
        super(DomainClassifier, self).__init__()
        dmn_len = 256
        dmn_hid = 64
        self.fc1 = nn.Linear(dmn_len, dmn_hid)
        self.fc2 = nn.Linear(dmn_hid, dmn_hid)
        self.fc3 = nn.Linear(dmn_hid, 4)  # 4 outputs for softmax

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        # x = self.sigmoid(x)
        # Binary Classification (BCE):
        # If using BCEWithLogitsLoss: No need to apply sigmoid.
        # If using BCELoss: Apply sigmoid before the loss function.
        # Multi-class Classification (CE):
        # If using CrossEntropyLoss: No need to apply softmax.
        # If using raw cross-entropy calculations, apply softmax first.
        return x 
    
    
class GradientReversalLayer(Function):
    
    @staticmethod
    def forward(ctx, input, lambda_):
        # Save lambda for later use in backward
        ctx.lambda_ = lambda_
        # Forward pass is identity, just return the input
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        # In the backward pass, retrieve lambda from ctx
        lambda_ = ctx.lambda_
        # Reverse the gradient by multiplying by -lambda
        grad_input = grad_output.neg() * lambda_
        return grad_input, None  # Return gradient for input, None for lambda


class Student(nn.Module):

    def __init__(self, device=None, teacher=None):
        super(Student, self).__init__()

        self.imgen = ImageEncoder(latent_dim=128)
        self.rimgde = ImageDecoder(latent_dim=128)
        self.cimgde = ImageDecoder(latent_dim=128)
        self.ctrde = CenterDecoder()
        self.csien = CSIEncoder()
        self.dmnde = DomainClassifier()

        if device is not None:
            self.imgen = self.imgen.to(device)
            self.rimgde = self.rimgde.to(device)
            self.cimgde = self.cimgde.to(device)
            self.ctrde = self.ctrde.to(device)
            self.csien = self.csien.to(device)
            self.dmnde = self.dmnde.to(device)
            
        if isinstance(teacher, Teacher):
            # Share with teacher
            self.imgen = teacher.imgen
            self.rimgde = teacher.rimgde
            self.cimgde = teacher.cimgde
            self.ctrde = teacher.ctrde

    def forward(self, csi, pd, rimg):
        s_fea, dann_fea, s_z, s_mu, s_logvar = self.csien(csi=csi, pd=pd)
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
        'dann_fea': dann_fea,
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
                 shapecoord='c',
                 *args, **kwargs):
        super(StudentTrainer, self).__init__(*args, **kwargs)

        self.modality = {'cimg', 'rimg', 'csi', 'center', 'depth', 'pd', 'tag', 'ctr', 'dpt', 'ind'}

        self.alpha = alpha
        self.lambda_ = 1.

        self.recon_lossfunc = recon_lossfunc
        self.sample_mse = nn.MSELoss(reduction='none')
        self.img_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.adv = nn.CrossEntropyLoss()
        self.iou_loss = PairwiseIoU()
        self.shapecoord = shapecoord
        self.shape_coord_loss = ShapeCoordLoss(mode=shapecoord, device=self.device)

        self.loss_terms = ('LOSS', 'MU', 'LOGVAR', 'LATENT', 'FEATURE', 'IMG', 'CTR', 'DPT', 'DOM', 'DOM_ACC', 'TG_LOSS', 'TG_FEA', 'TG_LAT', 'TG_CTR', 'TG_DPT')
        self.pred_terms = ('C_GT', 'R_GT',
                           'TR_PRED', 'R_PRED',
                           'TC_PRED', 'SC_PRED',
                           'T_LATENT', 'S_LATENT',
                           'GT_CTR', 'GT_DPT', 
                           'T_CTR', 'T_DPT',
                           'S_CTR', 'S_DPT',
                           'DOM_GT', 'DOM_PRED',
                           'TAG', 'IND')

        # FOR ADAPTING
        self.valid_phases = {
            'source': ValidationPhase(name='source', loader='valid', lossfunc=self.calculate_loss_main),
            'target': ValidationPhase(name='target', loader='valid2', lossfunc=self.calculate_loss_main),
            'test': ValidationPhase(name='test', loader='test', lossfunc=self.calculate_loss_main),
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
            'csien' : self.student.csien,
            'dmnde' : self.student.dmnde
                }

        self.calculate_losses = {
        'Feature_extractor': self.calculate_loss_fe,
        'Domain_classifier': self.calculate_loss_da,
        'Target_adaptation': self.calculate_loss_ta,
        }

        self.training_phases = {'Feature_extractor': TrainingPhase(name = 'Feature_extractor',
                                                                   train_module = Feature_extractor_train,
                                                                   eval_module = Feature_extractor_eval,
                                                                   verbose=False,
                                                                   lossfunc=self.calculate_loss_fe,                                                                   loss_arg={'reverse_feature': True},
                                                                   plot_terms=('LOSS', 'LATENT', 'MU', 'LOGVAR', 'FEATURE', 'IMG', 'CTR', 'DPT')
                                                                   ),
                                'Domain_classifier': TrainingPhase(name = 'Domain_classifier',
                                                                   train_module = Domain_classifier_train,
                                                                   eval_module = Domain_classifier_eval,
                                                                   loss = 'DOM',
                                                                   tolerance=1,
                                                                   conditioned_update=True,
                                                                   verbose=False,
                                                                   lossfunc=self.calculate_loss_da,
                                                                   loss_arg={'reverse_feature': False},
                                                                   plot_terms=('DOM', 'DOM_ACC')
                                                                   ),
                                'Target_adaptation': TrainingPhase(name = 'Target_adaptation',
                                                                   train_module = Feature_extractor_train,
                                                                   eval_module = Feature_extractor_eval,
                                                                   loss = 'TG_LOSS',
                                                                   tolerance=1,
                                                                   verbose=False,
                                                                   lossfunc=self.calculate_loss_ta,
                                                                   loss_arg={'reverse_feature': False},
                                                                   plot_terms=('TG_LOSS', 'TG_FEA', 'TG_LAT', 'TG_CTR', 'TG_DPT')
                                                                   ),

                                }

        
        self.latent_weight = 20
        self.rimg_weight = 1.e-4
        self.center_weight = 40.
        self.depth_weight = 50.
        self.feature_weight = 10

        self.domain_weight = 0.01

        self.target_fea_weight = 10.
        self.target_lat_weight = 20
        self.target_ctr_weight = 40.
        self.target_dpt_weight = 50.
        
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

    def calculate_lambda(self, max_iter=300):
        # Sigmoid schedule for lambda: 2 / (1 + exp(-10 * p)) - 1
        # where p is the proportion of iterations completed
        p = self.current_ep() / max_iter
        lambda_value = 2 / (1 + np.exp(-10 * p)) - 1
        self.lambda_ = min(lambda_value, 1)
        return min(lambda_value, 1)

    def dann_loss(self, source_tag, target_tag, source_feature, target_feature, reverse_feature):
        lambda_ = self.calculate_lambda() if reverse_feature else -1.
        
        if source_feature is not None:
            dann_features = torch.cat((source_feature, target_feature), dim=0)
        else:
            dann_features = target_feature
        
        # REVERSING DEPENDS ON LAMBDA
        dann_features = GradientReversalLayer.apply(dann_features, lambda_)
    
        domain_preds = self.models['dmnde'](dann_features.to(self.device))

        if source_tag is not None:
            domain_labels = torch.cat((source_tag[..., 0], target_tag[..., 0])).to(torch.int64).to(self.device)
        else:
            domain_labels = target_tag[..., 0].to(torch.int64).to(self.device)
        
        domain_loss = self.adv(domain_preds, domain_labels)

        with torch.no_grad():
            domain_acc_preds = torch.argmax(domain_preds, dim=1)
            domain_acc_loss = torch.sum(domain_acc_preds == domain_labels) / domain_preds.shape[0]
        
        return domain_loss, domain_acc_loss, domain_preds, domain_labels

    def kd_loss(self, mu_s, logvar_s, mu_t, logvar_t):
        mu_loss = self.recon_lossfunc(mu_s, mu_t)
        logvar_loss = self.recon_lossfunc(logvar_s, logvar_t)
        latent_loss = self.alpha * mu_loss + (1 - self.alpha) * logvar_loss

        return latent_loss, mu_loss, logvar_loss
    
    def feature_loss(self, feature_s, feature_t):
        feature_loss = self.recon_lossfunc(feature_s, feature_t)
        return feature_loss

    def weighted_loss(self, weight, est, gt):
        return (self.recon_lossfunc(est, gt) * weight.to(self.device)).sum() / weight.to(self.device).sum()

    def calculate_loss_main(self, data, reverse_feature=False):
        if len(data) == 2:
            # Source + Target
            source_data, target_data = data
            
            cimg = torch.where(source_data['cimg'] > 0, 1., 0.)
            rimg = target_data['rimg']
            ctr = target_data['center']
            dpt = target_data['depth']
            tag = target_data['tag']
            ind = target_data['ind']
            
            source_ret = self.student(source_data['csi'], source_data['pd'], source_data['rimg'])
            target_ret = self.student(target_data['csi'], target_data['pd'], rimg)
            ret = target_ret
            
            # 3-level loss
            source_feature_loss = self.feature_loss(source_ret['s_fea'], source_ret['t_fea'])
            feature_loss = source_feature_loss

            source_latent_loss, source_mu_loss, source_logvar_loss = self.kd_loss(source_ret['s_mu'], source_ret['s_logvar'], source_ret['t_mu'], source_ret['t_logvar'])
            
            latent_loss = source_latent_loss
            mu_loss = source_mu_loss
            logvar_loss = source_logvar_loss 

            domain_loss, domain_acc_loss, domain_preds, domain_labels = self.dann_loss(source_data['tag'], 
                                                                                       target_data['tag'], 
                                                                                       source_ret['dann_fea'],
                                                                                       target_ret['dann_fea'],
                                                                                       reverse_feature)

        else:
            # Single domain
            cimg = torch.where(data['cimg'] > 0, 1., 0.)
            rimg = data['rimg']
            ctr = data['center']
            dpt = data['depth']
            tag = data['tag']
            ind = data['ind']
            
            ret = self.student(data['csi'], data['pd'], rimg)
            
            # 3-level loss
            feature_loss = self.feature_loss(ret['s_fea'], ret['t_fea'])
            latent_loss, mu_loss, logvar_loss = self.kd_loss(ret['s_mu'], ret['s_logvar'], ret['t_mu'], ret['t_logvar'])

            domain_loss, domain_acc_loss, domain_preds, domain_labels = self.dann_loss(None, 
                                                                                    data['tag'], 
                                                                                    None,
                                                                                    ret['dann_fea'],
                                                                                    reverse_feature)
            
        image_loss = self.img_loss(ret['s_rimage'], rimg) / ret['s_rimage'].shape[0]
        center_loss = self.recon_lossfunc(ret['s_center'], torch.squeeze(ctr))
        depth_loss = self.recon_lossfunc(ret['s_depth'], torch.squeeze(dpt))

        LOSS = feature_loss * self.feature_weight
        LOSS += latent_loss * self.latent_weight
        LOSS += image_loss * self.rimg_weight
        LOSS += center_loss * self.center_weight
        LOSS += depth_loss * self.depth_weight

        TMP_LOSS = {
        'LOSS'   : LOSS,
        'LATENT' : latent_loss * self.latent_weight,
        'MU'     : mu_loss * self.alpha,
        'LOGVAR' : logvar_loss * (1 - self.alpha),
        'FEATURE': feature_loss * self.feature_weight,
        'IMG'    : image_loss * self.rimg_weight,
        'CTR'    : center_loss * self.center_weight,
        'DPT'    : depth_loss * self.depth_weight,
        'DOM'    : domain_loss,
        'DOM_ACC': domain_acc_loss
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
            'DOM_PRED': domain_preds,
            'DOM_GT' : domain_labels,
            'TAG'     : tag,
            'IND'     : ind
                }
        
        return PREDS, TMP_LOSS

    def calculate_loss_fe(self, data, reverse_feature=False):
        if len(data) == 2:
            # Source + Target
            source_data, target_data = data
            
            cimg = torch.where(source_data['cimg'] > 0, 1., 0.)
            rimg = target_data['rimg']
            ctr = target_data['center']
            dpt = target_data['depth']
            tag = target_data['tag']
            ind = target_data['ind']
            
            source_ret = self.student(source_data['csi'], source_data['pd'], source_data['rimg'])
            target_ret = self.student(target_data['csi'], target_data['pd'], rimg)
            ret = target_ret
            
            # 3-level loss
            source_feature_loss = self.feature_loss(source_ret['s_fea'], source_ret['t_fea'])
            feature_loss = source_feature_loss

            source_latent_loss, source_mu_loss, source_logvar_loss = self.kd_loss(source_ret['s_mu'], source_ret['s_logvar'], source_ret['t_mu'], source_ret['t_logvar'])
            
            latent_loss = source_latent_loss
            mu_loss = source_mu_loss
            logvar_loss = source_logvar_loss  
            
            center_loss = self.recon_lossfunc(source_ret['s_center'], torch.squeeze(source_data['center']))
            depth_loss = self.recon_lossfunc(source_ret['s_depth'], torch.squeeze(source_data['depth']))
            # center_loss, depth_loss = self.post_coord_loss(target_ret['s_rimage'], ctr, dpt)

            image_loss = self.img_loss(source_ret['s_cimage'], cimg) / source_ret['s_cimage'].shape[0]
                    
            LOSS = image_loss * self.rimg_weight
            LOSS += center_loss * self.center_weight
            LOSS += depth_loss * self.depth_weight
            LOSS += feature_loss * self.feature_weight
            LOSS += latent_loss * self.latent_weight

        TMP_LOSS = {
                'LOSS'   : LOSS,
                'LATENT' : latent_loss * self.latent_weight,
                'MU'     : mu_loss * self.alpha,
                'LOGVAR' : logvar_loss * (1 - self.alpha),
                'FEATURE': feature_loss * self.feature_weight,
                'IMG'    : image_loss * self.rimg_weight,
                'CTR'    : center_loss * self.center_weight,
                'DPT'    : depth_loss * self.depth_weight,
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

    def calculate_loss_da(self, data, reverse_feature=True):
        if len(data) == 2:
            # Source + Target
            source_data, target_data = data
            
            rimg = target_data['rimg']
            tag = target_data['tag']
            ind = target_data['ind']
            
            source_ret = self.student(source_data['csi'], source_data['pd'], source_data['rimg'])
            target_ret = self.student(target_data['csi'], target_data['pd'], rimg)

            # For Domain_classifier
            domain_loss, domain_acc_loss, domain_preds, domain_labels = self.dann_loss(source_data['tag'], 
                                                                                   target_data['tag'], 
                                                                                   source_ret['dann_fea'], 
                                                                                   target_ret['dann_fea'], 
                                                                                   reverse_feature)

            TMP_LOSS = {
                'DOM': domain_loss * self.domain_weight,
                'DOM_ACC': domain_acc_loss,
            }

            PREDS = {
            'DOM_PRED': domain_preds,
            'DOM_GT' : domain_labels,
            'TAG'     : tag,
            'IND'     : ind
                }

        return PREDS, TMP_LOSS

    def calculate_loss_ta(self, data, reverse_feature=False):
        if len(data) == 2:
            # Source + Target
            source_data, target_data = data
            
            cimg = torch.where(source_data['cimg'] > 0, 1., 0.)
            rimg = target_data['rimg']
            ctr = target_data['center']
            dpt = target_data['depth']
            tag = target_data['tag']
            ind = target_data['ind']
            
            source_ret = self.student(source_data['csi'], source_data['pd'], source_data['rimg'])
            target_ret = self.student(target_data['csi'], target_data['pd'], rimg)
            ret = target_ret

            # For Target_adaptation
            source_coord = torch.cat((source_data['center'][..., 0].reshape(-1, 1), source_data['depth'].reshape(-1, 1)), dim=-1)
            target_coord = torch.cat((target_data['center'][..., 0].reshape(-1, 1), target_data['depth'].reshape(-1, 1)), dim=-1)

            t_supervision = (source_ret['t_fea'], target_ret['s_fea'], 
                             source_ret['t_mu'], source_ret['t_logvar'],
                             target_ret['s_mu'], target_ret['s_logvar'])
            
            s_supervision = (source_ret['s_fea'], target_ret['s_fea'], 
                             source_ret['s_mu'], source_ret['s_logvar'],
                             target_ret['s_mu'], target_ret['s_logvar'])

            match_fea_loss, match_mu_loss, match_logvar_loss = self.shape_coord_loss(
                source_coord, target_coord, 
                source_data['cimg'], target_data['cimg'], 
                *t_supervision
            )

            match_lat_loss = self.alpha * match_mu_loss + (1 - self.alpha) * match_logvar_loss
            
            target_ctr_loss = self.recon_lossfunc(target_ret['s_center'], torch.squeeze(ctr))
            target_dpt_loss = self.recon_lossfunc(target_ret['s_depth'], torch.squeeze(dpt))

            TG_LOSS = match_fea_loss * self.target_fea_weight
            TG_LOSS += match_lat_loss * self.target_lat_weight
            TG_LOSS += target_ctr_loss * self.target_ctr_weight
            TG_LOSS += target_dpt_loss * self.target_dpt_weight

            TMP_LOSS = {
                'TG_LOSS'   : TG_LOSS,
                'TG_LAT' : match_lat_loss * self.target_lat_weight,
                'TG_FEA': match_fea_loss * self.target_fea_weight,
                'TG_CTR'    : target_ctr_loss * self.target_ctr_weight,
                'TG_DPT'    : target_dpt_loss * self.target_dpt_weight,
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