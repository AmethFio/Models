import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
# from torchvision.ops import complete_box_iou_loss
import numpy as np
import matplotlib.pyplot as plt
import os
from Trainer import BasicTrainer, TrainingPhase, ValidationPhase
from ModelVTS_v08F3 import *
from Loss import MyLossLog, MyLossCTR

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

class CenterDecoder(nn.Module):
    name = 'ctrde'

    def __init__(self):
        super(CenterDecoder, self).__init__()
        self.feature_length = 512

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


class ImageEncoder(nn.Module):
    name = 'imgen'
    
    def __init__(self, batchnorm='identity', latent_dim=16, *args, **kwargs):
        super(ImageEncoder, self).__init__(*args, **kwargs)

        self.batchnorm = batchnorm
        self.latent_dim = latent_dim
        
        block = [[1, 128, 3, 2, 1],
                [128, 128, 3, 1, 1],
                [128, 128, 3, 2, 1],
                [128, 128, 3, 1, 1],
                [128, 256, 3, 2, 1],
                [256, 256, 3, 1, 1],
                [256, 512, 3, 1, 1],
                [512, 512, 1, 1, 0],]
        
        cnn = []

        for [in_ch, out_ch, ks, st, pd] in block:
            cnn.extend([nn.Conv2d(in_ch, out_ch, ks, st, pd),
                        batchnorm_layer(out_ch, self.batchnorm),
                        nn.LeakyReLU(inplace=True)])
            
        self.cnn = nn.Sequential(*cnn)

        # 1 * 128 * 128
        # 128 * 64 * 64
        # Re
        # 128 * 32 * 32
        # Re
        # 256 * 16 * 16
        # Re
        # 512 * 16* 16
        # 1 * 1 conv
        # Re
        # GAP
        # 512 * 1

        self.fc_mu = nn.Sequential(
            nn.Linear(512, self.latent_dim)
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(512, self.latent_dim)
        )

    def __str__(self):
        return f"IMGEN{version}"

    def forward(self, x):
        out = self.cnn(x)
        # GAP
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(-1, 512)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        z = reparameterize(mu, logvar)

        return z, mu, logvar, out

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
        self.feature_length = 512
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
        dmn_len = 128
        dmn_hid = 32
        self.fc1 = nn.Linear(dmn_len, dmn_hid)
        self.fc2 = nn.Linear(dmn_hid, dmn_hid)
        self.fc3 = nn.Linear(dmn_hid, 2)  # 2 outputs for softmax

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
        self.models = {'imgen': self.teacher.imgen,
                       'cimgde': self.teacher.cimgde,
                       'rimgde': self.teacher.rimgde,
                       'ctrde': self.teacher.ctrde
                       }
        
        self.latent_weight = 0.01
        self.img_weight = 1.e-4
        self.center_weight = 100.
        self.depth_weight = 100.
        
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
        vae_loss = kl_loss * self.beta * self.latent_weight
        vae_loss += r_recon_loss * self.img_weight
        vae_loss += c_recon_loss * self.img_weight

        center_loss = self.center_loss(ret['center'], torch.squeeze(data['center']))
        depth_loss = self.depth_loss(ret['depth'], torch.squeeze(data['depth']))
        
        loss = vae_loss
        loss += center_loss * self.center_weight
        loss += depth_loss * self.depth_weight

        TEMP_LOSS = {'LOSS': loss,
              'KL': kl_loss * self.beta * self.latent_weight,
              'R_RECON': r_recon_loss * self.img_weight,
              'C_RECON': c_recon_loss * self.img_weight,
              'CTR': center_loss * self.center_weight, 
              'DPT': depth_loss * self.depth_weight
              }
        
        PREDS = {'R_GT': rimg,
                'C_GT': cimg,
                'R_PRED': ret['rimage'],
                'C_PRED': ret['cimage'],
                'GT_CTR': data['center'],
                'CTR_PRED': ret['center'],
                'GT_DPT': data['depth'],
                'DPT_PRED': ret['depth'],
                'LAT': torch.cat((ret['mu'], ret['logvar']), -1),
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
        self.csien = CSIEncoder()
        # self.dmnde = DomainClassifier()

        if device is not None:
            self.imgen = self.imgen.to(device)
            self.rimgde = self.rimgde.to(device)
            self.cimgde = self.cimgde.to(device)
            self.ctrde = self.ctrde.to(device)
            self.csien = self.csien.to(device)
            # self.dmnde = self.dmnde.to(device)
            
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
                 adapting=False,
                 *args, **kwargs):
        super(StudentTrainer, self).__init__(*args, **kwargs)

        self.modality = {'cimg', 'rimg', 'csi', 'center', 'depth', 'pd', 'tag', 'ctr', 'dpt', 'ind'}

        self.alpha = alpha
        self.lambda_ = 1.

        self.recon_lossfunc = recon_lossfunc
        self.sample_mse = nn.MSELoss(reduction='none')
        self.img_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.adv = nn.CrossEntropyLoss()

        self.loss_terms = (
            'LOSS', 
            'MU', 'LOGVAR', 
            'LATENT', 
            'FEATURE',
            #'FEATURE_MSE',
            'IMG', 'CTR', 'DPT',
            #'DOM', 'DOM_ACC'
          )

        self.pred_terms = ('C_GT', 'R_GT',
                           'TR_PRED', 'R_PRED',
                           'TC_PRED', 'SC_PRED',
                           'T_LATENT', 'S_LATENT',
                           'GT_CTR', 'GT_DPT', 
                           'T_CTR', 'T_DPT',
                           'S_CTR', 'S_DPT',
                           'DOM_GT', 'DOM_PRED',
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
            'csien' : self.student.csien,
            # 'dmnde' : self.student.dmnde
                }

        
        self.latent_weight = 0.1
        self.rimg_weight = 1.e-4
        self.center_weight = 40.
        self.depth_weight = 50.
        self.feature_weight = 10
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
    
    def feature_loss_contra(self, feature_s, feature_t):
        # CONTRASTIVE LEARNING VERSION
        feature_mse = self.recon_lossfunc(feature_s, feature_t)
        # Normalize the vectors for cosine similarity
        feature_s = feature_s.reshape(feature_s.shape[0], -1)
        feature_t = feature_t.reshape(feature_t.shape[0], -1)

        feature_s = feature_s / feature_s.norm(dim=1, keepdim=True)
        feature_t = feature_t / feature_t.norm(dim=1, keepdim=True)

        # Compute cosine similarity matrix (shape: (4,4))
        cos_sim = torch.matmul(feature_s[:, None], feature_t[None, :])

        # Create target labels: 1 for same index (positive pairs), 0 for others (negative pairs) 
        target = torch.eye(len(feature_t))  # Identity matrix (4x4) with 1s on the diagonal

        # Contrastive loss using a margin (hinge loss style)
        margin = 0.5  # Define a margin for dissimilar pairs
        positive_loss = (1 - cos_sim) * target   # Push similar pairs closer (minimize 1 - cos_sim)
        negative_loss = F.relu(cos_sim - margin) * (1 - target)  # Push dissimilar pairs apart

        # Final loss (sum of positive and negative parts)
        contrastive_loss = positive_loss.mean() + negative_loss.mean()

        return contrastive_loss, feature_mse

    def feature_loss_crossattn(self, feature_s, feasture_t):
        pass


    def calculate_loss(self, data, reverse_feature=False):
        # Single domain
        cimg = torch.where(data['cimg'] > 0, 1., 0.)
        rimg = data['rimg']
        ctr = data['center']
        dpt = data['depth']
        tag = data['tag']
        ind = data['ind']
        
        ret = self.student(data['csi'], data['pd'], rimg)
        
        # 3-level loss
        feature_loss, feature_mse = self.feature_loss(ret['s_fea'], ret['t_fea'])
        latent_loss, mu_loss, logvar_loss = self.kd_loss(ret['s_mu'], ret['s_logvar'], ret['t_mu'], ret['t_logvar'])

        # domain_loss, domain_acc_loss, domain_preds, domain_labels = self.dann_loss(None, 
        #                                                                         data['tag'], 
        #                                                                         None,
        #                                                                         ret['dann_fea'],
        #                                                                         reverse_feature)
            
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
            # 'FEATURE_MSE': feature_mse
            'IMG'    : image_loss * self.rimg_weight,
            'CTR'    : center_loss * self.center_weight,
            'DPT'    : depth_loss * self.depth_weight,
            # 'DOM'    : domain_loss,
            # 'DOM_ACC': domain_acc_loss
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
            # 'DOM_PRED': domain_preds,
            # 'DOM_GT' : domain_labels,
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