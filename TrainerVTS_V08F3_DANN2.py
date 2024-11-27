import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Function
import numpy as np
import matplotlib.pyplot as plt
import os
from Trainer import BasicTrainer, TrainingPhase, ValidationPhase
from Model import *
from Loss import MyLossLog, MyLossCTR

version = 'V08F3_DANN'

DANN_place =  'features'
dmn_len = 1536
dmn_hid = 256

def set_DANN(DANN_place):
    if DANN_place == 'LSTM_features':
        dmn_len = 128
        dmn_hid = 64
    elif DANN_place == 'concat_features':
        dmn_len = 256
        dmn_hid = 64
    elif DANN_place == 'features':
        dmn_len = 1536
        dmn_hid = 256
    return dmn_len, dmn_hid

Feature_extractor_train = ['csien']
Feature_extractor_eval = ['imgen', 'cimgde', 'rimgde', 'ctrde', 'dmnde']
Domain_classifier_train = ['dmnde']
Domain_classifier_eval = ['imgen', 'cimgde', 'rimgde', 'ctrde', 'csien']
        
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
        csi_features, (final_hidden_state, final_cell_state) = self.lstm.forward(
            fea_csi.view(-1, 512 * 7, self.lstm_steps).transpose(1, 2))
        # 256-dim output
        features = torch.cat((csi_features[:, -1, :].view(-1, self.csi_feature_length), fea_pd.view(-1, self.pd_feature_length)), -1)
        out = self.fc_feature(features)
        
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        z = reparameterize(mu, logvar)
        
        if DANN_place == 'features':
            return out, out.reshape(-1, dmn_len), z, mu, logvar
        elif DANN_place == 'LSTM_features':
            return out, csi_features.reshape(-1, dmn_len), z, mu, logvar
        elif DANN_place == 'concat_features':
            return out, features.reshape(-1, dmn_len), z, mu, logvar
    
    
class DomainClassifier(nn.Module):
    name = 'DmnDe'
    
    def __init__(self):
        super(DomainClassifier, self).__init__()
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
                 with_feature_loss=True,
                 lstm_steps=75,
                 *args, **kwargs):
        super(StudentTrainer, self).__init__(*args, **kwargs)

        self.modality = {'cimg', 'rimg', 'csi', 'center', 'depth', 'pd', 'tag', 'ctr', 'dpt', 'ind'}

        self.alpha = alpha
        self.lambda_ = 1.
        
        self.with_feature_loss = with_feature_loss
        self.mse_sum = nn.MSELoss(reduction='sum')
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.adv = nn.CrossEntropyLoss()

        self.loss_terms = ('LOSS', 'MU', 'LOGVAR', 'FEATURE', 'IMG', 'CTR', 'DPT', 'DOM', 'DOM_ACC')
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
        
        self.device2 = kwargs.get('device2', self.device)
        
        self.models = {
            'imgen' : ImageEncoder(latent_dim=128).to(self.device),
            'cimgde': ImageDecoder(latent_dim=128).to(self.device),
            'rimgde': ImageDecoder(latent_dim=128).to(self.device2),
            'csien' : CSIEncoder(latent_dim=128, lstm_steps=lstm_steps).to(self.device),
            'ctrde': CenterDecoder().to(self.device),
            'dmnde': DomainClassifier().to(self.device)
                }
        
        self.training_phases = {'Feature_extractor': TrainingPhase(name = 'Feature_extractor',
                                                                   train_module = Feature_extractor_train,
                                                                   eval_module = Feature_extractor_eval,
                                                                   verbose=False,
                                                                   loss_arg={'reverse_feature': True}
                                                                   ),
                                'Domain_classifier': TrainingPhase(name = 'Domain_classifier',
                                                                   train_module = Domain_classifier_train,
                                                                   eval_module = Domain_classifier_eval,
                                                                   loss = 'DOM',
                                                                   tolerance=1,
                                                                   conditioned_update=True,
                                                                   verbose=False,
                                                                   loss_arg={'reverse_feature': False}
                                                                   )
                                }
        self.valid_phases = {
            'source': ValidationPhase(name='source', loader='valid'),
            'target': ValidationPhase(name='target', loader='valid2')
        }
        
        self.early_stopping_trigger = 'target'
       
        
        # Modify dataloader dict to match the valid phases.
        
        self.latent_weight = 0.1
        self.rimg_weight = 0.5e-2
        self.center_weight = 40.
        self.depth_weight = 50.
        self.feature_weight = 10
        self.domain_weight = 0.01
        
    def data_preprocess(self, mode, data2):

        def to_device(data):
            if self.preprocess:
                data = self.preprocess(data, self.modality)
            data = {key: data[key].to(torch.float32).to(self.device) for key in self.modality if key in data}
            if 'tag' in data:
                data['tag'] = data['tag'].to(torch.int32).to(self.device)
                data['tag'][data['tag']==4] = 1  # Replace A308 with A308T
            return data
    
        # data is tuple of source and target
        source_data, target_data = data2
        
        source_data = to_device(source_data)
        target_data = to_device(target_data)
            
        return source_data, target_data
    
    def calculate_lambda(self, max_iter=300):
        # Sigmoid schedule for lambda: 2 / (1 + exp(-10 * p)) - 1
        # where p is the proportion of iterations completed
        p = self.current_ep() / max_iter
        lambda_value = 2 / (1 + np.exp(-10 * p)) - 1
        self.lambda_ = min(lambda_value, 1)
        return min(lambda_value, 1)
        
    def kd_loss(self, mu_s, logvar_s, mu_t, logvar_t):
        mu_loss = self.mse_sum(mu_s, mu_t) / mu_s.shape[0]
        logvar_loss = self.mse_sum(logvar_s, logvar_t) / logvar_s.shape[0]
        latent_loss = self.alpha * mu_loss + (1 - self.alpha) * logvar_loss

        return latent_loss, mu_loss, logvar_loss
    
    def feature_loss(self, feature_s, feature_t):
        feature_loss = self.mse(feature_s, feature_t)
        return feature_loss
    
    def dann_loss(self, source_data, target_data, s_feature, reverse_feature):
        lambda_ = self.calculate_lambda() if reverse_feature else -1.
        
        _, target_feature, target_z, target_mu, target_logvar = self.models['csien'](csi=target_data['csi'], pd=target_data['pd'])
        
        dann_features = torch.cat((s_feature, target_feature), dim=0)
        
        # REVERSING DEPENDS ON LAMBDA
        dann_features = GradientReversalLayer.apply(dann_features, lambda_)
    
        domain_preds = self.models['dmnde'](dann_features.to(self.device))
        domain_labels = torch.cat((source_data['tag'][..., 0], target_data['tag'][..., 0])).to(torch.int64).to(self.device)
        
        domain_loss = self.adv(domain_preds, domain_labels)

        with torch.no_grad():
            domain_acc_preds = torch.argmax(domain_preds, dim=1)
            domain_acc_loss = torch.sum(domain_acc_preds == domain_labels) / domain_preds.shape[0]
        
        return domain_loss, domain_acc_loss, domain_preds, domain_labels

    def calculate_loss(self, data2, reverse_feature=False):
        
        def outputs(data, mode='student'):
            if mode == 'student':
                feature, dann_feature, z, mu, logvar = self.models['csien'](csi=data['csi'], pd=data['pd'])
            elif mode == 'teacher':
                with torch.no_grad():
                    z, mu, logvar, feature = self.models['imgen'](rimg)
                    dann_feature = None
            center, depth = self.models['ctrde'](feature)
            cimage = self.models['cimgde'](z)
            rimage = self.models['rimgde'](z.to(self.device2))
            return {
                'feature': feature,
                'dann_feature': dann_feature,
                'z'      : z,
                'mu'     : mu,
                'logvar' : logvar,
                'center' : center,
                'depth'  : depth,
                'cimage' : cimage,
                'rimage' : rimage
                }
                
        def s_losses(s_out, t_out, data):
            # 3-level loss
            feature_loss = self.feature_loss(s_out['feature'], t_out['feature'])
        
            latent_loss, mu_loss, logvar_loss = self.kd_loss(s_out['mu'], s_out['logvar'], t_out['mu'], t_out['logvar'])
        
            center_loss = self.mse(s_out['center'], torch.squeeze(data['center']))
            depth_loss = self.mse(s_out['depth'], torch.squeeze(data['depth']))
            image_loss = self.mse_sum(s_out['rimage'].to(self.device), rimg) / s_out['rimage'].shape[0]
            
            loss = latent_loss * self.latent_weight +\
                    image_loss * self.rimg_weight +\
                    center_loss * self.center_weight +\
                    depth_loss * self.depth_weight
                    
            if self.with_feature_loss:
                loss += feature_loss * self.feature_weight
                
            return {
                'LOSS'   : loss,
                'MU'     : mu_loss * self.latent_weight,
                'LOGVAR' : logvar_loss * self.latent_weight,
                'FEATURE': feature_loss * self.feature_weight,
                'IMG'    : image_loss * self.rimg_weight,
                'CTR'    : center_loss * self.center_weight,
                'DPT'    : depth_loss * self.depth_weight,
                }
            
        source_data, target_data = data2
            
        cimg = torch.where(source_data['cimg'] > 0, 1., 0.)
        rimg = source_data['rimg']
        s_out = outputs(source_data, mode='student')
        with torch.no_grad():
            t_out = outputs(source_data, mode='teacher')
        s_loss = s_losses(s_out, t_out, source_data)
        domain_loss, domain_acc_loss, domain_preds, domain_labels = self.dann_loss(source_data, 
                                                                                   target_data, 
                                                                                   s_out['dann_feature'], 
                                                                                   reverse_feature)
        
        TMP_LOSS = {key: value for key, value in s_loss.items()}
        TMP_LOSS['DOM'] = domain_loss * self.domain_weight
        TMP_LOSS['DOM_ACC'] = domain_acc_loss
        TMP_LOSS['LOSS'] += domain_loss * self.domain_weight
        
        PREDS = {
            'R_GT'    : rimg,
            'C_GT'    : cimg,
            'T_LATENT': torch.cat((t_out['mu'], t_out['logvar']), -1),
            'S_LATENT': torch.cat((s_out['mu'], s_out['logvar']), -1),
            'TR_PRED' : t_out['rimage'],
            'R_PRED'  : s_out['rimage'],
            'TC_PRED' : t_out['cimage'],
            'SC_PRED' : s_out['cimage'],
            'GT_CTR'  : source_data['center'],
            'S_CTR'   : s_out['center'],
            'T_CTR'   : t_out['center'],
            'GT_DPT'  : source_data['depth'],
            'S_DPT'   : s_out['depth'],
            'T_DPT'   : t_out['depth'],
            'DOM_PRED': domain_preds,
            'DOM_GT' : domain_labels,
            'TAG'     : source_data['tag'],
            'IND'     : source_data['ind']
                }
        
        return PREDS, TMP_LOSS

    def plot_test(self, select_ind=None, select_num=8, autosave=False, **kwargs):
        figs: dict = {}
        self.losslog.generate_indices(select_ind, select_num)

        figs.update(self.losslog.plot_predict(plot_terms=('R_GT', 'TR_PRED', 'R_PRED'), title='RIMG_PRED'))
        figs.update(self.losslog.plot_predict(plot_terms=('C_GT', 'TC_PRED', 'SC_PRED'), title='CIMG_PRED'))
        figs.update(self.losslog.plot_latent(plot_terms=('T_LATENT', 'S_LATENT')))
        figs.update(self.losslog.plot_center())
        figs.update(self.losslog.plot_domain())
        figs.update(self.losslog.plot_test_cdf(plot_terms='all'))
        #figs.update(self.losslog.plot_tsne(plot_terms=('GT', 'T_LATENT', 'S_LATENT')))
        print(f"Domain accuracy = {np.mean(self.losslog.loss['DOM_ACC'].log['test'])}")
        print(f"Domain loss = {np.mean(self.losslog.loss['DOM'].log['test'])}")
        if autosave:
            for filename, fig in figs.items():
                fig.savefig(f"{self.save_path}{filename}")
            with open(f"{self.save_path}{self.name}_dann.txt", 'w') as logfile:
                logfile.write(f"{self.name}\n"
                    f"Domain accuracy = {np.mean(self.losslog.loss['DOM_ACC'].log['test'])}\n"
                    f"Domain loss = {np.mean(self.losslog.loss['DOM'].log['test'])}\n")

if __name__ == '__main__':
    cc = ImageEncoder(latent_dim=128).to(torch.device('cuda:7'))
    summary(cc, input_size=(1, 128, 128))
    
