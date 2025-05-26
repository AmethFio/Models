import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
import os
from Trainer import BasicTrainer, TrainingPhase, ValidationPhase
from Loss import MyLossLog
from Models.Structure.Model import *

##############################################################################
# -------------------------------------------------------------------------- #
# Notes
#
# 1) Wi2Vi uses 56x3x3x29 CSI, while we use 30x3x3x100
#
# 2) Video frames are aligned with the first packets of CSI
#
# 3) Wi2Vi video FPS = 30 -> 6, CSI rate = 100Hz
#
# 4) Wi2Vi train:test = 95:5
#
# 5) Wi2Vi lr=2e-3 and lower; epoch=1000; batch size=32
#
# 6) Wi2Vi outputs 320x240 images
# -------------------------------------------------------------------------- #
##############################################################################

version = 'CompV3'


class DropIn(nn.Module):
    def __init__(self, num_select):
        super(DropIn, self).__init__()
        self.num_select = num_select

    def forward(self, x):
        i = torch.randperm(x.shape[-1])[:self.num_select]
        return x[..., i]


class Wi2Vi(nn.Module):
    name = 'wi2vi'

    def __init__(self, batchnorm='instance'):
        super(Wi2Vi, self).__init__()

        # 56X29X18 (3x3xamp&phase)
        self.batchnorm = batchnorm
        self.Dropin = DropIn(17)
        # self.EncoderOriginal = nn.Sequential(
        #     # 56x17x18
        #     nn.Conv2d(18, 64, kernel_size=3, stride=1, padding=0),
        #     nn.InstanceNorm2d(64),
        #     nn.ReLU(),
        #     # 56x15x64
        #     nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
        #     nn.InstanceNorm2d(128),
        #     nn.ReLU(),
        #     # 26x7x128
        #     nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1),
        #     nn.InstanceNorm2d(256),
        #     nn.ReLU(),
        #     # 12x3x256
        #     nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=1),
        #     nn.InstanceNorm2d(512),
        #     nn.ReLU(),
        #     # 5x1x512
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm2d(512),
        #     nn.ReLU(),
        #     # 5x1x512
        # )

        self.Encoder = nn.Sequential(
            # 30x17x6
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 28x15x64
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # 13x7x128
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            # 11x5x256
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            # 5x2x512
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            # 5x2x512
        )

        self.Translator_A = nn.Sequential(
            # Fill in the flattened output shape of Encoder.
            nn.Linear(5120, 972),
            nn.LeakyReLU()
        )

        self.Translator_B = nn.Sequential(
            # 36x27
            nn.ReflectionPad2d(1),
            # 38x29
            nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32x23x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 16x12x64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # 8x6x128
        )

        self.Decoder = nn.Sequential(
            # 8x6x128
            # nn.ReflectionPad2d(1),
            # 10x8x128
            ResidualBlock(128, 128, self.batchnorm),
            # 8x6x128
            ResidualBlock(128, 128, self.batchnorm),
            # 8x6x128
            ResidualBlock(128, 128, self.batchnorm),
            # 8x6x128
            Interpolate(size=(12, 16)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            # 14x10x64
            Interpolate(size=(20, 28)),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
            # 26x18x32
            Interpolate(size=(36, 52)),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0),
            # 50x34x16
            Interpolate(size=(68, 100)),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=0),
            # 98x66x8
            Interpolate(size=(132, 196)),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=0),
            # 194x130x4
            Interpolate(size=(260, 388)),
            nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=0),
            # 386x258x2
            nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=0),
            nn.InstanceNorm2d(32),
            nn.Sigmoid()
            # 382x254x1
        )

    def forward(self, x):
        x = self.Dropin(x)
        x = self.Encoder(x)
        x = self.Translator_A(x.view(-1, 5120))
        x = self.Translator_B(x.view(-1, 1, 27, 36))
        x = self.Decoder(x)

        return x[..., 7:247, 31:351]

    def __str__(self):
        return 'Wi2Vi'



class CompTrainer(BasicTrainer):
    def __init__(self, mode='wi2vi',
                 mask=False,
                 *args, **kwargs):
        super(CompTrainer, self).__init__(*args, **kwargs)

        assert mode in ('wi2vi', 'ae', 'vae', 'ae_t', 'vae_t', 'cnnlstm')

        self.mode = mode
        self.mask = mask
        self.beta = kwargs['beta'] if 'beta' in kwargs.keys() else 0.5
        self.image_loss = nn.BCEWithLogitsLoss(reduction='sum') if self.mode=='ae' else nn.MSELoss(reduction='sum')
        self.mse = nn.MSELoss(reduction='sum')
        self.loss_terms = {'LOSS'}
        self.pred_terms = ('R_GT', 'R_PRED', 'TAG', 'IND') if mode == 'wi2vi' else ('R_GT', 'R_PRED', 'LAT', 'TAG', 'IND')

        self.models = {key: value.to(self.device) for key, value in self.models.items()}

        self.losslog = MyLossLog(name=self.name,
                           loss_terms=self.loss_terms,
                           pred_terms=self.pred_terms)
        
        # self.valid_phases = {
        #     'main': ValidationPhase(name='main', loader='valid'),
        #     'target': ValidationPhase(name='target', loader='valid2')
        # }

    def vae_loss(self, pred, gt, mu, logvar):
        recon_loss = self.image_loss(pred, gt) / pred.shape[0]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss * self.beta
        return loss, kl_loss, recon_loss

    def calculate_loss(self, data):
        img = torch.where(data['rimg'] > 0, 1., 0.) if self.mask else data['rimg']

        if self.mode == 'wi2vi':
            output = self.models['wi2vi'](data['csi'])
            loss = self.image_loss(output, img) / output.shape[0]

            PREDS = {'R_GT': img,
                    'R_PRED': output,
                    'TAG': data['tag'],
                    'IND': data['ind']}
            TMP_LOSS = {'LOSS': loss}

        elif self.mode == 'ae':
            latent, output = self.models['ae'](data['csi'])
            loss = self.image_loss(output, img) / output.shape[0]

            PREDS =  {'R_GT': img,
                    'R_PRED': output,
                    'LAT': latent,
                    'TAG': data['tag'],
                    'IND': data['ind']}
            TMP_LOSS = {'LOSS': loss}

        elif self.mode == 'vae':
            z, mu, logvar = self.models['csien'](data['csi'])
            output = self.models['imgde'](z)
            loss, kl_loss, recon_loss = self.vae_loss(output, img, mu, logvar)

            TMP_LOSS = {'LOSS': loss,
                              'KL': kl_loss,
                              'RECON': recon_loss
                              }
            PREDS =  {'R_GT': img,
                    'R_PRED': output,
                    'LAT': torch.cat((mu, logvar), -1),
                    'TAG': data['tag'],
                    'IND': data['ind']
                    }
            self.temp_loss = TMP_LOSS

        elif self.mode == 'ae_t':
            z = self.models['imgen'](img)
            output = self.models['imgde'](z)
            loss = self.image_loss(output, img) / output.shape[0]

            PREDS =  {'R_GT': img,
                    'R_PRED': output,
                    'LAT': z,
                    'TAG': data['tag'],
                    'IND': data['ind']}
            TMP_LOSS = {'LOSS': loss}
        
        elif self.mode == 'vae_t':
            z, mu, logvar = self.models['imgen'](img)
            output = self.models['imgde'](z)
            loss, kl_loss, recon_loss = self.vae_loss(output, img, mu, logvar)

            TMP_LOSS = {'LOSS': loss,
                            'KL': kl_loss,
                            'RECON': recon_loss
                            }
            PREDS =  {'R_GT': img,
                    'R_PRED': output,
                    'LAT': torch.cat((mu, logvar), -1),
                    'TAG': data['tag'],
                    'IND': data['ind']
                    }
            
        elif self.mode == 'cnnlstm':
            z = self.models['csien'](csi)
            output = self.models['imgde'](z)
            
            loss = self.image_loss(output, img) / output.shape[0]
            
            TMP_LOSS = {'LOSS': loss
                        }
            PREDS =  {'R_GT': img,
                    'R_PRED': output,
                    'LAT': z,
                    'TAG': data['tag'],
                    'IND': data['ind']
                    }

        return PREDS, TMP_LOSS

    def plot_test(self, select_ind=None, select_num=8, autosave=False, notion='', **kwargs):
        figs: dict = {}
        self.losslog.generate_indices(select_ind, select_num)

        figs.update(self.losslog.plot_predict(plot_terms=('R_GT', 'R_PRED')))
        figs.update(self.losslog.plot_test(plot_terms='all'))
        figs.update(self.losslog.plot_test_cdf(plot_terms='all'))
        if self.mode in ('ae', 'vae', 'ae_t', 'cnnlstm'):
            figs.update(self.losslog.plot_latent(plot_terms={'LAT'}))
            # figs.update(self.loss.plot_tsne(plot_terms=('GT', 'LAT', 'PRED')))
        # else:
            # figs.update(self.loss.plot_tsne(plot_terms=('GT', 'PRED')))

        if autosave:
            for filename, fig in figs.items():
                fig.savefig(f"{self.save_path}{filename}")


class CompStudentTrainer(BasicTrainer):
    def __init__(self, mask=False, mode='ae_s', alpha=0.8, *args, **kwargs):
        super(CompStudentTrainer, self).__init__(*args, **kwargs)

        self.mask = mask
        self.modality = {'csi', 'rimg', 'tag', 'ind'}

        self.mode = mode
        self.alpha = alpha
        self.image_loss = nn.BCELoss(reduction='sum') if self.mask else nn.MSELoss(reduction='sum')
        self.mse = nn.MSELoss(reduction='sum')
        self.loss_terms = ('LOSS', 'IMG')
        self.pred_terms = ('R_GT', 'T_PRED', 'R_PRED', 'T_LATENT', 'S_LATENT', 'TAG', 'IND')
        self.losslog = MyLossLog(name=self.name,
                           loss_terms=self.loss_terms,
                           pred_terms=self.pred_terms)
        
        self.valid_phases = {
            'main': ValidationPhase(name='main', loader='valid'),
            'target': ValidationPhase(name='target', loader='valid2')
        }
        
        self.training_phases = {
            'main': TrainingPhase(name = 'main',
                                  train_module = ['csien'],
                                  eval_module = ['imgen', 'imgde'],
                                  verbose=False
                                  )
        }
        
    def kd_loss(self, mu_s, logvar_s, mu_t, logvar_t):
        mu_loss = self.mse(mu_s, mu_t) / mu_s.shape[0]
        logvar_loss = self.mse(logvar_s, logvar_t) / logvar_s.shape[0]
        loss = self.alpha * mu_loss + (1 - self.alpha) * logvar_loss
        return loss, mu_loss, logvar_loss

    def calculate_loss(self, data):
        img = torch.where(data['rimg'] > 0, 1., 0.) if self.mask else data['rimg']
        if self.mode == 'ae_s':
            s_z = self.models['csien'](data['csi'])

            with torch.no_grad():
                t_z = self.models['imgen'](img)
                s_output = self.models['imgde'](s_z)
                t_output = self.models['imgde'](t_z)
                image_loss = self.image_loss(s_output, img)

            loss = self.mse(s_z, t_z) / s_z.shape[0]
            # LATENT loss = loss

            self.temp_loss = {'LOSS': loss,
                            'IMG': image_loss}
            return {'R_GT': img,
                    'T_LATENT': t_z,
                    'S_LATENT': s_z,
                    'T_PRED': t_output,
                    'R_PRED': s_output,
                    'TAG': data['tag'],
                    'IND': data['ind']}
            
        elif self.mode == 'vae_s':
            s_z, s_mu, s_logvar = self.models['csien'](data['csi'])

            with torch.no_grad():
                t_z, t_mu, t_logvar = self.models['imgen'](img)
                s_output = self.models['imgde'](s_z)
                t_output = self.models['imgde'](t_z)
                image_loss = self.image_loss(s_output, img)

            loss, mu_loss, logvar_loss = self.kd_loss(s_mu, s_logvar, t_mu, t_logvar)

            self.temp_loss = {'LOSS': loss,
                            'MU': mu_loss,
                            'LOGVAR': logvar_loss,
                            'IMG': image_loss}
            return {'R_GT': img,
                    'T_LATENT': torch.cat((t_mu, t_logvar), -1),
                    'S_LATENT': torch.cat((s_mu, s_logvar), -1),
                    'T_PRED': t_output,
                    'R_PRED': s_output,
                    'TAG': data['tag'],
                    'IND': data['ind']}
        

    def plot_test(self, select_ind=None, select_num=8, autosave=False, notion='', **kwargs):
        figs: dict = {}
        self.losslog.generate_indices(select_ind, select_num)

        figs.update(self.losslog.plot_predict(plot_terms=('R_GT', 'T_PRED', 'R_PRED'), title='RIMG_PRED'))
        figs.update(self.losslog.plot_latent(plot_terms=('T_LATENT', 'S_LATENT'), ylim=None))
        figs.update(self.losslog.plot_test(plot_terms='all'))
        figs.update(self.losslog.plot_test_cdf(plot_terms='all'))
        # figs.update(self.losslog.plot_tsne(plot_terms=('GT', 'T_LATENT', 'S_LATENT')))

        if autosave:
            for filename, fig in figs.items():
                fig.savefig(f"{self.save_path}{filename}")


if __name__ == "__main__":
    #m1 = Wi2Vi()
    #summary(m1, input_size=(6, 30, 30))

    m2 = CSIEncoder(middle_dim=512*7*75, latent_dim=128).to(torch.device('cuda:1'))
    summary(m2, input_size=(6, 30, 300))

