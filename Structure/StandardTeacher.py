import torch
import torch.nn as nn
import sys
sys.path.append('..')
from ModelVTS_v08F3 import *
from Trainer import BasicTrainer, TrainingPhase, ValidationPhase
from Loss import MyLossLog, MyLossCTR

class Teacher(nn.Module):

    def __init__(self, device=None):
        super(Teacher, self).__init__()

        self.imgen = ImageEncoder(latent_dim=128)
        self.rimgde = ImageDecoder(latent_dim=128)
        self.cimgde = ImageDecoder(latent_dim=128)
        self.ctrde = CenterDecoder()

        if device is not None:
            self.to(device)

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
        self.models = {'imgen': self.teacher.imgen,
                       'cimgde': self.teacher.cimgde,
                       'rimgde': self.teacher.rimgde,
                       'ctrde': self.teacher.ctrde
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