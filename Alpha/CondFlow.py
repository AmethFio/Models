import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
import os
from Trainer import BasicTrainer, TrainingPhase, ValidationPhase
from Loss import MyLoss
import torch.nn.functional as F

class TeacherTrainer(BasicTrainer):
    def __init__(self,
                 *args, **kwargs):
        super(TeacherTrainer, self).__init__(*args, **kwargs)

        self.modality = {'rimg', 'tag', 'ind'}
        self.beta = 0.5

        self.loss_terms = ('LOSS', 'KL', 'RECON')
        self.pred_terms = ('R_GT',
                           'R_PRED',
                           'TAG', 'IND')

        self.img_loss = nn.BCEWithLogitsLoss(reduction='sum')
        
        self.losslog = MyLoss(name=self.name,
                           loss_terms=self.loss_terms,
                           pred_terms=self.pred_terms,
                           depth=True)
        
        self.teacher = Teacher(device=self.device)
        self.models = {'imgen': self.teacher.imgen,
                       'imgde': self.teacher.imgde,
                       }

        self.weights = {
            'KL': self.beta,
            'RECON': 1.e-3,
        }
        
    def kl_loss(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss

    def calculate_loss(self, data):
        rimg = data['rimg']
        
        ret = self.teacher(rimg)

        kl_loss = self.kl_loss(ret['mu'], ret['logvar'])
        recon_loss = self.img_loss(ret['rimage'], rimg) / ret['rimage'].shape[0]

        LOSS = kl_loss * self.beta
        LOSS += recon_loss * self.weights['R_RECON']

        TEMP_LOSS = {'LOSS': LOSS,
                    'KL': kl_loss,
                    'RECON': r_recon_loss,
                    }
        
        PREDS = {'R_GT': rimg,
                'R_PRED': ret['rimage'],
                'LAT': torch.cat((ret['mu'], ret['logvar']), -1),
                'TAG': data['tag'],
                'IND': data['ind']
                }
        
        return PREDS, TEMP_LOSS

    def plot_test(self, select_ind=None, select_num=8, autosave=False, **kwargs):
        figs: dict = {}
        self.losslog.generate_indices(select_ind, select_num)

        figs.update(self.losslog.plot_predict(plot_terms=('R_GT', 'R_PRED')))
        figs.update(self.losslog.plot_latent(plot_terms={'LAT'}))
        # figs.update(self.loss.plot_test(plot_terms='all'))
        # figs.update(self.loss.plot_tsne(plot_terms=('GT', 'LAT', 'PRED')))

        if autosave:
            for filename, fig in figs.items():
                fig.savefig(f"{self.save_path}{filename}")


class MMDLoss(nn.Module):
	def __init__(self, kernel_mul = 2.0, kernel_num = 5):
		super(MMDLoss, self).__init__()
		self.kernel_num = kernel_num
		self.kernel_mul = kernel_mul
		self.fix_sigma = None
		return
	def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
		n_samples = int(source.size()[0])+int(target.size()[0])
    	total = torch.cat([source, target], dim=0)

    	total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    	total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    	L2_distance = ((total0-total1)**2).sum(2) 
    	if fix_sigma:
    		bandwidth = fix_sigma
    	else:
    		bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    	bandwidth /= kernel_mul ** (kernel_num // 2)
    	bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    	kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    	return sum(kernel_val)

    def forward(self, source, target):
    	batch_size = int(source.size()[0])
    	kernels = guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
    	XX = kernels[:batch_size, :batch_size]
    	YY = kernels[batch_size:, batch_size:]
    	XY = kernels[:batch_size, batch_size:]
    	YX = kernels[batch_size:, :batch_size]
    	loss = torch.mean(XX + YY - XY -YX)
    	return loss


class StudentTrainer(BasicTrainer):
    def __init__(self,
                 *args, **kwargs):
        super(StudentTrainer, self).__init__(*args, **kwargs)

        self.modality = {'cimg', 'rimg', 'csi', 'center', 'depth', 'pd', 'tag', 'ctr', 'dpt', 'ind'}

        self.loss_terms = ('LOSS', 'Z_PROB', 'JACOB', 'FEA_MMD', 'RECON')
        self.pred_terms = ('R_GT',
                           'TR_PRED',
                           'R_PRED',
                           'TAG', 'IND')

        self.mmd_loss = MMDLoss()
        self.recon_loss = nn.MSELoss()

        self.reverse = False

        self.losslog = MyLoss(name=self.name,
                              loss_terms=self.loss_terms,
                              pred_terms=self.pred_terms,
                              depth=True)

        self.teacher = Teacher(device=self.device)
        self.student = Student(device=self.device, teacher=self.teacher)
        
        self.models = {
            'imgen' : self.teacher.imgen,
            'imgde': self.teacher.imgde,
            'csien' : self.student.csien,
            'flow': self.student.flow
                }

        self.training_phases = {'main': TrainingPhase(name = 'main',
                                                        train_module = ['csien', 'flow'],
                                                        eval_module = ['imgen', 'imgde'],
                                                        verbose=False
                                                        )}

        self.flow_weight = 1.
        self.fea_weight = 1.
        self.recon_weight = 1.
        
    def calculate_loss(self, data):
        ret = self.student(data['csi'])
        
        z = ret['z']
        log_prob_z = -0.5 * (z ** 2).sum(dim=1) - 0.5 * z.size(1) * torch.log(torch.tensor(2 * torch.pi))

        log_det = ret['log_det']
        flow_loss = -(log_prob_z + log_det).mean()  # 负对数似然

        feature_loss = self.mmd_loss(ret['s_fea'], ret['t_fea'])

        recon_loss = self.recon_loss(ret['s_rimage'], ret['t_rimage'])

        loss = flow_loss * self.flow_weight
        loss += feature_loss * self.fea_weight
        loss += recon_loss * self.recon_weight

        # Consistency loss: forward (flow) + reverse (recon) + KD (feature)
        # IT IS RECOMMENDED THAT THE REVERSE LOSS (RECON) BE ADDED AFTER WARMING UP

        TMP_LOSS = {
            'LOSS': loss,
            'Z_PROB': log_prob_z * self.flow_weight,
            'JACOB': log_det * self.flow_weight,
            'FEA_MMD': feature_loss * self.fea_weight,
            'RECON': recon_loss * self.recon_weight
        }

        PREDS = {
            'R_GT': data['rimg'],
            'TR_PRED': ret['t_rimage'],
            'R_PRED': ret['s_rimage'],
            'TAG': data['tag'],
            'IND': data['ind']
        }

        return PREDS, TMP_LOSS

    def plot_test(self, select_ind=None, select_num=8, autosave=False, **kwargs):
            figs: dict = {}
            self.losslog.generate_indices(select_ind, select_num)

            figs.update(self.losslog.plot_predict(plot_terms=('R_GT', 'TR_PRED', 'R_PRED'), title='RIMG_PRED'))

            if autosave:
                for filename, fig in figs.items():
                    fig.savefig(f"{self.save_path}{filename}")