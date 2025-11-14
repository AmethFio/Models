import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
import os
from Trainer import BasicTrainer, TrainingPhase, ValidationPhase
from Loss import MyLoss
import torch.nn.functional as F
import torch.fft as fft
from CondFlowModels import *

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
 
 
class StudentPretrainer(BasicTrainer):
    def __init__(self, *args, **kwargs):
        super(StudentPretrainer, self).__init-_(*args, **kwargs)
        
        self.modality = {'csi'}
        
        self.loss_terms = ()
        self.pred_terms = ('CSI_GT', 'CSI_PRED')
        
        self.csi_loss = None
        
        self.losslog = MyLoss(name=self.name, loss_terms=self.loss_terms,pred_terms=self.pred_terms)
        
        self.encoder = SpatioTemporalCompressor()
        self.decoder = CSIDecoder()
        self.encoder.to(self.device)
        self.decoder.to(self.device)
       
        self.models = {'csien': self.encoder,
                       'cside': self.decoder}
        
        self.training_phases =  {'main': TrainingPhase(name = 'main',
                                                            train_module = ['csien', 'cside'],
                                                            eval_module = [],
                                                            verbose=False
                                                            )}
       
    def fft_loss(x, x_hat, lambda_t=0.3, lambda_f=0.7):
        # 时域 L1
        loss_t = F.l1_loss(x_hat, x)

        # 频域 L1（使用复频谱模值）
        X = torch.abs(fft.rfft(x, dim=-1))
        X_hat = torch.abs(fft.rfft(x_hat, dim=-1))
        loss_f = F.l1_loss(X_hat, X)

        return lambda_t * loss_t + lambda_f * loss_f
       
    def calculate_loss(self, data):
        csi_fea = self.encoder(data['csi'])
        recon = self.decoder(csi_fea)
        
        fft_loss = self.fft_loss(recon, data['csi'])
        
        


class StudentTrainer(BasicTrainer):
    def __init__(self,
                 *args, **kwargs):
        super(StudentTrainer, self).__init__(*args, **kwargs)

        self.modality = {'rimg', 'csi', 'tag', 'ind'}

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
                    

class CSIAugmentation:
    def __init__(self, noise_std=0.01, jitter_prob=0.5, jitter_shift=5, norm_mode='per_sample'):
        """
        Args:
            noise_std: 高斯噪声标准差（相对输入值幅度）
            jitter_prob: 时间抖动概率
            jitter_shift: 最大时间平移步数
            norm_mode: 'per_sample' 或 'global'，幅度归一化方式
        """
        self.noise_std = noise_std
        self.jitter_prob = jitter_prob
        self.jitter_shift = jitter_shift
        self.norm_mode = norm_mode

    def add_gaussian_noise(self, x):
        """加性高斯噪声"""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def time_jitter(self, x):
        """时间抖动（随机平移 + 环绕）"""
        if random.random() < self.jitter_prob:
            shift = random.randint(-self.jitter_shift, self.jitter_shift)
            x = torch.roll(x, shifts=shift, dims=-1)
        return x

    def amplitude_normalization(self, x):
        """幅度归一化"""
        if self.norm_mode == 'per_sample':
            mean = x.mean(dim=(-1, -2), keepdim=True)
            std = x.std(dim=(-1, -2), keepdim=True) + 1e-6
            return (x - mean) / std
        elif self.norm_mode == 'global':
            mean = x.mean()
            std = x.std() + 1e-6
            return (x - mean) / std
        else:
            raise ValueError("norm_mode should be 'per_sample' or 'global'")

    def __call__(self, x):
        """对输入 CSI 执行全部增强"""
        x = self.add_gaussian_noise(x)
        x = self.time_jitter(x)
        x = self.amplitude_normalization(x)
        return x