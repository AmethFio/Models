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

version = 'ThroughWall'

def reparameterize(mu, logvar):
    """
    Reparameterization trick in VAE.
    :param mu: mu vector
    :param logvar: logvar vector
    :return: reparameterized vector
    """
    eps = torch.randn_like(mu)
    return mu + eps * torch.exp(logvar / 2)


class ImageEncoder(nn.Module):
    
    name = 'CompTWimgen'
    
    def __init__(self, device=None):
        super(ImageEncoder, self).__init__()
        
        block = [1, 48, 96, 128, 192, 256, 512]
        cnn = []
        
        for in_ch, out_ch in zip(block[:-1], block[1:]):
            cnn.extend([
                nn.Conv2d(in_ch, out_ch, 3, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU()
            ])

        self.cnn = nn.Sequential(cnn)
        
        self.fc = nn.Linear(512* 4, 512 * 4 * 2)

        if device is not None:
            self.cnn = self.cnn.to(device)
            self.fc = self.fc.to(device)

    def forward(self, data):
        fea = self.cnn(data['rimg'])
        out = self.fc(fea.view(-1, 512 * 4))
        mu, logvar = out[..., :512 * 4], out[..., 512 * 4:]
        z = reparameterize(mu, logvar)

        return z, mu, logvar
    

class ImageDecoder(nn.Module):
    
    name = 'CompTWimgde'
    
    def __init__(self, device=None):
        super(ImageDecoder, self).__init__()
        
        block = [512, 256, 192, 128, 96, 48, 1]
        cnn = []
        
        for in_ch, out_ch in zip(block[:-1], block[1:]):
            cnn.extend([
                nn.ConvTranspose2d(in_ch, out_ch, 2, 2, 0),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU()
            ])
            
        cnn.extend([
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.Tanh()
        ])

        self.cnn = nn.Sequential(cnn)
        

        if device is not None:
            self.cnn = self.cnn.to(device)

    def forward(self, z):
        out = self.cnn(z.view(-1, 512, 2, 2))

        return out
    
    
class CSIEncoder(nn.Module):
    
    name = 'CompTWcsien'
    
    def __init__(self, device=None):
        super(CSIEncoder, self).__init__()
        
        self.fc1 = nn.Linear(30 * 3, 128)
        self.fc2 = nn.Linear(151 * (128 + 20), 512 * 4 * 2)
        self.batch_size = 32
        self.seq_len = 151
        
    def forward(self, data, time_emb):
        x = data['csi'].view(self.batch_size, self.seq_len, -1)
        
        x = self.fc1(x)
        x = torch.cat(x.view(self.batch_size, -1), time_emb)
        x = self.fc2(x) # 32 * (151 * (128 + 20)
        
        mu, logvar = out[..., :512 * 4], out[..., 512 * 4:]
        z = reparameterize(mu, logvar)

        return z, mu, logvar
        

class MoPoE(nn.Module):
    
    name = 'CompTWmopoe'
    
    def __init__(self, device=None):
        super(MoPoE, self).__init__()
        
        self.num_experts = 3
        # M1, M2, M1 + M2
        # Calculate the 3rd mu and sigma manually in the simple form
        
        self.logits = nn.Parameter(torch.randn(self.num_experts))
        # Mixture weights
        
    def poe_distribution(self, mu, logvar):
        """
        Calculate Product of Experts (PoE) for a single expert.
        This assumes Gaussian distribution for each expert.
        """
        sigma = torch.exp(0.5 * logvar)
        
        return torch.distributions.Normal(mu, sigma)
    
    def mixture_logvar(self, logvar1, logvar2, w1, w2):
        """
        Calculate the mixture log-variance from two log-variances using mixture weights.
        """
        # Convert logvar to variance
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        
        # Calculate the weighted mixture variance
        mixture_variance = w1 * var1 + w2 * var2
        
        # Return the log of the mixture variance (logvar_mixture)
        return torch.log(mixture_variance)
    
    def forward(self, mus, logvars):
        """
        Forward pass for MoPoE:
        1. Compute PoE for each expert of individual modality.
        2. Compute the PoE for mixture modality.
        3. Compute the mixture of experts by weighting each PoE.
        """
        # Step 1: Calculate PoE distributions for each expert
        expert_distributions = [self.poe_distribution(mu, logvar) for mu, logvar in zip(mus, logvars)]
        
        # Step 2: Calculate mixture weights (using softmax for stability)
        weights = torch.softmax(self.logits, dim=0)  # Softmax to get valid mixture weights
        
        # Step 3: Compute mixture log-variance using the learned weights
        # Calculate PoE distribution for the 3rd expert
        mixture_logvar_value = self.mixture_logvar(logvars[0], logvars[1], weights[0], weights[1])
        mixture_mu_value = (weights[0] * mu[0] + weights[1] * mu[1])  # Weighted means
        expert_distributions.append(self.poe_distribution(mixture_mu_value, mixture_logvar_value))
        
        # Step 4: Sample from the mixture of experts
        mixture_dist = torch.distributions.Categorical(weights)
        chosen_expert = mixture_dist.sample((mu.size(0),))  # Choose one expert per sample in the batch
        
        # Step 5: Sample from the selected expert's distribution
        expert_samples = torch.stack([expert_distributions[i].sample() for i in range(self.num_experts)], dim=-1)
        selected_samples = expert_samples.gather(-1, chosen_expert.view(-1, 1).expand(-1, 512 * 4))

        return selected_samples, chosen_expert


class Model(nn.Module):
    
    def __init__(self, device=None):
        super(Model, self).__init__()
        
        self.imgen = ImageEncoder()
        self.imgde = ImageDecoder()
        self.csien = CSIEncoder()
        self.mopoe = MoPoE()
        
        if device is not None:
            for module in ['imgen', 'imgde', 'csien', 'mopoe']:
                getattr(self, module).to(device)
                
        self._device = device
                
    def time_encode(self, x):
        L = 151 # window length
        F = 10 # Embedding length ( /2 )
        
        x -= x[..., 0] # Get relative timestamps
        time_emb = []
        
        for t in x:
            for f in range(F):
                time_emb.extend([torch.sin(2 ** f * torch.pi * t) / (3 * L), 
                            torch.cos(2 ** f * torch.pi * t) / (3 * L)])
                
        time_emb = torch.tensor(time_emb)
        if self._device is not None:
            time_emb.to(self._device)
            
        # 151 * 20 embedding
        return time_emb
                
    def forward(self, data):
        
        dimg = nn.functional.normalize(data['rimg'])
        csi = torch.amplitude(data['csi'])
        time = data['csitime']
        
        z_img, mu_img, logvar_img = self.imgen(dimg)
        z_csi, mu_csi, logvar_csi = self.csien(csi)
        z, exp = self.mopoe([mu_img, mu_csi], 
                               [logvar_img, logvar_csi])
        re_img = self.imgde(z)
        
        ret = {
            'z'         : z,
            'mu_img'    : mu_img,
            'logvar_img': logvar_img,
            'mu_csi'    : mu_csi,
            'logvar_csi': logvar_csi,
            're_img'    : re_img
        }
        
        return ret


class ThroughWallTrainer(BasicTrainer):
    
    def __init__(self,
                 beta=0.5,
                 recon_lossfunc=nn.BCEWithLogitsLoss(reduction='sum'),
                 *args, **kwargs
                 ):
        
        super(ThroughWallTrainer, self).__init__(*args, **kwargs)
    
        self.modality = {'rimg', 'csi', 'csitime', 'tag', 'ind'}

        self.beta = beta
        self.recon_lossfunc = recon_lossfunc
        
        self.loss_terms = ('LOSS', 'KL', 'RECON')
        self.pred_terms = ('GT', 'PRED', 'LAT', 'TAG', 'IND')
        
        self.losslog = MyLossLog(name=self.name,
                           loss_terms=self.loss_terms,
                           pred_terms=self.pred_terms)
        
        self.model = Model(device=self.device)
        self.models = vars(self.model)
        
        
    def kl_loss(self, mu, logvar, prior_mu=0, prior_logvar=0, weights=None):
        """
        Calculate the KL divergence between the learned distribution and the prior (standard normal).
        
        Parameters:
        - mu: Tensor of means for each expert in the mixture (batch_size x input_dim).
        - logvar: Tensor of log-variances for each expert in the mixture (batch_size x input_dim).
        - prior_mu: Mean of the prior (default 0 for standard normal).
        - prior_logvar: Log variance of the prior (default 0 for standard normal).
        - weights: Mixture weights for the experts (softmaxed logits, batch_size).
        
        Returns:
        - kl_loss: KL divergence between the mixture of experts and the prior.
        """
        
        # Standard normal distribution (mean=0, logvar=0)
        prior_var = torch.exp(prior_logvar)
        prior_std = torch.sqrt(prior_var)
        
        # Calculate variances for each expert in the mixture
        var = torch.exp(logvar)  # Variance of the current expert
        
        # KL divergence for each expert, given prior
        kl_experts = 0.5 * (logvar - prior_logvar + (prior_var + (mu - prior_mu)**2) / var - 1)
        
        # Weighted sum of KL divergences for the mixture (PoE)
        if weights is not None:
            kl_loss = torch.sum(weights * kl_experts, dim=0)
        else:
            # If no weights are provided, use equal contribution from each expert
            kl_loss = torch.mean(kl_experts, dim=0)
        
        return kl_loss
        
    def calculate_loss(self, data):
        
        ret = self.model(data)
        
        kl_loss = self.kl_loss(mu=torch.cat(ret['mu_img'], ret['mu_csi'], 0),
                               logvar=torch.cat(ret['logvar_img'], ret['logvar_csi'], 0))
        recon_loss = self.recon_lossfunc(ret['re_img'], data['rimg']) / ret['re_img'].shape[0]
        
        loss = kl_loss * self.beta + recon_loss

        TEMP_LOSS = {
            'LOSS' : loss,
            'KL'   : kl_loss,
            'RECON': recon_loss
              }   
        
        PREDS = {
            'GT': data['rimg'],
            'PRED': ret['re_img'],
            'LAT': ret['z'],
            'TAG': data['tag'],
            'IND': data['ind']
        }
        
        return PREDS, TEMP_LOSS
        
    def plot_test(self, select_ind=None, select_num=8, autosave=False, **kwargs):
        figs: dict = {}
        self.losslog.generate_indices(select_ind, select_num)

        figs.update(self.losslog.plot_predict(plot_terms=('GT', 'PRED')))
        figs.update(self.losslog.plot_latent(plot_terms={'LAT'}))

        if autosave:
            for filename, fig in figs.items():
                fig.savefig(f"{self.save_path}{filename}")


if __name__ == "__main__":

    m = ThroughWallTrainer()
    print(ThroughWallTrainer.preprocess(ThroughWallTrainer, 1))
