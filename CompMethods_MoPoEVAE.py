import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
import os
from Trainer import BasicTrainer, TrainingPhase, ValidationPhase
from Loss import MyLossLog
from Models.Structure.Model import *
import torch.nn.functional as F

version = 'MoPoEVAE'

import math
from torch.distributions import Normal as _Normal
from itertools import combinations
from scipy.stats import norm

EPS = 1e-8

# set numpy seed
np.random.seed(0)




class Preprocess:
    def __init__(self, new_size=(128, 128)):
        self.new_size = new_size
        self.batch_size = 32

    def transform(self, tensor):
        ten =  F.interpolate(tensor, size=self.new_size, mode='bilinear', align_corners=False)
        # Normalize
        # normalize with channel means and stds
        # ten = transforms.Normalize(self.imgMean, self.imgStd)(image)
        return ten
    
    @staticmethod
    def encode_time(x, L=10, window_size=151):
        window_size *= 3
        frequencies = torch.tensor([2**i for i in range(L)], dtype=torch.float3)
        x = x / window_size
        pos_enc = torch.cat([torch.sin(frequencies[:, None] * torch.pi * x),
                             torch.cos(frequencies[:, None] * torch.pi * x)], dim=0)
        return pos_enc
    
    def __call__(self, data, modalities):
        """
        Preprocess after retrieving data
        """
        
        #  Transform images
        if self.new_size and 'rimg' in modalities:
            data['rimg'] = self.transform(data['rimg'])

        if 'csi' in modalities:
            data['csi'] = torch.abs(data['csi']) # 3 Rx
            
        if 'csi_ind' in modalities:
            data['csi_ind'] = self.encode_time(data['csi_ind'])
            
        if 'img_ind' in modalities:
            data['img_ind'] = self.encode_time(data['img_ind'])

        return data


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)
    

# Image Variational Autoencoder
class ImageVAE(nn.Module):

    def __init__(self,
                 time_input_dim = 20,
                 z_dim = 128):
        super(ImageVAE, self).__init__()

        self.latent_dim = z_dim
        self.beta = 1.

        hidden_dims = [1, 48, 96, 128, 192, 256, 512]
        modules = []
    
        for ich, och in zip(hidden_dims[:-1], hidden_dims[1:]):
            modules.extend([
                nn.Conv2d(ich, och, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(och),
                nn.LeakyReLU()
            ])

        self.image_encoder = nn.Sequential(*modules)
        self.time_encoder = MLP(time_input_dim,
                                time_input_dim//2,
                                time_input_dim//4)
        
        self.latent_encoder = nn.Linear(hidden_dims[-1]*4 + time_input_dim//4, 
                                        z_dim * 2) # 2048 + 27 -> 256
        
        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(z_dim, hidden_dims[-1] * 4)

        for ich, och in zip(hidden_dims[::-1][:-2], hidden_dims[::-1][1:-1]):
            modules.extend([
                nn.ConvTranspose2d(ich, och, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(och),
                nn.LeakyReLU()]
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[1], hidden_dims[1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[1], out_channels=1,kernel_size=3, padding=1),
            nn.Tanh())
        
        self.logvar_out = nn.Parameter(torch.full((1, 1, 128, 128), -3.0, dtype=torch.float32))

    def encode(self, img_ind, rimg):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        image = self.image_encoder(rimg)
        time = self.time_encoder(img_ind.reshape(32,1,1,-1))
        image = torch.flatten(image, start_dim=1)
        time = time[:, 0, 0, :]
        x = torch.concat([time, image], dim=1)
        mu, logvar = self.latent_encoder(x).chunk(2, dim=-1)

        # clamp logvar to -10 and 10 to avoid numerical instability
        logvar = torch.clamp(logvar, -10, 10)

        return mu, logvar

    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result.view(-1, 512, 2, 2))
        result = self.final_layer(result)
        # s = torch.exp(0.5 * self.logvar_out)
        ll = Normal(result, self.logvar_out)
        return ll, result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, img_ind, rimg):
        mu, logvar = self.encode(img_ind, rimg)
        z = self.reparameterize(mu, logvar)
        result = self.decode(z)
        return mu, logvar
    

# CSI Variational Autoencoder
class CSIVAE(nn.Module):
    def __init__(self,
                 feature_input_dim = 30 * 3,
                 time_input_dim = 20,
                 sequence_length = 151,
                 z_dim = 128):
        super(CSIVAE, self).__init__()

        output = 64

        self.feature_encoder = MLP(input_dim=feature_input_dim,
                                   hidden_dim=feature_input_dim//2,
                                   output_dim=output)

        self.time_encoder = MLP(input_dim=time_input_dim,
                                hidden_dim=time_input_dim//8,
                                output_dim=time_input_dim//16)
        
        self.latent_encoder = MLP(input_dim=output*sequence_length+time_input_dim//16,
                                  hidden_dim=z_dim*2,
                                  output_dim=z_dim*2)
            
        self.latent_decoder = MLP(z_dim, z_dim*2, feature_input_dim)

        self.logvar_out = nn.Parameter(torch.full((1, feature_input_dim), -3.0, dtype=torch.float32))
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, csi_ind, csi):

        # CSI = 32 * 151 * 30 * 3 -> 32 * 151 * 64
        feature = self.feature_encoder(csi.view(32, 151, -1))
        time = self.time_encoder(csi_ind.reshape(32, 1, 1, -1))
 
        # feature aggregation
        feature = feature.reshape(feature.shape[0], -1)
        
        # concatenate features and temporal encoding
        time = time.view(32, -1)
        x = torch.concat([time, feature], dim=1)
        
        mu, logvar = self.latent_encoder(x).chunk(2, dim=-1)

        # clamp logvar to -10 and 10 to avoid numerical instability
        logvar = torch.clamp(logvar, -10, 10)

        return mu, logvar
    
    def decode(self, z):
        result = self.latent_decoder(z)
        # s = torch.exp(0.5 * self.logvar_out)
        ll = Normal(result, self.logvar_out)
        return ll, result

    def forward(self, csi_ind, csi):
        mu, logvar = self.encode(csi_ind, csi)
        z = self.reparameterize(mu, logvar)
        ll, result = self.decode(z)
        return mu, logvar, result
    

class MoPoEVAE(nn.Module):
    r"""
    Mixture-of-Product-of-Experts Variational Autoencoder.

    Code is based on: https://github.com/thomassutter/MoPoE

    Args:
        cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:

            - model.beta (int, float): KL divergence weighting term.
            - encoder.default._target_ (multiviewae.architectures.mlp.VariationalEncoder): Type of encoder class to use.
            - encoder.default.enc_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Encoding distribution.
            - decoder.default._target_ (multiviewae.architectures.mlp.VariationalDecoder): Type of decoder class to use.
            - decoder.default.init_logvar (int, float): Initial value for log variance of decoder.
            - decoder.default.dec_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Decoding distribution.

        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.

    References
    ----------
    Sutter, Thomas & Daunhawer, Imant & Vogt, Julia. (2021). Generalized Multimodal ELBO.
    """

    def __init__(
        self,
        device=None,
        sequence_length=151,
        z_dim=128,
        frequence_L=10,
        log=None,
        training=True,
    ):
        super(MoPoEVAE, self).__init__()
        self.prior_mean = torch.nn.Parameter(torch.zeros(z_dim), requires_grad=False)
        self.prior_logvar = torch.nn.Parameter(torch.zeros(z_dim), requires_grad=False)

        self.subsets = self.set_subsets()
        self.beta = 1
        self.training = training

        self.ll_weighting = 1
            
        self.csivae = CSIVAE(
                feature_input_dim=30 * 3,
                time_input_dim=frequence_L*2,
                sequence_length=sequence_length, 
                z_dim=z_dim
            )
        self.imgvae = ImageVAE(
                time_input_dim=frequence_L*2,
                z_dim=z_dim
            )
        
        if device is not None:
            self.csivae.to(device)
            self.imgvae.to(device)
            self.prior_mean.to(device)
            self.prior_logvar.to(device)
        
        self.prior = Normal(loc=self.prior_mean, logvar=self.prior_logvar)

    def encode(self, csi_ind, csi, img_ind=None, rimg=None):
        r"""Forward pass through encoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): list containing the MoE joint encoding distribution. 
            If training, the model also returns the encoding distribution for each subset. 
        """
        
        mu_csi, logvar_csi = self.csivae.encode(csi_ind, csi)
        if rimg is not None:
            mu_img, logvar_img = self.imgvae.encode(img_ind, rimg)

            mu = torch.stack([mu_csi, mu_img])
            logvar = torch.stack([logvar_csi, logvar_img])
        else:
            mu = [mu_csi]
            logvar = [logvar_csi]

        mu_out = []
        logvar_out = []

        qz_xs = []
        for subset in self.subsets:
            mu_s = mu[subset]
            logvar_s = logvar[subset]
            if len(subset) == 2:
                mu_ = self.prior.loc
                mu_ = mu_.expand(mu[0].shape).to(mu[0].device)
                logvar_ = torch.log(self.prior.variance).to(mu[0].device)
                logvar_ = logvar_.expand(logvar[0].shape)
                mu_ = mu_.unsqueeze(0)
                logvar_ = logvar_.unsqueeze(0)
                mu_s = torch.cat([mu_s, mu_], dim=0)
                logvar_s = torch.cat([logvar_s, logvar_], dim=0)

            mu_s, logvar_s = ProductOfExperts()(mu_s, logvar_s)
            mu_out.append(mu_s)
            logvar_out.append(logvar_s)
            qz_x = Normal(loc=mu_s, logvar=logvar_s)
            qz_xs.append(qz_x)
        mu_out = torch.stack(mu_out)
        logvar_out = torch.stack(logvar_out)

        moe_mu, moe_logvar = MixtureOfExperts()(mu_out, logvar_out)

        qz_x = Normal(loc=moe_mu, logvar=moe_logvar)
        return [qz_xs, qz_x]

    def decode(self, qz_x):
        r"""Forward pass of joint latent dimensions through decoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): A nested list of decoding distributions, px_zs. The outer list has a single element indicating the shared latent dimensions. 
            The inner list is a n_view element list with the position in the list indicating the decoder index.
        """

        px_zs_csi, x_hat_csi = self.csivae.decode(qz_x[0]._sample(training=self.training))
        px_zs_img, x_hat_img = self.imgvae.decode(qz_x[0]._sample(training=self.training))
        px_zs = [px_zs_csi, px_zs_img]
        x_hats = [x_hat_csi,x_hat_img]
        
        return [px_zs], [x_hats]

    def forward(self, data):
        r"""Apply encode and decode methods to input data to generate the joint and subset latent dimensions and data reconstructions. 

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.
        """
        
        qz_xs, qz_x = self.encode(data['csi_ind'], data['csi'], data['img_ind'], data['rimg'])

        px_zs, x_hats = self.decode([qz_x])
        
        x_hat_img = x_hats[0][1]  # Index [0][1] corresponds to img reconstruction
        
        ret = {
            "px_zs": px_zs, 
            "qz_xs_subsets": qz_xs, 
            "qz_x_joint": qz_x,
            're_img': x_hat_img
            }
        return ret

    def set_subsets(self):
        """Create combinations of subsets of views.

        Returns:
            subset_list (list): list of unique combinations of n_views.
        """
        n_views = 2
        xs = list(range(0, n_views))
        tmp = [list(combinations(xs, n+1)) for n in range(len(xs))]
        subset_list = [list(item) for sublist in tmp for item in sublist]
        return subset_list
    

def compute_log_alpha(mu, logvar):
    return (logvar - 2 * torch.log(torch.abs(mu) + 1e-8)).clamp(min=-8, max=8)

class MixtureOfExperts(nn.Module):
    """Return parameters for mixture of independent experts.
    Implementation from: https://github.com/thomassutter/MoPoE

    Args:
    mus (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvars (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mus, logvars):

        num_components = mus.shape[0]
        num_samples = mus.shape[1]
        weights = (1/num_components) * \
            torch.ones(num_components).to(mus[0].device)
        idx_start = []
        idx_end = []
        for k in range(0, num_components):
            if k == 0:
                i_start = 0
            else:
                i_start = int(idx_end[k-1])
            if k == num_components-1:
                i_end = num_samples
            else:
                i_end = i_start + int(torch.floor(num_samples*weights[k]))
            idx_start.append(i_start)
            idx_end.append(i_end)
        idx_end[-1] = num_samples

        mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(num_components)])
        logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(num_components)])

        return mu_sel, logvar_sel

class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mu, logvar):
        var = torch.exp(logvar) + EPS
        T = 1. / (var + EPS)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + EPS)
        return pd_mu, pd_logvar

class Normal(_Normal):
    """Univariate normal distribution. Inherits from torch.distributions.Normal.

    Args:
        loc (int, torch.Tensor): Mean of distribution.
        scale (int, torch.Tensor): Standard deviation of distribution.
    """

    def __init__(
        self,
        loc,
        logvar,
        **kwargs,
    ):
        self.loc = loc
        self.logvar = logvar
        self.scale = logvar.mul(0.5).exp_()+EPS

        super().__init__(loc=self.loc, scale=self.scale)

    @property
    def variance(self):
        return self.scale.pow(2)

    def kl_divergence(self, other):
        logvar0 = self.logvar
        mu0 = self.loc
        logvar1 = other.logvar
        mu1 = other.loc

        return -0.5 * (1 - logvar0.exp()/logvar1.exp() - (mu0-mu1).pow(2)/logvar1.exp() + logvar0 - logvar1)

    
    def sparse_kl_divergence(self):
        """
        Implementation from: https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/blob/master/KL%20approximation.ipynb

        """
        mu = self.loc
        logvar = torch.log(self.variance)
        log_alpha = compute_log_alpha(mu, logvar)
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        neg_KL = (
            k1 * torch.sigmoid(k2 + k3 * log_alpha)
            - 0.5 * torch.log1p(torch.exp(-log_alpha))
            - k1
        )
        return -neg_KL
    

    def log_likelihood(self, x):
        return self.log_prob(x)

    def _sample(self, *kwargs, training=False, return_mean=True):
        if training:
            return self.rsample(*kwargs)

        if return_mean:
            return self.loc
        return self.sample()



class MoPoEVAETrainer(BasicTrainer):
    
    def __init__(self,
                 *args, **kwargs
                 ):
        
        super(ThroughWallTrainer, self).__init__(*args, **kwargs)
    
        self.modality = {'rimg', 'csi', 'csi_ind', 'img_ind', 'tag', 'ind'}
        
        self.loss_terms = ('LOSS', 'KL', 'LL')
        self.pred_terms = ('GT', 'PRED', 'TAG', 'IND')
        
        self.losslog = MyLossLog(name=self.name,
                           loss_terms=self.loss_terms,
                           pred_terms=self.pred_terms)
        
        self.model = MoPoEVAE(device=self.device)
        self.models = vars(self.model)
        
    def calculate_loss(self, data, mode='train'):
        r"""Calculate MoPoE VAE loss.

        Args:
            x (list): list of input data of type torch.Tensor.
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.

        Returns:
            losses (dict): dictionary containing each element of the MoPoE VAE loss.
        """
        gt = data['rimg']
        
        if mode == 'test':
            data['rimg'] = None # Block out input img for test
            
        ret = self.model(data)

        kl = self.calc_kl_moe(fwd_rtn["px_zs"])
        ll = self.calc_ll(x, fwd_rtn["qz_xs_subsets"])

        total = self.beta * kl - ll

        TEMP_LOSS = {
            "LOSS": total, 
            "KL": kl, 
            'LL': ll
            }
        
        PREDS = {
            'GT': gt,
            'PRED': ret['re_img'],
            'IND': data['ind'],
            'TAG': data['tag']
        }
        
        return PREDS, TEMP_LOSS

    def calc_kl_moe(self, qz_xs):
        r"""Calculate KL-divergence between the each PoE subset posterior and the prior distribution.

        Args:
            qz_xs (list): list of encoding distributions.

        Returns:
            (torch.Tensor): KL-divergence loss.
        """
        weight = 1/len(qz_xs)
        kl = 0
        for qz_x in qz_xs:
            kl += qz_x.kl_divergence(self.prior).mean(0).sum()
        return kl*weight
        
    def calc_ll(self, inputs, decoded_distributions):
        """
        Calculate the log-likelihood loss.
        
        Args:
            inputs (dict): Dictionary containing input tensors.
            decoded_distributions (list): List of output distributions from the decoder.
        
        Returns:
            torch.Tensor: Log-likelihood loss.
        """
        log_likelihood = 0
        modality_index = 1  # Assuming index 1 corresponds to the relevant modality
        
        # Extract the appropriate decoded distribution
        decoded_dist = decoded_distributions[0][0][modality_index]
        
        # Retrieve the corresponding input data
        input_data = inputs[modality_index][1]
        
        # Compute log-likelihood and apply weighting
        log_likelihood += decoded_dist.log_likelihood(input_data).mean(0).sum() * self.ll_weighting
    
        return log_likelihood
        
    def plot_test(self, select_ind=None, select_num=8, autosave=False, **kwargs):
        figs: dict = {}
        self.losslog.generate_indices(select_ind, select_num)

        figs.update(self.losslog.plot_predict(plot_terms=('GT', 'PRED')))
        figs.update(self.losslog.plot_latent(plot_terms={'LAT'}))

        if autosave:
            for filename, fig in figs.items():
                fig.savefig(f"{self.save_path}{filename}")
