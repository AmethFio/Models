import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


# from torchvision.ops import complete_box_iou_loss
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append('..')
from Trainer import BasicTrainer, TrainingPhase, ValidationPhase
from Loss import MyLossLog, MyLossCTR
from Losses import NCCMSELoss

from Alpha.VViT_Model import *
from Structure.StandardStudent import Student, StudentTrainer

feature_length = 512 * 7
steps = 25

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