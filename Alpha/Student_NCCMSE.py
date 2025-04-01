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

from StandardTeacher import Teacher
from StandardStudent import Student, StudentTrainer

feature_length = 512 * 7
steps = 25

StudentTrainer.img_loss = NCCMSELoss(dims=(1,2,3), reduction='sum')

if __name__ == '__main__':
    pass
    
