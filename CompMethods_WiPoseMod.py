import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
import os
from Trainer import BasicTrainer, TrainingPhase, ValidationPhase
from Loss import MyLossLog
from Models.Structure.Model import *
import torch.nn.functional as F


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

version = 'WiPoseMod'

class Preprocess:
    def __init__(self, new_size=(64, 64)):
        self.new_size = new_size
        self.batch_size = 32

    def transform(self, tensor):
        return F.interpolate(tensor, size=self.new_size, mode='bilinear', align_corners=False)
    
    def __call__(self, data, modalities):
        """
        Preprocess after retrieving data
        """
        
        #  Transform images
        data['rimg'] = self.transform(data['rimg'])

        # CSI: Window length = 100, 3 rx
        data['csi'] = data['csi'].permute(0, 2, 1, 3) # batch * sub * packet * rx

        return data

class Encoder(nn.Module):
    name = 'csien'
    
    def __init__(self):
        super(Encoder, self).__init__()
        
        channels = [1, 64, 128, 64]
        cnn = []
        for in_chn, out_chn in zip(channels[:-1], channels[1:]):
            cnn.extend([
                nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_chn),
                nn.LeakyReLU(negative_slope=0.02),
                nn.Dropout(0.2)
            ])
        
        self.cnn = nn.Sequential(*cnn)
        
        self.lstm = nn.LSTM(30 * 3, 544, 3, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(544, 128)
        

    def forward(self, x):

        x = self.cnn(x.view(-1, 1, 64, 64))
        fea, _ = self.lstm(x.permute(0, 2, 1, 3).reshape(-1, 100, 30 * 3))
        out = self.fc(fea[:, -1, :])
        
        return out
    

class ImageDecoder(nn.Module):
    name = 'imgde'
    
    def __init__(self, batchnorm='identity', latent_dim=128, active_func=nn.Sigmoid(), *args, **kwargs):
        super(ImageDecoder, self).__init__(*args, **kwargs)

        self.batchnorm = batchnorm
        self.latent_dim = latent_dim
        self.active_func = active_func
        
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
    

class WiPoseMod(nn.Module):

    def __init__(self, device=None):
        super(WiPoseMod, self).__init__()

        self.csien = Encoder()
        self.imgde = ImageDecoder(latent_dim=128)

        if device is not None:
            self.csien = self.csien.to(device)
            self.imgde = self.imgde.to(device)

    def forward(self, data):
        
        z = self.csien(data['csi'])
        recon = self.imgde(z)

        ret = {
        'z'      : z,
        're_img' : recon
                }

        return ret


class WiPoseModTrainer(BasicTrainer):
    
    def __init__(self,
                 *args, **kwargs
                 ):
        
        super(WiPoseModTrainer, self).__init__(*args, **kwargs)
    
        self.modality = {'rimg', 'csi', 'tag', 'ind'}
        self.preprocess = Preprocess()

        self.dis_loss = nn.BCEWithLogitsLoss(reduction='sum')
        
        self.loss_terms = (['LOSS'])
        self.pred_terms = ('GT', 'PRED', 'TAG', 'IND')
        
        self.losslog = MyLossLog(name=self.name,
                           loss_terms=self.loss_terms,
                           pred_terms=self.pred_terms)

        
        self.model = LSTMCNN(device=self.device)
        self.models = {m: getattr(self.model, m) for m in ['csien', 'imgde']}
        
    def calculate_loss(self, data):
        
        ret = self.model(data)
        recon_loss = self.recon_lossfunc(ret['re_img'], data['rimg']) / ret['re_img'].shape[0]
            
        TEMP_LOSS = {
            'LOSS': recon_loss
        }
        
        PREDS = {
            'GT'      : data['rimg'],
            'PRED'    : ret['re_img'],
            'TAG'     : data['tag'],
            'IND'     : data['ind']
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

    pass
