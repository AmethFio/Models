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

version = 'LSTMCNN'

class Preprocess:
    def __init__(self, new_size=(128, 128)):
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

        # CSI: Window length = 200, select 1 antenna
        data['csi'] = data['csi'][..., 0]

        return data

class Encoder(nn.Module):
    name = 'csien'
    
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.lstm = nn.LSTM(30, 128, 2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(128, 64 * 64)
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        
        self.final = nn.Sequential(
            nn.Linear(32 * 8 * 8, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 128), # Changed output dim
            nn.Dropout(0.5),
            nn.LogSoftmax()
            )
        

    def forward(self, x):
        fea, _ = self.lstm(x)
        out = self.fc(fea[:, -1, :])
        out = self.conv(out.view(-1, 1, 64, 64))
        out = self.final(out.view(-1, 32 * 8 * 8))
        
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
    

class LSTMCNN(nn.Module):

    def __init__(self, device=None):
        super(LSTMCNN, self).__init__()

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


class LSTMCNNTrainer(BasicTrainer):
    
    def __init__(self,
                 *args, **kwargs
                 ):
        
        super(LSTMCNNTrainer, self).__init__(*args, **kwargs)
    
        self.modality = {'rimg', 'csi', 'tag', 'ind'}
        self.preprocess = Preprocess()

        self.recon_lossfunc = nn.BCEWithLogitsLoss(reduction='sum')
        
        self.loss_terms = (['LOSS'])
        self.pred_terms = ('GT', 'PRED', 'LAT', 'TAG', 'IND')
        
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
            'LAT'     : ret['z'],
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
