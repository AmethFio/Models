import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
import os
from Trainer import BasicTrainer, TrainingPhase, ValidationPhase
from Loss import MyLossLog
from Model import *
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

version = 'CSIImager'

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
        # CSI: Window length = 100, 3 rx
        # CSI: Extract amp and phase
        data['csi'] = torch.cat((torch.abs(data['csi']), torch.angle(data['csi'])), dim=-1)
        data['csi'] = data['csi'].reshape(data['csi'].shape[0], 200, -1) # batch * packet * (rx * sub * 2)
        return data
    
    
class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(MultiHeadSelfAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Multi-Head Self Attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_output)  # Add & LayerNorm
        return x


class Encoder(nn.Module):
    name='csien'
    
    def __init__(self):
        super(Encoder, self).__init__()
        
        # Initial Dense layer with ReLU
        self.dense1 = nn.Linear(180, 1000)
        
        # Stack of 8 attention blocks
        self.attention_blocks = nn.ModuleList([
            nn.Sequential(
                MultiHeadSelfAttentionBlock(embed_dim=1000, num_heads=4, dropout=0.1),
                nn.Linear(1000, 2000),  # Dense layer with GeLU
                nn.GELU(),
                nn.Linear(2000, 1000),  # Dense layer with GeLU
                nn.GELU(),
                nn.LayerNorm(1000)  # Add & LayerNorm
            )
            for _ in range(8)
        ])
        
        # Flatten, LayerNorm, Dropout
        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(200000),
            nn.Dropout(0.5)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(200000, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.LayerNorm(256), # modified bottleneck dim
            nn.ReLU()
        )


    def forward(self, x):
        # Initial dense layer
        x = F.relu(self.dense1(x))
        
        # Stack of 8 attention blocks
        for block in self.attention_blocks:
            x_residual = x
            x = block[0](x)  # Multi-Head Self Attention
            x = block[1](x)  # Dense 1
            x = block[2](x)  # GeLU 1
            x = block[3](x)  # Dense 2
            x = block[4](x)  # GeLU 2
            x = block[5](x + x_residual)  # Add & LayerNorm
        
        # Flatten, LayerNorm, and Dropout
        x = self.flatten(x)
        
        # Fully connected layers
        x = self.fc(x)
        return x
    
    
class Decoder(nn.Module):
    name = 'imgde'
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.cnn2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.cnn3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.cnn4 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = F.interpolate(self.cnn1(x.view(-1, 1, 16, 16)), scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(self.cnn2(x), scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(self.cnn3(x), scale_factor=2, mode='bilinear', align_corners=False)
        x = self.cnn4(x)
        
        return x
    
    
class CSIImager(nn.Module):
    name = 'csiimager'
    
    def __init__(self, device=None):
        super(CSIImager, self).__init__()
        
        self.csien = Encoder()
        self.imgde = Decoder()
        
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
        
class CSIImagerTrainer(BasicTrainer):
    
    def __init__(self,
                 *args, **kwargs
                 ):
        
        super(CSIImagerTrainer, self).__init__(*args, **kwargs)

        self.modality = {'rimg', 'csi', 'tag', 'ind'}
        self.preprocess = Preprocess()

        self.recon_lossfunc = nn.MSELoss()
        
        self.loss_terms = (['LOSS'])
        self.pred_terms = ('GT', 'PRED', 'LAT',  'TAG', 'IND')
        
        self.losslog = MyLossLog(name=self.name,
                           loss_terms=self.loss_terms,
                           pred_terms=self.pred_terms)

        
        self.model = CSIImager(device=self.device)
        self.models = {m: getattr(self.model, m) for m in ['csien', 'imgde']}
        
    def calculate_loss(self, data):
        
        ret = self.model(data)
        recon_loss = self.recon_lossfunc(ret['re_img'], data['rimg'])
            
        TEMP_LOSS = {
            'LOSS': recon_loss
        }
        
        PREDS = {
            'GT'  : data['rimg'],
            'PRED': ret['re_img'],
            'LAT' : ret['z'],
            'TAG' : data['tag'],
            'IND' : data['ind']
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
