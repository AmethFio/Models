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

version = 'P3DMod'

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

        # CSI: Extract amp and phase
        data['csi'] = torch.cat((torch.abs(data['csi']), torch.angle(data['csi'])), dim=2) 

        return data

class SpatialTemporalEncoder(nn.Module):
    name = 'csien'
    
    def __init__(self, input_dim=60, embed_dims=128, num_layers=6, num_heads=8, ffn_dims=1024, dropout=0.1):
        super(SpatialTemporalEncoder, self).__init__()

        # Linear layer to project input to `embed_dims`
        self.input_projection = nn.Linear(input_dim, embed_dims)

        # Linear layer to project input to `embed_dims`
        self.output_projection = nn.Linear(input_dim * embed_dims, embed_dims)

        # Spatial-Temporal Embedding (STE)
        self.spatial_temporal_embedding = nn.Parameter(torch.randn(1, 60, embed_dims))

        # Encoder Layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dims,
                nhead=num_heads,
                dim_feedforward=ffn_dims,
                dropout=dropout,
                activation='relu'
            )
            for _ in range(num_layers)
        ])
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(embed_dims)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (num_tokens, batch_size, embed_dims).
        
        Returns:
            Tensor: Output tensor of shape (num_tokens, batch_size, embed_dims).
        """
        x = x.permute(0, 2, 1, 3).reshape(32, 60, -1) # batch * (2 * sub) * (pkt * rx)

        # Project input to `embed_dims`
        x = self.input_projection(x)  # Shape: (batch_size, num_tokens, embed_dims)

        # Add spatial-temporal embedding to input
        x = x + self.spatial_temporal_embedding
        
        # Transform to sequence-first format for the Transformer (num_tokens, batch_size, embed_dims)
        x = x.permute(1, 0, 2)

        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x)
        
        # Transform back to batch-first format (batch_size, num_tokens, embed_dims)
        x = x.permute(1, 0, 2)

        # Apply final layer normalization
        x = self.layer_norm(x)

        x = self.output_projection(x.view(32, -1))
        return x
    

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
    

class Person3DMod(nn.Module):

    def __init__(self, num_layers=6, device=None):
        super(Person3DMod, self).__init__()

        self.csien = SpatialTemporalEncoder(embed_dims=128, num_layers=num_layers)
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


class P3DModTrainer(BasicTrainer):
    
    def __init__(self,
                 *args, **kwargs
                 ):
        
        super(P3DModTrainer, self).__init__(*args, **kwargs)
    
        self.modality = {'rimg', 'csi', 'tag', 'ind'}

        self.recon_lossfunc = nn.BCEWithLogitsLoss(reduction='sum')
        self.preprocess = Preprocess()
        
        self.loss_terms = (['LOSS'])
        self.pred_terms = ('GT', 'PRED', 'TAG', 'IND')
        
        self.losslog = MyLossLog(name=self.name,
                           loss_terms=self.loss_terms,
                           pred_terms=self.pred_terms)
        
        self.training_phases = {
            'main': TrainingPhase(name='main',
                 lr=2.e-5)}
        
        self.model = Person3DMod(device=self.device)
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

        if autosave:
            for filename, fig in figs.items():
                fig.savefig(f"{self.save_path}{filename}")


if __name__ == "__main__":
    from torchinfo import summary
    m = Person3DMod()
    summary(m, input_size=(32, 20, 30, 3))
