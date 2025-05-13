import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
import os
from Trainer import BasicTrainer, TrainingPhase, ValidationPhase
from Loss import MyLossGAN
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

version = 'Mods'

class ImageEncoder(nn.Module):
    name = 'imgen'
    
    def __init__(self, batchnorm='identity', latent_dim=16, input_channel=1, *args, **kwargs):
        super(ImageEncoder, self).__init__(*args, **kwargs)

        self.batchnorm = batchnorm
        self.latent_dim = latent_dim
        
        block = [[input_channel, 128, 3, 2, 1],
                [128, 128, 3, 1, 1],
                [128, 128, 3, 2, 1],
                [128, 128, 3, 1, 1],
                [128, 256, 3, 2, 1],
                [256, 256, 3, 1, 1],
                [256, 512, 3, 1, 1],
                [512, 512, 1, 1, 0],
                [512, 6, 1, 1, 0]]
        
        cnn = []

        for [in_ch, out_ch, ks, st, pd] in block:
            if in_ch != 512:
                cnn.extend([nn.Conv2d(in_ch, out_ch, ks, st, pd),
                            batchnorm_layer(out_ch, self.batchnorm),
                            nn.LeakyReLU(inplace=True)])
            else:
                cnn.extend([nn.Conv2d(in_ch, out_ch, ks, st, pd)])
            
        self.cnn = nn.Sequential(*cnn)

        # 1 * 128 * 128
        # 128 * 64 * 64
        # Re
        # 128 * 32 * 32
        # Re
        # 256 * 16 * 16
        # Re
        # 512 * 16* 16
        # 6 * 16 * 16

        self.fc_mu = nn.Sequential(
            nn.Linear(6 * 16 * 16, self.latent_dim)
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(6 * 16 * 16, self.latent_dim)
        )

    def __str__(self):
        return f"IMGEN{version}"

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(-1, 6 * 16 * 16)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        z = reparameterize(mu, logvar)

        return z, mu, logvar, out
    

class ImageDecoder(nn.Module):
    name = 'imgde'
    
    def __init__(self, batchnorm='identity', latent_dim=16, active_func=nn.Sigmoid(), *args, **kwargs):
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
    

class Teacher(nn.Module):

    def __init__(self, device=None):
        super(Teacher, self).__init__()

        self.imgen = ImageEncoder(latent_dim=128)
        self.imgde = ImageDecoder(latent_dim=128)

        if device is not None:
            self.imgen = self.imgen.to(device)
            self.imgde = self.rimgen.to(device)

    def forward(self, data):
        z, mu, logvar, feature = self.imgen(data['dimg'])
        recon = self.rimgde(z)

        ret = {
        'z'      : z,
        'mu'     : mu,
        'logvar' : logvar,
        'feature': feature,
        'dimage' : recon,
                }

        return ret
    

class DensePose(nn.Module):
    name = 'dpose'
    
    def __init__(self):
        super(DensePose, self).__init__()
        self.amplitude_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(450,2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
        )
        
        self.phase_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(450, 2048),
            nn.ReLU(),
            nn.Linear(2048,1024),
            nn.ReLU()
        )
        
        # Define the fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(1024* 2, 2048),
            nn.ReLU(),
            nn.Linear(2048, 576),
            nn.ReLU(),
        )
        
        # Define the convolution blocks
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: 6x6
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output size: 64*6*6
            nn.ReLU(),
        )
        
        # Define the deconvolution layers
        self.deconv_layers = nn.Sequential(           
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 32x12x12
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 16x24x24
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),   # Output: 8x48x48
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),    # Output: 4x96x96
            nn.ReLU(),
            nn.ConvTranspose2d(4, 3, kernel_size=3, stride=4, padding=1, output_padding=1),    # Output: 3x384x384
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # Output: 3x768x768
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1)     # Output: 3x1536x1536
        )
        
    def forward(self, data):
        amp = torch.abs(data['csi'])
        phs = torch.angle(data['csi'])
        
        # Encode the amplitude and phase tensors
        amplitude_features = self.amplitude_encoder(amplitude_tensor.view(-1, 5 * 30 * 3))
        phase_features = self.phase_encoder(phase_tensor.view(-1, 5 * 30 * 3))
        
        # Concatenate and fuse the features
        fused_features = torch.cat((amplitude_features, phase_features), dim=1)
        fused_features = self.fusion_mlp(fused_features)
        
        # Reshape and process through convolution blocks
        reshaped_features = fused_features.view(-1, 1, 24, 24)
        conv_output = self.conv_blocks(reshaped_features)
        
        # Process through deconvolution layers
        deconv_output = self.deconv_layers(conv_output)

        # target_height = 992
        # target_width = 736
        # # For a center crop
        # resized_output = F.interpolate(deconv_output, size=(target_height, target_width), mode='bilinear', align_corners=False)
        
        
        return {'feature': deconv_output}



class Preprocess:
    def __init__(self, new_size=(128, 128), filter_pd=False):
        self.new_size = new_size
        self.batch_size = 32

    def transform(self, tensor):
        return F.interpolate(tensor, size=self.new_size, mode='bilinear', align_corners=False)

    def calc_svd(self, csi):
        first_columns_of_V = []
        # 32 * 30 * 3 * 3 -> (32 * 30) * 3 * 3
        csi = csi.reshape(-1, 3, 3)
        for i in range(csi.shape[0]):
            U, S, Vh = torch.linalg.svd(csi, full_matrices=False)
            first_column_of_V = Vh.conj().T[:, 0]  # First column of V
            first_columns_of_V.append(torch.abs(first_column_of_V))

        first_columns_of_V = torch.stack(first_columns_of_V).reshape(self.batch_size, 150)
        return first_columns_of_V
    
    def __call__(self, data, modalities):
        """
        Preprocess after retrieving data
        """
        
        #  Transform images
        if self.new_size and 'rimg' in modalities:
            data['rimg'] = self.transform(data['rimg'])

        if 'csi' in modalities:
            data['csi'] = self.calc_svd(data['csi'])

        return data

        

class Model(nn.Module):
    
    def __init__(self, device=None, mode='gen'):
        super(Model, self).__init__()
        
        self.gen = Generator()
        self.dis = Discriminator()
        
        if device is not None:
            for module in ['gen', 'dis']:
                getattr(self, module).to(device)
                
        self._device = device
        self._mode = mode
        self._batch_size = 32
        
    def forward(self, data):
        
        if self._mode == 'gen':
            re_img = self.gen(data['csi'])
            ret = {
                're_img': re_img
            }
        
        elif self._mode == 'dis':
            noise = torch.randn((self._batch_size, 3 * 3 * 30)).to(self._device)
            fake_img = self.gen(noise)
            real_img = data['rimg']
            labels = torch.cat((torch.zeros(fake_img.shape[0], dtype=int), torch.ones(real_img.shape[0], dtype=int))).to(self._device)
            imgs = torch.cat(torch.cat((fake_img, real_img), dim=0))
            est = self.dis(imgs)
            
            ret = {
                'fake_img': fake_img,
                'label': labels,
                'est': est
            }
        
        elif self._mode == 'hyb':
            re_img = self.gen(data['csi'])
            est = self.dis(re_img)
            labels = torch.ones(re_img.shape[0], dtype=int).to(self._device)
            
            ret = {
                'gen_img': re_img,
                'label': labels,
                'est': est
            }
        
        return ret


class CSI2ImageTrainer(BasicTrainer):
    
    def __init__(self,
                 *args, **kwargs
                 ):
        
        super(CSI2ImageTrainer, self).__init__(*args, **kwargs)
    
        self.modality = {'rimg', 'csi', 'timestmap', 'tag', 'ind'}

        self.dis_loss = nn.BCEWithLogitsLoss()
        self.gen_loss = nn.MSELoss(reduction='sum')
        
        self.loss_terms = ('LOSS', 'GEN', 'DIS', 'HYB', 'HYB_GEN', 'HYB_DIS')
        self.pred_terms = ('GT', 'PRED', 'FAKE', 'DOM_GT', 'DOM_PRED', 'TAG', 'IND')
        
        self.losslog = MyLossGAN(name=self.name,
                           loss_terms=self.loss_terms,
                           pred_terms=self.pred_terms)
        
        self.model = Model(device=self.device)
        self.models = vars(self.model)
        
        self.training_phases = {
            'Generator': TrainingPhase(name='Generator',
                                       train_module=['gener'],
                                       eval_module=['discr'],
                                       verbose=False),
            'Discriminator': TrainingPhase(name='Discriminator',
                                           train_module=['discr'],
                                           eval_module=['gener'],
                                           verbose=False),
            'Hybrid': TrainingPhase(name='Hybrid',
                                    train_module=['gener'],
                                    eval_module=['discr'],
                                    verbose=False)
        }
        
        self.early_stopping_trigger = 'Hybrid'
        
    def phase_condition(self, name, epoch):
        if name == 'Hybrid' and epoch % 8 != 0:
            return False
        else:
            return True
        
    def calculate_loss(self, data):
        
        ret = self.model(data)
        
        if self.model._mode == 'gen':
            
            loss = self.gen_loss(ret['re_img'], data['rimg']) / ret['re_img'].shape[0]

            TEMP_LOSS = {
                'LOSS': loss,
                'GEN' : loss
                }   
            
            PREDS = {
                'GT': data['rimg'],
                'PRED': ret['re_img'],
                'TAG': data['tag'],
                'IND': data['ind']
            }
            
        elif self.model._mode == 'dis':
            
            loss = self.dis_loss(ret['est'], ret['labels'])
            
            TEMP_LOSS = {
                'DIS': loss
            }
            
            PREDS = {
                'GT'      : data['rimg'],
                'FAKE'    : ret['fake_img'],
                'DOM_GT'  : ret['label'],
                'DOM_PRED': ret['est'],
                'TAG'     : data['tag'],
                'IND'     : data['ind']
            }
            
        elif self.model._mode == 'hyb':
            
            dis_loss = self.dis_loss(ret['est'], ret['label'])
            gen_loss = self.gen_loss(ret['re_img'], data['rimg']) / ret['re_img'].shape[0]
            loss = dis_loss + gen_loss
            
            TEMP_LOSS = {
                'HYB_GEN': gen_loss,
                'HYB_DIS': dis_loss,             
                'HYB': loss
            }
            
            PREDS = {
                'GT'      : data['rimg'],
                'PRED'    : ret['gen_img'],
                'DOM_GT'  : ret['label'],
                'DOM_PRED': ret['est'],
                'TAG'     : data['tag'],
                'IND'     : data['ind']
            }
            
        return PREDS, TEMP_LOSS
        
    def plot_test(self, select_ind=None, select_num=8, autosave=False, **kwargs):
        figs: dict = {}
        self.losslog.generate_indices(select_ind, select_num)

        if self.model._mode == 'gen':
            figs.update(self.losslog.plot_predict(plot_terms=('GT', 'PRED')))
            
        elif self.model._mode == 'dis':
            figs.update(self.losslog.plot_predict(plot_terms=('GT', 'FAKE')))
            figs.update(self.losslog.plot_discriminate())
            
        elif self.model._mode == 'hyb':
            figs.update(self.losslog.plot_predict(plot_terms=('PRED')))
            figs.update(self.losslog.plot_discriminate())

        if autosave:
            for filename, fig in figs.items():
                fig.savefig(f"{self.save_path}{filename}")


if __name__ == "__main__":

    pass
