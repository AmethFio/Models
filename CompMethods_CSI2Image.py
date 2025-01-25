import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
import os
from Trainer import BasicTrainer, TrainingPhase, ValidationPhase
from Loss import MyLossGAN
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

version = 'CSI2Image'

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


class Generator(nn.Module):
    
    name = 'gener'
    
    def __init__(self):
        
        super(Generator, self).__init__()

        self.fc = nn.Linear(270, 65536)
        
        block = [1024, 512, 256, 128]
        cnn = []
        
        for in_ch, out_ch in zip(block[:-1], block[1:]):
            cnn.extend([
                nn.ConvTranspose2d(in_ch, in_ch, 2, 2, 0),
                nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                nn.BatchNorm2d(out_ch, momentum=0.8),
                nn.ReLU()
            ])
            
        cnn.extend([
            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Tanh()
        ])
        
        self.cnn = nn.Sequential(*cnn)
        
    def forward(self, x):
        x = self.fc(x)
        out = self.cnn(x.view(-1, 1024, 8, 8))
        
        return out
    
    
class Discriminator(nn.Module):
    
    name = 'discr'
    
    def __init__(self):
        
        super(Discriminator, self).__init__()
        
        block = [1, 64, 128, 256]
        cnn = []
        
        for in_ch, out_ch in zip(block[:-1], block[1:]):
            bn = nn.Identity(out_ch) if in_ch == 1 else nn.BatchNorm2d(out_ch, momentum=0.8)
            
            cnn.extend([
                nn.Conv2d(in_ch, out_ch, 3, 2, 1),
                nn.BatchNorm2d(out_ch, momentum=0.8),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout2d(0.25)
            ])
            
        self.cnn = nn.Sequential(*cnn)
        
        self.fc = nn.Sequential(
            nn.Linear(8 * 8 * 256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.cnn(x)
        out = self.fc(x.view(-1, 8 * 8 * 256))
        
        return out
        

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
