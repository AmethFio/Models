import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torch.distributed as dist
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
import pandas as pd
from scipy import signal
import os
from PIL import Image
import pickle
from misc import timer, file_finder, file_finder_multi
from joblib import Parallel, delayed
import time

from tqdm.notebook import tqdm


subject_code = {
    'higashinaka': 0,
    'zhang'      : 1,
    'chen'       : 2,
    'wang'       : 3,
    'jiao'       : 4,
    'qiao'       : 5,
    'zhang2'     : 6
    }

env_code = {
    'A208' : 0,
    'A308T': 1,
    'B211' : 2,
    'C605' : 3,
    'A308' : 4,
    'A208D1': 10,
    'A208D2': 11,
    'A208D3': 12,
    'A208X': 20
    }

MASK_CSI = False


class Raw:
    """
    Store raw data and avoid being changed
    """
    def __init__(self, value):
        self._value = value.copy()
        self._value.setflags(write=False)
        
    # Make sure to use copy() when assigning values!
    @property
    def value(self):
        return self._value

class FilterPD:
    def __init__(self, k=21):
        self.k = k
    
    def __call__(self, csi):
        def cal_pd(u):
            pd = u[:, 1:, 0] * u[:, :-1, 0].conj()
            
            # md = torch.zeros_like(pd, dtype=torch.cfloat)
            # for i in range(pd.shape[1]):
            #     real_filtered = torch.tensor(signal.medfilt(pd[:, i].real.cpu().numpy(), self.k), dtype=torch.float32, device=pd.device)
            #     imag_filtered = torch.tensor(signal.medfilt(pd[:, i].imag.cpu().numpy(), self.k), dtype=torch.float32, device=pd.device)
            #     print(real_filtered.shape, imag_filtered.shape)
            #     # md[:, i] = real_filtered + 1.j * imag_filtered
            # return torch.cat((real_filtered, imag_filtered), axis=-1).to(torch.float32)
            
            return torch.cat((torch.real(pd), torch.imag(pd)), axis=-1)
        
        try:
            # CSI shape = batch * 300 * 30 * 3
            # Reshape into batch * 3 * (30 * 300)
            u, *_ = torch.linalg.svd(csi.permute(0, 3, 2, 1).reshape(csi.shape[0], 3, -1), full_matrices=False)
            # AoA = batch * 4 (real & imag of 2)
            aoa = cal_pd(u)
            
            # Reshape into batch * 30 * (3 * 300)
            u, *_ = torch.linalg.svd(csi.permute(0, 2, 3, 1).reshape(csi.shape[0], 30, -1), full_matrices=False)
            # ToF = batch * 58 (real & imag of 29)
            tof = cal_pd(u)
            
            # Concatenate as a flattened vector
            pd = torch.cat((aoa, tof), axis=-1)

        except Exception as e:
            print(f'FilterPD aborted due to {e}')
        
        return pd
    
class MyDataset(Dataset):
    """
    DATASET wrapper
    Load CSI, IMG, IMG-related modalities (CIMG, DPT, CTR)
    """

    def __init__(self,
                 data,
                 label,
                 csi_len=300,
                 single_pd=True,
                 mask_csi=MASK_CSI,
                 simple_mode=False,
                 *args, **kwargs):

        self.data = data
        self.label = label
        self.alignment = 'tail'
        self.csi_len = csi_len
        self.single_pd = single_pd
        
        self.mask_csi = mask_csi
        self.csi_temporal_mask_prob = 0.1
        self.csi_spatial_mask_prob = 0.2
        
        self.subject_code = subject_code
        self.env_code = env_code
                
        self.simple_mode = simple_mode

    def __getitem__(self, index):
        """
        On-the-fly: select windowed CSI (and pd)
        """
        # Tag codes
        ret: dict = {}
        tag =  self.label.iloc[index][['env', 'subject', 'img_inds']]
        tag['env'] = self.env_code[tag['env']]
        tag['subject'] = self.subject_code[tag['subject']]
        ret['tag'] = tag.to_numpy().astype(int)
        
        # return the absolute index of sample
        ret['ind'] = self.label.index[index]
        
        # Label = ['env', 'subject', 'bag', 'csi', 
        # 'group', 'segment', 'timestamp', 'img_inds', 'csi_inds']

        bag = self.label.iloc[index]['bag']
        img_ind = int(self.label.iloc[index]['img_inds'])
        csi = self.label.iloc[index]['csi']
        csi_ind = int(self.label.iloc[index]['csi_inds'])
        pd_ind = int(self.label.iloc[index]['csi_inds'])


        for modality, value in self.data.items():
            if modality in ('rimg', 'cimg', 'bbx', 'ctr', 'dpt'):
                ret[modality] = np.copy(value[bag][img_ind])
                if modality == 'rimg':
                    ret[modality] = ret[modality][np.newaxis, ...] # Did not make spare axis
                # ctr and dpt combined
                if modality == 'ctr':
                    ret[modality] = ret[modality][..., :2]
                if modality == 'dpt':
                    ret[modality] = ret[modality][..., -1]


            elif modality in ('csi', 'csitime', 'csi2image'):

                if self.alignment == 'head':
                    csi_ind = np.arange(csi_ind, csi_ind + self.csi_len, dtype=int) 
                elif self.alignment == 'tail':
                    csi_ind = np.arange(csi_ind - self.csi_len, csi_ind, dtype=int)
                elif self.alignment == 'middle':
                    csi_ind = np.arange(csi_ind - self.csi_len // 2, 
                                        csi_ind - self.csi_len // 2 + self.csi_len, dtype=int)

                ret[modality] = np.copy(value[csi][csi_ind])
                    
                if self.mask_csi:
                    ret['csi'] = self.random_mask_csi(ret['csi'])
                
            elif modality == 'pd':
                if not self.single_pd and self.alignment == 'head':
                    pd_ind = np.arange(pd_ind, pd_ind + self.csi_len, dtype=int) 
                elif not self.single_pd and self.alignment == 'tail':
                    pd_ind = np.arange(pd_ind - self.csi_len, pd_ind, dtype=int)
                    
                ret[modality] = np.copy(value[csi][pd_ind])

                
        return ret

    def __len__(self):
        return len(self.label)
    
    def random_mask_csi(self, csi_data):
        T, S, R = csi_data.shape

        # Temporal mask: mask entire packets
        temporal_mask = torch.rand(T) < self.csi_temporal_mask_prob
        csi_data[temporal_mask, :, :] = 0  # Mask the entire packet

        # Spatial mask: mask specific subcarrier-receiver pairs
        spatial_mask = torch.rand(S, R) < self.csi_spatial_mask_prob
        csi_data[:, spatial_mask] = 0  # Mask the affected subcarrier-receiver pairs

        return csi_data
    
    
class Preprocess:
    def __init__(self, new_size=(128, 128), filter_pd=False):
        self.new_size = new_size
        self._filterpd = FilterPD()
        self.filter_pd = filter_pd

    def transform(self, tensor):
        return F.interpolate(tensor, size=self.new_size, mode='bilinear', align_corners=False)
    
    def __call__(self, data, modalities):
        """
        Preprocess after retrieving data
        """
        
        # Adjust key name
        if 'ctr' in data.keys():
            data['center'] = data['ctr']
        if 'dpt' in data.keys():
            data['depth'] = data['dpt']
        
        #  Transform images
        if self.new_size and 'rimg' in modalities:
            data['rimg'] = self.transform(data['rimg'])

        if 'csi' in modalities:
            if self.filter_pd:
                # Filter and reshape CSI, calculate phasediff
                # batch * packet * sub * rx
                csi_real = torch.tensor(signal.savgol_filter(np.real(data['csi']), 21, 3, axis=1), 
                                        dtype=torch.float32, device=data['csi'].device)  # denoise for real part
                csi_imag = torch.tensor(signal.savgol_filter(np.imag(data['csi']), 21, 3, axis=1), 
                                        dtype=torch.float32, device=data['csi'].device) # denoise for imag part
                csi_complex = csi_real + 1.j * csi_imag
                
                # batch *  packet * sub * (rx * 2)
                csi = torch.cat((csi_real, csi_imag), axis=-1)
                
                # batch *  (rx * 2) * sub * packet
                csi = csi.permute(0, 3, 2, 1)

                data['csi'] = csi
                data['pd'] = self._filterpd(csi_complex)
            
            else:
                csi = torch.cat((torch.real(data['csi']), torch.imag(data['csi'])), axis=-1)
                csi = csi.permute(0, 3, 2, 1)
                data['csi'] = csi
        
        return data
    
    