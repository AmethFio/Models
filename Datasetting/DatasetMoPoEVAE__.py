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
    
class MyDataset(Dataset):
    """
    DATASET wrapper
    Load CSI, IMG, IMG-related modalities (CIMG, DPT, CTR)
    """

    def __init__(self,
                 data,
                 label,
                 csi_len=151,
                 *args, **kwargs):

        self.data = data
        self.label = label
        self.csi_len = csi_len
        
        self.subject_code = subject_code
        self.env_code = env_code
        
        # Normalize
        # for bag, value in self.data['rimg'].items():
        #     tensor_value = torch.from_numpy(value.copy()).float()  # Ensure it's a float tensor
        #     mean = tensor_value.mean()
        #     std = tensor_value.std()

        #     self.data['rimg'][bag] = transforms.Normalize(mean, std)(tensor_value)
            
    def __getitem__(self, index):
        """
        On-the-fly: select windowed CSI
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
        ret['csi_ind'] = csi_ind
        ret['img_ind'] = img_ind
        
        ret['rimg'] = np.copy(self.data['rimg'][bag][img_ind])[np.newaxis, ...]
        
        csi_ind = np.arange(csi_ind - self.csi_len // 2, 
                                    csi_ind - self.csi_len // 2 + self.csi_len, dtype=int)
        ret['csi'] = np.copy(self.data['csi'][csi][csi_ind])

                
        return ret

    def __len__(self):
        return len(self.label)
    
    

    
    