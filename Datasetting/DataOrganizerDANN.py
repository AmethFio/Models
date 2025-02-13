import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torch.distributed as dist
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
import pandas as pd
from scipy import signal
import os
from glob import glob

from PIL import Image
import pickle
from pandas.compat import pickle_compat
from misc import timer, file_finder, file_finder_multi
import time
from Datasetting.Dataset import *
from DataOrganizer import *

from tqdm.notebook import tqdm

MASK_CSI = False
        
        
class GuidedLoader:
    def __init__(self, source_loader, target_loader, target_guide_num=1):
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.source_iter = iter(source_loader)
        self.target_iter = iter(target_loader)
        self.maximum_iter = len(source_loader)
        self.target_guide_num = target_guide_num
        self.current = -1

        self.guide_batch = [next(self.target_iter) for _ in range(self.target_guide_num)]
        self.guide_iter = iter(self.guide_batch)
        
    def __iter__(self):
        return self
        
    def __next__(self):
        self.current += 1
        if self.current > self.maximum_iter:
            # automatically reloop
            self.reset()
            raise StopIteration
            
        try:
            source_data = next(self.source_iter)
        except StopIteration:
            self.source_iter = iter(self.source_loader)  # Reset the iterator
            source_data = next(self.source_iter)         # Get the first batch again


        # ITERATE OVER TARGET GUIDE BATCHES, YIELD ONE
        try:
            target_guide = next(self.guide_iter)
        except StopIteration:
            self.guide_iter = iter(self.guide_batch)  # Reset the iterator
            target_guide = next(self.guide_iter)         # Get the first batch again
        
        source_data = {key: torch.cat([source_data[key], target_guide[key]], dim=0)
                        for key in source_data}
        return source_data

        
    def __len__(self):
        return self.maximum_iter
    
    def reset(self):
        self.source_iter = iter(self.source_loader)
        
        self.current = -1
        

class DANN_Loader:
    """
    Combines source and target loaders.
    Generates source smaples and target samples by n:1.
    Can combine guide batch with source.
    """
    
    def __init__(self, source_loader, target_loader, source_rate=3, target_guide=False, target_guide_num=1):
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.source_iter = iter(self.source_loader)
        self.target_iter = iter(self.target_loader)
        self.maximum_iter = len(source_loader) // 3
        self.current = -1
        
        self.source_rate = source_rate
        
        self.target_guide_batch = None
        self.target_guide = target_guide
        self.target_guide_num = target_guide_num
        if target_guide:
            self.guide_batch = [next(self.target_iter) for _ in range(self.target_guide_num)]
            self.guide_iter = iter(self.guide_batch)
        
    def __iter__(self):
        return self
        
    def __next__(self):
        self.current += 1
        if self.current > self.maximum_iter:
            # automatically reloop
            self.reset()
            raise StopIteration

            
        try:
            # Fetch 3 samples from the source loader
            source_samples = [next(self.source_iter) for _ in range(self.source_rate)]
            source_batch = {key: torch.cat([sample[key] for sample in source_samples], dim=0) 
                            for key in source_samples[0]}  # Combine into a single batch

        except StopIteration:
            self.source_iter = iter(self.source_loader)  # Reset the iterator
            source_samples = [next(self.source_iter) for _ in range(self.source_rate)]
            source_batch = {key: torch.cat([sample[key] for sample in source_samples], dim=0) 
                            for key in source_samples[0]}  # Get the first batch again
            
        if self.target_guide:
            # ITERATE OVER TARGET GUIDE BATCHES, YIELD ONE
            try:
                target_guide = next(self.guide_iter)
            except StopIteration:
                self.guide_iter = iter(self.guide_batch)  # Reset the iterator
                target_guide = next(self.guide_iter)         # Get the first batch again
            
            source_batch = {key: torch.cat([source_batch[key], target_guide[key]], dim=0)
                            for key in source_batch}

        try:
            target_data = next(self.target_iter)
        except StopIteration:
            self.target_iter = iter(self.target_loader)  # Reset the iterator
            target_data = next(self.target_iter)         # Get the first batch again

        return source_batch, target_data

        
    def __len__(self):
        return self.maximum_iter
    
    def reset(self):
        self.source_iter = iter(self.source_loader)
        self.target_iter = iter(self.target_loader)
        self.maximum_iter = len(self.source_loader) // 3
        self.current = -1
        if self.target_guide:
            self.guide_batch = [next(self.target_iter) for _ in range(self.target_guide_num)]
            self.guide_iter = iter(self.guide_batch)
        
    
class DataOrganizerDANN(DataOrganizer):
    def __init__(self, *args, **kwargs):
        super(DataOrganizerDANN, self).__init__(*args, **kwargs)
        
    def gen_loaders(self, mode='s', train_ratio=0.8, batch_size=64, csi_len=300, single_pd=True, num_workers=14, save_dataset=False, shuffle_test=False, pin_memory=True):

        print(f'\033[32mData Organizer DANN: Generating loaders for {mode}: level = {self.level}, current test = {self.current_test}\033[0m')
        data = self.data.copy()
        
        if mode == 't':
            data.pop('csi')
            data.pop('pd')

        else:
            # if mode == 's'
            self.train_labels = self.removal(self.train_labels)
            self.test_labels = self.removal(self.test_labels)
            
            if mode == 'c':
                data = self.data.copy()
                data.pop('pd')
                data.pop('cimg')
                
        dataset = self.dstype(data, self.train_labels, csi_len, single_pd)
        test_dataset = self.dstype(data, self.test_labels, csi_len, single_pd)
            
        print(f' Train dataset length = {len(dataset)}\n'
              f' Test dataset length = {len(test_dataset)}')
        
        # Generate loaders
        train_size = int(train_ratio * len(dataset))
        valid_size = len(dataset) - train_size
        train_set, valid_set = random_split(dataset, [train_size, valid_size])
        train_loader = DataLoader(train_set, 
                                  batch_size=batch_size, 
                                  num_workers=num_workers,
                                  drop_last=True, 
                                  pin_memory=pin_memory)
        valid_loader = DataLoader(valid_set, 
                                    batch_size=batch_size, 
                                    num_workers=num_workers,
                                    drop_last=True, 
                                    pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, 
                                    batch_size=batch_size, 
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    shuffle=shuffle_test)
        
        print(f" Exported train loader of len {len(train_loader)}, batch size = {batch_size}\n"
              f" Exported valid loader of len {len(valid_loader)}, batch size = {batch_size}\n"
              f" Exported test loader of len {len(test_loader)}, batch size = {batch_size}\n")
        
        return train_loader, valid_loader, test_loader, self.current_test
    
    
def gen_dann_loaders(data_organizer, subset_ratio=1, batch_size=64, num_workers=2, target_guide=False, target_guide_num=1):
    #if data_organizer.cross_validator and isinstance(data_organizer.cross_validator, CrossValidator):
    #    data_organizer.regen_plan()

    data_organizer.gen_plan(subset_ratio=subset_ratio)
    source_train_loader, source_valid_loader, target_test_loader, current_test = data_organizer.gen_loaders(mode='s', num_workers=num_workers, batch_size=batch_size)
    data_organizer.swap_train_test()
    target_train_loader, target_valid_loader, source_test_loader, _ = data_organizer.gen_loaders(mode='s', num_workers=num_workers, batch_size=batch_size)
    
    dann_train_loader = DANN_Loader(source_train_loader, target_train_loader, target_guide, target_guide_num)
    # dann_valid1 = DANN_Loader2source_valid_loader, target_valid_loader)
    # dann_valid2 = DANN_Loader(target_valid_loader, source_valid_loader)
    # dann_test_loader = DANN_Loader(target_test_loader, source_valid_loader)
    return dann_train_loader, source_valid_loader, target_valid_loader, target_test_loader, current_test


def gen_double_valid_loaders(data_organizer, subset_ratio=1, batch_size=64, num_workers=2, target_guide=False, target_guide_num=1):

    data_organizer.gen_plan(subset_ratio=subset_ratio)
    source_train_loader, source_valid_loader, target_test_loader, current_test = data_organizer.gen_loaders(mode='s', num_workers=num_workers, batch_size=batch_size)
    data_organizer.swap_train_test()
    target_train_loader, target_valid_loader, source_test_loader, _ = data_organizer.gen_loaders(mode='s', num_workers=num_workers, batch_size=batch_size)
    
    if target_guide:
        source_train_loader = GuidedLoader(source_train_loader, target_train_loader, target_guide_num)
    
    return source_train_loader, source_valid_loader, target_valid_loader, target_test_loader, current_test
    
    
    

    