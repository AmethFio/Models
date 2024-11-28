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
from misc import file_finder

from tqdm.notebook import tqdm

remov = [('0709A10', 1, 8),
         ('0709A10', 1, 9),
         ('0709A53', 6, 6),
         ('0709A53', 6, 7)]


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
                 *args, **kwargs):

        self.data = data
        self.label = label
        self.alignment = 'tail'
        self.csi_len = csi_len
        self.single_pd = single_pd
        self.subject_code = {'higashinaka': 0,
                             'zhang': 1,
                             'chen': 2,
                             'wang': 3,
                             'jiao': 4,
                             'qiao': 5,
                             'zhang2': 6}
        self.env_code = {'A208': 0,
                         'A308T': 1,
                         'B211': 2,
                         'C605': 3,
                         'A308': 4}

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

            elif modality == 'csi':

                if self.alignment == 'head':
                    csi_ind = np.arange(csi_ind, csi_ind + self.csi_len, dtype=int) 
                elif self.alignment == 'tail':
                    csi_ind = np.arange(csi_ind - self.csi_len, csi_ind, dtype=int)

                ret[modality] = np.copy(value[csi][csi_ind]) # Assume csi is n * 30 * 3
                
            elif modality == 'pd':
                if not self.single_pd:
                    if self.alignment == 'head':
                        pd_ind = np.arange(pd_ind, pd_ind + self.csi_len, dtype=int) 
                    elif self.alignment == 'tail':
                        pd_ind = np.arange(pd_ind - self.csi_len, pd_ind, dtype=int)
                        
                ret[modality] = np.copy(value[csi][pd_ind])
                
        return ret

    def __len__(self):
        return len(self.label)
    
class Preprocess:
    def __init__(self, new_size=(128, 128), filter_pd=False):
        self.new_size = new_size
        self._filterpd = FilterPD()
        self.filter_pd = filter_pd

    def transform(self, tensor):
        return F.interpolate(tensor, size=self.new_size, mode='bilinear', align_corners=False)
    
    def __call__(self, data, modalities):
        """
        Manually perform lazy preprocess
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


class CrossValidator:
    """
    Generate labels for cross validation
    """
    def __init__(self, labels, level, train=None, test=None, subset_ratio=1):
        self.labels = labels
        self.level = level
        self.subset_ratio = subset_ratio
        self.train = train
        self.test = test
        # train and test are str
        
        self.iter_range()
        self.current = -1
        self.current_test = None
        
    def iter_range(self):
        # Total range including train and test
        if self.train and self.test:
            if isinstance(self.train, list):
                self.range = self.train
            else:
                self.range = [self.train]
        else:
            self.range = ['A308', 'A308T'] if self.level == 'day' else list(set(self.labels.loc[:, self.level].values))
        

    def __iter__(self):
        return self
    
    def __next__(self):
        self.current += 1
        
        if self.current >= len(self.range):
            raise StopIteration
        
        train_range = self.range
        if self.train and self.test:
            self.current_test = self.test
        else:
            self.current_test = self.range[self.current]
            train_range = [x for x in self.range if x != self.current_test]
             
        print(f"\033[32mCross-validator: Fetched level {self.level}, {self.current + 1} of {len(self.range)}, current test = {self.current_test}\033[0m")
                            
    
        if self.level == 'day':
            train_labels = self.labels[self.labels['env']!=self.current_test]
            test_labels = self.labels[self.labels['env']==self.current_test]

        else:
            # Select one as leave-1-out test
            if self.train and self.test:
                train_labels = self.labels[self.labels[self.level].isin(self.range)]
                test_labels = self.labels[self.labels[self.level]==self.test]
            else:              
                train_labels = self.labels[self.labels[self.level]!=self.current_test]
                test_labels = self.labels[self.labels[self.level]==self.current_test]
                 

        if self.subset_ratio < 1:
            
            train_subset_size = int(len(train_labels) * self.subset_ratio)
            test_subset_size = int(len(test_labels) * self.subset_ratio) 
            
            print(f" Train set range = {train_range}, len = {train_subset_size} from {len(train_labels)}\n"
                  f" Test set range = {self.current_test}, len = {test_subset_size} from {len(test_labels)}"
                  )

            train_subset_indices = torch.randperm(len(train_labels))[:train_subset_size]
            test_subset_indices = torch.randperm(len(test_labels))[:test_subset_size]

            train_labels = train_labels.iloc[train_subset_indices]
            test_labels = test_labels.iloc[test_subset_indices]
            
        else:
            print(f" Train set range = {train_range}, len = {len(train_labels)}\n"
                  f" Test set range = {self.current_test}, len = {len(test_labels)}")

        return (train_labels, test_labels, self.current_test)
    
    def current_train(self):
        pass
    
    def reset(self):
        self.iter_range()
        self.current = -1
        self.current_test = None
    

class Removal:
    """
    Remove unpaired segments from dataset.
    """
    def __init__(self, conditions):
        self.conditions = conditions
        # (csi, group, segment) tuples
        
    def __call__(self, labels):
        for (csi, group, segment) in self.conditions:
            removal = (labels['csi']==csi) & (labels['group']==group) & (labels['segment']==segment)
            labels = labels.loc[~removal]
        return labels


class DataOrganizer:
    def __init__(self, name, data_path, level=None, train=None, test=None):
        self.name = name
        self.data_path = data_path
        # Specify the exact range of envs
        
        self.level = level
        assert level in {'env', 'subject', 'day'}
        print(f'Cross validation plan at {self.level} level')
        
        self.train = train
        self.test = test
        
        self.batch_size = 64
        
        # Data-Modality-Name
        self.data: dict = {}
        self.modalities = ['csi', 'rimg', 'cimg', 'bbx', 'ctr', 'dpt', 'pd']
        
        # Put all labels into one DataFrame
        self.total_segment_labels = pd.DataFrame(columns=['env', 'subject', 'bag', 'csi', 'group', 'segment', 'timestamp', 'img_inds', 'csi_inds'])
        
        self.train_indicies = None
        self.test_indicies = None        
        self.train_labels = None
        self.test_labels = None
        self.current_test = None
        
        self.cross_validator = None
        
        self.removal = Removal(remov)
        
    def load(self):
        def load_single(file_path, file_name_, ext):
            try:
                # Load Label <subject>_matched.csv
                if ext == '.csv' and 'matched' in file_name_ and 'checkpoint' not in file_name_:
                    sub = file_name_[8:]
                    sub_label = pd.read_csv(file_path)
                    self.total_segment_labels = pd.concat([self.total_segment_labels, sub_label], ignore_index=True)
                    print(f'Loaded {file_name_}{ext} of len {len(sub_label)}')
                    
                # Load CSI and IMGs <name>-<modality>.npy
                elif ext == '.npy':
                    if not ('env' in file_name_ or 'test' in file_name_ or 'checkpoint' in file_name_):
                        name, modality = file_name_.split('-')
                        if modality in self.modalities:     
                            if modality not in self.data.keys():
                                self.data[modality]: dict = {}
                            self.data[modality][name] = np.load(file_path, mmap_mode='r')
                            print(f'Loaded {file_name_}{ext} of shape {self.data[modality][name].shape}')
            except Exception as e:
                print(f"\033[31mError: {e} for {file_name_}{ext}\033[0m")
        
        for dpath in self.data_path:
            file_finder(dpath, load_single, process_name='Data Organizer')
                    
        print(f"\nLoad complete!")         
        # self.total_segment_labels['csi_inds'] = self.total_segment_labels['csi_inds'].apply(lambda x: list(map(int, x.strip('[]').split())))
            
    def regen_plan(self, **kwargs):
        # reset in crossvalidator
        if kwargs:
            for key, value in kwargs.items():
                setattr(self.cross_validator, key, value)       
        self.cross_validator.reset()
        print("\033[32mData Organizer: Data iterator reset!\033[0m")
    
    def gen_plan(self, subset_ratio=1, save=False, notion=''):
        if not self.cross_validator:
            self.cross_validator = CrossValidator(self.total_segment_labels, self.level, self.train, self.test, subset_ratio)
        
        if save:
            print(f'\033[32mData Organizer: Saving plan {self.level} @ {subset_ratio}...\033[0m')
            if notion:
                notion = '_' + str(notion)
            cross_validator = CrossValidator(self.total_segment_labels, self.level, self.train, self.test, subset_ratio) 
            with open(f'../dataset/Door_EXP/{self.level}_r{subset_ratio}_{self.current_test}{notion}.pkl', 'wb') as f:
                plan = list(cross_validator)
                pickle.dump(plan, f)
                
            print('Plan saved!')
            
        else:
            # Divide train and test
            if self.train is None and self.test is None:
                self.train_labels, self.test_labels, self.current_test = next(self.cross_validator)
                self.test = self.current_test
                self.train = ['A208', 'A308T', 'B211', 'C605']
                self.train.remove(self.current_test)
            else:
                while True:
                    train_labels, test_labels, current_test = next(self.cross_validator)
                    if current_test == self.test:
                        self.train_labels, self.test_labels, self.current_test = train_labels, test_labels, current_test
                        self.test = self.current_test
                        self.train = ['A208', 'A308T', 'B211', 'C605']
                        self.train.remove(self.current_test)
                        break

    
    def load_plan(self, path):
        with open(path, 'rb') as f:
            plan = pickle.load(f)
        self.cross_validator = iter(plan)
        print(f'\033[32mData Organizer: Loaded plan!\033[0m')
    
    def gen_loaders(self, mode='s', train_ratio=0.8, batch_size=64, csi_len=300, single_pd=True, num_workers=14, save_dataset=False, shuffle_test=True, pin_memory=True):

        print(f'\033[32mData Organizer: Generating loaders for {mode}: level = {self.level}, current test = {self.current_test}\033[0m')
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
                
        dataset = MyDataset(data, self.train_labels, csi_len, single_pd)
        test_dataset = MyDataset(data, self.test_labels, csi_len, single_pd)
            
        print(f' Train dataset length = {len(dataset)}\n'
              f' Test dataset length = {len(test_dataset)}')
        
        # Generate loaders
        train_size = int(train_ratio * len(dataset))
        valid_size = len(dataset) - train_size
        train_set, valid_set = random_split(dataset, [train_size, valid_size])
        
        def worker_init_fn(worker_id):
            np.random.seed(worker_id)
            
        train_loader = DataLoader(train_set, 
                                  batch_size=batch_size, 
                                  num_workers=num_workers,
                                  drop_last=True, 
                                  pin_memory=pin_memory,
                                  worker_init_fn=worker_init_fn
                                  )
        valid_loader = DataLoader(valid_set, 
                                    batch_size=batch_size, 
                                    num_workers=num_workers,
                                    drop_last=True, 
                                    pin_memory=pin_memory,
                                    worker_init_fn=worker_init_fn
                                  )
        test_loader = DataLoader(test_dataset, 
                                    batch_size=batch_size, 
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    shuffle=shuffle_test,
                                    worker_init_fn=worker_init_fn
                                  )
        
        print(f" Exported train loader of len {len(train_loader)}, batch size = {batch_size}\n"
              f" Exported valid loader of len {len(valid_loader)}, batch size = {batch_size}\n"
              f" Exported test loader of len {len(test_loader)}, batch size = {batch_size}\n")
        
        return train_loader, valid_loader, test_loader, self.current_test
    
    def swap_train_test(self):
        self.train_labels, self.test_labels = self.test_labels, self.train_labels
        if self.train and self.test:
            self.train, self.test = self.test, self.train
        self.current_test = self.test
        print("Train and Test labels swapped!")
    

class DANN_Loader:
    def __init__(self, source_loader, target_loader):
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.source_iter = iter(source_loader)
        self.target_iter = iter(target_loader)
        self.maximum_iter = len(source_loader)
        self.current = -1
        
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

        try:
            target_data = next(self.target_iter)
        except StopIteration:
            self.target_iter = iter(self.target_loader)  # Reset the iterator
            target_data = next(self.target_iter)         # Get the first batch again

        return source_data, target_data

        
    def __len__(self):
        return self.maximum_iter
    
    def reset(self):
        self.source_iter = iter(self.source_loader)
        self.target_iter = iter(self.target_loader)
        self.maximum_iter = len(self.source_loader)
        self.current = -1
        
        
class GuidedLoader:
    def __init__(self, source_loader, target_loader):
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.source_iter = iter(source_loader)
        self.target_iter = iter(target_loader)
        self.maximum_iter = len(source_loader)
        self.current = -1
        
        self.target_guide_batch = None
        # TRY: FIRST BATCH LEVEL, THEN SAMPLE LEVEL
        
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

        if self.target_guide_batch is None:
            self.target_guide_batch = next(iter(self.target_loader))
        source_data = {key: torch.cat([source_data[key], self.target_guide_batch[key]], dim=0)
                        for key in source_data}
        return source_data

        
    def __len__(self):
        return self.maximum_iter
    
    def reset(self):
        self.source_iter = iter(self.source_loader)
        self.current = -1
        

class DANN_Loader2:
    """
    Generates source smaples and target samples by 3:1.
    """
    
    def __init__(self, source_loader, target_loader, target_guide=False):
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.source_iter = iter(self.source_loader)
        self.target_iter = iter(self.target_loader)
        self.maximum_iter = len(source_loader) // 3
        self.current = -1
        
        self.target_guide_batch = None
        self.target_guide = target_guide
        # TRY: FIRST BATCH LEVEL, THEN SAMPLE LEVEL
        
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
            source_samples = [next(self.source_iter) for _ in range(3)]
            source_batch = {key: torch.cat([sample[key] for sample in source_samples], dim=0) 
                            for key in source_samples[0]}  # Combine into a single batch
            
            if self.target_guide:
                if self.target_guide_batch is None:
                    self.target_guide_batch = next(iter(self.target_loader))
                source_batch = {key: torch.cat([source_batch[key], self.target_guide_batch[key]], dim=0)
                                for key in source_batch}

        except StopIteration:
            self.source_iter = iter(self.source_loader)  # Reset the iterator
            source_samples = [next(self.source_iter) for _ in range(3)]
            source_batch = {key: torch.cat([sample[key] for sample in source_samples], dim=0) 
                            for key in source_samples[0]}  # Get the first batch again

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
        
    
class DataOrganizerDANN(DataOrganizer):
    def __init__(self, *args, **kwargs):
        super(DataOrganizerDANN, self).__init__(*args, **kwargs)
        
    def gen_loaders(self, mode='s', train_ratio=0.8, batch_size=64, csi_len=300, single_pd=True, num_workers=14, save_dataset=False, shuffle_test=True, pin_memory=True):

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
                
        dataset = MyDataset(data, self.train_labels, csi_len, single_pd)
        test_dataset = MyDataset(data, self.test_labels, csi_len, single_pd)
            
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
    
    
def gen_dann_loaders(data_organizer, train=None, test=None, subset_ratio=1, batch_size=64, num_workers=2, target_guide=False):
    #if data_organizer.cross_validator and isinstance(data_organizer.cross_validator, CrossValidator):
    #    data_organizer.regen_plan()
    data_organizer.train = train
    data_organizer.test = test

    data_organizer.gen_plan(subset_ratio=subset_ratio)
    source_train_loader, source_valid_loader, target_test_loader, current_test = data_organizer.gen_loaders(mode='s', num_workers=num_workers, batch_size=batch_size)
    data_organizer.swap_train_test()
    target_train_loader, target_valid_loader, source_test_loader, _ = data_organizer.gen_loaders(mode='s', num_workers=num_workers, batch_size=batch_size)
    dann_train_loader = DANN_Loader2(source_train_loader, target_train_loader, target_guide)
    dann_valid1 = DANN_Loader2(source_valid_loader, target_valid_loader)
    dann_valid2 = DANN_Loader2(target_valid_loader, source_valid_loader)
    dann_test_loader = DANN_Loader2(target_test_loader, source_valid_loader)
    return dann_train_loader, dann_valid1, dann_valid2, dann_test_loader, current_test


def gen_double_valid_loaders(data_organizer, train=None, test=None, subset_ratio=1, batch_size=64, num_workers=2, target_guide=False):
    data_organizer.train = train
    data_organizer.test = test

    data_organizer.gen_plan(subset_ratio=subset_ratio)
    source_train_loader, source_valid_loader, target_test_loader, current_test = data_organizer.gen_loaders(mode='s', num_workers=num_workers, batch_size=batch_size)
    data_organizer.swap_train_test()
    target_train_loader, target_valid_loader, source_test_loader, _ = data_organizer.gen_loaders(mode='s', num_workers=num_workers, batch_size=batch_size)
    
    if target_guide:
        source_train_loader = GuidedLoader(source_train_loader, target_train_loader)
    
    return source_train_loader, source_valid_loader, target_valid_loader, target_test_loader, current_test
    
    
    

    