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
# from joblib import Parallel, delayed
import time
from Datasetting.Dataset import *
from Datasetting.DatasetMoPoEVAE import MyDataset as MoPoEDataset

from tqdm.notebook import tqdm

remov = [('0709A10', 1, 8),
         ('0709A10', 1, 9),
         ('0709A53', 6, 6),
         ('0709A53', 6, 7)]


MASK_CSI = False

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
    


class CrossValidator:
    """
    Generate labels for cross validation
    """
    def __init__(self, level, all_range=[None], subset_ratio=1):
        self.labels = None
        self.level = level
        self.all_range = all_range
        self.subset_ratio = subset_ratio

        self.cur = -1
        self.current_test = None
        
    def set_label(self, labels):
        self.labels = labels
        print(f'Cross validator got {len(labels)} labels')

    def __iter__(self):
        return self
    
    def __next__(self):
        self.cur += 1
        
        if self.cur >= len(self.all_range):
            raise StopIteration
        
        else:
            self.current_test = self.all_range[self.cur]
            
        # Set all_range = [None] to gen only train
        # Set all_range = [something] to gen only test
        
        select_range = [x for x in self.all_range if x != self.current_test]
             
        print(f"\033[32mCross-validator: Fetched level {self.level}, "
              f"{self.cur + 1} of {len(self.all_range)}, "
              f"current test = {self.current_test}\033[0m")

        # Select one as leave-1-out test        
        select_labels = self.labels[self.labels[self.level].isin(select_range)]
        leave_labels = self.labels[self.labels[self.level]==self.current_test]
                 
        if self.subset_ratio < 1:
            
            select_subset_size = int(len(select_labels) * self.subset_ratio)
            leave_subset_size = int(len(leave_labels) * self.subset_ratio) 
            
            print(f" Select set range = {select_range}, len = {select_subset_size} from {len(select_labels)}\n"
                  f" Leave set range = {self.current_test}, len = {leave_subset_size} from {len(leave_labels)}"
                  )

            select_subset_size = torch.randperm(len(select_labels))[:select_subset_size]
            leave_subset_size = torch.randperm(len(leave_labels))[:leave_subset_size]

            select_labels = select_labels.iloc[select_subset_size]
            leave_labels = leave_labels.iloc[leave_subset_size]
            
        else:
            print(f" Select set range = {select_range}, len = {len(select_labels)}\n"
                  f" Leave set range = {self.current_test}, len = {len(leave_labels)}")

        return (select_labels, leave_labels, select_range, self.current_test)
    
    def current_train(self):
        pass
    
    def reset(self):
        self.cur = -1
        self.current_test = None


class DataOrganizer:
    def __init__(self, name, data_path, crossvalidator, 
                 modalities=['csi', 'rimg', 'cimg', 'ctr', 'dpt', 'pd'],
                 removal=Removal(remov),
                 dstype='default'):
        self.name = name
        self.data_path = data_path
        self.cross_validator = crossvalidator
        
        # Controls MyDataset type
        if dstype == 'default':
            self.dstype = MyDataset
        elif dstype == 'mopoe':
            self.dstype = MoPoEDataset
        
        
        print(f'Cross validation plan at {crossvalidator.level} level')
        
        # Data-Modality-Name
        self.data: dict = {}
        self.modalities = modalities
        
        # Put all labels into one DataFrame
        self.total_segment_labels = pd.DataFrame(columns=['env', 'subject', 'bag', 'csi', 'group', 'segment', 'timestamp', 'img_inds', 'csi_inds'])
        
        self.train_indicies = None
        self.test_indicies = None        
        self.train_labels = None
        self.test_labels = None
        
        self.removal = removal
        
    def file_condition(self, fname):
        ret = False
        typ = None
        modality = None
        name = None

        if fname[-3:] == 'csv' and 'matched' in fname and 'checkpoint' not in fname:
            ret = True
            typ = 'label'

        if fname[-3:] == 'npy' and not ('env' in fname or 'test' in fname or 'checkpoint' in fname):
            name, modality = fname[:-4].split('-')
            if modality in self.modalities:
                ret = True
                typ = 'data'
                
        return ret, typ, (modality, name)
                
    @timer
    def load(self, multi=False):
        
        def load_single(fname, fpath, atm, etc):
            data = None
            label = None
            loaded = False
            start = time.time()

            if typ == 'data':
                # Load CSI and IMGs <name>-<modality>.npy
                d = np.load(fpath, mmap_mode='r')
                data = (*etc, d) # modality, name, data
                
                end = time.time()
                lg = f"Loaded {fname} of {d.shape}{atm}, elapsed {(end - start):.4f} sec"
                loaded = True
                
            elif typ == 'label':
                # Load Label <subject>_matched.csv
                label = pd.read_csv(fpath)
                
                end = time.time()
                lg = f"Loaded {fname} of len {len(label)}{atm}, elapsed {(end - start):.4f} sec"
                loaded = True
                
            print(lg)
            return label, data, loaded, lg
        
        def load_attempt(file_path, file_name_, ext):
            label = None
            data = None
            att_max = 6
            loaded = False
            start = time.time()
            for attempt in range(1, att_max):
                if loaded:
                    break
                
                atm = '' if attempt == 1 else f" at attempt {attempt}"
                loadable, typ, etc = self.file_condition(file_name_, ext)
                
                if not loadable:
                    end = time.time()
                    print(f"\033[33mSkipping {file_name_}{ext}, elapsed {(end - start):.4f} sec\033[0m")
                    loaded = True
                    
                else:
                    try:
                        label, data, loaded, lg = load_single(file_name_, ext, file_path, atm, start, etc)

                    except Exception as e:
                        print(f"\033[31mError: {e} for {file_name_}{ext} (Attempt {attempt})\033[0m")
                        if attempt == att_max:
                            print(f"\033[31mFailed to load {file_name_}{ext} after {att_max} attempts.\033[0m")
                            return -1, f"{file_name_}{ext}"

            return label, data
        
        def unpack_results(label, data, fail):
            if label is not None:
                if isinstance(label, int) and label == -1:
                    # Collect failure
                    fail.append(data)
                else:
                    self.total_segment_labels = pd.concat((self.total_segment_labels, label), ignore_index=True)
                    
            if data is not None:
                modality, name, d = data
                if modality not in self.data.keys():
                    self.data[modality] = {}
                    
                if modality in ('rimg', 'rgbimg', 'cimg'):
                    max_value = 1.
                    
                    if d.dtype == np.uint8:
                        max_value = 255
                    elif d.dtype == np.uint16:
                        max_value = 65535
                    # else:
                    #     raise ValueError(f"Only support uint8 or uint16, got {d.dtype}")
                    d = d.astype(np.float32) / max_value
                    
                self.data[modality][name] = d
                
        # Main loop
        
        fail = []
        load_logs = []
        skip_logs = []
        
        for dpath in self.data_path:
            # if multi:
                # print('Multi-process loading...')
                # files = file_finder_multi(dpath, process_name="Data Organizer")
                # results = Parallel(n_jobs=8)(
                #     delayed(load_attempt)(f, file_name_, ext) for f, file_name_, ext in files
                # )
                
                # for label, data in results:
                #     unpack_results(label, data)
                        
            # else:
            print('Single-process loading...')
            
            # results = file_finder(dpath, load_single, process_name="Data Organizer")
            files = glob(f"{dpath}/**/*", recursive=True)
            for file_path in files:
                file_name = os.path.basename(file_path)

                loadable, typ, etc = self.file_condition(file_name)
                if not loadable:
                    skip_logs.append(f"\033[33mSkipping {file_name}\033[0m")
                
                else:
                    label, data, etc, lg = load_single(file_name, file_path, '', etc)
                    unpack_results(label, data, fail)
                    load_logs.append(lg)
        
        # print("\n".join(load_logs))
        print(f"\nLoad complete!")
        if len(fail) > 0:
            print(f"Failed to load: {fail}")
        # self.total_segment_labels['csi_inds'] = self.total_segment_labels['csi_inds'].apply(lambda x: list(map(int, x.strip('[]').split())))
        
        self.cross_validator.set_label(self.total_segment_labels)
        
    def reset_plan(self, **kwargs):
        # reset in crossvalidator
        if kwargs:
            for key, value in kwargs.items():
                setattr(self.cross_validator, key, value)       
        self.cross_validator.reset()
        print("\033[32mData Organizer: Data iterator reset!\033[0m")
    
    def gen_plan(self, specify_test=None, save=False, notion=''):
        
        if save:
            print(f'\033[32mData Organizer: Saving plan {self.cross_validator.level} @ {subset_ratio}...\033[0m', end='')
            
            with open(f'../dataset/Door_EXP/{self.cross_validator.level}_r{subset_ratio}_{self.current_test}{notion}.pkl', 'wb') as f:
                plan = list(self.cross_validator)
                pickle.dump(plan, f)
                
            print('Done!')
            
        else:
            # Divide train and test
            current_test = None
            plan = next(self.cross_validator)
            if len(plan) == 3:
                train_labels, test_labels, current_test = plan
            elif len(plan) == 4:
                train_labels, test_labels, train_range, current_test = plan
            while True:
                if specify_test is not None and current_test != specify_test:
                    plan = next(self.cross_validator)
                    if len(plan) == 3:
                        train_labels, test_labels, current_test = plan
                    elif len(plan) == 4:
                        train_labels, test_labels, train_range, current_test = plan
                else:
                    break
            
            self.train_labels, self.test_labels, self.current_test = train_labels, test_labels, current_test
    
    def load_plan(self, path):
        with open(path, 'rb') as f:
            plan = pickle_compat.load(f)
        self.cross_validator = iter(plan)
        print(f'\033[32mData Organizer: Loaded plan!\033[0m')
    
    def gen_loaders(self, mode='s', 
                    train_ratio=0.8, 
                    batch_size=64, 
                    csi_len=300, 
                    single_pd=True, 
                    num_workers=14, 
                    save_dataset=False, 
                    shuffle_test=False, 
                    pin_memory=True):

        print(f'\033[32mData Organizer: Generating loaders for {mode}: '
              f'current test = {self.current_test}\033[0m')
        
        data = self.data.copy()
        
        if mode == 't':
            data.pop('csi')
            data.pop('pd')

        elif mode == 's':
            if self.removal:
                self.train_labels = self.removal(self.train_labels)
                self.test_labels = self.removal(self.test_labels)
                
        tv_dataset = self.dstype(data, self.train_labels, csi_len, single_pd)
        test_dataset = self.dstype(data, self.test_labels, csi_len, single_pd)
            
        print(f' Train/Valid dataset length = {len(tv_dataset)}\n'
              f' Test dataset length = {len(test_dataset)}')
        
        # Generate loaders
        train_size = int(train_ratio * len(tv_dataset))
        valid_size = len(tv_dataset) - train_size
        train_set, valid_set = random_split(tv_dataset, [train_size, valid_size])
        test_size = len(test_dataset)
        
        def worker_init_fn(worker_id):
            np.random.seed(worker_id)
        
        if train_size > 0:
            train_loader = DataLoader(train_set, 
                                    batch_size=batch_size, 
                                    num_workers=num_workers,
                                    drop_last=True, 
                                    pin_memory=pin_memory,
                                    worker_init_fn=worker_init_fn
                                    )
            print(f" Exported train loader of len {len(train_loader)}, batch size = {batch_size}\n")
        else:
            train_loader = None
            
        if valid_size > 0:
            valid_loader = DataLoader(valid_set, 
                                        batch_size=batch_size, 
                                        num_workers=num_workers,
                                        drop_last=True, 
                                        pin_memory=pin_memory,
                                        worker_init_fn=worker_init_fn
                                    )
            print(f" Exported valid loader of len {len(valid_loader)}, batch size = {batch_size}\n")
        else:
            valid_loader = None
            
        if test_size > 0:
            test_loader = DataLoader(test_dataset, 
                                        batch_size=batch_size, 
                                        num_workers=num_workers,
                                        pin_memory=pin_memory,
                                        drop_last=False,
                                        shuffle=shuffle_test,
                                        worker_init_fn=worker_init_fn
                                    )
        
            print(f" Exported test loader of len {len(test_loader)}, batch size = {batch_size}\n")
        else:
            test_loader = None
        
        return train_loader, valid_loader, test_loader, self.current_test
        
        
class DataOrganizerEXT(DataOrganizer):
    
    def __init__(self, *args, **kwargs):
        super(DataOrganizerEXT, self).__init__(*args, **kwargs)
    
    def save_planned_data(self, save_path):
        # Re-organize data from plan, reindex, and save
        train_labels, test_labels, current_test = next(self.cross_validator)
        re_labels = pd.DataFrame(columns=['env', 'subject', 'bag', 'csi', 'group', 'segment', 'timestamp', 'img_inds', 'csi_inds'])
        re_data = {mod: [] for mod in self.modalities}
        for _, row in df.iterrows():
            csi = row['csi']
            bag = row['bag']
            
            pass
        
    def gen_same_amount_plan(self, path, subset_ratio=0.1561):
        with open(path, 'rb') as f:
            plan = pickle.load(f)
        cross_validator = iter(plan)
        print(f'\033[32mData Organizer: Loaded plan!\033[0m')
        t_labels = {}
        # PRESERVE TEST
        for i in range(4):
            train_labels, test_labels, current_test = next(cross_validator)
            t_labels[current_test] = test_labels
            self.total_segment_labels.drop(test_labels.index, inplace=True)
            print(len(self.total_segment_labels), len(test_labels))
        
        cross_validator = CrossValidator(self.total_segment_labels, 'env', None, None, subset_ratio)
                
        new_plan = []
        
        # REGENERATE TRAIN
        for c_test in t_labels.keys():
            train_labels = self.total_segment_labels[self.total_segment_labels['env']==c_test]
            train_subset_size = int(len(train_labels) * subset_ratio)
            
            print(f" {c_test} Train len = {train_subset_size} from {len(train_labels)}\n"
                  )

            train_subset_indices = torch.randperm(len(train_labels))[:train_subset_size]
            train_labels = train_labels.iloc[train_subset_indices]
            
            new_plan.append([train_labels, t_labels[c_test], c_test])


        with open(f'../dataset/Door_EXP/single_env_same_amount.pkl', 'wb') as f:
            pickle.dump(new_plan, f)
                
            print('Plan saved!')
            
    def swap_train_test(self):
        self.train_labels, self.test_labels = self.test_labels, self.train_labels
        if self.train and self.test:
            self.train, self.test = self.test, self.train
        self.current_test = self.test
        print("Train and Test labels swapped!")
        