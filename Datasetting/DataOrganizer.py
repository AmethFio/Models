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
from Datasetting.Dataset import *

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
        select_labels = self.labels[self.labels[self.level]!=self.current_test]
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
        self.iter_range()
        self.current = -1
        self.current_test = None


class DataOrganizer:
    def __init__(self, name, data_path, crossvalidator, 
                 modalities=['csi', 'rimg', 'cimg', 'ctr', 'dpt', 'pd'],
                 removal=Removal(remov)):
        self.name = name
        self.data_path = data_path
        self.cross_validator = crossvalidator
        
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
        
    def file_condition(self, fname, fext):
        ret = False
        typ = None
        modality = None
        name = None
        
        if fext == '.csv':
            if 'matched' in fname and 'checkpoint' not in fname:
                ret = True
                typ = 'label'

        elif fext == '.npy':
            if not ('env' in fname or 'test' in fname or 'checkpoint' in fname):
                name, modality = fname.split('-')
                if modality in self.modalities:
                    ret = True
                    typ = 'data'
                
        return ret, typ, (modality, name)
                
    @timer
    def load(self, multi=False):
        
        def load_single(fname, fext, fpath, atm, start, etc):
            data = None
            label = None
            loaded = False

            if typ == 'data':
                # Load CSI and IMGs <name>-<modality>.npy
                d = np.load(fpath, mmap_mode='r')
                data = (*etc, d) # modality, name, data
                
                end = time.time()
                print(f"Loaded {fname}{fext} of {d.shape}{atm}, elapsed {(end - start):.4f} sec")
                loaded = True
                
            elif typ == 'label':
                # Load Label <subject>_matched.csv
                label = pd.read_csv(fpath)
                
                end = time.time()
                print(f"Loaded {fname}{fext} of len {len(label)}{atm}, elapsed {(end - start):.4f} sec")
                loaded = True
 
            return label, data, loaded
        
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
                        label, data, loaded = load_single(file_name_, ext, file_path, atm, start, etc)

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
        
        for dpath in self.data_path:
            if multi:
                print('Multi-process loading...')
                files = file_finder_multi(dpath, process_name="Data Organizer")
                results = Parallel(n_jobs=8)(
                    delayed(load_attempt)(f, file_name_, ext) for f, file_name_, ext in files
                )
                
                for label, data in results:
                    unpack_results(label, data)
                        
            else:
                print('Single-process loading...')
                # results = file_finder(dpath, load_single, process_name="Data Organizer")
                for p, _, file_lst in os.walk(dpath):
                    for file_name in file_lst:
                        file_name_, ext = os.path.splitext(file_name)
                        loadable, typ, etc = self.file_condition(file_name_, ext)
                
                        if not loadable:
                            print(f"\033[33mSkipping {file_name_}{ext}\033[0m")
                            
                        else:
                            start = time.time()
                            label, data, _ = load_single(file_name_, ext, os.path.join(p, file_name), '', start, etc)
                            unpack_results(label, data, fail)
                    
        print(f"\nLoad complete!")
        if fail is not None:
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
            train_labels, test_labels, train_range, current_test = next(self.cross_validator)
            
            while True:
                if specify_test is not None and current_test != specify_test:
                    train_labels, test_labels, train_range, current_test = next(self.cross_validator)
                    
                else:
                    break
            
            self.train_labels, self.test_labels = train_labels, test_labels
    
    def load_plan(self, path):
        with open(path, 'rb') as f:
            plan = pickle.load(f)
        self.cross_validator = iter(plan)
        print(f'\033[32mData Organizer: Loaded plan!\033[0m')
    
    def gen_loaders(self, mode='s', 
                    train_ratio=0.8, 
                    batch_size=64, 
                    csi_len=300, 
                    single_pd=True, 
                    num_workers=14, 
                    save_dataset=False, 
                    shuffle_test=True, 
                    pin_memory=True):

        print(f'\033[32mData Organizer: Generating loaders for {mode}: '
              f'level = {self.cross_validator.level}, current test = {self.cross_validator.current_test}\033[0m')
        
        data = self.data.copy()
        
        if mode == 't':
            data.pop('csi')
            data.pop('pd')

        elif mode == 's':
            if self.removal:
                self.train_labels = self.removal(self.train_labels)
                self.test_labels = self.removal(self.test_labels)
                
        tv_dataset = MyDataset(data, self.train_labels, csi_len, single_pd)
        test_dataset = MyDataset(data, self.test_labels, csi_len, single_pd)
            
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
                                        shuffle=shuffle_test,
                                        worker_init_fn=worker_init_fn
                                    )
        
            print(f" Exported test loader of len {len(test_loader)}, batch size = {batch_size}\n")
        else:
            test_loader = None
        
        return train_loader, valid_loader, test_loader, self.cross_validator.current_test
        
        
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
        

class DANN_Loader2:
    """
    Generates source smaples and target samples by 3:1.
    """
    
    def __init__(self, source_loader, target_loader, target_guide=False, target_guide_num=1):
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.source_iter = iter(self.source_loader)
        self.target_iter = iter(self.target_loader)
        self.maximum_iter = len(source_loader) // 3
        self.current = -1
        
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
            source_samples = [next(self.source_iter) for _ in range(3)]
            source_batch = {key: torch.cat([sample[key] for sample in source_samples], dim=0) 
                            for key in source_samples[0]}  # Combine into a single batch

        except StopIteration:
            self.source_iter = iter(self.source_loader)  # Reset the iterator
            source_samples = [next(self.source_iter) for _ in range(3)]
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
    
    
def gen_dann_loaders(data_organizer, train=None, test=None, subset_ratio=1, batch_size=64, num_workers=2, target_guide=False, target_guide_num=1):
    #if data_organizer.cross_validator and isinstance(data_organizer.cross_validator, CrossValidator):
    #    data_organizer.regen_plan()
    data_organizer.train = train
    data_organizer.test = test

    data_organizer.gen_plan(subset_ratio=subset_ratio)
    source_train_loader, source_valid_loader, target_test_loader, current_test = data_organizer.gen_loaders(mode='s', num_workers=num_workers, batch_size=batch_size)
    data_organizer.swap_train_test()
    target_train_loader, target_valid_loader, source_test_loader, _ = data_organizer.gen_loaders(mode='s', num_workers=num_workers, batch_size=batch_size)
    dann_train_loader = DANN_Loader2(source_train_loader, target_train_loader, target_guide, target_guide_num)
    dann_valid1 = DANN_Loader2(source_valid_loader, target_valid_loader)
    dann_valid2 = DANN_Loader2(target_valid_loader, source_valid_loader)
    dann_test_loader = DANN_Loader2(target_test_loader, source_valid_loader)
    return dann_train_loader, dann_valid1, dann_valid2, dann_test_loader, current_test


def gen_double_valid_loaders(data_organizer, train=None, test=None, subset_ratio=1, batch_size=64, num_workers=2, target_guide=False, target_guide_num=1):
    data_organizer.train = train
    data_organizer.test = test

    data_organizer.gen_plan(subset_ratio=subset_ratio)
    source_train_loader, source_valid_loader, target_test_loader, current_test = data_organizer.gen_loaders(mode='s', num_workers=num_workers, batch_size=batch_size)
    data_organizer.swap_train_test()
    target_train_loader, target_valid_loader, source_test_loader, _ = data_organizer.gen_loaders(mode='s', num_workers=num_workers, batch_size=batch_size)
    
    if target_guide:
        source_train_loader = GuidedLoader(source_train_loader, target_train_loader, target_guide_num)
    
    return source_train_loader, source_valid_loader, target_valid_loader, target_test_loader, current_test
    
    
    

    