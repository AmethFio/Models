import torch
try:
    from torch.amp import autocast, GradScaler
    _autocast_arg = {'device_type': 'cuda'}
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    _autocast_arg = {}

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from LossLayer import LossTracker
from Trainers import *
from misc import timer, file_finder
import time
from datetime import timedelta, datetime

class ModelNotFoundError(Exception):
    def __init__(self, message="\033[31mCheck path: did not load any model!\033[0m"):
        self.message = message
        super().__init__(self.message)

class ModelTrainer:
    """
    ModelTrainer = Trainer + Validator + Tester
    """
    def __init__(self,
                name='Model', dataloader: dict={}, model: dict={}, preprocess=None,
                notion=''):

        self.name = name
        self.notion = notion
        self.save_path = f'../saved/{notion}/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.dataloader = dataloader
        self.model = model
        self.optimizer = torch.optim.Adam()

        self.preprocess = preprocess

        self.trainer = Trainer(f"{self.name}_{self.notion}_TRAIN")
        self.validator = Validator(f"{self.name}_{self.notion}_VALID")
        self.tester = Tester(f"{self.name}_{self.notion}_TEST")

        self.final_log: str = ""

    @timer
    def train(self, epochs=100, early_stop=True, lr_decay=True, save_model=True, *args, **kwargs):
        if early_stop:
            epochs = 999
        start = datetime.fromtimestamp(time.time())
        self.final_log += f"{self.name}_{self.notion}\n"
        self.final_log += f"Start time = {start.strftime('%Y-%m-%d %H:%M:%S')}\n"

        print(f"\033[32m=========={self.start_time.strftime('%Y-%m-%d %H:%M:%S')} {self.notion} {self.name} Training starting==========\033[0m")
        for epoch in tqdm(range(1, epochs + 1), initial=1, dynamic_ncols=True):
            
            for idx, loss in self.trainer(self.dataloader['train'], self.model, self.optimizer, self.preprocess):
                continue

            for idx, loss in self.validator(self.dataloader['valid'], self.model, self.optimizer, self.preprocess, early_stop, lr_decay):
                continue
        
        if save_model:
            self.save(mode='best')

    def plot_train_valid_loss(self):
        filename, out_fig = self.trainer.loss_tracker.plot_loss_track('train')
        filename, out_fig = self.validator.loss_tracker.plot_loss_track('valid')
        plt.show()

    def test(self):
        start = time.time()
        start_time = datetime.fromtimestamp(start)
        print(f"\033[32m=========={start_time.strftime('%Y-%m-%d %H:%M:%S')} {self.notion} {self.name} Test starting==========\033[0m")
            
        for idx, loss in self.tester(self.dataloader['test'], self.model, self.preprocess):
            continue

        print(f"\nTest finished.")

    def plot_test(self, pred_terms):
        filename, out_fig = self.tester.loss_tracker.plot_preds(pred_terms)
        plot.show()
        filename, out_fig = self.tester.loss_tracker.plot_cdf()
        plt.show()

    def save(self, mode='checkpoint'):
        # mode = 'best' or 'chechpoint'
        print("Saving models...")
        for modelname, model in self.model.items():
            print(f"Saving {modelname}...")
            torch.save({
                'model_state_dict': model.state_dict(),
                }, 
                       f"{self.save_path}{self.name}_models_{modelname}_{mode}.pth")
        
        print(f"Saving optimizer...")
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict()
            }, 
                    f"{self.save_path}{self.name}_optimizer_{name}_{mode}.pth")
        
        print("All saved!")

    def load(self, path, name='Student', mode='checkpoint', load_optimizer=False):
        print(f"\033[32m=========={self.notion} {self.name} Loading==========\033[0m")
        hit = False
        # Collect all matching file paths for each model
        model_files = {model_name: None for model_name in self.models.keys()}
        
        def find_path(file_path, file_name_, ext):
            if ext == '.pth' and name in file_name_ and mode in file_name_:
                # Match file with model names
                for model_name in self.models.keys():
                    if model_name in file_name_:
                        model_files[model_name] = file_path

        file_finder(path, find_path)

        # Load each model's checkpoint if available
        for model_name, model in self.models.items():
            file_path = model_files.get(model_name)
            if file_path:
                hit = True
                checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)

                model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                model.to(self.device)

                print(f"Loaded model {model_name} from {file_path}!")

        # Warning
        if not hit:
            raise ModuleNotFoundError

    