import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from LossLayer import LossTracker
from Trainers import *
import time
from datetime import timedelta, datetime
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} elapsed: {end - start} sec")
        return result

    return wrapper

def file_walker(path, process_name=None, *args, **kwargs):
    process_name = f"{process_name}: " if process_name else ''
    print(f"\033[32m{process_name}Loading {path}\033[0m")

    for p, _, file_lst in os.walk(path):
        for file_name in file_lst:
            file_name_, ext = os.path.splitext(file_name)
            filepath = os.path.join(p, file_name)
            yield file_name_, ext, filepath

class ModelNotFoundError(Exception):
    def __init__(self, message="\033[31mCheck path: did not load any model!\033[0m"):
        self.message = message
        super().__init__(self.message)

class ModelTrainer:
    """
    ModelTrainer = Trainer + Validator + Tester
    """
    def __init__(self,
                name='Model', dataloader: dict={}, model: nn.Module=None, 
                train_module='all',
                pred_terms='all',
                lr=1.e-4, device=0,
                notion=''):

        self.name = name
        self.notion = notion
        self.save_path = f'../saved/{notion}/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.dataloader = dataloader

        self.model = model
        self.lr = lr
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

        self.trainer = [Trainer(self.device, f"{self.name}_{self.notion}_TRAIN", train_module)]
        self.validator = [Validator(self.device, f"{self.name}_{self.notion}_VALID")]
        self.tester = Tester(self.device, f"{self.name}_{self.notion}_TEST")

        # For early stopping, changeable
        self.main_validator = self.validator[0]

        self.final_log: str = ""
        self.model_paths = {module_name: None for (module_name, _) in self.model.named_children()}
        self.pred_terms = pred_terms

        self.model.to(self.device)
        self.set_optimizer = False

        self.line_color = {
            'train': 'blue',
            'valid': 'orange',
            }

    def set_optimizer(self):
        for trainer in self.trainer:
            trainer.set_optimizer(self.model, self.lr)

    def plot_train_valid_loss(self):
        filename, fig = self.trainer.loss_tracker.plot_loss_track('blue', 'train')
        for validator in self.validator:
            color = self.line_color.get(validator.valid_set_name, 'orange')
            filename, fig = validator.loss_tracker.plot_loss_track(color, validator.valid_set_name, fig, True)
        return fig

    def add_validator(self, val_name='VALID2', val_set='valid'):
        self.validator.append(Validator(self.device, f"{self.name}_{self.notion}_{val_name}",
                                valid_set_name=val_set))

        for c in ['blue', 'orange', 'green', 'red', 'cyan']:
            if c not in self.line_color.values():
                self.line_color[val_set] = c
                break

    def add_trainer(self, train_name='TRAIN2', train_set='train', loss_to_update='LOSS'):
        self.trainer.append(Trainer(self.device, f"{self.name}_{self.notion}_{train_name}",
                                train_module='all',
                                train_set_name=train_set,
                                loss_to_update=loss_to_update))

    def trainer_organizer(self):
        pass

    @timer
    def train(self, epochs=100, early_stop=True, lr_decay=True, save_model=True, *args, **kwargs):
        if not self.set_optimizer:
            self.set_optimizer()
            self.set_optimizer = True
        if early_stop:
            epochs = 999
        start = datetime.fromtimestamp(time.time())
        self.final_log += f"{self.name}_{self.notion}\n"
        self.final_log += f"Start time = {start.strftime('%Y-%m-%d %H:%M:%S')}\n"

        print(f"\033[32m=========={start.strftime('%Y-%m-%d %H:%M:%S')} {self.notion} {self.name} Training starting==========\033[0m")
        for epoch in tqdm(range(1, epochs + 1), initial=1, dynamic_ncols=True):
            
            for trainer in self.trainer:
                for idx, loss in trainer(epoch, self.dataloader, self.model, self.optimizer):
                    continue

            for validator in self.validator:
                for idx, loss in validator(epoch, self.dataloader, self.model, self.optimizer, early_stop, lr_decay):
                    continue

            if early_stop and self.main_validator.stop_flag:
                break

            if epoch % 10 == 0:
                self.plot_train_valid_loss()
                for validator in self.validator:
                    filename, fig = validator.loss_tracker.plot_preds(self.pred_terms)
                    if self.pred_terms_cimg is not None:
                        filename, fig = validator.loss_tracker.plot_preds(self.pred_terms_cimg)
                    if self.pred_terms_ctr is not None:
                        filename, fig = validator.loss_tracker.plot_ctr(self.pred_terms_ctr)
        
        if save_model:
            self.save(mode='best')

    def test(self, *args, **kwargs):
        start = time.time()
        start_time = datetime.fromtimestamp(start)
        print(f"\033[32m=========={start_time.strftime('%Y-%m-%d %H:%M:%S')} {self.notion} {self.name} Test starting==========\033[0m")
            
        for idx, loss in self.tester(self.dataloader, self.model):
            continue

        self.plot_test()
        self.tester.loss_tracker.save(*args, **kwargs)

        print(f"\nTest finished.")

    def plot_test(self):
        # Change the way of plotting preds specifically.
        filename, out_fig = self.tester.loss_tracker.plot_preds(self.pred_terms)
        if self.pred_terms_cimg is not None:
            filename, fig = self.tester.loss_tracker.plot_preds(self.pred_terms_cimg)
        if self.pred_terms_ctr is not None:
            filename, fig = self.tester.loss_tracker.plot_ctr(self.pred_terms_ctr)
        filename, out_fig = self.tester.loss_tracker.plot_cdf()


    def save(self, mode='checkpoint'):
        # mode = 'best' or 'chechpoint'
        print("Saving model...")
        for modelname, model in self.model.named_children():
            if 'loss' in modelname:
                continue
            print(f"Saving {modelname}...")
            torch.save({
                'model_state_dict': model.state_dict(),
                }, 
                    f"{self.save_path}{self.name}_model_{modelname}_{mode}.pth")
            
        print(f"Saving optimizer...")
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict()
            }, 
                    f"{self.save_path}{self.name}_optimizer_{modelname}_{mode}.pth")
        
        print("All saved!")

    def load(self, path, name='Student', mode='checkpoint', load_optimizer=False):
        print(f"\033[32m=========={self.notion} {self.name} Loading==========\033[0m")
        hit = False
        # Collect all matching file paths for each model
        
        for file_name, ext, file_path in file_walker(path):

            is_ext = (ext == '.pth')
            is_name = (name in file_name)
            is_mode = (mode in file_name)
            isn_opt = ('optimizer' not in file_name)
            isn_loss = ('loss' not in file_name)

            if is_ext and is_name and is_mode and isn_opt and isn_loss:
                # Match file with model names
                for module_name in self.model_paths.keys():
                    if module_name in file_name:
                        self.model_paths[module_name] = file_path

        # Load each model's checkpoint if available
        for module_name, file_path in self.model_paths.items():
            if file_path:
                hit = True
                checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)

                getattr(self.model, module_name).load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                getattr(self.model, module_name).to(self.device)

                print(f"Loaded model {module_name} from {file_path}!")

        # Warning
        if not hit:
            raise ModuleNotFoundError
