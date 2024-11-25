import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from IPython.display import clear_output

from Loss import MyLossLog
from misc import timer, file_finder
import time
from datetime import timedelta, datetime


class ExtraParams:
    def __init__(self, device):
        self.device = device
        self.params = {}
        self.track = {}
        self.updatable = False

    def add(self, **kwargs):
        if kwargs:
            self.updatable = True
        for key, value in kwargs.items():
            self.params[key] = torch.nn.Parameter(torch.tensor(value, device=self.device), requires_grad=True)
            self.track[key] = [kwargs[key]]

    def update(self):
        if self.updatable:
            for param, value in self.params.items():
                self.track[param].append(value.cpu().detach().tolist())

    def plot_track(self, *args: str):
        fig = plt.figure(constrained_layout=True)
        fig.suptitle("Extra Parameters")

        for param in args:
            if param in self.params.keys():
                plt.plot(self.track[param], label=param)
        plt.xlabel("#Epoch")
        plt.ylabel("Value")
        plt.grid()
        plt.legend(fontsize="20")
        plt.show()
        return fig, "Extra_Parameters.jpg"


class EarlyStopping:

    def __init__(self, min_epoch=200, early_stop_max=10, lr_decay_max=5, verbose=True, delta=0, *args, **kwargs):

        self.min_epoch = min_epoch

        self.early_stop_max = early_stop_max
        self.early_stop_counter = 0
        self.stop_flag = False

        self.verbose = verbose
        self.delta = delta
        self.total_epochs = 0
        self.best_valid_loss = np.inf

        self.decay_flag = False
        self.lr_decay_counter = 0
        self.lr_decay_max = lr_decay_max

    def __call__(self, val_loss, early_stop=True, lr_decay=True):
        self.total_epochs += 1
        self.decay_flag = False
        if self.min_epoch !=0 and self.total_epochs < self.min_epoch:
            return
        
        if early_stop:
            if val_loss >= self.best_valid_loss:
                self.early_stop_counter += 1
                if self.verbose:
                    print(f"\n\033[32mEarly Stopping reporting: {self.early_stop_counter} out of {self.early_stop_max}\033[0m", end='')
                if self.early_stop_counter >= self.early_stop_max:
                    if lr_decay:
                        self.lr_decay_counter += 1
                        if self.verbose:
                            print(f"\n\033[32mLr decay reporting: {self.lr_decay_counter} out of {self.lr_decay_max}. "
                                f"Decay rate = {0.5 ** self.lr_decay_counter}\033[0m", end='')
                        if self.lr_decay_counter >= self.lr_decay_max:
                            self.stop_flag = True
                        else:
                            self.decay_flag = True
                            self.early_stop_counter = 0
                    else:
                        self.stop_flag = True
            else:
                self.best_valid_loss = val_loss
                self.early_stop_counter = 0
                

class TrainingPhase:
    def __init__(self,
                 name = None,
                 train_module='all',
                 eval_module=[],
                 loss='LOSS',
                 lr=1.e-4,
                 optimizer=torch.optim.Adam,
                 scaler=GradScaler(),
                 freeze_params=None,
                 tolerance=1,
                 conditioned_update=False,
                 verbose=False,
                 loss_arg=None):
        
        self.name = name
        self.loss = loss
        self.optimizer_def = optimizer
        self.optimizer = None
        self.scaler = scaler
        
        self.train_module = train_module
        self.eval_module = eval_module
        self.trainable_params = None
        self.freeze_params = freeze_params
        
        if self.eval_module is None:
            self.eval_module = []
        
        # IF USE FREEZE_PARAMS:
        # e.g. freeze_params = {'csien': ['fc_mu', 'fc_logvar']}
        
        self.tolerance = tolerance
        self.current_best = float("inf")
        self.lr = lr
        self.lr_decay_rate = 0.5
        self.conditioned_update = conditioned_update
        
        self.verbose = verbose
        self.show_trainable_params = False
        self.PREDS = None
        self.TMP_LOSS = None
        
        self.loss_arg = loss_arg
    
    def __call__(self, models, data, calculate_loss):
        
        def update(TMP_LOSS):
            if torch.isnan(TMP_LOSS[self.loss]):
                print(f"Phase {self.name}: NaN value in loss {self.loss}, skipping update.")
            elif not torch.isfinite(TMP_LOSS[self.loss]):
                print(f"Phase {self.name}: Infinite value in loss {self.loss}, skipping update.")
                
            else:
                self.scaler.scale(TMP_LOSS[self.loss]).backward()

                # GRADIENT CLIPPING
                # self.scaler.unscale_(self.optimizer)  # Unscale gradients for clipping if needed
                # for group in self.optimizer.param_groups:
                #     torch.nn.utils.clip_grad_norm_(group['params'], max_norm=1.0)
                # Another way:
                # for model in models.values():
                #    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
        # SET TRAINABLE PARAMS & LEARNING RATE
        if not self.optimizer:
            self.optimizer = self.optimizer_def(self.set_params(models), self.lr)
        else:
            _ = self.set_params(models)
        
        self.train_mode(models)
        
        progress_bar = None
        if self.verbose:
            bar_format = '{desc}{percentage:3.0f}%|{bar}|[{postfix}]'
            progress_bar = tqdm(total=self.tolerance, bar_format=bar_format)
                
        if not self.show_trainable_params:
            for key, model in models.items():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(f"{key} {name}: requires_grad={param.requires_grad}")
            self.show_trainable_params = True
        
        for i in range(self.tolerance):
            # Perform loss calculation
            with autocast():
                if self.loss_arg is not None:   
                    PREDS, TMP_LOSS = calculate_loss(data, self.loss_arg)
                else:
                    PREDS, TMP_LOSS = calculate_loss(data) 
            
            # Optionally update based on the loss
            if not self.conditioned_update:
                update(TMP_LOSS)

            # Handle progress bar updates
            if self.verbose:
                progress_bar.set_description(f" {self.name} phase: iter {i + 1}/{self.tolerance}")
                progress_bar.set_postfix({self.loss: f"{TMP_LOSS[self.loss].item():.4f}"})
                progress_bar.update(1)

            # Check for improvement in the loss
            if TMP_LOSS[self.loss] < self.current_best:
                self.current_best = TMP_LOSS[self.loss]
                # CONDITIONED UPDATE
                if self.conditioned_update:
                    update(TMP_LOSS)
                break

        # Close progress bar if used
        if self.verbose and progress_bar is not None:
            progress_bar.close()
            
        self.PREDS, self.TMP_LOSS = PREDS, TMP_LOSS
        return PREDS, TMP_LOSS
    
    def set_params(self, models):
        self.train_module = list(models.keys()) if self.train_module == 'all' else self.train_module
        trainable_params = []
        for model in self.train_module:
            for name, param in models[model].named_parameters():
                if self.freeze_params and name in self.freeze_params[model]:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    trainable_params.append({'params': param, 'lr': self.lr})
        
        self.eval_module = list(models.keys()) if self.eval_module == 'all' else self.eval_module
        if self.eval_module:
            for model in self.eval_module:
                for param in models[model].parameters():
                    param.requires_grad = False
        
        if self.trainable_params is None:
            self.trainable_params = trainable_params
            return trainable_params
    
    def train_mode(self, models):
        for model in self.train_module + self.eval_module:
            models[model].train()
            # EVAL MODULES SHOULD ALSO BE SET AS TRAIN
                    
    def lr_decay(self):
        self.lr *= self.lr_decay_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        
        
class ValidationPhase:
    def __init__(self, name, loader='valid', best_loss='LOSS', loss_arg=None):
        self.name = name
        self.loader = loader
        self.best_loss = best_loss
        self.best_val_loss = float("inf")
        self.best_vloss_ep = 0
        self.loss_arg = loss_arg
        
    def __call__(self, models, data, calculate_loss):

        with torch.no_grad():
            if self.loss_arg is not None:   
                PREDS, TMP_LOSS = calculate_loss(data, self.loss_arg)
            else:
                PREDS, TMP_LOSS = calculate_loss(data)            
        return PREDS, TMP_LOSS
    


class BasicTrainer:
    def __init__(self, name, 
                 epochs, cuda,
                 dataloaders,
                 loss_optimizer: dict,
                 networks = None,
                 preprocess = None,
                 modality = {'csi', 'rimg', 'tag', 'ind'},
                 train_module = 'all',
                 eval_module = None,
                 notion = None,
                 *args, **kwargs
                 ):
        self.name = name
        self.start_ep = 1
        self.epochs = epochs
        self.loss_optimizer = loss_optimizer
        self.scaler = GradScaler()

        # Please define optimizers in this way
        # self.optimizer: dict = {'LOSS1': ['optimizer1', lr1],
        #                        'LOSS2': ['optimizer2', lr2],
        #                        'LOSS3': '...'}
        self.thread = 'single'

        self.dataloader = dataloaders
        # {'train': train_loader,
        # 'valid': valid_loader,
        # 'test': test_loader}

        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        self.extra_params = ExtraParams(self.device)
            
        self.modality = modality
        
        self.preprocess = preprocess
        
        self.train_module =  train_module
        self.eval_module = eval_module

        self.loss_terms = ('loss1', 'loss2', '...')
        self.pred_terms = ('predict1', 'predict2', '...')
        self.losslog = MyLossLog(self.name, self.loss_terms, self.pred_terms)
        
        self.current_epoch = 0
        self.early_stopping_trigger = 'main'
        
        self.train_batches = len(self.dataloader['train'])
        self.valid_batches = len(self.dataloader['valid'])
        self.train_sampled_batches = None
        self.valid_sampled_batches = None
        
        self.training_phases = {
            'main': TrainingPhase(name='main',
                 train_module='all',
                 eval_module=None,
                 loss='LOSS',
                 lr=1.e-4,
                 optimizer=torch.optim.Adam,
                 scaler=GradScaler(),
                 freeze_params=None)}
        
        self.valid_phases = {
            'main': ValidationPhase(name='main')
        }
        
        self.on_test = 'train'

        self.notion = notion
        self.save_path = f'../saved/{notion}/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.early_stopping = None
        
    def data_preprocess(self, mode, data):
        # mode in ('train', 'valid', 'test')
        # Use 16-bit type to save memory and speed up
        if self.preprocess:
            data = self.preprocess(data, self.modality)

        data = {key: data[key].to(torch.float32).to(self.device) for key in self.modality if key in data}
        if 'tag' in data:
            data['tag'] = data['tag'].to(torch.int32).to(self.device)
            
        return data

    def current_ep(self):
        return self.losslog.current_epoch
    
    def current_lr(self):
        return {loss: lr for loss, lr in self.losslog.loss.items()}
    
    def calculate_loss(self, *inputs):
        # mode in ('train', 'valid', 'test')
        # --- Return losses in this way ---
        PREDS = {pred: None for pred in self.pred_terms}
        TMP_LOSS = {loss: None for loss in self.loss_terms}
        return PREDS, TMP_LOSS
                
    def assign_params(self):
        pass

    @timer
    def train(self, early_stop=True, lr_decay=True, subsample_fraction=1, *args, **kwargs):
        
        # TO BE MOVED ELSEWHERE
        # if self.extra_params.updatable:
        #     for param, value in self.extra_params.params.items():
        #         trainable_params.append({'params': value})

        # ===============MANUALLY SET PARAMS==============
        self.assign_params()

        # ===============SET TRAINING FLOW==============
        bar_format = '{desc}{percentage:3.0f}%|{bar}|[{elapsed}<{remaining}{postfix}]'

        if subsample_fraction < 1:
            self.train_batches = int(self.train_batches * subsample_fraction)
            self.train_sampled_batches = np.random.choice(len(self.dataloader['train']), self.train_batches, replace=False)
            
            self.valid_batches = int(self.valid_batches * subsample_fraction)
            self.valid_sampled_batches = np.random.choice(len(self.dataloader['valid']), self.valid_batches, replace=False)
            
        self.epochs = 1000 if early_stop else self.epochs
        self.early_stopping = EarlyStopping(*args, **kwargs)
        self.temp_loss = {loss: None for loss in self.loss_terms}
        # ===============TRAIN & VALIDATE EACH EPOCH==============
        start = time.time()
        start_time = datetime.fromtimestamp(start)
        print(f"\033[32m=========={start_time.strftime('%Y-%m-%d %H:%M:%S')} {self.notion} {self.name} Training starting==========\033[0m")
        
        for epoch in tqdm(range(self.start_ep, self.epochs)):
            print('')
            self.current_epoch = epoch
            # =====================train============================
                
            EPOCH_LOSS = {loss: [] for loss in self.loss_terms}

            with tqdm(total=self.train_batches, bar_format=bar_format) as _tqdm:
                _tqdm.set_description(f"{self.notion} {self.name} train: ep {epoch}/{self.epochs}")
                
                for idx, data in enumerate(self.dataloader['train'], 0):
                    TMP_LOSS = {loss: [] for loss in self.loss_terms}
                    
                    # Randomly select samples
                    if self.train_sampled_batches is not None and idx not in self.train_sampled_batches:
                        continue
                        
                    data_ = self.data_preprocess('train', data)
                    
                    for name, phase in self.training_phases.items():
                        _, TMP_LOSS_ = phase(self.models, data_, self.calculate_loss)
                        TMP_LOSS.update(TMP_LOSS_)

                    for key in TMP_LOSS.keys():
                        EPOCH_LOSS[key].append(TMP_LOSS[key].item())

                    _tqdm.set_postfix({'batch': f"{idx}/{self.train_batches}",
                                        'loss': f"{TMP_LOSS['LOSS'].item():.4f}"})
                    _tqdm.update(1)

            for key, value in EPOCH_LOSS.items():
                EPOCH_LOSS[key] = np.average(value)
            self.losslog('train', EPOCH_LOSS)
            self.extra_params.update()
            # clear_output(wait=True) 

            # =====================valid============================
            print('')
            for model in self.models:
                self.models[model].eval()
                
            for name, phase in self.valid_phases.items():
                
                EPOCH_LOSS = {loss: [] for loss in self.loss_terms}

                with tqdm(total=self.valid_batches, bar_format=bar_format) as _tqdm:
                    _tqdm.set_description(f"{self.notion} {self.name} {name} valid: ep {epoch}/{self.epochs}")
                    
                    for idx, data in enumerate(self.dataloader.get(phase.loader, 'valid'), 0):
                        # Randomly select samples
                        if self.valid_sampled_batches is not None and idx not in self.valid_sampled_batches:
                            continue
                        
                        data_ = self.data_preprocess('valid', data)

                        PREDS, TMP_LOSS = phase(self.models, data_, self.calculate_loss)
                        
                        for key in EPOCH_LOSS.keys():
                            EPOCH_LOSS[key].append(TMP_LOSS[key].item())
                            
                        if epoch % 10 == 0 and idx == 0:
                            self.losslog.reset('pred', dataset='VALID')
                            self.losslog('pred', PREDS)
                            self.plot_test(autosave=False)

                        val_loss = np.average(EPOCH_LOSS['LOSS'])
                        if 0 < val_loss < phase.best_val_loss:
                            phase.best_val_loss = val_loss
                            phase.best_vloss_ep = self.current_epoch
                            
                        _tqdm.set_postfix({'batch': f"{idx}/{self.valid_batches}",
                                        'loss': f"{TMP_LOSS['LOSS'].item():.4f}",
                                        'current best': f"{phase.best_val_loss:.4f} @ epoch {phase.best_vloss_ep}"})
                        _tqdm.update(1)

                    with open(f"{self.save_path}{self.name}_{name}_trained.txt", 'w') as logfile:
                        logfile.write(f"{self.notion}_{self.name}\n"
                                    f"Start time = {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                    f"Total epochs = {self.current_ep()}\n"
                                    f"Best val_loss = {phase.best_val_loss} @ epoch {phase.best_vloss_ep}\n"
                                    f"Final validation losses:\n"
                                    f"{' '.join([key + ': ' + str(TMP_LOSS[key].item()) for key in self.loss_terms])}\n"
                                    )
                for key in EPOCH_LOSS.keys():          
                    EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
                self.losslog(phase.loader, EPOCH_LOSS)
                    
            # Check output every 10 epochs
            if epoch % 10 == 0:
                self.plot_train_loss(autosave=False)
                
                    
            # Save checkpoint every 50 epochs
            if epoch % 50 == 0:
                self.save()

            self.early_stopping(self.valid_phases.get(self.early_stopping_trigger).best_val_loss, early_stop, lr_decay)
            if lr_decay and self.early_stopping.decay_flag:
                self.losslog.decay(0.5)
                for phase in self.training_phases.values():
                    phase.lr_decay()
                    
            if early_stop and self.early_stopping.stop_flag:
                self.losslog.in_training = False
                if 'save_model' in kwargs.keys() and kwargs['save_model'] is False:
                    break
                else:
                    end = time.time()
                    end_time = datetime.fromtimestamp(end)
                    print(f"\n\033[32mEarly Stopping triggered. Saving @ epoch {epoch}...\033[0m")
                    for model in self.train_module:
                        torch.save(self.models[model].state_dict(),
                                   f"{self.save_path}{self.name}_models_{model}_best.pth")
                        
                    with open(f"{self.save_path}{self.name}_trained.txt", 'a') as logfile:
                        logfile.write(f"End time = {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                      f"Total training time = {str(timedelta(seconds=end-start))}\n"
                                      f"\nModules:\n{list(self.models.values())}\n")
                    break


    @timer
    def test(self, single_test=False, loader: str = 'test', subsample_fraction=1, control_speed=False, *args, **kwargs):

        for model in self.models:
            self.models[model].eval()

        EPOCH_LOSS = {loss: [] for loss in self.loss_terms}
        self.losslog.reset('test', 'pred', dataset=loader)
        
        bar_format = '{desc}{percentage:3.0f}%|{bar}|[{elapsed}<{remaining}{postfix}]'
        self.on_test = loader
        
        test_sampled_batches = None
        test_batches = len(self.dataloader[loader])
        if subsample_fraction < 1:
            if loader == 'test':
                test_batches = int(test_batches * subsample_fraction)
                test_sampled_batches = np.random.choice(len(self.dataloader[loader]), test_batches, replace=False)
            else:
                test_sampled_batches = self.train_sampled_batches
        
        start = time.time()
        start_time = datetime.fromtimestamp(start)
        
        print(f"\033[32m=========={start_time.strftime('%Y-%m-%d %H:%M:%S')} {self.notion} {self.name} Test starting==========\033[0m")

        with tqdm(total=test_batches, bar_format=bar_format) as _tqdm:
            _tqdm.set_description(f'{self.notion} {self.name} test')
            
            for idx, data in enumerate(self.dataloader[loader], 0):
                # Randomly select samples
                if test_sampled_batches is not None and idx not in test_sampled_batches:
                    continue
                
                data_ = self.data_preprocess('test', data)

                with torch.no_grad():
                    if single_test:
                        for sample in range(len(list(data_.values())[0])):
                            data_i = {key: data_[key][sample][np.newaxis, ...] for key in data_.keys()}
                            PREDS, TMP_LOSS = self.calculate_loss(data_i)

                            for key in EPOCH_LOSS.keys():
                                EPOCH_LOSS[key].append(TMP_LOSS[key].item())
                    else:
                        PREDS, TMP_LOSS = self.calculate_loss('test',data_)
                        
                        for key in EPOCH_LOSS.keys():
                            EPOCH_LOSS[key].append(np.average(TMP_LOSS[key].item()))
                            
                    self.losslog('pred', PREDS)
                        
                _tqdm.set_postfix({'batch': f"{idx}/{test_batches}",
                                   'loss': f"{TMP_LOSS['LOSS'].item():.4f}"})
                _tqdm.update(1)
                
                if control_speed:
                    time.sleep(1)

        self.losslog('test', EPOCH_LOSS)
        for key in EPOCH_LOSS.keys():
            EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])

        print(f"\nTest finished. Average loss={EPOCH_LOSS}")

    def plot_train_loss(self, title=None, double_y=False, plot_terms='all', autosave=False, **kwargs):
        fig = self.losslog.plot_train(title, plot_terms, double_y)
        if autosave:
            for filename, fig in fig.items():
                fig.savefig(f"{self.save_path}{filename}")

    def plot_test(self, select_inds=None, select_num=8, autosave=False, **kwargs):
        # According to actual usages
        self.losslog.generate_indices(select_inds, select_num)
        pass

    def save(self):
        print("Saving models...")
        for modelname, model in self.models.items():
            print(f"Saving {modelname}...")
            torch.save({
                'epoch': self.current_ep(),
                'lr': self.current_lr(),
                'model_state_dict': model.state_dict(),
                }, 
                       f"{self.save_path}{self.name}_models_{modelname}_checkpoint.pth")
            
        for name, phase in self.training_phases.items():
            print(f"Saving {name} optimizer...")
            torch.save({
                'epoch': self.current_ep(),
                'lr': self.current_lr(),
                'optimizer_state_dict': phase.optimizer.state_dict()
                }, 
                       f"{self.save_path}{self.name}_optimizer_{name}_checkpoint.pth")
        
        print("All saved!")
        
    def load(self, path, name='Student', mode='checkpoint', load_optimizer=True, gpu=None):
        print(f"\033[32m=========={self.notion} {self.name} Loading==========\033[0m")

        # Collect all matching file paths for each model
        model_files = {model_name: None for model_name in self.models.keys()}
        optimizer_files = {phase_name: None for phase_name in self.training_phases.keys()}
        
        def find_path(file_path, file_name_, ext):
            if ext == '.pth' and name in file_name_ and mode in file_name_:
                # Match file with model names
                for model_name in self.models.keys():
                    if model_name in file_name_:
                        model_files[model_name] = file_path
                # Match file with optimizer names
                for phase_name in self.training_phases.keys():
                    if phase_name in file_name_:
                        optimizer_files[phase_name] = file_path

        file_finder(path, find_path)

        # Load each model's checkpoint if available
        for model_name, model in self.models.items():
            file_path = model_files.get(model_name)
            if file_path:
                checkpoint = torch.load(file_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                if model_name == 'rimgde':
                    model.to(self.device2)
                else:
                    model.to(self.device)
                ep = f" at epoch {checkpoint.get('epoch')}" if checkpoint.get('epoch') else ''
                self.start_ep = checkpoint.get('epoch', 0)
                print(f"Loaded model {model_name}{ep} from {file_path}!")

        # Load each optimizer's checkpoint if available
        if load_optimizer:
            for phase_name, phase in self.training_phases.items():
                file_path = optimizer_files.get(phase_name)
                if file_path:
                    checkpoint = torch.load(file_path, map_location=f"cuda:{gpu}" if isinstance(gpu, int) else None)
                    if 'optimizer_state_dict' in checkpoint:
                        phase.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    ep = f" at epoch {checkpoint.get('epoch')}" if checkpoint.get('epoch') else ''
                    lr = f" lr {checkpoint.get('lr')}" if checkpoint.get('lr') else ''
                    print(f"Loaded optimizer {phase_name}{ep}{lr} from {file_path}!")


    def schedule(self, autosave=True, *args, **kwargs):
        # Training, testing and saving
        model = self.train(autosave=autosave, notion=self.notion, *args, **kwargs)
        self.plot_train_loss(autosave=autosave)
        self.test(loader='train', *args, **kwargs)
        self.plot_test(select_num=8, autosave=autosave)
        self.test(loader='test', *args, **kwargs)
        self.plot_test(select_num=8, autosave=autosave)
        self.losslog.save('preds', self.save_path)
        print(f'\n\033[32m{self.name} schedule Completed!\033[0m')
        return model

