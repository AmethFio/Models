import torch
try:
    from torch.amp import autocast, GradScaler
    _autocast_arg = {'device_type': 'cuda'}
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    _autocast_arg = {}

from LossLayer import LossTracker
from functools import wraps
from tqdm.notebook import tqdm

def with_progress_bar(bar_format=None):
    if bar_format is None:
        bar_format = '{desc}{percentage:3.0f}%|{bar}|[{elapsed}<{remaining}{postfix}]'

    def decorator(func):
        @wraps(func)
        def wrapped(self, dataloader, *args, **kwargs):
            total = len(dataloader)
            print('')
            with tqdm(total=total, dynamic_ncols=True, bar_format=bar_format) as progress_bar:
                for idx, loss, *_ in func(self, dataloader, *args, **kwargs):
                    progress_bar.set_postfix({
                        'batch': f"{idx}/{total}",
                        'loss': f"{loss:.4f}"
                    })
                    progress_bar.update(1)   # ← 正确推进进度条
                    yield idx, loss, *_
        return wrapped
    return decorator


class ModelStep:
    def __init__(self):
        pass

    def preprocess_gpu(self, data):
        return data

    def data_preprocess(self, data, device):
        # Put batch onto GPU
        data = {key: value.to(device=device, dtype=torch.float32) for key, value in data.items()}

        if 'tag' in data:
            data['tag'] = data['tag'].to(torch.int32)

        data = self.preprocess_gpu(data)
            
        return data

    def __call__(self, dataloader, model, device, loss_tracker):
        for idx, data in enumerate(dataloader, 1):
            # Prepare data
            _data_ = self.data_preprocess(data, device)
            # Calculate model output
            with autocast(**_autocast_arg):
                preds, loss = model(_data_)
            # Log into buffer
            loss_tracker.log_loss(loss)
            loss_tracker.log_preds(preds)

            # Return the total loss for update
            yield idx, loss['LOSS']


class Trainer:
    def __init__(self, device, name='TRAIN',
                train_module='all', eval_module=[]):
        self.name = name
        self.device = device
        self.loss_tracker = LossTracker(name, self.device)
        self.step = ModelStep()
        self.scaler = GradScaler()

        self.train_module = train_module
        self.eval_module = eval_module
        self.already_set_optimizer = False

    def set_optimizer(self, model, optimizer, lr):
        if self.train_module == 'all':
            self.train_module = list(model.get_modules().keys())

        if self.eval_module == 'all':
            self.eval_module = list(model.get_modules().keys())

        trainable_params = []
        for module in self.train_module:
            for name, param in getattr(model, module).named_parameters():
                param.requires_grad = True
                trainable_params.append({'params': param, 'lr': lr})

        for module in self.eval_module:
            for name, param in getattr(model, module).named_parameters():
                param.requires_grad = False

        opt = optimizer(trainable_params, lr, amsgrad=False)
        return opt

    def epoch_behavior(self, optimizer):
        lr = self.loss_tracker.log_lr_change(optimizer)
        epoch_loss = self.loss_tracker.get_epoch_mean('train')

    @with_progress_bar()
    def __call__(self, dataloader, model, optimizer):
        for module in model.get_modules().keys():
            getattr(model, module).train()

        for idx, loss in self.step(dataloader, model, self.device, self.loss_tracker):
            
            # Update
            if torch.isnan(loss):
                print(f"\033[31mPhase {self.name}: NaN value in loss, skipping update.\033[0m")
            elif not torch.isfinite(loss):
                print(f"\033[31mPhase {self.name}: Infinite value in loss, skipping update.\033[0m")
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
            yield idx, loss

        self.epoch_behavior(optimizer)


class Validator:
    def __init__(self, device, name='VALID'):
        self.name = name
        self.device = device
        self.loss_tracker = LossTracker(name, self.device)
        self.step = ModelStep()

        self.best_val_loss = torch.inf
        self.best_val_epoch = 0

        self.early_stopper = EarlyStopper()

    def epoch_behavior(self, loss, optimizer, early_stop, lr_decay):
        self.loss_tracker.current_epoch += 1
        epoch_loss = self.loss_tracker.get_epoch_mean()

        loss = epoch_loss['LOSS']
        # Log best loss
        if 0 < loss < self.best_val_loss:
            self.best_val_loss = loss
            self.best_val_epoch = loss_tracker.current_epoch

        if early_stop:
            self.early_stopper(loss, lr_decay, optimizer)

    @with_progress_bar()
    def __call__(self, dataloader, model, optimizer, early_stop, lr_decay):
        for module in model.get_modules().keys():
            getattr(model, module).eval()

        for idx, loss in self.step(dataloader, model, self.device, self.loss_tracker):
            yield idx, loss, self.early_stopper.stop_flag
        self.epoch_behavior(loss, optimizer, early_stop, lr_decay)


class Tester:
    def __init__(self, device, name='TEST'):
        self.name = name
        self.device = device
        self.loss_tracker = LossTracker(name, self.device)
        self.step = ModelStep()

    def epoch_behavior(self):
        for key, value in self.loss_tracker.loss_buffer.buffer.items():
            self.loss_tracker.loss_buffer.buffer[key] = value.squeeze()

    @with_progress_bar()
    def __call__(self, dataloader, model):
        for module in model.get_modules().keys():
            getattr(model, module).eval()

        for idx, loss in self.step(dataloader, model, self.device, self.loss_tracker):
            yield idx, loss
        self.epoch_behavior()


class EarlyStopper:
    def __init__(self, min_epoch=100, tolerance=10, verbose=True, *args, **kwargs):

        self.min_epoch = min_epoch
        self.tolerance = tolerance

        self.early_stop_counter = 0
        self.stop_flag = False

        self.verbose = verbose
        self.best_valid_loss = torch.inf

        self.lr_decayer = LrDecayer()

    def __call__(self, val_loss, lr_decay, optimizer):
        # Early stopping flag
        if val_loss >= self.best_valid_loss:
            self.early_stop_counter += 1
            if self.verbose:
                print(f"\033[32mEarly Stopping reporting: {self.early_stop_counter} out of {self.tolerance}\033[0m")
            
            if self.early_stop_counter > self.tolerance:
                self.early_stop_counter = 0
                if lr_decay:
                    self.lr_decayer(optimizer)
                    if self.lr_decayer.stop_flag:
                        self.stop_flag = True
                else:
                    self.stop_flag = True
        else:
            self.best_valid_loss = val_loss
            self.early_stop_counter = 0

class LrDecayer:
    def __init__(self, tolerance=5, verbose=True, *args, **kwargs):

        self.tolerance = tolerance
        self.lr_decay_counter = 0
        self.stop_flag = False
        self.decay_rate = 0.5
        self.verbose = verbose

    def __call__(self, optimizer):
        self.lr_decay_counter += 1
        for param_group in optimizer.param_groups:
            param_group['lr'] *= self.decay_rate

        if self.verbose:
            print(f"\033[32mLr decay reporting: {self.lr_decay_counter} out of {self.tolerance}. "
                f"Decay rate = {self.decay_rate ** self.lr_decay_counter}\033[0m")
        if self.lr_decay_counter > self.tolerance:
            self.stop_flag = True
