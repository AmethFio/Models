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
from IPython.display import clear_output

class AutoCleanTqdm(tqdm):
    # Not proper
    def close(self):
        super().close()
        clear_output(wait=True)

class ProgressBar:
    def __init__(self, desc="", bar_format=None):
        self.desc_name = desc
        if bar_format is None:
            bar_format = '{desc}: {percentage:3.0f}%|{bar}|[{elapsed}<{remaining}{postfix}]'
        self.bar_format = bar_format

    def __call__(self, func):
        """
        func must yield (idx, loss)
        """

        def wrapped(instance, dataloader, *args, **kwargs):
            total = len(dataloader)
            interval = max(total // 10, 1)
            desc = f"{self.desc_name} ep{instance.epoch}"

            with tqdm(
                total=total,
                dynamic_ncols=True,
                bar_format=self.bar_format,
                desc=desc,
                leave=False
            ) as progress_bar:

                for idx, loss in func(dataloader, *args, **kwargs):
                    # 10-step update + always finish at total
                    if idx % interval == 0 or idx == total:
                        progress_bar.set_postfix({
                            "batch": f"{idx}/{total}",
                            "loss": f"{loss:.4f}"
                        })
                        progress_bar.n = idx
                        progress_bar.refresh()

                    yield idx, loss

        return wrapped


class ModelStep:
    def __init__(self):
        # For progress bar
        self.epoch = 0

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
        self.epoch += 1
        for idx, data in enumerate(dataloader, 1):
            # Prepare data
            _data_ = self.data_preprocess(data, device)
            # Calculate model output
            with autocast(**_autocast_arg):
                preds, loss = model(_data_)
                if 'ind' in _data_:
                    preds['IND'] = _data_['ind']
            # Log into buffer
            loss_tracker.log_loss(loss)
            loss_tracker.log_preds(preds)

            # Return the total loss for update
            yield idx, loss['LOSS']


class Trainer:
    name = 'TRAIN'
    def __init__(self, device, name='TRAIN',
                train_module='all'):
        self.name = name
        self.device = device
        self.loss_tracker = LossTracker(name, self.device)
        self.scaler = GradScaler()

        self.train_module = train_module

        self.progress_bar = ProgressBar(self.name)
        self.modelstep = ModelStep()
        # Decorate with progress bar
        self.progress_step = self.progress_bar(self.modelstep)

    def set_optimizer(self, model, optimizer, lr):
        print(f'Setting optimizer... trainable = {self.train_module}')
        reqs = []
        if self.train_module == 'all':
            self.train_module = [module_name for module_name, _ in model.named_children()]

        trainable_params = []
        for name, module in model.named_children():
            
            requires_grad = name in self.train_module
            print(f'Setting {name} as {requires_grad}')
            for p in module.parameters():
                p.requires_grad = requires_grad

            if requires_grad:
                trainable_params += list(module.parameters())
        print('Setting over')

        opt = None
        if len(trainable_params) > 0:
            opt = optimizer(trainable_params, lr, amsgrad=False)
        return opt

    def epoch_behavior(self, optimizer):
        lr = self.loss_tracker.log_lr_change(optimizer)
        epoch_loss = self.loss_tracker.get_epoch_mean('train')
        
    def __call__(self, dataloader, model, optimizer):
        model.train()

        for idx, loss in self.progress_step(self.modelstep, dataloader, model, self.device, self.loss_tracker):
            
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

        self.best_val_loss = torch.inf
        self.best_val_epoch = 0

        self.early_stopper = EarlyStopper()
        self.lr_decayer = LrDecayer()
        self.stop_flag = False

        self.progress_bar = ProgressBar(self.name)
        self.modelstep = ModelStep()
        # Decorate with progress bar
        self.progress_step = self.progress_bar(self.modelstep)

    def epoch_behavior(self, loss, optimizer, early_stop, lr_decay):
        self.loss_tracker.current_epoch += 1
        epoch_loss = self.loss_tracker.get_epoch_mean()

        loss = epoch_loss['LOSS']
        # Log best loss
        if 0 < loss < self.best_val_loss:
            self.best_val_loss = loss
            self.best_val_epoch = self.loss_tracker.current_epoch

        if early_stop:
            self.early_stopper(self.loss_tracker.current_epoch, loss, optimizer, 
            lr_decay, self.lr_decayer)

        if self.early_stopper.stop_flag:
            self.stop_flag = True

    def __call__(self, dataloader, model, optimizer, early_stop, lr_decay):
        model.eval()

        for idx, loss in self.progress_step(self.modelstep, dataloader, model, self.device, self.loss_tracker):
            yield idx, loss
        self.epoch_behavior(loss, optimizer, early_stop, lr_decay)


class Tester:
    def __init__(self, device, name='TEST'):
        self.name = name
        self.device = device
        self.loss_tracker = LossTracker(name, self.device)
        
        self.progress_bar = ProgressBar(self.name)
        self.modelstep = ModelStep()
        # Decorate with progress bar
        self.progress_step = self.progress_bar(self.modelstep)

    def epoch_behavior(self):
        epoch_loss = self.loss_tracker.get_epoch_mean()

    def __call__(self, dataloader, model):
        model.eval()

        for idx, loss in self.progress_step(self.modelstep, dataloader, model, self.device, self.loss_tracker):
            yield idx, loss
        self.epoch_behavior()


class EarlyStopper:
    def __init__(self, min_epoch=30, tolerance=10, verbose=True, *args, **kwargs):

        self.min_epoch = min_epoch
        self.tolerance = tolerance

        self.early_stop_counter = 0
        self.stop_flag = False

        self.verbose = verbose
        self.best_valid_loss = torch.inf

    def __call__(self, epoch, val_loss, optimizer, lr_decay, lr_decayer):
        if epoch < self.min_epoch:
            return

        accumulate = False
        process = False

        if val_loss >= self.best_valid_loss:
            accumulate = True
        else:
            self.best_valid_loss = val_loss

        if accumulate:
            self.early_stop_counter += 1
            if self.verbose:
                print(f"\033[32mEarly Stopping reporting: {self.early_stop_counter} out of {self.tolerance}\033[0m")
            
        if self.early_stop_counter > self.tolerance:
            if lr_decay:
                process = True
            else:
                self.stop_flag = True

        if process:
            lr_decayer(optimizer)
            self.early_stop_counter = 0

        if lr_decayer.stop_flag:
            self.stop_flag = True


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
