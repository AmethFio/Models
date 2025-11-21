import torch
from LossLayer import LossTracker
from functools import wraps
from tqdm import tqdm

def with_progress_bar(bar_format=None):
    """
    Decorator for wrapping a generator function with tqdm progress bar.
    The decorated function must yield (idx, loss).
    """
    if bar_format is None:
        bar_format = '{desc}{percentage:3.0f}%|{bar}|[{elapsed}<{remaining}{postfix}]'

    def decorator(func):
        @wraps(func)
        def wrapped(self, dataloader, *args, **kwargs):
            total = len(dataloader)
            print('')
            with tqdm(total=total, dynamic_ncols=True, bar_format=bar_format) as progress_bar:
                for idx, loss in func(self, dataloader, *args, **kwargs):
                    progress_bar.set_postfix({
                        'batch': f"{idx}/{total}",
                        'loss': f"{loss:.4f}"
                    })
                    progress_bar.n = idx
                    progress_bar.refresh()
                    yield idx, loss
        return wrapped
    return decorator


class ModelStep:
    def __init__(self, train_module, eval_module):
        self.train_module = train_module
        self.eval_module = eval_module

    def data_preprocess(self, data, device, preprocess=None):
        if preprocess:
            data = preprocess(data)

        data = {key: value.to(torch.float32).to(device) for key, value in data.items()}

        if 'tag' in data:
            data['tag'] = data['tag'].to(torch.int32).to(device)
            
        return data)

    def __call__(self, dataloader, model, optimizer, preprocess):
        if self.train_module = 'all':
            self.train_module = model.modules.keys()

        if self.eval_module = 'all':
            self.eval_module = model.modules.keys()

        for train_m in self.train_module:
            model.get(train_m).train()

        for eval_m in self.eval_module:
            model.get(eval_m).eval()

        for idx, data in enumerate(dataloader, 1):
            # Prepare data
            _data_ = self.data_preprocess(data, preprocess)
            # Calculate model output
            loss, preds = model(_data_)
            # Log loss to buffer
            self.loss_tracker.log_loss(loss)

            yield idx, loss['LOSS'].item()


class Trainer:
    def __init__(self, name='TRAIN', train_module='all', eval_module=[]):
        self.name = name
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        self.loss_tracker = LossTracker(name, self.device)
        self.preprocess = DataPreprocess()
        self.step = ModelStep(train_module, eval_module)
        self.scaler = GradScaler()
        self.traim_module = train_module,
        self.eval_module = eval_module

    def epoch_bahavior(self, optimizer):
        if len(self.loss_tracker.lr_change_log) == 0:
            self.lr_log(optimizer)

        epoch_loss = self.loss_tracker.get_epoch_mean()
        loss = epoch_loss['LOSS']
        # Update
        if torch.isnan(loss):
                print(f"Phase {self.name}: NaN value in loss, skipping update.")
            elif not torch.isfinite(loss):
                print(f"Phase {self.name}: Infinite value in loss, skipping update.")
                
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

    @with_progress_bar()
    def __call__(self, dataloader, model, optimizer, preprocess):
        for idx, loss in self.step(dataloader, model, optimizer, preprocess):
            yield idx, loss
        self.epoch_behavior(optimizer)


class Validator:
    def __init__(self, name='VALID'):
        self.name = name
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        self.loss_tracker = LossTracker(name, self.device)
        self.preprocess = DataPreprocess()
        self.step = ModelStep(train_module=[], eval_module='all')

        self.best_val_loss = torch.inf
        self.best_val_epoch = 0

        self.early_stopper = EarlyStopper()

    def epoch_bahavior(self, loss, optimizer, early_stop, lr_decay):
        self.loss_tracker.current_epoch += 1
        epoch_loss = self.loss_tracker.get_epoch_mean()
        loss = epoch_loss['LOSS']
        # Log best loss
        if 0 < loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_epoch = loss_tracker.current_epoch

        if early_stop:
            lr, lr_change = self.early_stopper(loss, lr_decay, optimizer)
            if lr_change:
                self.loss_tracker.lr_change_log[self.loss_tracker.current_epoch] = lr
            

    @with_progress_bar()
    def __call__(self, dataloader, model, optimizer, preprocess, early_stop=False):
        # Wrap the model step with other functions
        for idx in self.step(dataloader, model, optimizer, preprocess):
            yield idx, loss
        self.epoch_behavior(loss, optimizer, early_stop, lr_decay)


class Tester:
    def __init__(self, name='TEST'):
        self.name = name
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        self.loss_tracker = LossTracker(name, self.device)
        self.preprocess = DataPreprocess()
        self.step = ModelStep(train_module=[], eval_module='all')

    def epoch_behavior(self):
        for key, value in self.loss_tracker.loss_buffer.buffer.items():
            self.loss_tracker.loss_buffer.buffer[key] = value.squeeze()

    @with_progress_bar()
    def __call__(self, dataloader, model, preprocess):
        # Wrap the model step with other functions
        for idx in self.step(dataloader, model, optimizer, preprocess):
            yield idx, loss
        self.epoch_behavior()


class EarlyStopper:
    def __init__(self, min_epoch=100, tolerance=10, verbose=True, *args, **kwargs):

        self.min_epoch = min_epoch
        self.tolerance = tolerance

        self.early_stop_counter = 0
        self.stop_flag = False

        self.verbose = verbose
        self.current_epoch = 0
        self.best_valid_loss = torch.inf

        self.lr_decayer = LrDecayer()

    def __call__(self, val_loss, lr_decay, optimizer):
        self.current_epoch += 1
        lr_change = False
        
        # Early stopping flag
        if val_loss >= self.best_valid_loss:
            self.early_stop_counter += 1
            if self.verbose:
                print(f"\033[32mEarly Stopping reporting: {self.early_stop_counter} out of {self.tolerance}\033[0m")
            
            if self.early_stop_counter >= self.early_stop_max:
                if lr_decay:
                    lr_change = True
                    lr = self.lr_decayer(optimizer)
                    if self.lr_decayer.stop_flag:
                        self.stop_flag = True
                    else:
                        self.early_stop_counter = 0
                else:
                    self.stop_flag = True
        else:
            self.best_valid_loss = val_loss
            self.early_stop_counter = 0

        return lr, lr_change

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
                f"Decay rate = {self.decay_rate} ** {self.lr_decay_counter}\033[0m")
        if self.lr_decay_counter >= self.tolerance:
            self.stop_flag = True

        return param_group['lr']
