import torch
import torch.nn as nn
import os
import numpy as np
from misc import plot_settings

# Examplar Loss Layer
class MyLossModule(nn.Module):
    def __init__(self, w_mse=1.0, w_l1=1.0):
        super().__init__()
        self.w_mse = w_mse
        self.w_l1 = w_l1
        self.mse = nn.MSELoss(reduction='mean')
        self.l1 = nn.L1Loss(reduction='mean')

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        total_loss = self.w_mse * mse_loss + self.w_l1 * l1_loss
        # 返回所有 loss，避免 CPU 侧拆分
        return {
            "total": total_loss,
            "mse": mse_loss,
            "l1": l1_loss
        }


# Buffers batch loss on GPU
class LossBuffer:
    def __init__(self, device='cuda'):
        self.device = device
        self.buffer = {}  # {"mse": [tensor], "l1": [tensor], ...}
        self.whole_log = {}

    def add(self, loss_dict):
        for k, v in loss_dict.items():
            if k not in self.buffer:
                self.buffer[k] = []
            # v 是 GPU tensor，不 detach 不拉回 CPU
            self.buffer[k].append(v.detach())

    def epoch_mean(self):
        # 只在这里才转成 CPU numpy，减少 GPU->CPU copy
        out = {}
        for k, v_list in self.buffer.items():
            losses = torch.stack(v_list)  # GPU 上堆叠

            # Keep track of epoch loss
            if k not in self.whole_log:
                self.whole_log[k] = []
            self.whole_log[k].append(losses.mean())

            out[k] = losses.mean().item()  # CPU 上标量

        return out

    def reset(self):
        self.buffer = {}


# Buffers predictions on GPU
class PredBuffer:
    def __init__(self, device='cuda'):
        self.device = device
        self.buffer = {}  # {"mse": [tensor], "l1": [tensor], ...}
        self.whole_log = {}

    def add(self, loss_dict):
        for k, v in loss_dict.items():
            if k not in self.buffer:
                self.buffer[k] = []
            # v 是 GPU tensor，不 detach 不拉回 CPU
            self.buffer[k].append(v.detach())

    def epoch_set(self):
        # 只在这里才转成 CPU numpy，减少 GPU->CPU copy
        out = {}
        for k, v_list in self.buffer.items():
            losses = torch.stack(v_list)  # GPU 上堆叠
            out[k] = losses.item()  # CPU 上标量

        self.whole_log = {}
        return out

    def reset(self):
        self.buffer = {}


class IndexGenerator:
    def __init__(self):
        self.select_ind = None

    def generate(self, select_ind: list=None, ind_range=8, select_num=8):
        if select_ind:
            self.select_ind = np.array(select_ind)
        else:
            if not np.any(self.select_ind):
                inds = np.random.choice(np.arange(ind_range), select_num, replace=False).astype(int)
                inds = np.sort(inds)
                select_ind = inds
                self.select_ind = inds
            else:
                return self.select_ind

        return select_ind


class LossPlotter:
    def __init__(self):
        self.plot_settings = plot_settings

    @staticmethod
    def colors(arrays):
        """
        Color solution for plotting loss
        :param arrays: array of learning rates
        :return: variation of colors
        """
        arr = -np.log(arrays)
        norm = plt.Normalize(arr.min(), arr.max())
        map_vir = cm.get_cmap(name='viridis')
        c = map_vir(norm(arr))
        return c

    def plot(self, losses:dict, title=None, lr_change_epoch: dict=None):
        fig = self.plot_settings()
        fig.suptitle(title)
        line_color = ['b', 'orange', 'green']

        if len(plot_terms) == 1:
            axes = [plt.gca()]
        elif len(plot_terms) > 3:
            axes = fig.subplots(2, np.ceil(len(plot_terms)/2).astype(int))
        else:
            axes = fig.subplots(1, len(plot_terms))
            
        axes = axes.flatten()

        for i, loss in enumerate(losses.keys()):
            if lr_change_epoch:
                stage_color = self.colors(list(lr_change_epoch.values()))
                for j, lr in enumerate(lr_change_epoch.keys()):
                    axes[i].axvline(lr_change_epoch[lr],
                                    linestyle='--',
                                    color=stage_color[j],
                                    label=f'lr={lr}')

            axes[i].plot(losses[loss], )




# Loss Tracker on CPU
class LossTracker:
    def __init__(self, device, loss_terms, pred_terms):

        self.loss_buffer = {
            'train': LossBuffer(device),
            'valid': LossBuffer(device),
            'test': LossBuffer(device)
        }
        self.pred_buffer = {
            'train': PredBuffer(device),
            'valid': PredBuffer(device),
            'test': PredBuffer(device)
        }

        self.current_epoch = 0
        self.lr_change_epoch = []

        self.index_generator = IndexGenerator()

    def log_loss(self, mode, losses:dict):
        if mode == 'train':
            self.current_epoch += 1

        self.loss_buffer[mode].add(losses)


    def log_preds(self, mode, preds:dict):

        # Only keep the latest epoch preds
        self.pred_buffer[mode].reset()
        self.pred_buffer[mode].add(preds)

    def lr_decay(self, rate=0.5):
        # Really needed?
        return

    def save(self, target, mode, save_path):
        targets = {
            'loss': self.loss_buffer,
            'preds': self.pred_buffer}

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # self.loss_buffer['train'].whole_log = {'loss1': [...]}
 
        for key, value in self.targets[target][mode].whole_log.items():
            print(f"Saving {target}: {key} of len {len(value)}...")
            np.save(f"{save_path}_{target}_{mode}_{key}.npy", value.cpu().numpy())
        
        print('All saved!')