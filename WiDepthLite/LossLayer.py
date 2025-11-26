import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

# Figure Control
class PlotSettings:
    def __init__(self, figsize=(20, 10)):
        mpl.rcParams['figure.figsize'] = figsize
        mpl.rcParams["figure.titlesize"] = 35
        mpl.rcParams['lines.markersize'] = 10
        mpl.rcParams['axes.titlesize'] = 26
        mpl.rcParams['axes.labelsize'] = 26
        mpl.rcParams['xtick.labelsize'] = 20
        mpl.rcParams['ytick.labelsize'] = 20

    def __call__(self, title=None, ax_num=1):
        _ = plt.figure()
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(title)
        fig, axes = self.gen_axes(fig, ax_num)
        
        return fig, axes

    def gen_axes(self, fig, ax_num):
        if ax_num > 3:
            axes = fig.subplots(2, np.ceil(ax_num / 2).astype(int))
            axes = axes.flatten()
        elif ax_num > 1 and ax_num <= 3:
            axes = fig.subplots(1, ax_num)
            axes = axes.flatten()
        else:
            axes = fig.get_axes()

        return fig, axes


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
        self.epoch_log = {}

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

            # Keep track of epoch loss
            if k not in self.epoch_log:
                self.epoch_log[k] = []
            self.epoch_log[k].append(losses.mean().item())

            out[k] = losses.mean().item()  # CPU 上标量
        
        self.reset()
        return out

    def reset(self):
        self.buffer = {}


# Buffers predictions on GPU
class PredBuffer:
    def __init__(self, device='cuda'):
        self.device = device
        self.buffer = {}
        self.epoch_log = {}

    def add(self, loss_dict):
        for k, v in loss_dict.items():
            if k not in self.buffer:
                self.buffer[k] = []
            # v 是 GPU tensor，不 detach 不拉回 CPU
            self.buffer[k].append(v.detach())

    def epoch_set(self):
        self.epoch_log = {}
        # 只在这里才转成 CPU numpy，减少 GPU->CPU copy
        for k, v_list in self.buffer.items():
            preds = torch.cat(v_list)  # GPU 上堆叠
            self.epoch_log[k] = preds.squeeze()

        self.buffer = {}
        return self.epoch_log

    def reset(self):
        self.buffer = {}
        self.epoch_log = {}


class IndexGenerator:
    def __init__(self, fixed_inds=True):
        self.select_ind = None
        self.select_tag = None
        self.fixed_inds = fixed_inds

    def __call__(self, inds: list=None, tags: list=None, select_ind: list=None, ind_range=8, select_num=8):
        if select_ind:
            self.select_ind = np.array(select_ind)
        else:
            inds = np.random.choice(np.arange(len(inds)), select_num, replace=False).astype(int)
            inds = np.sort(inds)

        if not np.any(self.select_ind):
            self.select_ind = inds

        if not self.fixed_inds:
            self.select_ind = inds

        if tags:
            self.select_tag = tags[self.select_ind]

        return self.select_ind, self.select_tag

class LossPlotter:
    def __init__(self):
        self.plot_settings = PlotSettings()

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

    def plot_track(self, losses:dict, line_color: str='blue', line_label='', fig=None, title=None, lr_change_log: dict=None):

        if not fig:
            fig, axes = self.plot_settings(title, len(losses))

            if lr_change_log:
                stage_color = self.colors(list(lr_change_log.values()))
                for ax in axes:
                    for j, lr in enumerate(lr_change_log.keys()):
                        ax.axvline(lr_change_log[lr],
                            linestyle='--',
                            color=stage_color[j],
                            label=f'lr={lr}')

                        ax.set_xlabel('#Epoch')
                        ax.set_ylabel('Loss')
                        ax.grid()

        else:
            axes = fig.get_axes()

        for ax, loss in zip(axes, list(losses.keys())):
            ax.plot(losses[loss], line_color, label=line_label)
            ax.set_title(loss, fontweight="bold")
            ax.legend()

        return fig

    def plot_cdf(self, losses:dict, title=None):

        fig, axes = self.plot_settings(title, len(losses))

        for ax, loss in zip(axes, list(losses.keys())):
            hist, bin_edges = np.histogram(losses[loss])
            width = (bin_edges[1] - bin_edges[0]) * 0.8
            cdf = np.cumsum(hist / sum(hist))

            ax.bar(bin_edges[1:], hist / max(hist), width=width, color='blue')
            ax.plot(bin_edges[1:], cdf, '-*', color='orange')
            ax.set_title(loss, fontweight="bold")
            ax.set_ylim([0, 1])
            ax.set_xlabel('Per-sample Loss')
            ax.set_ylabel('Frequency')
            ax.grid()

        plt.show()
        return fig


class PredPlotter:
    def __init__(self):
        self.plot_settings = PlotSettings()

    def plot_images(self, preds, inds, tags=None, title=None):
        rows = len(preds)
        if 'IND' in preds:
            rows -= 1
        if 'TAG' in preds:
            rows -= 1
        fig, axes = self.plot_settings(title)
        subfigs = fig.subfigures(nrows=rows, ncols=1)

        for subfig, pred in zip(subfigs, preds.keys()):
            subfig.suptitle(pred)
            axes = subfig.subplots(nrows=1, ncols=len(inds))
            for i, (ax, ind) in enumerate(zip(axes, inds)):
                img = preds[pred][ind]
                img = ax.imshow(img, vmin=0, vmax=1)
                ax.axis('off')
                if tags is None:
                    subtitle = str(ind)
                else:
                    f"{'-'.join(map(str, map(int, tags[i])))}"
                ax.set_title(subtitle)

            subfig.colorbar(img, ax=axes, shrink=0.8)

        plt.show()

        return fig


# Loss Tracker on GPU
# Plotting and saving involves GPU-to-CPU transition
class LossTracker:
    def __init__(self, name, device):

        self.name = name
        self.loss_buffer = LossBuffer(device)
        self.pred_buffer = PredBuffer(device)

        self.loss_plotter = LossPlotter()
        self.pred_plotter = PredPlotter()

        self.current_epoch = 0
        self.current_lr = 0
        self.lr_change_log: dict = {}

        self.index_generator = IndexGenerator()

    def to_cpu(self, loss: dict):
        out = {key: torch.tensor(value, device='cpu').numpy() for key, value in loss.items()}
        return out

    def log_loss(self, losses:dict):
        self.loss_buffer.add(losses)

    def log_preds(self, preds:dict):
        self.pred_buffer.add(preds)

    def log_lr_change(self, optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break

        if lr != self.current_lr:
            self.lr_change_log[lr] = self.current_epoch
            self.current_lr = lr
        return lr

    def get_epoch_mean(self, mode=''):
        if mode == 'train':
            self.current_epoch += 1
        ret = self.loss_buffer.epoch_set()
        preds = self.pred_buffer.epoch_set()
        return ret

    def plot_loss_track(self, line_color, line_label, fig=None, show=False):
        # prepare for iterative use
        cpu_losses = self.to_cpu(self.loss_buffer.epoch_log)
        title = f"{self.name}_LOSS@ep{self.current_epoch}"
        filename = f"{title}.jpg"
        out_fig = self.loss_plotter.plot_track(cpu_losses, line_color, line_label, fig, title, self.lr_change_log)
        if show:
            plt.show()
        return filename, out_fig

    def plot_cdf(self):
        cpu_losses = self.to_cpu(self.loss_buffer)

        title = f"{self.name}_{mode.upper()}@ep{self.current_epoch}"
        filename = f"{title}.jpg"
        out_fig = self.loss_plotter.plot_cdf(test_losses, title)
        return filename, out_fig

    def plot_preds(self, pred_terms='all', show=True):
        preds = self.pred_buffer.epoch_log
        if pred_terms == 'all':
            pred_terms = list(preds.keys())
        pred_terms.extend(['IND', 'TAG'])
        preds = {key: value for key, value in preds.items() if key in pred_terms}
        preds = self.to_cpu(preds)

        title = f"{self.name}_PREDS@{self.current_epoch}"
        filename = f"{title}.jpg"

        inds, tags = self.index_generator(preds['IND'], preds.get('TAG', None))
        out_fig = self.pred_plotter.plot_images(preds, inds, tags, title)

        if show:
            plt.show()
        return filename, out_fig

    def save(self, target, mode, save_path):
        targets = {
            'loss': self.loss_buffer,
            'preds': self.pred_buffer}

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # self.loss_buffer['train'].epoch_log = {'loss1': [...]}
 
        for key, value in self.targets[target][mode].epoch_log.items():
            print(f"Saving {target}: {key} of len {len(value)}...")
            np.save(f"{save_path}_{target}_{mode}_{key}.npy", value.cpu().numpy())
        
        print('All saved!')