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
        elif ax_num > 1 and ax_num <= 3:
            axes = fig.subplots(1, ax_num)
        else:
            axes = flg.get_axes()
        axes = axes.flatten()
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

    def epoch_mean(self):
        # 只在这里才转成 CPU numpy，减少 GPU->CPU copy
        out = {}
        for k, v_list in self.buffer.items():
            losses = torch.stack(v_list)  # GPU 上堆叠

            # Keep track of epoch loss
            if k not in self.epoch_log:
                self.epoch_log[k] = []
            self.epoch_log[k].append(losses.mean())

            out[k] = losses.mean().item()  # CPU 上标量
        
        self.buffer = {}
        return out

    def reset(self):
        self.buffer = {}


# Buffers predictions on GPU
class PredBuffer:
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
            out[k] = losses.item()  # CPU 上标量

        self.epoch_log = out
        return out

    def reset(self):
        self.buffer = {}


class IndexGenerator:
    def __init__(self, fixed_inds=True):
        self.select_ind = None
        self.select_tag = None
        self.fixed_inds = fixed_inds

    def generate(self, tags: list=None, inds: list=None, select_ind: list=None, ind_range=8, select_num=8):
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

    def plot_train_valid_loss(self, typed_losses: dict, title=None, lr_change_log: dict=None):
        fig = None
        for loss_type, loss in typed_losses.items():
            ofig = self.plot_track(loss, loss_type, fig, title, lr_change_log)
        plt.show()
        return fig

    def plot_track(self, losses:dict, loss_type: str='train', fig=None, title=None, lr_change_log: dict=None):

        line_color = {
            'train': 'blue',
            'valid': 'orange',
            'valid_target': 'green'
        }

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
            axes = axes.flatten()

        for ax, loss in zip(axes, list(losses.keys())):
            ax.plot(losses[loss], line_color[loss_type], label=loss_type)
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

        fig, axes = self.plot_settings(title)
        subfigs = fig.subfigures(nrows=len(preds), ncols=1)

        for subfig, pred in zip(subfigs, preds.keys()):
            subfig.suptitle(pred)
            axes = subfig.subplots(nrows=1, ncols=len(tags))
            for ax, ind, tag in zip(axes, inds, tags):
                img = ax.imshow(preds[pred][ind], vmin=0, vmax=1)
                ax.axis('off')
                if tags is None:
                    ax.set_title(f"{ind}")
                else:
                    ax.set_title(f"{'-'.join(map(str, map(int, tag)))}")
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
        self.lr_change_log: dict = {}

        self.index_generator = IndexGenerator()

    def to_cpu(self, loss: dict):
        out = {key: value.cpu().numpy() for key, value in loss.items()}
        return out

    def log_loss(self, losses:dict):
        self.loss_buffer.add(losses)

    def log_preds(self, preds:dict):
        self.pred_buffer.add(preds)

    def log_lr_change(self, lr, ep):
        self.lr_change_log[lr] = ep

    def get_epoch_mean(self):
        if mode == 'train':
            self.current_epoch += 1
        ret = self.loss_buffer.epoch_mean()
        return ret

    def reset_preds(self):
        # Only keep the latest epoch preds
        self.pred_buffer.reset()

    def reset_loss(self):
        self.loss_tracker.reset()

    def plot_loss_track(self):
        # prepare for iterative use
        cpu_losses = self.to_cpu(self.loss_buffer.epoch_log)

        title = f"{self.name}_TRAIN_LOSS@ep{self.current_epoch}"
        filename = f"{title}.jpg"
        out_fig = self.loss_plotter.plot_train_valid_loss(typed_losses, title, self.lr_change_log)
        return filename, out_fig

    def plot_cdf(self):
        cpu_losses = self.to_cpu(self.loss_buffer)

        title = f"{self.name}_{mode.upper()}@ep{self.current_epoch}"
        filename = f"{title}.jpg"
        out_fig = self.loss_plotter.plot_cdf(test_losses, title)
        return filename, out_fig

    def plot_preds(self, pred_terms='all', mode='train'):
        if pred_terms == 'all':
            pred_terms = list(self.pred_buffer.keys())
        plot_preds = {key: value.epoch_log for key, value in self.pred_buffer.items() if key in pred_terms}
        plot_preds = self.to_cpu(plot_preds)

        title = f"{self.name}_PREDS_{mode.upper()}@{self.current_epoch}"
        filename = f"{title}.jpg"

        inds, tags = self.index_generator(plot_preds['TAG'], plot_preds['IND'])
        out_fig = self.pred_plotter.plot_images(plot_preds, inds, tags, title)
        return filename, out_fig

    def lr_decay(self, rate=0.5):
        # Really needed?
        return

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