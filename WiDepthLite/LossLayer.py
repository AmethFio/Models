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
            loss = torch.stack(v_list)  # GPU 上堆叠

            # Keep track of epoch loss
            if k not in self.epoch_log:
                self.epoch_log[k] = []
            self.epoch_log[k].append(loss.mean().item())

            out[k] = loss.mean().item()  # For show
        
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

        if tags is not None:
            self.select_tag = tags[self.select_ind]

        return self.select_ind, self.select_tag

class LossPlotter:
    def __init__(self):
        self.plot_settings = PlotSettings()

    def plot_lr_change(self, axes, lr_change_log):

        def colors(lrs):
            lrs = -np.log(lrs)
            norm = plt.Normalize(np.min(lrs), np.max(lrs))
            map_vir = cm.get_cmap(name='viridis')
            c = map_vir(norm(lrs))
            return c

        stage_color = colors(list(lr_change_log.keys()))

        for ax in axes:
            for c, (lr, ep) in zip(stage_color, lr_change_log.items()):
                ax.axvline(ep,
                            linestyle='--',
                            color=c,
                            label=f'lr={lr:.2e}')

        return axes

    def plot_track(self, epoch_axis: list, losses: dict, line_color: str='blue', line_label='', fig=None, title=None, lr_change_log: dict=None):

        if not fig:
            fig, axes = self.plot_settings(title, len(losses))
            for ax in axes:
                ax.set_xlabel('#Epoch')
                ax.set_ylabel('Loss')
                ax.grid()
            if lr_change_log:
                axes = self.plot_lr_change(axes, lr_change_log)

        else:
            axes = fig.get_axes()

        for ax, loss in zip(axes, list(losses.keys())):
            ax.plot(epoch_axis, losses[loss], line_color, label=line_label)
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
        self.centercolor = ['blue', 'orange', 'green']

    def plot_images(self, preds, inds, tags=None, title=None):
        rows = len(preds) - 2

        fig, axes = self.plot_settings(title)
        subfigs = fig.subfigures(nrows=rows, ncols=1)
        if rows == 1:
            subfigs = [subfigs]

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
                    subtitle = f"{'-'.join(map(str, map(int, tags[i])))}"
                ax.set_title(subtitle)

            subfig.colorbar(img, ax=axes, shrink=0.8)

        plt.show()

        return fig

    def plot_center(self, preds, inds, tags=None, title=None):
        fig, axes = self.plot_settings(title)
        axes = fig.subplots(nrows=2, ncols=4)
        axes = axes.flatten()

        for j, (ax, ind) in enumerate(zip(axes, inds)):
            ax.set_xlim([0, 226])
            ax.set_ylim([0, 128])
            ax.set_title(f"{'-'.join(map(str, map(int, tags[i])))}")

            for ctr in preds.values():
                x, y = ctr[j]
                x = int(x * 226)
                y = int(y * 128)
                ax.scatter(x, y, c=self.centercolor[j % len(self.centercolor)], marker=(5, 1), alpha=0.5, linewidths=5, label=ctr)
             
            ax.axis('off')
            ax.add_patch(Rectangle((0, 0), 226, 128, facecolor="#F0FFFF",
                                        transform=ax.transAxes, zorder=-1))

        axes[0].legend()
        plt.show()

        return fig


# Loss Tracker on GPU
# Plotting and saving involves GPU-to-CPU transition
class LossTracker:
    def __init__(self, name, device, activate_interval=1):

        self.name = name
        self.loss_buffer = LossBuffer(device)
        self.pred_buffer = PredBuffer(device)
        self.activate_interval = activate_interval
        self.epoch_axis = []

        self.loss_plotter = LossPlotter()
        self.pred_plotter = PredPlotter()

        self.current_epoch = 0
        self.current_lr = 0
        self.lr_change_log: dict = {}

        self.index_generator = IndexGenerator()

    def to_cpu(self, loss: dict):
        out = {key: torch.tensor(value).cpu().numpy() if value is not None else None for key, value in loss.items()}
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
            self.epoch_axis.append(self.current_epoch)
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

    def plot_cdf(self, show=True):
        cpu_losses = self.to_cpu(self.loss_buffer.buffer)

        title = f"{self.name}_CDF@ep{self.current_epoch}"
        filename = f"{title}.jpg"
        out_fig = self.loss_plotter.plot_cdf(cpu_losses, title)

        if show:
            plt.show()
        return filename, out_fig

    def plot_preds(self, pred_terms='all', show=True):
        preds = self.pred_buffer.epoch_log
        if pred_terms == 'all':
            pred_terms = list(preds.keys())
        preds = {key: value for key, value in preds.items() if key in pred_terms}
        preds['IND'] = self.pred_buffer.epoch_log.get('IND', None)
        preds['TAG'] = self.pred_buffer.epoch_log.get('TAG', None)
        preds = self.to_cpu(preds)

        title = f"{self.name}_PREDS@ep{self.current_epoch}"
        filename = f"{title}.jpg"

        inds, tags = self.index_generator(preds['IND'], preds['TAG'])
        out_fig = self.pred_plotter.plot_images(preds, inds, tags, title)

        if show:
            plt.show()
        return filename, out_fig

    def plot_ctr(self, pred_terms='all', show=True):
        preds = self.pred_buffer.epoch_log
        if pred_terms == 'all':
            pred_terms = ['GT_CTR', 'T_CTR', 'S_CTR']

        preds = {key: value for key, value in preds.items() if key in pred_terms}
        preds['IND'] = self.pred_buffer.epoch_log.get('IND', None)
        preds['TAG'] = self.pred_buffer.epoch_log.get('TAG', None)
        preds = self.to_cpu(preds)

        title = f"{self.name}_CENTER@ep{self.current_epoch}"
        filename = f"{title}.jpg"
        
        inds, tags = self.index_generator(preds['IND'], preds['TAG'])
        out_fig = self.pred_plotter.plot_center(preds, inds, tags, title)

        if show:
            plt.show()
        return filename, out_fig

    def save(self, target='preds', save_path=None):
        targets = {
            'loss': self.loss_buffer,
            'preds': self.pred_buffer}

        if not save_path:
            save_path = f"../saved/{self.name}/"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # self.loss_buffer.epoch_log = {'loss1': [...]}
 
        for key, value in targets[target].epoch_log.items():
            print(f"Saving {target}: {key} of len {len(value)}...")
            np.save(f"{save_path}{target}_{key}.npy", value.cpu().numpy())
        
        print('All saved!')