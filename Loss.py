import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
import os
from misc import plot_settings
import torch.nn.functional as F
# from sklearn.manifold import TSNE

"""
These definitions are not for loss functions.\n
These are MyLoss classes that can store and plot losses.\n
"""

class LossUnit:
    def __init__(self, name):
        self.name = name
        self.lr = []
        self.log = {'train': [],
                    'valid': [],
                    'valid2': [],
                    'test': []}
        self.tsne = None
        self.optimizer = None
        
    def __call__(self, mode, loss_value): 
        self.log[mode].append(loss_value)
        
    def set_optimizer(self, optimizer, lr, params):
        self.optimizer = optimizer(params, lr)
        if not self.lr or self.lr[-1] != lr:
            self.lr.append(lr)
            
    def set_lr(self, lr):
        if self.optimizer:
            self.lr.append(lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def lr_decay(self, rate=0.5):
        if self.optimizer:
            self.lr.append(self.lr[-1] * 0.5)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr[-1]
        
    def reset(self, *modes):
        for mode in modes:
            self.log[mode] = []
        
    def Tsne(self):
        self.tsne = TSNE(n_components=2, random_state=33).fit_transform(
            np.array(self.log).reshape(len(self.log), -1))


class MyLossLog:
    def __init__(self, name, loss_terms, pred_terms, dataset: str = 'TRAIN'):
        self.name = name
        self.dataset = dataset
        self.loss = {term: LossUnit(term) for term in loss_terms}
        self.preds = {term: [] for term in pred_terms}
        self.in_training = False

        self.epochs = []
        self.current_epoch = 0
        self.loss_terms = loss_terms
        self.pred_terms = pred_terms
        self.select_inds = None
        self.select_num = 8
        self.ind_range = 0
        self.__plot_settings__ = plot_settings

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

    def __call__(self, mode, losses:dict):
        if mode == 'train':
            self.current_epoch += 1
            if not self.in_training:
                self.epochs.append(self.current_epoch)
                self.in_training = True
        if mode in ('train', 'valid', 'valid2', 'test'):
            for key in losses.keys():
                self.loss[key](mode, np.squeeze(losses[key]))
        elif mode == 'pred':
            for key in losses.keys():
                # PRED_TERMS
                lss = losses[key].cpu().detach().numpy().squeeze()
                # Unpack batches
                try:
                    for ls in lss:
                        self.preds[key].append(ls)
                except Exception:
                    self.preds[key].append(lss)
                
    def reset(self, *modes, dataset: str = 'TRAIN'):
        for mode in modes:
            if mode in ('train', 'valid', 'test'):
                for loss in self.loss.values():
                    loss.reset(mode)
            else:
                self.preds = {term: [] for term in self.pred_terms}
        self.dataset = dataset.upper()

    def decay(self, rate=0.5):
        self.epochs.append(self.current_epoch)
        for loss in self.loss.values():
            loss.lr_decay(rate)

    def save(self, save_term: str = 'preds', save_path=None):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_term == 'preds':
            for key, value in self.preds.items():
                #if 'GT' in key:
                #    continue
                print(f"Saving {save_term}: {key}...")
                np.save(f"{save_path}{self.name}_{save_term}_{key}.npy", value)
        else:
            for key, value in self.loss.items():
                print(f"Saving {save_term}: {key}...")
                np.save(f"{save_path}{self.name}_{save_term}_{key}.npy", value.log[save_term])
        print('All saved!')

    def generate_indices(self, select_ind: list = None, select_num=8):
        if select_ind:
            self.select_inds = np.array(select_ind)
        else:
            if not np.any(self.select_inds) or self.ind_range != len(self.preds['TAG']):
                self.ind_range = len(self.preds['TAG'])
                inds = np.random.choice(np.arange(self.ind_range), select_num, replace=False).astype(int)
                inds = np.sort(inds)
                self.select_inds = inds
                self.select_num = select_num

    def plot_train(self, title=None, plot_terms='all', double_y=False):
        line_color = ['b', 'orange', 'green']
        if title:
            title = f"{title} @ep{self.current_epoch}"
        else:
            title = f"{self.name} Training Status @ep{self.current_epoch}"

        if plot_terms == 'all':
            plot_terms = self.loss_terms

        fig = self.__plot_settings__()
        fig.suptitle(title)

        if len(plot_terms) == 1:
            axes = [plt.gca()]
        elif len(plot_terms) > 3:
            axes = fig.subplots(2, np.ceil(len(plot_terms)/2).astype(int))
            axes = axes.flatten()
        else:
            axes = fig.subplots(1, len(plot_terms))
            axes = axes.flatten()

        for i, loss in enumerate(plot_terms):
            if self.loss[loss].lr:
                stage_color = self.colors(self.loss[loss].lr)
                for j, lr in enumerate(self.loss[loss].lr):
                    axes[i].axvline(self.epochs[j],
                                    linestyle='--',
                                    color=stage_color[j],
                                    label=f'lr={lr}')

            axes[i].plot(
                         self.loss[loss].log['valid'],
                         line_color[1], label='Valid')
            axes[i].plot(
                self.loss[loss].log['valid2'],
                         line_color[2], label='Valid2'
            )
            if double_y:
                ax_r = axes[i].twinx()
            else:
                ax_r = axes[i]
            ax_r.plot(
                      self.loss[loss].log['train'],
                      line_color[0], label='Train')
            axes[i].set_title(loss, fontweight="bold")
            axes[i].set_xlabel('#Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].grid()
            axes[i].legend()
        plt.show()
        filename = f"{self.name}_TRAIN@ep{self.current_epoch}.jpg"
        return {filename: fig}

    def plot_test(self, title=None, plot_terms='all'):
        if title:
            title = f"{title} @ep{self.current_epoch}"
        else:
            title = f"{self.name} Test Loss on {self.dataset} @ep{self.current_epoch}"

        if plot_terms == 'all':
            plot_terms = self.loss_terms

        fig = self.__plot_settings__()
        fig.suptitle(title)
        plt.yscale('log', base=2)
        # plt.ylim([2 ** -18, 2])
        plt.axhline(1, linestyle='--', color='lightgreen', label='1.0')
        for i, loss in enumerate(plot_terms):
            plt.boxplot(self.loss[loss].log['test'], labels=[loss], positions=[i+1], vert=True, showmeans=True,
                        patch_artist=True,
                        boxprops={'facecolor': 'lightblue'})

        plt.show()
        filename = f"{self.name}_TEST_{self.dataset}SET@ep{self.current_epoch}.jpg"
        return {filename: fig}

    def plot_test_cdf(self, title=None, plot_terms='all'):
        if title:
            title = f"{title} @ep{self.current_epoch}"
        else:
            title = f"{self.name} Test PDF-CDF on {self.dataset} @ep{self.current_epoch}"

        if plot_terms == 'all':
            plot_terms = self.loss_terms

        fig = self.__plot_settings__()
        fig.suptitle(title)
        if len(plot_terms) == 1:
            axes = [plt.gca()]
        elif len(plot_terms) > 3:
            axes = fig.subplots(2, np.ceil(len(plot_terms) / 2).astype(int))
            axes = axes.flatten()
        else:
            axes = fig.subplots(1, len(plot_terms))
            axes = axes.flatten()

        for i, loss in enumerate(plot_terms):
            hist, bin_edges = np.histogram(self.loss[loss].log['test'])
            width = (bin_edges[1] - bin_edges[0]) * 0.8
            cdf = np.cumsum(hist / sum(hist))

            axes[i].bar(bin_edges[1:], hist / max(hist), width=width, color='blue')
            axes[i].plot(bin_edges[1:], cdf, '-*', color='orange')
            axes[i].set_ylim([0, 1])
            axes[i].set_title(loss, fontweight="bold")
            axes[i].set_xlabel('Per-sample Loss')
            axes[i].set_ylabel('Frequency')
            axes[i].grid()

        plt.show()
        filename = f"{self.name}_PDF_{self.dataset}SET@ep{self.current_epoch}.jpg"
        return {filename: fig}

    def plot_predict(self, plot_terms, title=None):
        if title:
            title = f"{title} on {self.dataset} @ep{self.current_epoch}"
            filename = f"{title}_{self.dataset}SET@ep{self.current_epoch}.jpg"
        else:
            title = f"{self.name} Image Predicts on {self.dataset} @ep{self.current_epoch}"
            filename = f"{self.name}_PRED_{self.dataset}SET@ep{self.current_epoch}.jpg"
            
        samples = np.array(self.preds['TAG']).astype(int)[self.select_inds]

        fig = self.__plot_settings__()
        fig.suptitle(title)
        subfigs = fig.subfigures(nrows=len(plot_terms), ncols=1)

        for i, item in enumerate(plot_terms):
            subfigs[i].suptitle(item)
            axes = subfigs[i].subplots(nrows=1, ncols=self.select_num)
            for j in range(len(axes)):
                img = axes[j].imshow(self.preds[item][self.select_inds[j]], vmin=0, vmax=1)
                axes[j].axis('off')
                axes[j].set_title(f"{'-'.join(map(str, map(int, samples[j])))}", fontweight="bold")
            subfigs[i].colorbar(img, ax=axes, shrink=0.8)
        plt.show()
        
        return {filename: fig}

    def plot_latent(self, plot_terms, title=None, ylim: tuple = (-1, 1)):
        if title:
            title = f"{title} on {self.dataset} @ep{self.current_epoch}"
        else:
            title = f"{self.name} Latent Predicts on {self.dataset} @ep{self.current_epoch}"
        samples = np.array(self.preds['TAG']).astype(int)[self.select_inds]
        colors = ('blue', 'orange')

        fig = self.__plot_settings__()
        fig.suptitle(title)
        axes = fig.subplots(nrows=2, ncols=np.ceil(self.select_num / 2).astype(int))
        axes = axes.flatten()
        for j in range(self.select_num):
            for no, item in enumerate(plot_terms):
                axes[j].bar(range(len(self.preds[item][self.select_inds[j]])),
                            self.preds[item][self.select_inds[j]],
                            width=1, fc=colors[no], alpha=0.8, label=item)
            if ylim:
                axes[j].set_ylim(*ylim)

            axes[j].set_title(f"{'-'.join(map(str, map(int, samples[j])))}", fontweight="bold")
            axes[j].grid()

        axes[0].legend()
        plt.show()
        filename = f"{self.name}_LAT_{self.dataset}SET@ep{self.current_epoch}.jpg"
        return {filename: fig}

    # TODO: pred unit
    def plot_tsne(self, plot_terms, title=None):
        from sklearn.manifold import TSNE
        # plt.style.use('dark_background')
        if title:
            title = f"{title} @ep{self.current_epoch}"
        else:
            title = f"{self.name} T-SNE on {self.dataset} @ep{self.current_epoch}"
        samples = np.array(self.preds['TAG']).astype(int)[self.select_inds]

        fig = self.__plot_settings__()
        fig.suptitle(title)
        axes = fig.subplots(nrows=1, ncols=len(plot_terms))
        for i, item in enumerate(plot_terms):
            axes[i].set_title(item, fontweight="bold")
            self.preds[item].Tsne()
            axes[i].scatter(self.preds[item].tsne[:, 0], self.preds[item].tsne[:, 1],  alpha=0.6)
            for j, ind in enumerate(self.select_inds):
                axes[i].scatter(self.preds[item].tsne[ind, 0], 
                                self.preds[item].tsne[ind, 1],
                                c='magenta', marker=(5, 1), linewidths=4)
                axes[i].annotate(f"{'-'.join(map(str, map(int, samples[j])))}",
                                 (self.preds[item].tsne[ind, 0], 
                                  self.preds[item].tsne[ind, 1]),
                                 fontsize=20)

        plt.show()
        filename = f"{self.name}_TSNE_{self.dataset}SET@ep{self.current_epoch}.jpg"
        return {filename: fig}


class MyLossBBX(MyLossLog):
    def __init__(self, bbx='S_BBX', depth='S_DPT', *args, **kwargs):
        super(MyLossBBX, self).__init__(*args, **kwargs)
        self.bbx = bbx # or change to 'T_BBX'
        self.depth = depth # or False or change to 'T_DPT'

    def plot_bbx(self, title=None):
        if title:
            title = f"{title} on {self.dataset} @ep{self.current_epoch}"
        else:
            title = f"{self.name} Bounding Box Predicts on {self.dataset} @ep{self.current_epoch}"
        samples = np.array(self.preds['TAG']).astype(int)[self.select_inds]
        
        fig = self.__plot_settings__()
        fig.suptitle(title)
        axes = fig.subplots(nrows=2, ncols=np.ceil(self.select_num / 2).astype(int))
        axes = axes.flatten()
        for j, ind in enumerate(self.select_inds):
            axes[j].set_xlim([0, 226])
            axes[j].set_ylim([0, 128])
            axes[j].set_title(f"{'-'.join(map(str, map(int, samples[j])))}", fontweight="bold")
            x1, y1, x2, y2 = self.preds['GT_BBX'][ind]
            x = int(x1 * 226)
            y = int(y1 * 128)
            w = int((x2 - x1) * 226)
            h = int((y2 - y1) * 128)
            axes[j].add_patch(Rectangle((x, y), w, h, edgecolor='blue', fill=False, lw=4, label='GroundTruth'))
            if self.depth:
                axes[j].annotate(f"GT={self.preds['GT_DPT'][ind]:.2f}",
                                 (48, 20),
                                 fontsize=20, color='blue')
            x1, y1, x2, y2 = self.preds[self.bbx][ind]
            x = int(x1 * 226)
            y = int(y1 * 128)
            w = int((x2 - x1) * 226)
            h = int((y2 - y1) * 128)
            axes[j].add_patch(Rectangle((x, y), w, h, edgecolor='orange', fill=False, lw=4, label='Student'))
            if self.depth:
                axes[j].annotate(f"Pred={self.preds[self.depth][ind]:.2f}",
                                 (48, 10),
                                 fontsize=20, color='orange')
                axes[j].scatter(48, 20,
                                c='blue', marker=(5, 1), linewidths=4)
                axes[j].scatter(48, 10,
                                c='orange', marker=(5, 1), linewidths=4)
            axes[j].axis('off')
            axes[j].add_patch(Rectangle((0, 0), 226, 128, facecolor="#eafff5",
                                            transform=axes[j].transAxes, zorder=-1))

        axes[0].legend()
        plt.show()
        filename = f"{self.name}_BBX_{self.dataset}SET@ep{self.current_epoch}.jpg"
        return {filename: fig}


class MyLossCTR(MyLossLog):
    def __init__(self, depth=False, *args, **kwargs):
        super(MyLossCTR, self).__init__(*args, **kwargs)
        self.depth = depth
        self.ctr = ['GT_CTR', 'S_CTR']
        self.dpt = ['GT_DPT', 'S_DPT']
        self.color = ['blue', 'orange', 'green', 'pink']

    def plot_center(self, title=None):
        if title:
            title = f"{title} on {self.dataset} @ep{self.current_epoch}"
        else:
            title = f"{self.name} Center Predicts on {self.dataset} @ep{self.current_epoch}"
        samples = np.array(self.preds['TAG']).astype(int)[self.select_inds]
        
        fig = self.__plot_settings__()
        fig.suptitle(title)
        axes = fig.subplots(nrows=2, ncols=np.ceil(self.select_num / 2).astype(int))
        axes = axes.flatten()
        for j, ind in enumerate(self.select_inds):
            
            axes[j].set_xlim([0, 226])
            axes[j].set_ylim([0, 128])
            axes[j].set_title(f"{'-'.join(map(str, map(int, samples[j])))}", fontweight="bold")
            
            for ci, ctr in enumerate(self.ctr):
                x, y = self.preds[ctr][ind]
                x = int(x * 226)
                y = int(y * 128)
                axes[j].scatter(x, y, c=self.color[ci], marker=(5, 1), alpha=0.5, linewidths=5, label=ctr)
                
            if self.depth:
                for di, dpt in enumerate(self.dpt):
                    axes[j].annotate(f"{dpt}={self.preds[dpt][ind]:.2f}",
                                    (48, 30 - 10 * di),
                                    fontsize=20, color=self.color[di])

            axes[j].axis('off')
            axes[j].add_patch(Rectangle((0, 0), 226, 128, facecolor="#F0FFFF",
                                        transform=axes[j].transAxes, zorder=-1))

        axes[0].legend()
        plt.show()
        filename = f"{self.name}_CTR_{self.dataset}SET@ep{self.current_epoch}.jpg"
        return {filename: fig}
    
    def plot_domain(self):

        title = f"{self.name} Domain Predicts on {self.dataset} @ep{self.current_epoch}"
        samples = np.array(self.preds['TAG']).astype(int)[self.select_inds]

        fig = self.__plot_settings__()
        fig.suptitle(title)
        ax = plt.gca()

        bar_colors = ['skyblue', 'orange', 'yellowgreen', 'pink']  # Colors for each segment
        dom_pred = []
        dom_gt = []
        xtick = []

        for j, ind in enumerate(self.select_inds):
            soft_logits = F.softmax(torch.tensor(self.preds['DOM_PRED'][ind]))
            dom_pred.append(torch.argmax(soft_logits) / len(soft_logits) + 0.1)
            dom_gt.append(self.preds['DOM_GT'][ind] / len(soft_logits) + 0.13)
            xtick.append(f"{'-'.join(map(str, map(int, samples[j])))}")
            # Bar Plot
            ax.bar(j, soft_logits[0], color=bar_colors[0])  # Base segment
            for i in range(1, len(soft_logits)):
                ax.bar(j, soft_logits[i], 
                        bottom=sum(soft_logits[:i]),
                        color=bar_colors[i])
                
        ax.scatter(list(range(self.select_num)), 
                    dom_pred, 
                    marker=(5, 1), alpha=0.8, linewidths=5, color='blue',
                    label="PRED")

        ax.scatter(list(range(self.select_num)),
                    dom_gt, 
                    marker=(5, 1), alpha=0.8, linewidths=5, color='red',
                    label="GT")     
            
        ax.set_xticks(list(range(self.select_num)))  # Positions
        ax.set_xticklabels(xtick)  # Custom labels
        ax.legend()

        plt.grid()
        plt.show()
        filename = f"{self.name}_DOM_{self.dataset}SET@ep{self.current_epoch}.jpg"
        return {filename: fig}
    
    
class MyLossGAN(MyLossLog):
    
    def __init__(self, *args, **kwargs):
        super(MyLossGAN, self).__init__(*args, **kwargs)
    
    def plot_discriminate(self):

        title = f"{self.name} Domain Predicts on {self.dataset} @ep{self.current_epoch}"
        samples = np.array(self.preds['TAG']).astype(int)[self.select_inds]

        fig = self.__plot_settings__()
        fig.suptitle(title)
        ax = plt.gca()

        bar_colors = ['skyblue', 'orange']  # Colors for each segment
        dom_pred = np.array(self.preds['DOM_PRED'])[self.select_inds]
        dom_gt = np.array(self.preds['DOM_GT'])[self.select_inds]
        dom_gt[dom_gt > 0.5] = 0.8
        dom_gt[dom_gt < 0.5] = 0.2 # Adjust position
        dom_pred[dom_pred > 0.5] = 0.75
        dom_pred[dom_pred < 0.5] = 0.25 # Adjust position
        xtick = []

        for j, ind in enumerate(self.select_inds):
            xtick.append(f"{'-'.join(map(str, map(int, samples[j])))}")
            # Bar Plot
            ax.bar(j, 1-dom_pred[j], color=bar_colors[0])  # Fake segment
            ax.bar(j, dom_pred[j], bottom=1-dom_pred[j], color=bar_colors[1])  # Real segment
                
        ax.scatter(list(range(self.select_num)), 
                    dom_pred, 
                    marker=(5, 1), alpha=0.8, linewidths=5, color='blue',
                    label="PRED")

        ax.scatter(list(range(self.select_num)),
                    dom_gt, 
                    marker=(5, 1), alpha=0.8, linewidths=5, color='red',
                    label="GT")     
            
        ax.set_xticks(list(range(self.select_num)))  # Positions
        ax.set_xticklabels(xtick)  # Custom labels
        ax.set_ylim(0, 1)
        ax.legend()

        plt.grid()
        plt.show()
        filename = f"{self.name}_DOM_{self.dataset}SET@ep{self.current_epoch}.jpg"
        return {filename: fig}

