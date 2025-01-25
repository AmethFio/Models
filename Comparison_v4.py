import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import pandas as pd
from tqdm.notebook import tqdm

from joblib import Parallel, delayed
cp_flag = False
if torch.cuda.is_available():
    import cupy as cp
    import cupyx.scipy.ndimage as cnd
    import cupyx.scipy.signal as cps
    from cupy.cuda import Device
    cp_flag = True
else:
    from scipy import signal
    
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from misc import plot_settings
from PIL import Image
from scipy.ndimage import zoom

from skimage.metrics import structural_similarity as ssim

criteria = ['mse', 'snr', 'ssim', 'soft_iou', 'matched_iou', 'matched_iou_mask', 'matched_mae', 'matched_mae_mask', 'average_depth_mse', 'est_depth', 'gt_depth', 'hist_mse', 'distance', 'dx', 'dy']


def print_result(attribute_name='result'):
    def decorator(func):
        def wrapper(obj, *args, **kwargs):
            # Check if the object has the specified attribute
            if hasattr(obj, attribute_name):
                print(f"{attribute_name}: {getattr(obj, attribute_name)}")
            else:
                print(f"Attribute '{attribute_name}' not found in the class.")
            return func(obj, *args, **kwargs)
        return wrapper
    return decorator



class Tester:
    
    # Notice: set shuffle=False for test loader!
    # Test on all test samples
    def __init__(self, name, trainer, total_length, save_path=None):
        self.name = name
        self.trainer = trainer
        self.preds = None
        self.results = None
        self.total_length = total_length  # lenth of total matched labels of data organizer
        self.save_path = save_path
        
    def fetch_preds(self, pred='R_PRED'):
        
        self.preds = pd.DataFrame(index=list(range(self.total_length)), columns=['gt', 'pred', 'tag', 'matched', 'center'])
        
        gt = self.trainer.losslog.preds['R_GT']
        r_preds = self.trainer.losslog.preds[pred]
        tags = self.trainer.losslog.preds['TAG']
        abs_ind = self.trainer.losslog.preds['IND']
        
        # Store results by absolute indicies
        for i, ind in enumerate(abs_ind):
            self.preds.loc[int(ind), ['gt', 'pred', 'tag']] = [gt[i], r_preds[i], tags[i]]

        # Important: remove nan rows
        self.preds = self.preds.dropna(how='all')

class ResultCalculator(Tester):
    def __init__(self, *args, **kwargs):
        super(ResultCalculator, self).__init__(*args, **kwargs)
        self.image_size = (128, 128)
    
    @staticmethod
    def find_center(img):
        (T, timg) = cv2.threshold((np.squeeze(img) * 255).astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(timg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
            contour = max(contours, key=lambda x: cv2.contourArea(x))
            area = cv2.contourArea(contour)
            if area > 0:
                x, y, w, h = cv2.boundingRect(contour)
                return np.array([x + w / 2, y + h / 2]).astype(int)
        else:
            return np.zeros((1, 4))
        
    @staticmethod
    def iou_loss(pred_mask, true_mask, smooth=1e-6):
        # Flatten the masks to 1D arrays
        pred_mask_flat = pred_mask.flatten()
        true_mask_flat = true_mask.flatten()

        # Calculate the intersection and union
        intersection = np.sum(pred_mask_flat * true_mask_flat)
        union = np.sum(pred_mask_flat) + np.sum(true_mask_flat) - intersection

        # Compute IoU
        iou = (intersection + smooth) / (union + smooth)
        
        return 1 - iou
    
    @staticmethods
    def snr_gt(reconstructed, ground_truth):
        signal_power = np.sum(ground_truth**2)
        noise_power = np.sum((ground_truth - reconstructed)**2)
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    @staticmethod
    def histogram_matching_loss(pred, gt):
        # Function to calculate the histogram of depth values
        def calculate_histogram(depth_map, bins=50):
            histogram, _ = np.histogram(depth_map, bins=bins, range=(0, 1), density=False)
            histogram = histogram / np.sum(histogram)
            return histogram
        
        def histogram_mse_loss(hist_pred, hist_gt):
            return np.mean((hist_pred - hist_gt) ** 2)

        # Calculate histograms for predicted and ground truth depth maps
        hist_pred = calculate_histogram(pred, bins=50)
        hist_gt = calculate_histogram(gt, bins=50)

        # Calculate the histogram matching loss (MSE)
        loss = histogram_mse_loss(hist_pred, hist_gt)
        return loss
        
    def matched_loss(self, im, gt, scale, cuda, mask_threshold=0.1): 
        # threshold 0.1 => 36cm
        im_mask = np.where(im > mask_threshold, 1., 0.)
        gt_mask = np.where(gt > mask_threshold, 1., 0.)
        
        im_depth = im.sum() / (im_mask.sum() + 1.e-6)
        gt_depth = gt.sum() / (gt_mask.sum() + 1.e-6)
        average_depth = np.abs(gt_depth - im_depth)
        
        new_size = int(128*scale), int(128*scale)
        
        _im = Image.fromarray((im * 255).astype(np.uint8))
        small_im = _im.resize(new_size, Image.BILINEAR)
        small_im = np.array(small_im).astype(float)
        
        _gt = Image.fromarray((gt * 255).astype(np.uint8))
        small_gt = _gt.resize(new_size, Image.BILINEAR)
        small_gt = np.array(small_gt).astype(float)
    
        if cp_flag:
            with Device(cuda):
                tmp = cps.correlate2d(cp.array(small_gt), cp.array(small_im), mode='full')
                tmp = cp.asnumpy(tmp)
        else:
            tmp = signal.correlate2d(small_gt, small_im, mode='full')
        
        dy, dx = np.unravel_index(np.argmax(tmp), tmp.shape)

        dy = int(dy / scale) - 128
        dx = int(dx / scale) - 128
    
        im_re = _im.rotate(0, translate = (dx, dy))
        im_re = np.array(im_re).astype(float) / 255.
        
        matched_mae = np.mean(np.abs(gt - im_re))
        matched_iou = self.iou_loss(im_re, gt)
        matched_mae_mask = np.mean(np.abs(im_mask - gt_mask))
        matched_iou_mask = self.iou_loss(im_mask, gt_mask)
        
        distance = np.sqrt(dx ** 2 + dy ** 2)

        return {
            'average_depth_mse': average_depth,
            'est_depth': im_depth,
            'gt_depth': gt_depth,
            'matched_mae': matched_mae,
            'matched_iou': matched_iou,
            'matched_mae_mask': matched_mae_mask,
            'matched_iou_mask': matched_iou_mask,
            'distance': distance / 128,
            'dx': dx / 128,
            'dy': dy / 128
            }
    
    @staticmethod
    def mse_loss(im, gt, find_center=False):
        e = np.mean((np.array(gt) - np.array(im)) ** 2)

        if find_center:
            center = self.find_center(im)
            # To be finalized
        return e
        
    def evaluate(self, scale=0.2, cuda=0, find_center=False):
        def process_row(i, gt, pred, scale, cuda):
            # Calculate losses
            mse = self.mse_loss(pred, gt)
            soft_iou = self.iou_loss(pred, gt)
            snr = self.snr_gt(pred, gt)
            mssm = ssim(pred, gt)
            hist_mse = self.histogram_matching_loss(pred, gt)
            matched_res = self.matched_loss(pred, gt, scale, cuda)

            # Prepare the result dictionary
            result = {
                'index'   : i,
                'mse'     : mse,
                'snr'     : snr,
                'ssim'    : ssim,
                'soft_iou': soft_iou,
                'hist_mse': hist_mse
                      }
            result.update(matched_res)
            return result
        
        self.results = pd.DataFrame(index=list(range(self.total_length)), columns=criteria)
        
        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(process_row)(i, gt, pred, scale, cuda)
            for i, gt, pred, *_ in tqdm(self.preds.itertuples(), desc="Evaluating", total=len(self.preds))
        )

        # Store results in DataFrame
        for res in results:
            self.results.loc[res.pop('index')] = res

        self.results = self.results.dropna(how='all')
        self.results.replace(np.inf, 1, inplace=True)  # Some avg_depth may be inf

    
    def save(self, name):
        if not os.path.exists('../results'):
            os.makedirs('../results')
        self.results.to_csv(os.path.join('../results', f"{name}.csv"))
        print("Results saved")
            
    def plot_example(self, selected=None, matched=False, source='vanilla'):
        fig = plot_settings()
        fig.suptitle(f"{self.name} Depth Image Estimates")
        filename = f"{self.name}_Depth_Image.jpg"

        plot_terms = {'Ground Truth': self.preds['gt'],
                      'Estimates': self.preds['pred']}
        if matched:
            plot_terms['Matched Estimates'] = self.preds['matched']
            filename = f"{self.name}_Depth_Image_Matched.jpg"
        subfigs = fig.subfigures(nrows=len(list(plot_terms.keys())), ncols=1)

        if not selected:
            selected = np.choice(np.arange(self.len), 8)

        zoom_factors = (1, 226 / 128) 

        for i, (key, value) in enumerate(plot_terms.items()):
            subfigs[i].suptitle(key, fontweight="bold")
            axes = subfigs[i].subplots(nrows=1, ncols=len(selected))
            for ind, ax in zip(selected, axes):
                ax.imshow(zoom(value[ind].squeeze(), zoom_factors), vmin=0, vmax=1)
                
                if key == 'Matched Estimates':
                    st = self.preds.loc[ind, 'center'], 
                    ed = [st[0] + self.result.loc[ind, 'x'], 
                          st[1] + self.result.loc[ind, 'y']]
                    ax.plot([st[0], ed[0]], [st[1], ed[1]], color='red', linewidth=2)
                    
                ax.axis('off')
                ax.set_title(self.tags[ind])

        plt.show()
        return {filename: fig}
    
    
class ResultProcess:
    def __init__(self, subjects, labels, cuda):
        # subject = {name: trainer}
        # labels = total_segment_labels (as in DataOrganizer)
        self.subjects = subjects
        self.testers = {}
        self.labels = labels
        self.cuda = cuda
        self.selected = None
        self.table = None
    
        self.vis_count = 0
        self.box_count = 0
        self.cdf_count = 0
        self.nsamples = 12
        self.figs: dict = {}
        
    def test(self):
        for sub, trainer in self.subjects.items():
            trainer.test()
            tester = ResultCalculator(name=sub, 
                trainer=trainer, 
                total_length = len(self.labels)
               )
            tester.fetch_preds()
            tester.evaluate(cuda=self.cuda)
            self.testers[sub] = tester
            
            
    def visualization(self, scope='all', selected=None, matched=False):
        # TODO: REWRITE!
        
        if selected is None:
            if self.selected is None:
                selected = self.univ_tag.sample(n=self.nsamples)
                self.selected = selected
            else:
                selected = self.selected
        
        scope = list(self.testers.keys()) if scope=='all' else scope
        self.figsize = (2.5 * self.nsamples, 2.5 * (len(scope) + 1))
        fig = plot_settings(self.figsize)
        mch = ' - Matched' if matched else ''
        
        fig.suptitle(f'Comparison Visualization{mch}')
        filename = f"Comparison_Visual_Matched_{self.vis_count}.jpg" if matched else f"Comparison_Visual_{self.vis_count}.jpg"
        zoom_factors = (1, 226 / 128)
         
        subfigs = fig.subfigures(nrows=len(scope) + 1, ncols=1)
        
        # Show GT
        subfigs[0].suptitle("Ground Truth", fontweight="bold")
        axes = subfigs[0].subplots(nrows=1, ncols=self.nsamples)
        for ind, ax in zip(selected, axes):
            ax.imshow(self.subjets[scope[0]].gt[ind].squeeze(), vmin=0, vmax=1)
            ax.set_title(self.subjets[scope[0]].tags[ind])
            ax.axis('off')
            
        # Show pred
        for sub, subfig in zip(scope, subfigs[1:]):
            _sub = self.testers[sub]

            subfig.suptitle(f'{_sub.name}', fontweight="bold")
            axes = subfig.subplots(nrows=1, ncols=self.nsamples)
            for ind, ax in zip(selected, axes):
                im = getattr(_sub, 'matched' if matched else 'resized')[ind].squeeze()
                ax.imshow(zoom(im, zoom_factors), vmin=0, vmax=1)
                ax.axis('off')
                ax.set_title(_sub.tags[ind])

                if matched:
                    st = _sub.center[ind][0]
                    ed = [st[0] + _sub.result.loc[ind, 'x'].values, 
                        st[1] + _sub.result.loc[ind, 'y'].values]
                    ax.plot([st[0], ed[0]], [st[1], ed[1]], color='red', linewidth=2)
        plt.show()
        self.vis_count += 1
        self.figs[filename] = fig
    
    def gathercdf(self, scope='all', item='mse', show_pdf=True):
        scope = list(self.testers.keys()) if scope=='all' else scope
        fig = plot_settings((20, 10)) if show_pdf else plot_settings((6, 4))

        filename = f"Comparison_CDF_{item.upper()}_{self.cdf_count}.jpg"
        fig.suptitle(f'Test PDF-CDF - {item.upper()}', fontweight="bold")

        _mmin = 1
        _mmax = 0

        data_values = np.concatenate([self.testers[sub].results[item].values for sub in scope])
        mmin, mmax = np.min(data_values), np.max(data_values)
        nbins = 50 if show_pdf else 30
        bins = np.linspace(mmin, mmax, nbins)
        zorder_base = len(scope) + 2
        
        for i, sub in enumerate(scope):
            
            hist_, bin_edges = np.histogram(self.testers[sub].results[item].values, bins)
            width = (bin_edges[1] - bin_edges[0]) * 0.8
            if sum(hist_) > 0:
                cdf = np.cumsum(hist_ / sum(hist_))
            else:
                cdf = np.zeros_like(hist_)        
            _mmax = max(_mmax, np.max(bin_edges))
            _mmin = min(_mmin, np.min(bin_edges))

            if show_pdf:
                plt.bar(bin_edges[1:], hist_ / max(hist_), width=width, label=sub, zorder=i)
            plt.plot(bin_edges[1:], cdf, '-*', label=sub, zorder=i + zorder_base)

        ax = plt.gca()
        ax.fill_between(np.linspace(_mmin, _mmax, 100), 1.02, color='white', alpha=0.5, zorder=zorder_base)

        plt.xlabel(f'Per-sample {item}')
        plt.ylabel('Frequency')
        plt.grid()
        plt.legend()
        plt.show()
        self.cdf_count += 1
        self.figs[filename] = fig
    
    def gatherbox(self, scope='all', item='mse'):
        scope = list(self.testers.keys()) if scope=='all' else scope
        width = len(scope) + 1 if len(scope) > 9 else 6
        fig = plot_settings((width, 4))
 
        filename = f"Comparison_BoxPlot_{item.upper()}_{self.box_count}.jpg"
        # fig.suptitle(f'Impact of Guide Data Amount', fontweight="bold")
        
        for i, sub in enumerate(scope):
            _sub = self.testers[sub]
            plt.boxplot(_sub.results[item].values, 
                        labels=[sub], 
                        positions=[i], 
                        vert=True, 
                        showmeans=True,
                        patch_artist=True, 
                        boxprops={'facecolor': 'lightblue'}
                        )
        plt.setp(plt.gca().get_xticklabels(), 
                 #rotation=45, 
                 horizontalalignment='right')    
        # plt.yscale('log', base=2)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Method Variations", fontsize=14)
        plt.ylabel(item, fontsize=14)
        plt.show()
        self.box_count += 1
        self.figs[filename] = fig
    
    def average_table(self, scope='all'):
        scope = list(self.testers.keys()) if scope=='all' else scope
        keys = criteria
        comparison_table = pd.DataFrame(columns=keys)
                
        for sub in scope:
            for key in keys:
                comparison_table.loc[sub, key] = np.mean(self.testers[sub].results[key].values)

        self.table = comparison_table
        print(comparison_table)
