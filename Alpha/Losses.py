import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np


class NCCLoss:
    """
    Normalized Cross Correlation Loss
    """
    def __init__(self):
        self.eps = 1e-8

    def __forward__(self, source_shape, target_shape):

        target_shape = target_shape.reshape(target_shape.shape[0], -1)
        source_shape = source_shape.reshape(source_shape.shape[0], -1)

        # Zero-mean
        target_shape = target_shape - target_shape.mean(dim=1, keepdim=True)
        source_shape = source_shape - source_shape.mean(dim=1, keepdim=True)

        # Normalize (L2 norm)
        target_shape_norm = target_shape / (target_shape.norm(dim=1, keepdim=True) + self.eps)
        source_shape_norm = source_shape / (source_shape.norm(dim=1, keepdim=True) + self.eps)

        ncc = torch.matmul(target_shape_norm, source_shape_norm.T)

        return ncc
        

class NCCMSELoss:
    """
    Normalized Cross Correlation-MSE Loss
    """
    def __init__(self, dims=-1, lambda_ncc=0.7, epsilon=1e-10, reduction=None):
        self.lambda_ncc = lambda_ncc
        self.epsilon = epsilon
        self.dims = dims
        self.reduction = reduction
        # FOR VECTORS: dims=-1
        # FOR IMAGES: dims=(1, 2, 3)

    def ncc(self, pred, target):
        """Compute batch-wise Normalized Cross-Correlation (NCC)."""
        pred = pred.float()
        target = target.float()

        mean_pred = torch.mean(pred, dim=self.dims, keepdim=True)
        mean_target = torch.mean(target, dim=self.dims, keepdim=True)

        numerator = torch.sum((pred - mean_pred) * (target - mean_target), dim=self.dims)
        denominator = torch.sqrt(
            torch.sum((pred - mean_pred) ** 2, dim=self.dims) * 
            torch.sum((target - mean_target) ** 2, dim=self.dims)
        )

        ncc_value = numerator / (denominator + self.epsilon)  # Avoid division by zero
        return ncc_value.mean()

    def mse(self, pred, target):
        """Compute Mean Squared Error (MSE)."""
        mse_loss = F.mse_loss(pred, target, reduction=self.reduction)
        if self.reduction == 'sum':
            mse_loss = mse_loss / pred.shape[0]
        return mse_loss

    def __call__(self, pred, target):
        ncc_loss = 1 - self.ncc(pred, target)  # Convert NCC to loss
        mse_loss = self.mse(pred, target)

        # Combined loss: lambda * NCC + (1 - lambda) * MSE
        total_loss = self.lambda_ncc * ncc_loss + (1 - self.lambda_ncc) * mse_loss
        return total_loss


class GradientReversalLayer(Function):
    
    @staticmethod
    def forward(ctx, input, lambda_):
        # Save lambda for later use in backward
        ctx.lambda_ = lambda_
        # Forward pass is identity, just return the input
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        # In the backward pass, retrieve lambda from ctx
        lambda_ = ctx.lambda_
        # Reverse the gradient by multiplying by -lambda
        grad_input = grad_output.neg() * lambda_
        return grad_input, None  # Return gradient for input, None for lambda


class DANNLoss:
    """
    Domain Adversarial Loss
    """
    def __init__(self, lambda_, max_iter=300, adv_loss=nn.CrossEntropyLoss()):
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.grl = GradientReversalLayer
        self.adv_loss = adv_loss

    def calculate_lambda(self, current_ep):
        # Sigmoid schedule for lambda: 2 / (1 + exp(-10 * p)) - 1
        # where p is the proportion of iterations completed
        p = current_ep / self.max_iter
        lambda_value = 2 / (1 + np.exp(-10 * p)) - 1
        self.lambda_ = min(lambda_value, 1)
        return min(lambda_value, 1)

    def __call__(self, critic, current_ep,
                source_label, target_label, 
                source_feature, target_feature, 
                reverse_feature):

        lambda_ = self.calculate_lambda() if reverse_feature else -1.

        dann_features = target_feature if not source_feature else torch.cat(
                        (source_feature, target_feature), dim=0)
        
        # REVERSING DEPENDS ON LAMBDA
        dann_features = GradientReversalLayer.apply(dann_features, lambda_)
    
        domain_preds = critic(dann_features)

        if source_label is not None:
            domain_labels = torch.cat((source_label, 
                target_label)).to(torch.int64).to(dann_features.device)
        else:
            domain_labels = target_label.to(torch.int64).to(dann_features.device)
        
        domain_loss = self.adv_loss(domain_preds, domain_labels)

        with torch.no_grad():
            domain_acc_preds = torch.argmax(domain_preds, dim=1)
            domain_acc = torch.sum(domain_acc_preds == domain_labels) / domain_preds.shape[0]
        
        return domain_loss, domain_acc, domain_preds, domain_labels


class PostCoordLoss:
    """
    Posterior Coordinate Loss
    """
    def __init__(self, mask_threshold=0.01, reduction=None):
        self.mask_threshold = mask_threshold
        self.mse = nn.MSELoss(reduction=reduction)

    def __call__(self, rimg, gt_center, gt_depth):
        mask = torch.where(rimg > self.mask_threshold, 1., 0.)
        N, C, H, W = mask.shape
        y_coords = torch.arange(H, device=mask.device).view(1, 1, H, 1)
        x_coords = torch.arange(W, device=mask.device).view(1, 1, 1, W)

        x_center = (x_coords * mask).sum(dim=[2, 3]) / mask.sum(dim=[2, 3])
        y_center = (y_coords * mask).sum(dim=[2, 3]) / mask.sum(dim=[2, 3])

        post_center = torch.stack((x_center, y_center), dim=-1)  # Shape: (N, 2)

        post_depth = torch.mean(rimg[rimg != 0])

        center_loss = self.mse(post_center, gt_center)
        depth_loss = self.mse(post_depth, gt_depth)

        return center_loss, depth_loss


class PairwiseIoU:
    def __init__(self, threshold=0.5, eps=1e-6):
        self.threshold = threshold
        self.eps = eps

    def __call__(self, pred, target):
        """
        Computes pairwise IoU between two sets of binary masks.

        Args:
            group_a: Tensor of shape (m, H, W) - first group of masks
            group_b: Tensor of shape (n, H, W) - second group of masks
            eps: Small epsilon to avoid division by zero

        Returns:
            iou_matrix: Tensor of shape (n, m) with IoU between each b_i and a_j
        """

        # Flatten masks to (m, H*W) and (n, H*W)
        a_flat = pred.view(pred.shape[0], -1).float()
        b_flat = target.view(target.shape[0], -1).float()

        # Compute intersection: (n, m)
        intersection = torch.matmul(b_flat, a_flat.T)

        # Compute areas
        area_a = a_flat.sum(dim=1)  # (m,)
        area_b = b_flat.sum(dim=1)  # (n,)

        # Compute union: (n, m)
        union = area_a.unsqueeze(0) + area_b.unsqueeze(1) - intersection

        # Compute IoU
        iou = (intersection + self.eps) / (union + self.eps)
        return iou
    
class MMDLoss:
    def __init__(self, sigma=1.0):
        self.sigme = sigma

    def gaussian_kernel(self, x, y):
        x = x.unsqueeze(1)  # (n, 1, d)
        y = y.unsqueeze(0)  # (1, m, d)
        dist = ((x - y) ** 2).sum(2)
        return torch.exp(-dist / (2 * self.sigma ** 2))

    def __call__(self, x, y):
        xx = self.gaussian_kernel(x, x)
        yy = self.gaussian_kernel(y, y)
        xy = self.gaussian_kernel(x, y)
        
        m = x.size(0)
        n = y.size(0)

        loss = (xx.sum() - xx.diag().sum()) / (m * (m - 1)) \
             + (yy.sum() - yy.diag().sum()) / (n * (n - 1)) \
             - 2 * xy.mean()

        return loss
