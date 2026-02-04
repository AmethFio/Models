import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PoolingFlow import SpatialLatentFlow, TemporalSlotAttention, SpatialLearnablePooling
from TrainerLite import ModelTrainer

class DiceLossWithLogits(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        """
        logits: (B, 1, H, W)
        target: (B, 1, H, W)
        return: (B,)
        """
        pred = torch.sigmoid(logits)

        pred = pred.flatten(start_dim=1)
        target = target.flatten(start_dim=1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        loss = 1.0 - dice            # (B,)

        return loss


class SoftXORLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        pred = torch.sigmoid(pred)
        return (pred * (1 - gt) + (1 - pred) * gt).mean()


class ImageGradientLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def gradient(img):
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        return dx.abs().mean() + dy.abs().mean()

    def forward(self, pred, gt):
        delta_img = pred - gt
        return self.gradient(delta_img)


class SilhouetteWeightLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        fg = (gt > 0).float()              # 人体区域
        kernel = torch.ones(1, 1, 9, 9).to(gt.device)
        fg_dilate = F.conv2d(fg, kernel, padding=4) > 0
        weight = fg_dilate.float()

        pixel_loss = F.l1_loss(pred, gt, reduction='none')
        loss = (pixel_loss * weight).sum() / (weight.sum() + 1.e-6)

        return loss


class CenterWeightLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        mask = (gt > 0.1).float()       # 人体
        weight = 1.0 + 4.0 * mask      # 人体 ×5 权重

        loss = ((pred - gt).abs() * weight).mean()

        return loss


class XorWeightLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        mask = (gt > 0.1).float()       # 人体
        weight = 1.0 + 4.0 * mask      # 人体 ×5 权重

        shape_loss = torch.logical_xor(
            (pred > 0.5), (gt > 0.5)
        ).float()

        loss = (shape_loss * weight).mean()

        return loss


class EdgeWightLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
        edge = F.pad(dx.abs(), (0,1,0,0)) + F.pad(dy.abs(), (0,0,0,1))
        weight = torch.clamp(edge * 5.0 + 1.0, max=10.0)

        pixel_loss = F.l1_loss(pred, gt, reduction='none')
        loss = (pixel_loss * weight).mean()

        return loss


class CropbyGT:
    def __init__(self, threshold=0.1, margin=5):
        self.threshold = threshold
        self.margin = margin

    def bbox_from_gt(self, gt):
        mask = gt > self.threshold
        # mask: (H, W), bool
        ys, xs = torch.where(mask)
        if len(xs) == 0:
            return None
        return ys.min(), ys.max(), xs.min(), xs.max()

    def __call__(self, pred, gt):
        B, C, H, W = pred.shape
        crops_pred, crops_gt = [], []

        for b in range(B):
            box = self.bbox_from_gt(gt[b, 0])
            if box is None:
                crops_pred.append(pred[b:b+1])
                crops_gt.append(gt[b:b+1])
                continue

            y1, y2, x1, x2 = box
            y1 = max(0, y1 - self.margin)
            x1 = max(0, x1 - self.margin)
            y2 = min(H, y2 + self.margin)
            x2 = min(W, x2 + self.margin)

            crops_pred.append(pred[b:b+1, :, y1:y2, x1:x2])
            crops_gt.append(gt[b:b+1, :, y1:y2, x1:x2])

        return crops_pred, crops_gt



class GEGLU_proj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GEGLU_proj, self).__init__()
        self.proj = nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x):
        x = self.proj(x)
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=128, patch_size=16, in_chans=1, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B*T, C, H, W)
        x = self.proj(x) # (B*T, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B*T, N, D)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTVideoEncoder(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_chans=1, embed_dim=512, num_heads=16, num_slots=4, depth=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads) for _ in range(depth)
        ])

        self.pooling = SpatialLearnablePooling(embed_dim)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.slot_attention = TemporalSlotAttention(dim=embed_dim, num_slots=num_slots, num_heads=num_heads)
        
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: (B, K, C, H, W)
        B, K, C, H, W = x.shape
        
        # 1. Fold time into batch for spatial encoding
        x = x.view(B * K, C, H, W)
        
        # 2. Patch Embed
        x = self.patch_embed(x) # (B*K, N, D)
        
        # 3. Add Positional Embed
        x = x + self.pos_embed

        # embeds = x
        
        # # 4. Transformer Encoder
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # 4. Spatial Learnable Pooling
        x = self.pooling(x)
        x = self.norm(x)
        
        # 5. Unfold time
        # (B*K, N, D) -> (B, K*N, D)
        # We flatten all spatial temporal tokens into one sequence per batch item
        x = x.view(B, K * x.shape[1], self.embed_dim)
        uni_embeds = x.view(B, K, -1, self.embed_dim)
        
        # 6. Temporal Slot Attention
        slots = self.slot_attention(x) # (B, 4, D)
        
        return uni_embeds, slots


class TimeConditionalDecoder(nn.Module):
    """
    Decodes a frame at a specific time t given the slot tokens.
    Designed for robustness: simple Cross-Attention to force encoder to learn good slots.
    """
    def __init__(self, img_size=128, patch_size=16, embed_dim=512, num_heads=8, in_chans=1, num_time_steps=6):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim
        
        # Spatial Queries: learnable parameters for each patch position
        self.spatial_query = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.spatial_query, std=.02)
        
        # Time Embedding: Learnable vectors for each discrete time step [0, num_time_steps-1]
        self.time_embed = nn.Embedding(num_time_steps, embed_dim)
        
        # Cross Attention: Queries (Spatial+Time) -> Key/Value (Slots)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Stronger attention
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Output Projection: Project back to pixel space
        self.output_proj = nn.Linear(embed_dim, patch_size * patch_size * in_chans)
        
        # More informative
        # self.output_proj = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, embed_dim * 4),
        #     nn.GELU(),
        #     nn.Linear(embed_dim * 4, patch_size * patch_size * in_chans),
        # )
        
        # Refine conv
        self.conv_refine = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, slots, t):
        """
        slots: (B, 4, D)
        t: (B,) long/int tensor with values in [0, num_time_steps-1]
        """
        B = slots.shape[0]
        
        # 1. Prepare Queries
        # Spatial Queries
        spatial_q = self.spatial_query.expand(B, -1, -1) # (B, N, D)
        
        # Time Embeddings
        # t is (B,), so time_emb is (B, D)
        time_emb = self.time_embed(t).unsqueeze(1) # (B, 1, D)
        
        # Add Time to Spatial Queries (Broadcasting time across all patches)
        q = spatial_q + time_emb # (B, N, D)
        #q = spatial_q
        
        # 2. Cross Attention
        # Connect to Slots
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(slots)
        
        # Note: PyTorch MultiheadAttention expects (B, S, E) if batch_first=True
        x, _ = self.attn(q_norm, kv_norm, kv_norm)
        x = x + self.ffn(x)
        
        # 3. Reconstruct Patches
        x = self.output_proj(x) # (B, N, P*P*C)
        
        # 4. Unpatchify
        # (B, H*W, P*P*C) -> (B, C, H, W)
        p = self.patch_size
        h, w = self.grid_size
        x = x.transpose(1, 2).reshape(B, 1, p, p, h, w)
        x = torch.einsum('nchpwq->ncwhqp', x)
        x = x.reshape(B, 1, h * p, w * p)
        x = self.conv_refine(x)
        
        return x


class ViTTeacher(nn.Module):
    def __init__(self, img_size=128, patch_size=16, embed_dim=512, num_slots=4, num_heads=8):
        super().__init__()
        self.encoder = ViTVideoEncoder(img_size, patch_size, embed_dim=embed_dim, num_slots=num_slots, num_heads=num_heads)
        self.decoder = TimeConditionalDecoder(img_size, patch_size, embed_dim=embed_dim)
        self.flow = SpatialLatentFlow(embed_dim)

        self.crop = CropbyGT()

        self.dice_loss = DiceLossWithLogits()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.xor_loss = SoftXORLoss()
        self.edge_loss = ImageGradientLoss()
        self.silweight_loss = SilhouetteWeightLoss()
        self.edgeweight_loss = EdgeWightLoss()
        self.centerweight_loss = CenterWeightLoss()
        self.xorweight_loss = XorWeightLoss()

        self.recon_weight = 1.
        self.flow_weight = 0.

        self.bce_weight = 1.
        self.mse_weight = 0
        self.dice_weight = 0.5 * 0
        self.xor_weight = 1.* 0
        self.edge_weight = 10. 

        self.silweight_weight = 0.
        self.edgeweight_weight = 0.
        self.centerweight_weight = 0.
        self.xorweight_weight = 1.

        self.foreground_weight = 2. * 0
        self.constant_weight = 0.5 * 0

    def foreground_area_loss(self, logits, target, min_ratio=0.3):
        """
        Background regularization.
        logits: (B, 1, H, W)
        target: (B, 1, H, W)
        """
        pred = torch.sigmoid(logits)

        pred_area = pred.flatten(1).mean(dim=1)        # (B,)
        gt_area   = target.flatten(1).mean(dim=1)      # (B,)

        # 要求 pred_area >= min_ratio * gt_area
        loss = torch.relu(min_ratio * gt_area - pred_area)
        return loss    # (B,)

    def spatial_variance_loss(self, logits):
        """
        Constant regularization.
        logits: (B, 1, H, W)
        """
        x = logits.flatten(2)      # (B, 1, HW)
        var = x.var(dim=2)         # (B, 1)
        return torch.relu(1e-3 - var.squeeze(1))

    def recon_loss(self, recon, gt):

        bce_loss = self.bce_loss(recon, gt).flatten(start_dim=1).mean(dim=1)
        mse_loss = self.mse_loss(recon, gt).flatten(start_dim=1).mean(dim=1)
        dice_loss = self.dice_loss(recon, gt)
        xor_loss = self.xor_loss(recon, gt)
        edge_loss = self.edge_loss(recon, gt)
        silweight_loss = self.silweight_loss(recon, gt)
        edgeweight_loss = self.edgeweight_loss(recon, gt)
        centerweight_loss = self.centerweight_loss(recon, gt)
        xorweight_loss = self.xorweight_loss(recon, gt)

        recon_loss = bce_loss * self.bce_weight
        recon_loss += mse_loss * self.mse_weight
        recon_loss += dice_loss * self.dice_weight
        recon_loss += xor_loss * self.xor_weight
        recon_loss += edge_loss * self.edge_weight

        recon_loss += silweight_loss * self.silweight_weight
        recon_loss += edgeweight_loss * self.edgeweight_weight
        recon_loss += centerweight_loss * self.centerweight_weight
        recon_loss += xorweight_loss * self.xorweight_weight

        foreground_loss = self.foreground_area_loss(recon, gt)
        constant_loss = self.spatial_variance_loss(recon)
        recon_loss += foreground_loss * self.foreground_weight
        recon_loss += constant_loss * self.constant_weight

        # r, g = self.crop(recon, gt)
        # crop_loss = 0.
        # for r_, g_ in zip(r, g):
        #     crop_loss += F.l1_loss(r_, g_)
        # crop_loss = crop_loss / len(r_)

        # recon_loss += crop_loss * 10
        return recon_loss
    
    def forward(self, data):
        # Find number of samples T in x
        # Randomly generate t < T
        uni_embeds, slots = self.encoder(data['dimg'])

        B, T = data['dimg'].size(0), data['dimg'].size(1)
        t = torch.randint(0, T, (B,), device=data['dimg'].device)
        t0 = t[0]
        gt = data['dimg'][:, t0]
        recon = self.decoder(slots, t)

        z, log_det_sum, flow_loss = self.flow(uni_embeds)

        flow_loss = flow_loss * self.flow_weight
        recon_loss = self.recon_loss(recon, gt) * self.recon_weight

        # Reduce batch due to being kept by flow
        flow_loss = torch.mean(flow_loss)
        recon_loss = torch.mean(recon_loss)
        loss = flow_loss + recon_loss

        LOSS = {
            'LOSS': loss,
            'FLOW': flow_loss,
            'RECON': recon_loss
        }

        RET = {
            'PRED': recon,
            'GT': gt,
            'FRAME': t0,
            'TOKEN': uni_embeds,
            'IND': data['ind'],
            'TAG': data['tag']
        }

        return RET, LOSS


class TeacherTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(name='WiHAR', 
        model=ViTTeacher(num_slots=16, num_heads=8), 
        pred_terms = ('GT', 'PRED'), 
        lr=1.e-4, 
        *args, **kwargs)



if __name__ == '__main__':
    # Simple sanity check
    model = ViTVideoAutoencoder(embed_dim=512, num_slots=4)
    B, T = 2, 5
    dummy_input = torch.randn(B, T, 1, 128, 128) 
    dummy_time = torch.randint(0, 6, (B,)) # Random time indices [0, 5]
    
    recon, slots = model(dummy_input, dummy_time)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Recon shape: {recon.shape}")
    print(f"Slots shape: {slots.shape}")
    
    assert recon.shape == (B, 1, 128, 128)
    assert slots.shape == (B, 4, 512)
    print("Autoencoder Check passed!")
