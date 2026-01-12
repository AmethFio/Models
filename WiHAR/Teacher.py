import torch
import torch.nn as nn
import math
from PoolingFlow import SpatialLatentFlow, TemporalSlotAttention

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
    def __init__(self, img_size=128, patch_size=16, in_chans=1, embed_dim=512, num_heads=8, num_slots=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # self.blocks = nn.ModuleList([
        #     Block(embed_dim, num_heads) for _ in range(depth)
        # ])

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
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # 1. Fold time into batch for spatial encoding
        x = x.view(B * T, C, H, W)
        
        # 2. Patch Embed
        x = self.patch_embed(x) # (B*T, N, D)
        
        # 3. Add Positional Embed
        x = x + self.pos_embed

        # embeds = x
        
        # # 4. Transformer Encoder
        # for block in self.blocks:
        #     x = block(x)
        # x = self.norm(x)

        # 4. Spatial Learnable Pooling
        x = self.pooling(x)
        x = self.norm(x)
        uni_embeds = x
        
        # 5. Unfold time
        # (B*T, N, D) -> (B, T*N, D)
        # We flatten all spatial temporal tokens into one sequence per batch item
        x = x.view(B, T * x.shape[1], self.embed_dim)
        
        # 6. Temporal Slot Attention
        slots = self.slot_attention(x) # (B, 4, D)
        
        return uni_embeds, slots

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / (10000 ** (2 * (hid_idx // 2) / d_hid))

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = torch.tensor([get_posi_angle_vec(pos_i) for pos_i in range(n_position)], dtype=torch.float)

    sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return sinusoid_table


class SimpleTimeDecoder(nn.Module):
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
        
        # Output Projection: Project back to pixel space
        self.output_proj = nn.Linear(embed_dim, patch_size * patch_size * in_chans)
        
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
        
        # 2. Cross Attention
        # Connect to Slots
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(slots)
        
        # attn_output, _ = self.attn(query=q_norm, key=kv_norm, value=kv_norm)
        # Note: PyTorch MultiheadAttention expects (B, S, E) if batch_first=True
        x, _ = self.attn(q_norm, kv_norm, kv_norm)
        
        # 3. Reconstruct Patches
        x = self.output_proj(x) # (B, N, P*P*C)
        
        # 4. Unpatchify
        # (B, H*W, P*P*C) -> (B, C, H, W)
        p = self.patch_size
        h, w = self.grid_size
        x = x.transpose(1, 2).reshape(B, 1, p, p, h, w)
        x = torch.einsum('nchpwq->ncwhqp', x)
        x = x.reshape(B, 1, h * p, w * p)
        
        return x

class ViTVideoAutoencoder(nn.Module):
    def __init__(self, img_size=128, patch_size=16, embed_dim=512, num_slots=4):
        super().__init__()
        self.encoder = ViTVideoEncoder(img_size, patch_size, embed_dim=embed_dim, num_slots=num_slots)
        self.decoder = SimpleTimeDecoder(img_size, patch_size, embed_dim=embed_dim)
        self.recon_loss = nn.MSELoss(reduction='sum')
        
    def forward(self, data, t):
        # x: (B, T, C, H, W)
        # t: (B,) target time for reconstruction [0, T]
        uni_embeds, slots = self.encoder(data['dimg'])
        recon = self.decoder(slots, t)

        loss = self.recon_loss(recon, data['dimg'][:, t[0]])

        LOSS = {
            'LOSS': loss
        }

        RET = {
            'PRED': recon,
            'GT': data['dimg'][:, t[0]]
        }


        return RET, LOSS


class ViTTeacher(nn.Module):
    def __init__(self, img_size=128, patch_size=16, embed_dim=512, num_slots=4):
        super().__init__()
        self.encoder = ViTVideoEncoder(img_size, patch_size, embed_dim=embed_dim, num_slots=num_slots)
        self.decoder = SimpleTimeDecoder(img_size, patch_size, embed_dim=embed_dim)
        self.flow = SpatialLatentFlow(embed_dim)

        self.recon_loss = nn.BCEWithLogitsLoss()

        self.recon_weight = 1.
        self.flow_weight = 1.
    
    def forward(self, data):
        # Find number of samples T in x
        # Randomly generate t < T
        uni_embeds, slots = self.encoder(data['dimg'])

        B, T = data['dimg'].size(0), data['dimg'].size(1)
        t = torch.randint(0, T, (B,))
        t0 = t[0]
        recon = self.decoder(slots, t)

        z, log_det_sum, flow_loss = self.flow(uni_embeds)

        flow_loss = flow_loss * self.flow_weight
        recon_loss = self.recon_loss(recon, data['dimg'][:, t0]) * self.recon_weight

        LOSS = {
            'LOSS': flow_loss + recon_loss,
            'FLOW': flow_loss,
            'RECON': recon_loss
        }

        RET = {
            'PRED': recon,
            'GT': data['dimg'][:, t0],
            'FRAME': t0,
            'TOKEN': uni_embeds
        }

        return RET, LOSS




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
