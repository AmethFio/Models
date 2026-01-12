import torch
import torch.nn as nn
import math
from PoolingFlow import SpatialLearnablePooling, TemporalSlotAttention, SpatialLatentFlow

class TubeletTokenize(nn.Module):
    def __init__(self, in_channels=12, out_channels=512, kernel_size=(13, 5), stride=(13, 5)):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride
        )

    def forward(self, x):
        x = self.conv(x)
        # (B, D, S/P_s, T/P_t)
        x = torch.einsum('bdst->bstd', x)
        B, D = x.shape(0), x.shape(-1)
        x = x.reshape(B, -1, D)

        return x


class SpatioTemporalEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_slots=4):
        super().__init__()
        self.tokenize = TubeletTokenize()
        self.pooling = SpatialLearnablePooling(embed_dim)
        self.slot_attention = TemporalSlotAttention(dim=embed_dim, num_slots=num_slots, num_heads=num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # (B, C, S, T, K)
        x = self.tokenize(x)
        # (B*K, N, D)
        x = self.pooling(x)
        x = self.norm(x)
        uni_embeds = x
        # (B*K, M, D)
        x = self.slot_attention(x)

        return uni_embeds, x

# Permutation unwritten

class Student(nn.Module):
    def __init__(self, embed_dim=512, num_slots=4):
        super().__init__()
        self.encoder = SpatioTemporalEncoder(embed_dim=embed_dim, num_slots=num_slots)
        self.flow = SpatialLatentFlow(embed_dim)

        self.flow_weight = 1.

    def forward(self, data):
        uni_embeds, slots = self.encoder(data['csi'])

        z, log_det_sum, pooled_tokens, flow_loss = self.flow(uni_embeds)
        
        flow_loss = flow_loss * self.flow_weight
        
        LOSS = {
            'LOSS': flow_loss
        }

        RET = {
            'TOKEN': uni_embeds
        }

        return RET, LOSS

# Decoder?