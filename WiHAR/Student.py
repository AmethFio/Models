import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PoolingFlow import SpatialLearnablePooling, TemporalSlotAttention, SpatialLatentFlow
from Teacher import ViTTeacher


class TubeletTokenize(nn.Module):
    def __init__(self, in_channels=12, out_channels=512, kernel_size=(19, 5), stride=(19, 5)):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride
        )

    def forward(self, x):
        B, C, S, T = x.shape
        # CSI shape = (B, 12, 114, 10)
        x = self.conv(x)
        # (B, D, S/P_s, T/P_t) = (B, 512, 6, 2)
        x = torch.einsum('bdst->bstd', x)
        B, D = x.shape[0], x.shape[-1]
        x = x.reshape(B, -1, D)
        # (B, N, D) = (B, 12, 512)
        return x


class SpatioTemporalEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_slots=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.tokenize = TubeletTokenize()
        self.pooling = SpatialLearnablePooling(embed_dim)
        self.slot_attention = TemporalSlotAttention(dim=embed_dim, num_slots=num_slots, num_heads=num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # (B, K, C, S, T)
        B, K, C, S, T = x.shape
        x = self.tokenize(x.reshape(B * K, C, S, T))
        # (B*K, N, D)
        x = self.pooling(x)
        x = self.norm(x)
        # (B*K, M, D)
        x = x.view(B, K * x.shape[1], self.embed_dim)
        uni_embeds = x.view(B, K, -1, self.embed_dim) # (B, K, M, D)
        x = self.slot_attention(x)  # (B, 4, D)

        return uni_embeds, x

class HarHead(nn.Module):
    def __init__(self, embed_dim=512, num_classes=27):
        super().__init__()
        C = embed_dim * 4 # (Total_Tokens)
        self.har_head = nn.Sequential(
            nn.Linear(C, C // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(C // 2, num_classes)
        )

    def forward(self, x):
        x = self.har_head(x.view(x.shape[0], -1))
        return x

class Student(nn.Module):
    def __init__(self, embed_dim=512, num_slots=4):
        super().__init__()
        self.encoder = SpatioTemporalEncoder(embed_dim=embed_dim, num_slots=num_slots)
        self.flow = SpatialLatentFlow(embed_dim)
        self.har_head = HarHead(embed_dim)
        self.teacher = ViTTeacher(embed_dim=embed_dim, num_slots=num_slots)

        self.har_loss = nn.CrossEntropyLoss()
        self.flow_weight = 1.
        self.firstorder_weight = 1.
        self.secondorder_weight = 1.
        self.har_weight = 1.

    @staticmethod
    def first_order_dynamic_loss_agg(student_tokens, teacher_tokens):
        """
        student_tokens, teacher_tokens: (B, K, M, D)
        """
        # 1) mean over M
        # (B, K, D)
        S_agg = student_tokens.mean(dim=2)
        T_agg = teacher_tokens.mean(dim=2)

        # 2) temporal difference over K
        # (B, K-1, D)
        dS = S_agg[:, 1:] - S_agg[:, :-1]
        dT = T_agg[:, 1:] - T_agg[:, :-1]

        # 3) Alignment loss
        loss = F.mse_loss(dS, dT)

        return loss

    @staticmethod
    def multiscale_first_order_dynamic_loss_agg(
        student_tokens,
        teacher_tokens,
        scales=(1, 2, 4),
        weights=(0.5, 0.3, 0.2)
    ):
        """
        Multi-scale 1st-order dynamic loss (aggregation level)

        Args:
            student_tokens: (B, K, M, D)
            teacher_tokens: (B, K, M, D)
            scales: iterable of k values (e.g., (1, 2, 4))
            weights: list or tuple of same length as scales
                    if None, uniform weights are used

        Returns:
            scalar loss
        """
        assert student_tokens.shape == teacher_tokens.shape
        B, K, M, D = student_tokens.shape

        # 1) token aggregation
        # (B, K, D)
        S = student_tokens.mean(dim=2)
        T = teacher_tokens.mean(dim=2)

        if weights is None:
            weights = [1.0 / len(scales)] * len(scales)

        loss = 0.0

        for k, w in zip(scales, weights):
            if k >= K:
                continue

            # (B, K-k, D)
            dS = S[:, k:] - S[:, :-k]
            dT = T[:, k:] - T[:, :-k]

            loss_k = F.mse_loss(dS, dT)
            loss = loss + w * loss_k

        return loss

    @staticmethod
    def multiscale_second_order_dynamic_loss_agg(
        student_tokens,
        teacher_tokens,
        scales=(1, 2),
        weights=None
    ):
        """
        Multi-scale 2nd-order dynamic loss (aggregation level)

        Args:
            student_tokens: (B, K, M, D)
            teacher_tokens: (B, K, M, D)
            scales: k values, e.g. (1, 2)
            weights: same length as scales; if None, uniform

        Returns:
            scalar loss
        """
        assert student_tokens.shape == teacher_tokens.shape
        B, K, M, D = student_tokens.shape

        # 1) token aggregation
        # (B, K, D)
        S = student_tokens.mean(dim=2)
        T = teacher_tokens.mean(dim=2)

        if weights is None:
            weights = [1.0 / len(scales)] * len(scales)

        loss = 0.0

        for k, w in zip(scales, weights):
            # need t, t+k, t+2k
            if 2 * k >= K:
                continue

            # (B, K-2k, D)
            ddS = S[:, 2*k:] - 2 * S[:, k:-k] + S[:, :-2*k]
            ddT = T[:, 2*k:] - 2 * T[:, k:-k] + T[:, :-2*k]

            loss_k = F.mse_loss(ddS, ddT)
            loss = loss + w * loss_k

        return loss

    def forward(self, data):
        uni_embeds, slots = self.encoder(data['csi'])
        har = self.har_head(slots)
        har_loss = self.har_loss(har, data['act']) * self.har_weight

        z, log_det_sum, flow_loss = self.flow(uni_embeds)
        flow_loss = flow_loss * self.flow_weight

        with torch.no_grad():
            t_ret, t_loss = self.teacher(data)
        
        first_order_loss = self.multiscale_first_order_dynamic_loss_agg(uni_embeds, t_ret['TOKEN']) * self.firstorder_weight
        second_order_loss = self.multiscale_second_order_dynamic_loss_agg(uni_embeds, t_ret['TOKEN']) * self.secondorder_weight

        loss = first_order_loss + second_order_loss + har_loss
        
        LOSS = {
            'LOSS': loss,
            'FLOW': flow_loss,
            '1-DYNAMIC': first_order_loss,
            '2-DYNAMIC': second_order_loss,
            'HAR': har_loss
        }

        RET = {
            'TOKEN': uni_embeds,
            'HAR': har
        }

        return RET, LOSS

# Decoder?