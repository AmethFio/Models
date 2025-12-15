import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from WiDepthLite.TrainerLite import ModelTrainer

"""
Teacher = TimeSformer
loss = RECON

Student = (Raw+FFT)CNN + CONDFLOW
loss = FLOW + MMD + RECON

StudentPretrain = (Raw+FFT)CNN + CNN
loss = RECON + FFT
"""

# ----------------------------
# Helper: simple phase unwrap (per-sample, per-channel, along dim)
# ----------------------------
def unwrap_phase(phase, dim=-1):
    """
    phase: tensor (..., L)
    returns unwrapped phase along dim
    """
    diff = torch.diff(phase, dim=dim)
    # wrap jump > pi or < -pi
    jump = (diff > math.pi).to(phase.dtype) * (-2 * math.pi) + (diff < -math.pi).to(phase.dtype) * (2 * math.pi)
    # cumulative sum of jumps to correct
    # pad 0 to match length
    jump = torch.cat([torch.zeros_like(jump.take(indices=torch.tensor([0]), dim=dim)), jump], dim=dim)
    corr = torch.cumsum(jump, dim=dim)
    return phase + corr

#
# Temporal Attention Pooling for fea_global
#
class TemporalAttentionPooling(nn.Module):
    """
    输入:  x -> (B, C, T)
    输出: out -> (B, C)
    原理: 对时间维进行 attention，加权求和
    """

    def __init__(self, C, hidden_dim=128):
        super().__init__()

        # 注意力网络：C -> hidden -> 1（对每个时间点生成一个权重）
        self.att_mlp = nn.Sequential(
            nn.Conv1d(C, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, x):
        """
        x: (B, C, T)
        """
        # 得到注意力得分: (B, 1, T)
        att_logits = self.att_mlp(x)

        # softmax 得到归一化权重
        att_weight = F.softmax(att_logits, dim=-1)   # (B, 1, T)

        # 加权求和得到全局特征
        out = (x * att_weight).sum(dim=-1)           # (B, C)

        return out


# ----------------------------
# CSI Encoder: raw + FFT branches -> fusion -> temporal 1D conv
# ----------------------------
class CSIEncoder2D1D(nn.Module):
    """
    Expected input formats (choose one):
    - x shape: (B, C, Ns, Packets), where C = 2 * Rx * Tx (2 for amp/phase per antenna pair)
      e.g. if Tx=1,Rx=3 -> C=6 (amp/phase per Rx flattened)
      if Tx=2,Rx=3 -> C=12

    - or x shape: (B, Tx, Rx, 2, Ns, Packets)  (if you keep explicit Tx/Rx/chan dims)
      This constructor will flatten it to (B, C, Ns, Packets) automatically.

    Outputs:
      feat_time: (B, out_dim, T_out)  # temporal feature sequence
      feat_global: (B, out_dim)       # temporal-pooled global vector (optional)
      feat2d_fused: (B, Cf, Hf, Tf)   # fused spatio-temporal 2D feature (if keep spatial)
    """
    def __init__(self,
                 out_dim=256,   # Recommended 512 or 768
                 base_channels=32,
                 use_fft=True,
                 target_T=10,    # desired temporal length to align to (interpolation possible)
                 ns_fft_len=None  # if you want zero-pad FFT along subcarrier to get higher res
                 ):
        super().__init__()
        self.use_fft = use_fft
        self.out_dim = out_dim
        self.target_T = target_T
        self.ns_fft_len = ns_fft_len

        self.tap = None

        # --- raw-branch 2D convs (input channels = C_raw) ---
        # We'll build small conv stack; final spatial dims depend on Ns/Packets and strides
        self.raw_conv = nn.Sequential(
            nn.Conv2d(in_channels= None, out_channels=base_channels, kernel_size=(3,3), padding=(1,1), stride=(1,2)),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels*2, kernel_size=(3,3), padding=(1,1), stride=(2,2)),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=(3,3), padding=(1,1), stride=(2,2)),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
        )

        # --- fft-branch 2D convs (input channels = C_fft) ---
        self.fft_conv = nn.Sequential(
            nn.Conv2d(in_channels=None, out_channels=base_channels, kernel_size=(3,3), padding=(1,1), stride=(1,2)),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels*2, kernel_size=(3,3), padding=(1,1), stride=(2,2)),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=(3,3), padding=(1,1), stride=(2,2)),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
        )

        self.att_fc = nn.Sequential(

        )

        # after concat of raw+fft, fuse channels
        fused_channels = base_channels*4 * (1 + int(self.use_fft))
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels, base_channels*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True)
        )

        # temporal 1D conv stack: input channels = base_channels*8 * H_spatial
        # We'll create after seeing a forward pass (so dynamic), but define layers here as modules to use once initialized
        self.temporal = None
        self._temporal_inited = False

        # final projection if need global feature
        self.project = nn.Linear(512, out_dim)  # placeholder; will re-init on first forward if needed

    def _init_temporal(self, C_spatial, H_spatial):
        """
        C_spatial: fuse_conv 输出的 channel 数
        H_spatial: fuse_conv 输出的空间 dim（高度）
        
        输入给 temporal conv 的形状为:
            B, (C_spatial * H_spatial), T
        """

        in_ch = C_spatial * H_spatial
        hidden = max(in_ch // 2, 256)

        # -------- 1. Temporal Conv Module -------- #
        # 不 flatten C 和 H，在通道方向直接拼成 in_ch
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv1d(hidden, self.out_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(self.out_dim),
            nn.ReLU(inplace=True)
        )

        # -------- 2. Temporal Attention Pooling -------- #
        # attention 输入的是 conv 输出的通道维（self.out_dim），
        # 每个时间步得到一个 attention logit，然后 softmax。
        self.temporal_attn = nn.Sequential(
            nn.Conv1d(self.out_dim, self.out_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.out_dim // 2, 1, kernel_size=1)  # 输出 (B, 1, T')
        )

        # 输出 proj（可选）
        self.project = nn.Linear(self.out_dim, self.out_dim)

        self._temporal_inited = True

    def _prepare_input(self, x):
        """
        Accept either:
          (B, C, Ns, Packets)
        or
          (B, Tx, Rx, 2, Ns, Packets)  where 2 = (amp, phase)
        Return:
          real_imag_raw: (B, C_raw, Ns, Packets)   # channels=2*Tx*Rx
          fft_amp_phase: (B, C_fft, Ns_fft, Packets)  (or Ns_fft same as Ns)
        """
        B = x.shape[0]
        if x.dim() == 6:
            # (B, Tx, Rx, 2, Ns, Packets) -> flatten Tx,Rx,2 into channel
            B, Tx, Rx, two, Ns, P = x.shape
            assert two == 2
            x = x.view(B, Tx * Rx * 2, Ns, P)  # channels = Tx*Rx*2

        # now x is (B, C_chan, Ns, P)
        # assume channel order: for each antenna pair (amp, phase) flattened
        # build real/imag raw:
        # x channels: [amp1, phase1, amp2, phase2, ...]  or user can give other order
        C = x.shape[1]
        assert C % 2 == 0, "expected pairs of (amp,phase) per antenna"
        pair = C // 2
        amps = x[:, 0::2, :, :]   # (B, pair, Ns, P)
        phases = x[:, 1::2, :, :] # (B, pair, Ns, P)

        # normalize amplitude & phase (phase in [-pi, pi] assumed)
        eps = 1e-6
        amps_norm = (amps - amps.mean(dim=(-2, -1), keepdim=True)) / (amps.std(dim=(-2, -1), keepdim=True) + eps)
        phases_unwrap = unwrap_phase(phases, dim=2)  # unwrap along Ns
        phases_norm = phases_unwrap / math.pi  # normalize to about [-1,1]

        # raw real/imag
        real = amps_norm * torch.cos(phases_norm * math.pi)
        imag = amps_norm * torch.sin(phases_norm * math.pi)
        real_imag = torch.cat([real, imag], dim=1)  # (B, 2*pair, Ns, P)

        # FFT along subcarrier axis (Ns) -> get delay domain features
        # build complex from amps & phase for FFT:
        complex_signal = amps * torch.cos(phases) + 1j * (amps * torch.sin(phases))
        # optional zero-pad for higher resolution
        ns = complex_signal.shape[2]
        if self.ns_fft_len is None:
            ns_fft = ns
        else:
            ns_fft = self.ns_fft_len
        # compute FFT along subcarrier axis (dim=2)
        fft = torch.fft.fft(complex_signal, n=ns_fft, dim=2)  # (B, pair, ns_fft, P), complex
        fft_amp = torch.abs(fft)
        fft_phase = torch.angle(fft)
        fft_phase_unwrap = unwrap_phase(fft_phase, dim=2)
        fft_phase_norm = fft_phase_unwrap / math.pi
        # normalize fft_amp
        fft_amp_norm = (fft_amp - fft_amp.mean(dim=(-2, -1), keepdim=True)) / (fft_amp.std(dim=(-2, -1), keepdim=True) + 1e-6)

        # stack FFT channels as [amp, phase]
        fft_ch = torch.cat([fft_amp_norm, fft_phase_norm], dim=1)  # (B, 2*pair, ns_fft, P)

        return real_imag, fft_ch

    def forward(self, x):
        """
        x: either (B, C, Ns, Packets) or (B, Tx, Rx, 2, Ns, Packets)
        returns:
          feat_time: (B, out_dim, T_out)
          feat_global: (B, out_dim)
          feat2d_fused: (B, C_fused, Hf, Tf)
        """
        # prepare channels
        real_imag, fft_ch = self._prepare_input(x)  # (B, C_raw, Ns, P), (B, C_fft, Ns_fft, P)

        # run raw branch
        B = real_imag.shape[0]

        # dynamic init of raw_conv / fft_conv input channels if needed
        # (Conv2d layers were created with in_channels=None placeholders earlier: set them now)
        if getattr(self.raw_conv[0], 'in_channels', None) is None or self.raw_conv[0].in_channels != real_imag.shape[1]:
            # re-create raw_conv with correct in_channels
            in_ch = real_imag.shape[1]
            self.raw_conv = nn.Sequential(
                nn.Conv2d(in_ch, self.raw_conv[0].out_channels, kernel_size=(3,3), padding=(1,1), stride=(1,2)),
                nn.BatchNorm2d(self.raw_conv[0].out_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(self.raw_conv[0].out_channels, self.raw_conv[3].out_channels, kernel_size=(3,3), padding=(1,1), stride=(2,2)),
                nn.BatchNorm2d(self.raw_conv[3].out_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(self.raw_conv[3].out_channels, self.raw_conv[6].out_channels, kernel_size=(3,3), padding=(1,1), stride=(2,2)),
                nn.BatchNorm2d(self.raw_conv[6].out_channels),
                nn.ReLU(inplace=True),
            )

        if self.use_fft:
            if getattr(self.fft_conv[0], 'in_channels', None) is None or self.fft_conv[0].in_channels != fft_ch.shape[1]:
                in_ch2 = fft_ch.shape[1]
                self.fft_conv = nn.Sequential(
                    nn.Conv2d(in_ch2, self.fft_conv[0].out_channels, kernel_size=(3,3), padding=(1,1), stride=(1,2)),
                    nn.BatchNorm2d(self.fft_conv[0].out_channels),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(self.fft_conv[0].out_channels, self.fft_conv[3].out_channels, kernel_size=(3,3), padding=(1,1), stride=(2,2)),
                    nn.BatchNorm2d(self.fft_conv[3].out_channels),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(self.fft_conv[3].out_channels, self.fft_conv[6].out_channels, kernel_size=(3,3), padding=(1,1), stride=(2,2)),
                    nn.BatchNorm2d(self.fft_conv[6].out_channels),
                    nn.ReLU(inplace=True),
                )

        feat_raw = self.raw_conv(real_imag)        # (B, C1, H1, T1)
        if self.use_fft:
            feat_fft = self.fft_conv(fft_ch)       # (B, C2, H2, T2)
            # if H/T differ slightly, we could center-crop or interpolate; here we'll align T by interpolation
            # unify spatial dims by interpolation to the minimum common size
            # simple strategy: downsample larger T to smaller T via interpolation
            # find spatial shapes
            _, C1, H1, T1 = feat_raw.shape
            _, C2, H2, T2 = feat_fft.shape
            # align time dimension by interpolation (linear on last dim)
            if T1 != T2:
                if T2 > T1:
                    feat_fft = F.interpolate(feat_fft, size=(H2, T1), mode='bilinear', align_corners=False)
                    _, C2, H2, T1 = feat_fft.shape
                else:
                    feat_raw = F.interpolate(feat_raw, size=(H1, T2), mode='bilinear', align_corners=False)
                    _, C1, H1, T2 = feat_raw.shape

            # now we can concat on channel dimension (after possible spatial align)
            feat_cat = torch.cat([feat_raw, feat_fft], dim=1)  # (B, C1+C2, H_common, T_common)
        else:
            feat_cat = feat_raw

        feat_fused = self.fuse_conv(feat_cat)  # (B, C_fuse, Hf, Tf)
        B, Cf, Hf, Tf = feat_fused.shape

        # prepare temporal conv init if not inited
        if not self._temporal_inited:
            self._init_temporal(C_spatial=Cf, H_spatial=Hf)

        # reshape for temporal conv: collapse channel and spatial height dims -> (B, Cf*Hf, Tf)
        feat_for_time = feat_fused.view(B, Cf * Hf, Tf)  # (B, in_ch, T)
        feat_time = self.temporal_conv(feat_for_time)    # (B, out_dim, T')

        # optional: temporal interpolation to target_T
        if self.target_T is not None and feat_time.shape[-1] != self.target_T:
            feat_time = F.interpolate(feat_time, size=self.target_T, mode='linear', align_corners=False)

        attn_logits = self.temporal_attn(feat_time)    # (B, 1, T')
        attn = torch.softmax(attn_logits, dim=-1)  # (B, 1, T')
        feat_global = (x * attn).sum(dim=-1)      # (B, out_dim)

        # Mean pooling is time invariant, replaced by attn
        # feat_global = feat_time.mean(dim=-1)  # (B, out_dim) 
        
        feat_global = self.project(feat_global)

        return {
            'feat_time': feat_time,         # (B, out_dim, T_out or target_T)
            'feat_global': feat_global,     # (B, out_dim)
            'feat2d_fused': feat_fused      # (B, Cf, Hf, Tf)
        }


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Example: Tx=1, Rx=3, amp/phase -> channels = 2*1*3 = 6
    B = 2
    Tx = 1
    Rx = 3
    Ns = 30
    P = 300
    # create dummy input in shape (B, Tx, Rx, 2, Ns, P)
    x = torch.randn(B, Tx, Rx, 2, Ns, P) * 0.1
    enc = CSIEncoder2D1D(out_dim=256, base_channels=32, use_fft=True, target_T=10)
    out = enc(x)
    print("feat_time", out['feat_time'].shape)      # (B, out_dim, target_T)
    print("feat_global", out['feat_global'].shape)  # (B, out_dim)
    print("feat2d_fused", out['feat2d_fused'].shape) # (B, Cf, Hf, Tf)