import torch
import torch.nn as nn

class SpatialLearnablePooling(nn.Module):
    """
    Learns a fixed number of tokens from variable spatial inputs using cross-attention.
    """
    def __init__(self, dim, num_tokens=8, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.queries = nn.Parameter(torch.zeros(1, num_tokens, dim))
        
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.queries, std=.02)

    def forward(self, x):
        # x: (B, N, D)
        B, N, C = x.shape
        
        q = self.to_q(self.norm_q(self.queries)).expand(B, -1, -1)
        
        kv = self.to_kv(self.norm_kv(x))
        k, v = kv.chunk(2, dim=-1)
        
        q = q.reshape(B, self.num_tokens, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, self.num_tokens, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TemporalSlotAttention(nn.Module):
    """
    Aggregates tokens from variable number of frames into fixed number of slots.
    """
    def __init__(self, dim=512, num_slots=4, num_heads=8):
        super().__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.dim = dim

        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, num_slots, dim))
        
        # Uses cross-attention: Q comes from slots, K,V come from encoded video tokens
        # Standard implementation often uses LayerNorm on queries and inputs
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_input = nn.LayerNorm(dim)
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        self.proj = nn.Linear(dim, dim)
        
        # Initialize slots
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.proj.weight)


    def forward(self, x):
        # x: (B, Total_Tokens, D) - Flattened over time and space
        B, N, D = x.shape
        
        # Initialize slots for the batch
        slots = self.slots_mu.expand(B, -1, -1) + \
                self.slots_log_sigma.exp().expand(B, -1, -1) * torch.randn_like(self.slots_mu.expand(B, -1, -1))
        
        q = self.to_q(self.norm_slots(slots)).reshape(B, self.num_slots, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        k = self.to_k(self.norm_input(x)).reshape(B, N, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        v = self.to_v(self.norm_input(x)).reshape(B, N, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        
        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1) # (B, H, S, N)
        
        out = (attn @ v).transpose(1, 2).reshape(B, self.num_slots, D)
        out = self.proj(out)
        
        return out


class AffineCoupling(nn.Module):
    """
    Affine Coupling Layer for RealNVP.
    Splits input into two halves. Transforms strictly the second half based on the first.
    """
    def __init__(self, dim, hidden_dim=512):
        super().__init__()
        self.dim = dim
        self.sub_dim = dim // 2
        
        # Simple MLP for s and t
        self.net = nn.Sequential(
            nn.Linear(self.sub_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * (dim - self.sub_dim)) # Output s and t
        )
        
        # Initialize final layer to zero for identity start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        # x: (B, Dim)
        x1, x2 = x[:, :self.sub_dim], x[:, self.sub_dim:]
        
        params = self.net(x1)
        s, t = params.chunk(2, dim=-1)
        
        # Stabilize s with tanh and scaling
        s = torch.tanh(s)
        
        y1 = x1
        y2 = x2 * torch.exp(s) + t
        
        y = torch.cat([y1, y2], dim=-1)
        log_det = torch.sum(s, dim=-1) # Sum over dimensions
        
        return y, log_det

    def inverse(self, y):
        y1, y2 = y[:, :self.sub_dim], y[:, self.sub_dim:]
        
        params = self.net(y1)
        s, t = params.chunk(2, dim=-1)
        s = torch.tanh(s)
        
        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)
        
        x = torch.cat([x1, x2], dim=-1)
        return x

class SpatialLatentFlow(nn.Module):
    """
    Module that inputs Embeddings, pools them to fixed tokens, 
    and projects them to a Gaussian via Normalizing Flow.
    """
    def __init__(self, input_dim=512, num_tokens=8, flow_depth=4, flow_hidden_dim=512):
        super().__init__()
        self.num_tokens = num_tokens
        self.input_dim = input_dim
        self.flat_dim = num_tokens * input_dim

        # Construct Flow
        self.flow_layers = nn.ModuleList()
        for i in range(flow_depth):
            self.flow_layers.append(AffineCoupling(self.flat_dim, hidden_dim=flow_hidden_dim))
            
        # Fixed Permutation (Reverse) to mix information between layers
        # In a real RealNVP, we'd use random or learned permutations. 
        # Here we just flip the channel order every other layer implicitly 
        # by how we handle the next coupling.
        # Actually simpler: standard implementation often swaps or shuffles.
        # Let's implement a simple reverse permutation block if needed, 
        # but for simplicity, we will just rely on the fact that we can 
        # re-order in the coupling? No, coupling is fixed split.
        # We MUST permute.
        
    def forward(self, x):
        """
        x: (B, N, D) or (B*T, N, D)
        Returns:
            z: (B, M*D) - Latent Gaussian vector
            log_det: (B,) - Log determinant of Jacobian
            pooled: (B, M, D) - Intermediate pooled tokens
        """
        # (B, M, D)
        B = x.shape[0]
        
        # Flatten
        h = x.view(B, -1) # (B, M*D)
        
        log_det_sum = 0
        
        for i, flow in enumerate(self.flow_layers):
            # Apply permutation (Reverse) before coupling, except first layer maybe?
            # Or after? Strategy: Flip -> Couple -> Flip -> Couple
            if i % 2 == 1:
                h = torch.flip(h, dims=[-1])
            
            h, ld = flow(h)
            log_det_sum = log_det_sum + ld
            
        z = h
        return z, log_det_sum, flow_loss
        
    def inverse(self, z):
        """
        Samples from Gaussian z -> Reconstructs flat tokens
        """
        h = z
        for i in reversed(range(len(self.flow_layers))):
            flow = self.flow_layers[i]
            
            h = flow.inverse(h)
            
            if i % 2 == 1:
                h = torch.flip(h, dims=[-1])
                
        return h.view(z.shape[0], self.num_tokens, self.input_dim)
