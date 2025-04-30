import torch
import torch.nn as nn

class ImageAttentionPooler(nn.Module):
    def __init__(self, patch_size=8, embed_dim=128, num_heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding: flatten 8x8 patches (1 * 8 * 8 = 64) to embed_dim
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.patch_proj = nn.Linear(patch_size * patch_size, embed_dim)

        # CLS token and positional encoding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, (128 // patch_size)**2 + 1, embed_dim))  # +1 for CLS

        # Attention pooling
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape  # should be (B, 1, 128, 128)
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image size must be divisible by patch size"

        # Step 1: Patchify
        patches = self.unfold(x)  # shape: (B, patch_dim, num_patches)
        patches = patches.permute(0, 2, 1)  # (B, num_patches, patch_dim)

        # Step 2: Linear projection
        tokens = self.patch_proj(patches)  # (B, num_patches, embed_dim)

        # Step 3: Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, num_patches+1, embed_dim)

        # Step 4: Positional embedding
        tokens = tokens + self.pos_embed[:, :tokens.size(1)]

        # Step 5: Attention pooling (CLS token attends to all others)
        cls_output, _ = self.attn(tokens[:, :1, :], tokens, tokens)  # Q=CLS, K=V=tokens
        return cls_output.squeeze(1)  # (B, embed_dim)
    

class VAEImageEncoder(nn.Module):
    def __init__(self, patch_size=8, embed_dim=128, latent_dim=32, num_heads=4):
        super().__init__()
        self.pooler = ImageAttentionPooler(patch_size, embed_dim, num_heads)
        
        self.to_mu = nn.Linear(embed_dim, latent_dim)
        self.to_logvar = nn.Linear(embed_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        pooled = self.pooler(x)              # shape: (B, embed_dim)
        mu = self.to_mu(pooled)              # (B, latent_dim)
        logvar = self.to_logvar(pooled)      # (B, latent_dim)
        z = self.reparameterize(mu, logvar)  # (B, latent_dim)
        return z, mu, logvar
    