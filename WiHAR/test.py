import torch
from Teacher import ViTVideoEncoder, ViTVideoAutoencoder

import numpy as np
import torch.nn as nn
from PoolingFlow import SpatialLatentFlow

def test_vit_slot_encoder():
    print("Testing ViTVideoEncoder...")
    
    # Configuration
    B = 2
    C, H, W = 1, 128, 128
    D = 512
    model = ViTVideoEncoder(img_size=H, embed_dim=D, num_slots=4)
    model.eval()

    # Test 1: Fixed number of frames
    print("\nTest 1: Fixed T=5")
    T1 = 5
    x1 = torch.randn(B, T1, C, H, W)
    with torch.no_grad():
        out1 = model(x1)
    print(f"Input: {x1.shape}, Output: {out1.shape}")
    assert out1.shape == (B, 4, D)

    # Test 2: Different number of frames
    print("\nTest 2: Fixed T=10")
    T2 = 10
    x2 = torch.randn(B, T2, C, H, W)
    with torch.no_grad():
        out2 = model(x2)
    print(f"Input: {x2.shape}, Output: {out2.shape}")
    assert out2.shape == (B, 4, D)

    print("\nEncoder tests passed!")

def test_vit_autoencoder():
    print("\nTesting ViTVideoAutoencoder (with SimpleTimeDecoder - Discrete)...")
    B = 2
    T = 6
    C, H, W = 1, 128, 128
    D = 512
    
    model = ViTVideoAutoencoder(img_size=H, embed_dim=D, num_slots=4)
    model.eval()
    
    x = torch.randn(B, T, C, H, W)
    t = torch.randint(0, 6, (B,)) # Random discrete time indices [0, 5]
    
    with torch.no_grad():
        recon, slots = model(x, t)
        
    print(f"Input Video: {x.shape}")
    print(f"Target Times: {t.shape}")
    print(f"Reconstructed Frame: {recon.shape}")
    print(f"Slots: {slots.shape}")
    
    assert recon.shape == (B, C, H, W)
    assert slots.shape == (B, 4, D)
    print("Autoencoder tests passed!")


def test_flow():
    print("Testing SpatialLatentFlow Module...")
    B, N, D = 4, 196, 64 # Use smaller dim for speed
    M = 8
    
    # Initialize implementation
    # Note: flow_hidden_dim is usually keeping D or 2*D.
    model = SpatialLatentFlow(input_dim=D, num_tokens=M, flow_depth=4, flow_hidden_dim=D)
    
    x = torch.randn(B, N, D)
    
    print(f"Input Shape: {x.shape}")
    
    # Forward
    z, log_det, pooled = model(x)
    
    print(f"Latent Shape: {z.shape}")
    print(f"Pooled Shape: {pooled.shape}")
    print(f"LogDet Shape: {log_det.shape}")
    
    expected_dim = M * D
    assert z.shape == (B, expected_dim)
    assert pooled.shape == (B, M, D)
    
    # Check Invertibility for the Flow part
    # Inverse takes z -> reconstructed flat pooled tokens
    recon_pooled = model.inverse(z)
    print(f"Reconstructed Pooled Shape: {recon_pooled.shape}")
    
    # Check difference between pooling output and reconstruction
    # Should be close to numerical precision
    diff = (pooled - recon_pooled).abs().max().item()
    print(f"Max reconstruction error: {diff}")
    
    if diff < 1e-4:
        print("Invertibility Test Passed!")
    else:
        print("Invertibility Test Failed (error too high)")
        
    print("-" * 20)



if __name__ == "__main__":
    test_vit_slot_encoder()
    test_vit_autoencoder()
    test_flow()