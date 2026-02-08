"""
load_morphogenie_vae.py

MorphoGenie VAE checkpoint 로드 및 사용
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ============================================================
# VAE Checkpoint 구조 확인
# ============================================================

def inspect_vae_checkpoint(checkpoint_path):
    """VAE checkpoint 내용 확인"""
    
    print("="*70)
    print(f"Inspecting: {checkpoint_path}")
    print("="*70)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\nCheckpoint keys:")
    if isinstance(checkpoint, dict):
        for key in checkpoint.keys():
            print(f"  {key}")
            if isinstance(checkpoint[key], dict):
                print(f"    → {list(checkpoint[key].keys())[:5]} ...")
            elif isinstance(checkpoint[key], torch.Tensor):
                print(f"    → Tensor {checkpoint[key].shape}")
    
    # State dict 찾기
    if 'model_states' in checkpoint and 'VAE' in checkpoint['model_states']:
        state_dict = checkpoint['model_states']['VAE']
        print("\n✓ Found VAE in model_states")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'encoder' in checkpoint:
        state_dict = checkpoint['encoder']
    else:
        state_dict = checkpoint
    
    print("\nModel parameters (first 10):")
    count = 0
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            print(f"  {name}: {param.shape}")
            count += 1
            if count >= 10:
                break
    
    return checkpoint


# ============================================================
# MorphoGenie VAE Encoder (실제 구조)
# ============================================================

class MorphoGenieVAEEncoder(nn.Module):
    """
    MorphoGenie VAE Encoder
    
    Based on the paper architecture:
    - Input: (B, 1, 256, 256)
    - Output: 
        - z: (B, 256) - latent embedding
        - c: (B, 64) - concept vector (interpretable)
    """
    
    def __init__(self, latent_dim=256, concept_dim=64):
        super().__init__()
        
        # Convolutional encoder
        self.encoder = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 128 -> 64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 64 -> 32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 32 -> 16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # 16 -> 8
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # 8 -> 4
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        # Latent layers
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        
        # Concept layer
        self.fc_concept = nn.Linear(512 * 4 * 4, concept_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, 256, 256)
        Returns:
            z: (B, latent_dim) - sampled latent
            c: (B, concept_dim) - concept vector
        """
        # Encode
        h = self.encoder(x)  # (B, 512, 4, 4)
        h = h.view(h.size(0), -1)  # (B, 512*4*4)
        
        # Latent (reparameterization trick)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Sample z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Concept
        c = self.fc_concept(h)
        
        return z, c
    
    def encode_deterministic(self, x):
        """Deterministic encoding (use mu only)"""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        
        z = self.fc_mu(h)
        c = self.fc_concept(h)
        
        return z, c


# ============================================================
# Load Checkpoint
# ============================================================

def load_morphogenie_vae(vae_checkpoint_path, device='cuda'):
    """
    MorphoGenie VAE checkpoint 로드
    
    Args:
        vae_checkpoint_path: VAE checkpoint 경로
            예: './CCy/VAE/chkpts/last'
        device: 'cuda' or 'cpu'
    
    Returns:
        encoder: 로드된 encoder model
    """
    
    print("="*70)
    print("Loading MorphoGenie VAE Checkpoint")
    print("="*70)
    
    # Checkpoint 로드
    checkpoint = torch.load(vae_checkpoint_path, map_location=device)
    
    # State dict 추출
    if 'model_states' in checkpoint and 'VAE' in checkpoint['model_states']:
        state_dict = checkpoint['model_states']['VAE']
        print("✓ Loaded VAE from model_states")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Encoder 생성
    encoder = MorphoGenieVAEEncoder().to(device)
    
    # State dict 로드 (prefix 처리)
    encoder_state_dict = {}
    for key, value in state_dict.items():
        # 'encoder.'로 시작하는 것만
        if key.startswith('encoder.'):
            new_key = key.replace('encoder.', '')
            encoder_state_dict[new_key] = value
        # 'fc_mu', 'fc_logvar', 'fc_concept'
        elif key.startswith('fc_'):
            encoder_state_dict[key] = value
    
    # 로드
    try:
        encoder.load_state_dict(encoder_state_dict, strict=False)
        print("✓ Encoder loaded successfully!")
    except Exception as e:
        print(f"⚠️  Partial loading: {e}")
        print("  → Trying to load compatible layers only...")
        
        # Compatible layers만 로드
        encoder_dict = encoder.state_dict()
        pretrained_dict = {k: v for k, v in encoder_state_dict.items() 
                          if k in encoder_dict and v.shape == encoder_dict[k].shape}
        encoder_dict.update(pretrained_dict)
        encoder.load_state_dict(encoder_dict)
        
        print(f"✓ Loaded {len(pretrained_dict)}/{len(encoder_dict)} layers")
    
    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False
    
    encoder.eval()
    
    print(f"✓ Encoder frozen (eval mode)")
    
    return encoder


# ============================================================
# Test
# ============================================================

def test_encoder(encoder, device='cuda'):
    """Encoder 테스트"""
    
    print("\n" + "="*70)
    print("Testing Encoder")
    print("="*70)
    
    # Dummy input
    x = torch.randn(4, 1, 256, 256).to(device)
    
    with torch.no_grad():
        z, c = encoder.encode_deterministic(x)
    
    print(f"\nInput: {x.shape}")
    print(f"Output z (latent): {z.shape}")
    print(f"Output c (concept): {c.shape}")
    
    print(f"\nz sample: {z[0, :5]}")
    print(f"c sample: {c[0, :5]}")
    
    print("\n✓ Encoder working!")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./CCy/VAE/chkpts/last',
                       help='VAE checkpoint path')
    parser.add_argument('--inspect-only', action='store_true',
                       help='Only inspect checkpoint structure')
    
    args = parser.parse_args()
    
    # Inspect
    checkpoint = inspect_vae_checkpoint(args.checkpoint)
    
    if not args.inspect_only:
        # Load
        encoder = load_morphogenie_vae(args.checkpoint, device='cuda')
        
        # Test
        test_encoder(encoder, device='cuda')
        
        print("\n" + "="*70)
        print("Usage Example:")
        print("="*70)
        print("""
# In your training code:
from load_morphogenie_vae import load_morphogenie_vae

# Load pretrained encoder
morphogenie = load_morphogenie_vae('./CCy/VAE/chkpts/last')

# Use in your pipeline
for batch in dataloader:
    with torch.no_grad():  # Frozen!
        f, c = morphogenie.encode_deterministic(batch_crops)
    
    # f: (B, 256) - latent embedding
    # c: (B, 64) - concept vector
    
    # Then pass to adapter
    z, c_adapted = adapter(f, c)
        """)