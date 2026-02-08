"""
train_final_working.py

FactorVAE256 사용 (공식 모델)
모델 불러오기, checkpoint적용 완벽. 
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from pathlib import Path
from tqdm import tqdm
import argparse

from MorphoGenie.dVAE.model import FactorVAE256

# ============================================================
# Load VAE
# ============================================================

def load_vae(checkpoint_path, z_dim=10, nc=3, device='cuda'):
    """
    FactorVAE256 checkpoint 로드
    """
    
    print("="*70)
    print("Loading FactorVAE256")
    print("="*70)
    
    # Checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_states']['VAE']
    
    # Model
    vae = FactorVAE256(z_dim=z_dim, nc=nc).to(device)
    
    # Load (strict=False, decoder missing OK)
    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    
    print(f"✓ Loaded FactorVAE256")
    print(f"  Missing: {len(missing)} (decoder - OK)")
    print(f"  Unexpected: {len(unexpected)}")
    
    # Check encoder
    encoder_keys = [k for k in state_dict.keys() if k.startswith('encode')]
    model_encoder_keys = [k for k in vae.state_dict().keys() if k.startswith('encode')]
    
    matched = 0
    for key in encoder_keys:
        if key in vae.state_dict():
            if state_dict[key].shape == vae.state_dict()[key].shape:
                matched += 1
    
    print(f"  Encoder: {matched}/{len(encoder_keys)} layers matched")
    
    if matched < len(encoder_keys):
        print("\n⚠️  Warning: Not all encoder layers matched!")
        for key in encoder_keys:
            if key in vae.state_dict():
                if state_dict[key].shape != vae.state_dict()[key].shape:
                    print(f"  Mismatch: {key}")
                    print(f"    Checkpoint: {state_dict[key].shape}")
                    print(f"    Model: {vae.state_dict()[key].shape}")
    else:
        print(f"\n✓ All encoder layers loaded correctly!")
    
    # Freeze
    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()
    
    return vae


# ============================================================
# Extract Features
# ============================================================

def extract_features(vae, crops, device):
    """Features 추출"""
    
    print("\n" + "="*70)
    print("Extracting Features")
    print("="*70)
    
    # RGB conversion
    crops_tensor = torch.from_numpy(crops).float() / 255.0
    crops_tensor = crops_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
    
    print(f"Input: {crops_tensor.shape}")
    
    dataset = TensorDataset(crops_tensor)
    dataloader = DataLoader(
        dataset, batch_size=512, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    all_z = []
    
    with torch.no_grad():
        for (batch,) in tqdm(dataloader, desc="Extracting"):
            batch = batch.to(device, non_blocking=True)
            
            # VAE forward (no_dec=True → encoder only)
            z = vae(batch, no_dec=True)
            all_z.append(z.cpu().numpy())
    
    z_all = np.vstack(all_z)
    
    print(f"✓ Features: {z_all.shape}")
    
    return z_all


# ============================================================
# Analysis
# ============================================================

def analyze(z_all, is_target):
    """분석"""
    
    z_t = z_all[is_target]
    z_nt = z_all[~is_target]
    
    print(f"\nTarget ({len(z_t)}):")
    print(f"  Mean: {z_t.mean(axis=0)}")
    print(f"  Std: {z_t.std(axis=0)}")
    
    print(f"\nNon-target ({len(z_nt)}):")
    print(f"  Mean: {z_nt.mean(axis=0)}")
    print(f"  Std: {z_nt.std(axis=0)}")
    
    dist = np.linalg.norm(z_t.mean(axis=0) - z_nt.mean(axis=0))
    
    print(f"\n{'='*70}")
    print(f"Distance: {dist:.6f}")
    print(f"{'='*70}")
    
    if dist > 1.0:
        print("✓ EXCELLENT separation!")
    elif dist > 0.5:
        print("✓ GOOD separation")
    elif dist > 0.1:
        print("○ Moderate separation")
    else:
        print("✗ WEAK separation - check model!")
    
    return dist


# ============================================================
# K-means
# ============================================================

def kmeans_clustering(z_all, is_target, K=3):
    """K-means"""
    
    print(f"\n{'='*70}")
    print(f"K-means Clustering (K={K})")
    print(f"{'='*70}")
    
    pseudo_labels = np.zeros(len(z_all), dtype=int)
    pseudo_labels[is_target] = 0
    
    if is_target.sum() < len(is_target):
        kmeans = KMeans(n_clusters=K-1, random_state=42, n_init=10)
        pseudo_labels[~is_target] = kmeans.fit_predict(z_all[~is_target]) + 1
    
    print(f"\nCluster distribution:")
    for k in range(K):
        count = (pseudo_labels == k).sum()
        center = z_all[pseudo_labels == k].mean(axis=0)
        print(f"  Class {k}: {count:,} cells")
        print(f"    Center: {center}")
    
    # Distances
    print(f"\nCluster distances:")
    centers = [z_all[pseudo_labels == k].mean(axis=0) for k in range(K)]
    
    for i in range(K):
        for j in range(i+1, K):
            d = np.linalg.norm(centers[i] - centers[j])
            print(f"  Class {i} <-> Class {j}: {d:.6f}")
    
    return pseudo_labels


# ============================================================
# Main
# ============================================================

def main(args):
    device = torch.device('cuda')
    
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Data
    crops = np.load(f'{args.data_dir}/crops.npy')
    is_target = np.load(f'{args.data_dir}/is_target.npy')
    
    print(f"\nData:")
    print(f"  Crops: {crops.shape}")
    print(f"  Target: {is_target.sum():,} / {len(is_target):,}")
    
    # Load VAE
    vae = load_vae(args.vae_checkpoint, z_dim=args.z_dim, nc=3, device=device)
    
    # Extract
    z_all = extract_features(vae, crops, device)
    
    # Analyze
    dist = analyze(z_all, is_target)
    
    # K-means
    pseudo_labels = kmeans_clustering(z_all, is_target, K=args.K)
    
    # Save
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    np.save(f'{args.output_dir}/features.npy', z_all)
    np.save(f'{args.output_dir}/pseudo_labels.npy', pseudo_labels)
    
    print(f"\n✓ Saved to {args.output_dir}/")
    
    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    
    if dist > 0.5:
        print("\n✓ Good separation - ready for training!")
    else:
        print("\n⚠️  Weak separation - may need improvement")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--vae-checkpoint', 
                       default='./checkpoint/CCy/VAE/chkpts/last')
    parser.add_argument('--output-dir', default='./vae_features')
    parser.add_argument('--z-dim', type=int, default=10)
    parser.add_argument('--K', type=int, default=3)
    
    args = parser.parse_args()
    main(args)