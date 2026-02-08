"""
train_semisupervised_complete.py

FactorVAE256 + Semi-supervised Learning
Step 1-4: 완전 구현
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

from MorphoGenie.dVAE.model import FactorVAE256

# ============================================================
# Models
# ============================================================

class Adapter(nn.Module):
    """Residual Adapter (z_dim에 맞춤)"""
    def __init__(self, embedding_dim=10):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim, 32), 
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(32, embedding_dim)
        )
    
    def forward(self, z):
        return z + self.adapter(z)


class PrototypicalHead(nn.Module):
    """Prototypical classification"""
    def __init__(self, embedding_dim=10, num_classes=3):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim))
    
    def forward(self, z):
        distances = torch.cdist(z, self.prototypes)
        logits = -distances
        return logits


# ============================================================
# Load VAE
# ============================================================

def load_vae(checkpoint_path, z_dim=10, device='cuda'):
    """FactorVAE256 로드"""
    
    print("="*70)
    print("Loading FactorVAE256")
    print("="*70)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_states']['VAE']
    
    vae = FactorVAE256(z_dim=z_dim, nc=3).to(device)
    vae.load_state_dict(state_dict, strict=False)
    
    print("✓ FactorVAE256 loaded")
    
    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()
    
    return vae


# ============================================================
# Step 1: Feature Extraction
# ============================================================

def step1_extract_features(crops, is_target, vae, device):
    """VAE로 features 추출"""
    
    print("\n" + "="*70)
    print("Step 1: Feature Extraction")
    print("="*70)
    
    # RGB conversion
    crops_tensor = torch.from_numpy(crops).float() / 255.0
    crops_tensor = crops_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
    
    dataset = TensorDataset(crops_tensor)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    all_z = []
    
    with torch.no_grad():
        for (batch,) in tqdm(dataloader, desc="Extracting"):
            batch = batch.to(device, non_blocking=True)
            z = vae(batch, no_dec=True)
            all_z.append(z.cpu().numpy())
    
    z_all = np.vstack(all_z)
    
    print(f"✓ Features: {z_all.shape}")
    
    # Initialize pseudo labels
    pseudo_labels = np.zeros(len(z_all), dtype=int)
    pseudo_labels[is_target] = 0
    
    confidence = np.zeros(len(z_all))
    confidence[is_target] = 0.8
    
    return z_all, pseudo_labels, confidence


# ============================================================
# Step 2: K-means
# ============================================================

def step2_kmeans(z_all, is_target, pseudo_labels, confidence, K=3):
    """K-means clustering"""
    
    print("\n" + "="*70)
    print(f"Step 2: K-means Clustering (K={K})")
    print("="*70)
    
    z_nontarget = z_all[~is_target]
    
    if len(z_nontarget) == 0:
        return pseudo_labels, confidence
    
    kmeans = KMeans(n_clusters=K-1, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(z_nontarget)
    
    pseudo_labels = pseudo_labels.copy()
    pseudo_labels[~is_target] = cluster_labels + 1
    
    # Confidence based on distance
    confidence = confidence.copy()
    for i in range(K-1):
        mask = (cluster_labels == i)
        indices = np.where(~is_target)[0][mask]
        
        z_cluster = z_nontarget[mask]
        center = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(z_cluster - center, axis=1)
        
        max_dist = distances.max()
        conf = 1.0 - (distances / max_dist) * 0.5
        confidence[indices] = conf
    
    # Statistics
    print(f"\nCluster distribution:")
    for k in range(K):
        count = (pseudo_labels == k).sum()
        avg_conf = confidence[pseudo_labels == k].mean()
        print(f"  Class {k}: {count:,} cells (conf: {avg_conf:.3f})")
    
    # Distances
    print(f"\nCluster distances:")
    centers = [z_all[pseudo_labels == k].mean(axis=0) for k in range(K)]
    for i in range(K):
        for j in range(i+1, K):
            d = np.linalg.norm(centers[i] - centers[j])
            print(f"  Class {i} <-> Class {j}: {d:.4f}")
    
    return pseudo_labels, confidence


# ============================================================
# Step 3: Anchor Selection
# ============================================================

def step3_anchor_selection(z_all, pseudo_labels, percentile=10, 
                           output_dir='./anchors'):
    """Anchor 선정 & 시각화"""
    
    print("\n" + "="*70)
    print(f"Step 3: Anchor Selection (Top {percentile}%)")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    K = len(np.unique(pseudo_labels))
    anchor_indices = {}
    anchor_centers = []
    
    # Select anchors
    for k in range(K):
        mask = (pseudo_labels == k)
        z_k = z_all[mask]
        indices_k = np.where(mask)[0]
        
        # Center
        center = z_k.mean(axis=0)
        
        # Distances
        distances = np.linalg.norm(z_k - center, axis=1)
        
        # Top percentile
        threshold = np.percentile(distances, percentile)
        anchor_mask = distances <= threshold
        
        anchor_indices[k] = indices_k[anchor_mask]
        anchor_centers.append(z_k[anchor_mask].mean(axis=0))
        
        print(f"\nClass {k}:")
        print(f"  Total: {len(z_k):,}")
        print(f"  Anchors: {anchor_mask.sum():,} ({anchor_mask.mean()*100:.1f}%)")
        print(f"  Threshold: {threshold:.4f}")
    
    anchor_centers = np.array(anchor_centers)
    
    # Visualization
    visualize_anchors(z_all, pseudo_labels, anchor_indices, anchor_centers, 
                     output_dir, K, percentile)
    
    return anchor_indices, anchor_centers

def visualize_anchors(z_all, pseudo_labels, anchor_indices, anchor_centers,
                     output_dir, K, percentile):
    """Anchor 시각화"""
    
    print(f"\nVisualizing anchors...")
    
    # Sample
    n_sample = min(300, len(z_all))
    sample_idx = np.random.choice(len(z_all), n_sample, replace=False)
    z_sample = z_all[sample_idx]
    labels_sample = pseudo_labels[sample_idx]
    
    # NaN/Inf 제거
    valid_mask = np.all(np.isfinite(z_sample), axis=1)
    z_sample = z_sample[valid_mask]
    labels_sample = labels_sample[valid_mask]
    sample_idx = sample_idx[valid_mask]
    n_sample = len(z_sample)
    
    # Outlier 제거 (z-score > 3)
    z_mean = z_sample.mean(axis=0)
    z_std = z_sample.std(axis=0) + 1e-8
    z_scores = np.abs((z_sample - z_mean) / z_std)
    inlier_mask = np.all(z_scores < 3, axis=1)
    z_sample = z_sample[inlier_mask]
    labels_sample = labels_sample[inlier_mask]
    sample_idx = sample_idx[inlier_mask]
    n_sample = len(z_sample)
    
    print(f"  After filtering: {n_sample} samples")
    
    if n_sample < 10:
        print("  ⚠ Too few samples after filtering, skipping visualization")
        return
    
    # Anchor mask
    is_anchor_sample = np.zeros(n_sample, dtype=bool)
    for k in range(K):
        anchor_set = set(anchor_indices[k])
        for i, idx in enumerate(sample_idx):
            if idx in anchor_set:
                is_anchor_sample[i] = True
    
    # PCA
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_sample)
    
    plt.figure(figsize=(12, 10))
    colors = ['red', 'blue', 'green']
    
    for k in range(K):
        mask = (labels_sample == k) & (~is_anchor_sample)
        plt.scatter(z_pca[mask, 0], z_pca[mask, 1],
                   c=colors[k], alpha=0.3, s=10, label=f'Class {k}')
    
    for k in range(K):
        mask = (labels_sample == k) & is_anchor_sample
        plt.scatter(z_pca[mask, 0], z_pca[mask, 1],
                   c=colors[k], alpha=0.8, s=100,
                   edgecolors='black', linewidths=2,
                   marker='*', label=f'Anchor {k}')
    
    # Centers (also filter)
    centers_valid = anchor_centers[np.all(np.isfinite(anchor_centers), axis=1)]
    if len(centers_valid) > 0:
        centers_pca = pca.transform(centers_valid)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
                   c='yellow', s=300, marker='X',
                   edgecolors='black', linewidths=3,
                   label='Centers', zorder=10)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('PCA Projection with Anchors', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'anchors_pca.png', dpi=150)
    plt.close()
    
    print(f"  ✓ Saved: {output_dir / 'anchors_pca.png'}")
# ============================================================
# Step 4: Self-Distillation
# ============================================================

def step4_self_distillation(crops, pseudo_labels, anchor_indices, anchor_centers,
                            vae, adapter, proto_head, device,
                            epochs=10, lr=1e-3, output_dir='./output'):
    """Self-distillation training"""
    
    print("\n" + "="*70)
    print(f"Step 4: Self-Distillation (Epochs: {epochs})")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Anchor mask
    N = len(crops)
    K = len(anchor_indices)
    is_anchor = np.zeros(N, dtype=bool)
    for k in range(K):
        is_anchor[anchor_indices[k]] = True
    
    print(f"\nAnchors: {is_anchor.sum():,} / {N:,} ({is_anchor.mean()*100:.1f}%)")
    
    # Teacher embeddings
    teacher_embeddings = {}
    for k in range(K):
        teacher_embeddings[k] = torch.from_numpy(anchor_centers[k]).float().to(device)
    
    # DataLoader
    crops_tensor = torch.from_numpy(crops).float() / 255.0
    crops_tensor = crops_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
    labels_tensor = torch.from_numpy(pseudo_labels).long()
    is_anchor_tensor = torch.from_numpy(is_anchor)
    
    dataset = TensorDataset(crops_tensor, labels_tensor, is_anchor_tensor)
    dataloader = DataLoader(dataset, batch_size=256, num_workers=4, pin_memory=True, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(adapter.parameters()) + list(proto_head.parameters()),
        lr=lr
    )
    
    # Loss
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    
    # Training
    adapter.train()
    proto_head.train()
    vae.eval()
    
    history = {'loss': [], 'loss_ce': [], 'loss_distill': [], 'accuracy': []}
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_ce = []
        epoch_distill = []
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_crops, batch_labels, batch_is_anchor in pbar:
            batch_crops = batch_crops.to(device)
            batch_labels = batch_labels.to(device)
            batch_is_anchor = batch_is_anchor.to(device)
            
            # VAE (frozen)
            with torch.no_grad():
                z = vae(batch_crops, no_dec=True)
            
            # Adapter
            z_adapted = adapter(z)
            logits = proto_head(z_adapted)
            
            # Loss 1: CE (non-anchors)
            if (~batch_is_anchor).sum() > 0:
                loss_ce_batch = ce_loss(logits[~batch_is_anchor],
                                       batch_labels[~batch_is_anchor])
            else:
                loss_ce_batch = torch.tensor(0.0).to(device)
            
            # Loss 2: Distillation (anchors)
            if batch_is_anchor.sum() > 0:
                z_student = z_adapted[batch_is_anchor]
                z_teacher = torch.stack([teacher_embeddings[l.item()]
                                        for l in batch_labels[batch_is_anchor]])
                loss_distill_batch = mse_loss(z_student, z_teacher)
            else:
                loss_distill_batch = torch.tensor(0.0).to(device)
            
            # Total
            loss = loss_ce_batch + loss_distill_batch
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Stats
            epoch_losses.append(loss.item())
            epoch_ce.append(loss_ce_batch.item())
            epoch_distill.append(loss_distill_batch.item())
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.3f}'
            })
        
        # Summary
        avg_loss = np.mean(epoch_losses)
        avg_ce = np.mean(epoch_ce)
        avg_distill = np.mean(epoch_distill)
        accuracy = correct / total
        
        history['loss'].append(avg_loss)
        history['loss_ce'].append(avg_ce)
        history['loss_distill'].append(avg_distill)
        history['accuracy'].append(accuracy)
        
        print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}, CE={avg_ce:.4f}, "
              f"Distill={avg_distill:.4f}, Acc={accuracy:.4f}")
    
    # Save
    torch.save({
        'adapter': adapter.state_dict(),
        'proto_head': proto_head.state_dict(),
        'history': history,
        'anchor_centers': anchor_centers
    }, output_dir / 'model.pth')
    
    print(f"\n✓ Saved: {output_dir / 'model.pth'}")
    
    # Plot
    plot_training(history, output_dir)
    
    return adapter, proto_head


def plot_training(history, output_dir):
    """Training curves"""
    
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Total', linewidth=2)
    plt.plot(history['loss_ce'], label='CE', linewidth=2)
    plt.plot(history['loss_distill'], label='Distill', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['accuracy'], linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.title('Accuracy')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['loss_distill'], linewidth=2, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Distillation Loss')
    plt.grid(True, alpha=0.3)
    plt.title('Distillation Loss')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training.png', dpi=150)
    plt.close()
    
    print(f"✓ Saved: {output_dir / 'training.png'}")


# ============================================================
# Main
# ============================================================

def main(args):
    device = torch.device('cuda')
    
    print(f"Device: {device}")
    
    # Data
    crops = np.load(f'{args.data_dir}/crops.npy')
    is_target = np.load(f'{args.data_dir}/is_target.npy')
    
    print(f"\nData: {crops.shape}")
    print(f"Target: {is_target.sum()} / {len(is_target)}")
    
    # Load VAE
    vae = load_vae(args.vae_checkpoint, z_dim=args.z_dim, device=device)
    
    # Step 1: Extract features
    z_all, pseudo_labels, confidence = step1_extract_features(
        crops, is_target, vae, device
    )
    
    # Step 2: K-means
    pseudo_labels, confidence = step2_kmeans(
        z_all, is_target, pseudo_labels, confidence, K=args.K
    )
    
    # Step 3: Anchors
    anchor_indices, anchor_centers = step3_anchor_selection(
        z_all, pseudo_labels, 
        percentile=args.anchor_percentile,
        output_dir=f'{args.output_dir}/anchors'
    )
    
    # Models
    adapter = Adapter(embedding_dim=args.z_dim).to(device)
    proto_head = PrototypicalHead(embedding_dim=args.z_dim, num_classes=args.K).to(device)
    
    # Step 4: Training
    adapter, proto_head = step4_self_distillation(
        crops, pseudo_labels, anchor_indices, anchor_centers,
        vae, adapter, proto_head, device,
        epochs=args.epochs, lr=args.lr,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--vae-checkpoint', 
                       default='./checkpoint/CCy/VAE/chkpts/last')
    parser.add_argument('--output-dir', default='./train_result/0204_semisupervised')
    parser.add_argument('--z-dim', type=int, default=10)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--anchor-percentile', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    args = parser.parse_args()
    main(args)