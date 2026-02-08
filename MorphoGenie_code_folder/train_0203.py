"""
train_semisupervised.py

Semi-supervised Learning with Self-Distillation
Step 1-4: 꼼꼼한 구현 (O)
Step 5-8: 간단한 구현
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import argparse
from MorphoGenie.dVAE.model import FactorVAE1
# ============================================================
# Model Definitions (간단한 버전)
# ============================================================
class MorphoGenieEncoder(nn.Module):
    def __init__(self, latent_dim=256, concept_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_concept = nn.Linear(512 * 4 * 4, concept_dim)
    
    def encode_deterministic(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_concept(h)

class Adapter(nn.Module):
    """Residual Adapter"""
    def __init__(self, embedding_dim=256, concept_dim=64):
        super().__init__()
        self.emb_adapter = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(64, embedding_dim)
        )
        self.concept_adapter = nn.Sequential(
            nn.Linear(concept_dim, 32), nn.ReLU(),
            nn.Linear(32, concept_dim)
        )
    
    def forward(self, f, c):
        delta_f = self.emb_adapter(f)
        z = f + delta_f
        delta_c = self.concept_adapter(c)
        c_adapted = c + delta_c
        return z, c_adapted


class PrototypicalHead(nn.Module):
    """Prototypical classification"""
    def __init__(self, embedding_dim=256, num_classes=3):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim))
    
    def forward(self, z):
        distances = torch.cdist(z, self.prototypes)  # (B, K)
        logits = -distances
        return logits
def load_morphogenie(ckpt_path, device='cuda'):
    print("="*70)
    print("Loading official FactorVAE1 checkpoint (STRICT)")
    print("="*70)

    ckpt = torch.load(ckpt_path, map_location=device)

    # 1) ckpt 구조 통일
    if isinstance(ckpt, dict) and 'model_states' in ckpt and 'VAE' in ckpt['model_states']:
        state = ckpt['model_states']['VAE']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    elif isinstance(ckpt, dict):
        # 이미 state_dict일 수도 있음
        state = ckpt
    else:
        raise ValueError("Unsupported checkpoint format")

    # 2) 모델 생성: z_dim이 체크포인트와 반드시 동일해야 함
    #    (대부분 checkpoint에 z_dim이 저장되어 있지 않아서 기본값/코드 기반으로 맞춰야 함)
    model = FactorVAE1(z_dim=10).to(device)  # ⚠️ z_dim이 다르면 strict에서 바로 에러 나고 그게 정상

    # 3) strict load
    missing, unexpected = model.load_state_dict(state, strict=True)
    # strict=True면 원래 missing/unexpected가 나면 에러가 나야 정상인데,
    # torch 버전에 따라 반환될 수 있음. 혹시라도 반환되면 체크.
    if len(missing) or len(unexpected):
        raise RuntimeError(f"State dict mismatch. missing={missing}, unexpected={unexpected}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print("✓ Loaded checkpoint with strict=True")
    return model


# ============================================================
# Step 1: Initial Clustering
# ============================================================

def step1_initial_setup(crops, is_target, morphogenie, device):
    print("\n" + "="*70)
    print("Step 1: Extracting Features with Pretrained VAE")
    print("="*70)
    
    N = len(crops)
    
    pseudo_labels = np.zeros(N, dtype=int)
    pseudo_labels[is_target] = 0
    
    confidence = np.zeros(N, dtype=float)
    confidence[is_target] = 0.8
    confidence[~is_target] = 0.0
    
    print(f"\nInitial setup:")
    print(f"  Target: {np.sum(is_target):,} ({np.mean(is_target)*100:.2f}%)")
    print(f"  Non-target: {np.sum(~is_target):,}")
    
    # Extract features with pretrained VAE
    print(f"\nExtracting features with Pretrained VAE...")
    
    crops_tensor = torch.from_numpy(crops).float().unsqueeze(1) / 255.0
    dataset = TensorDataset(crops_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_f = []
    all_c = []
    
    morphogenie.eval()
    with torch.no_grad():
        for (batch_crops,) in tqdm(dataloader, desc="Extracting"):
            batch_crops = batch_crops.to(device)
            f, c = morphogenie.encode_deterministic(batch_crops)
            all_f.append(f.cpu().numpy())
            all_c.append(c.cpu().numpy())
    
    all_f = np.vstack(all_f)
    all_c = np.vstack(all_c)
    
    print(f"✓ Features extracted: f {all_f.shape}, c {all_c.shape}")
    
    # 통계
    f_target = all_f[is_target]
    f_nontarget = all_f[~is_target]
    
    print(f"\nFeature statistics:")
    print(f"  Target mean: {f_target.mean(axis=0)[:5]} ...")
    print(f"  Non-target mean: {f_nontarget.mean(axis=0)[:5]} ...")
    
    # 거리 계산
    center_target = f_target.mean(axis=0)
    center_nontarget = f_nontarget.mean(axis=0)
    distance = np.linalg.norm(center_target - center_nontarget)
    
    print(f"\n✓ Target vs Non-target distance: {distance:.4f}")
    
    if distance > 0.1:
        print(f"  → Good separation! (distance > 0.1)")
    else:
        print(f"  → Weak separation (distance < 0.1)")
    
    return pseudo_labels, confidence, all_f, all_c


# ============================================================
# Step 2: K-means (train_0203.py와 동일)
# ============================================================

def step2_kmeans_clustering(all_f, is_target, pseudo_labels, confidence, K=3):
    print("\n" + "="*70)
    print(f"Step 2: K-means Clustering (K-1={K-1} clusters)")
    print("="*70)
    
    f_nontarget = all_f[~is_target]
    print(f"\nClustering {len(f_nontarget):,} non-target cells...")
    
    if len(f_nontarget) == 0:
        print("\n⚠️  All cells are target! Skipping clustering.")
        return pseudo_labels, confidence, None
    
    kmeans = KMeans(n_clusters=K-1, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(f_nontarget)
    
    pseudo_labels_copy = pseudo_labels.copy()
    pseudo_labels_copy[~is_target] = cluster_labels + 1
    
    confidence_copy = confidence.copy()
    
    for i in range(K-1):
        cluster_mask_local = (cluster_labels == i)
        cluster_indices_global = np.where(~is_target)[0][cluster_mask_local]
        
        f_cluster = f_nontarget[cluster_mask_local]
        center = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(f_cluster - center, axis=1)
        
        max_dist = distances.max()
        conf = 1.0 - (distances / max_dist) * 0.5
        confidence_copy[cluster_indices_global] = conf
    
    print(f"\nPseudo label distribution:")
    for k in range(K):
        count = np.sum(pseudo_labels_copy == k)
        avg_conf = confidence_copy[pseudo_labels_copy == k].mean()
        center_k = all_f[pseudo_labels_copy == k].mean(axis=0)
        print(f"  Class {k}: {count:,} cells (conf: {avg_conf:.3f})")
        print(f"    Center: {center_k[:5]} ...")
    
    # Cluster 간 거리
    print(f"\nCluster distances:")
    centers = []
    for k in range(K):
        centers.append(all_f[pseudo_labels_copy == k].mean(axis=0))
    
    for i in range(K):
        for j in range(i+1, K):
            dist = np.linalg.norm(centers[i] - centers[j])
            print(f"  Class {i} <-> Class {j}: {dist:.4f}")
    
    return pseudo_labels_copy, confidence_copy, kmeans


# ============================================================
# Step 3: Anchor Selection & Visualization
# ============================================================

def step3_anchor_selection(all_f, pseudo_labels, confidence, 
                           percentile=10, output_dir='./anchors'):
    """
    Step 3: 각 클래스의 anchor (상위 10%) 선정 및 시각화
    
    Returns:
        anchor_indices: dict {class_k: indices}
        anchor_centers: (K, 256) - anchor centers
    """
    print("\n" + "="*70)
    print(f"Step 3: Anchor Selection (Top {percentile}%)")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    K = len(np.unique(pseudo_labels))
    anchor_indices = {}
    anchor_centers = np.zeros((K, all_f.shape[1]))
    
    # 각 클래스별 anchor 선정
    for k in range(K):
        class_mask = (pseudo_labels == k)
        f_k = all_f[class_mask]
        indices_k = np.where(class_mask)[0]
        
        # Prototype (cluster center)
        p_k = f_k.mean(axis=0)
        
        # 거리 계산
        distances = np.linalg.norm(f_k - p_k, axis=1)
        
        # Top percentile (가장 가까운)
        threshold = np.percentile(distances, percentile)
        anchor_mask_local = distances <= threshold
        
        # Global indices
        anchor_indices_k = indices_k[anchor_mask_local]
        anchor_indices[k] = anchor_indices_k
        
        # Anchor center
        anchor_centers[k] = f_k[anchor_mask_local].mean(axis=0)
        
        # 통계
        print(f"\nClass {k}:")
        print(f"  Total: {len(f_k):,}")
        print(f"  Anchors: {len(anchor_indices_k):,} ({len(anchor_indices_k)/len(f_k)*100:.1f}%)")
        print(f"  Distance threshold: {threshold:.4f}")
        print(f"  Anchor center: {anchor_centers[k][:5]} ...")
    
    # ===== 시각화 =====
    
    print(f"\nVisualizing anchors...")
    
    # Sampling (전체는 너무 많음)
    n_sample = min(5000, len(all_f))
    sample_indices = np.random.choice(len(all_f), n_sample, replace=False)
    f_sample = all_f[sample_indices]
    labels_sample = pseudo_labels[sample_indices]
    
    # Anchor 포함 여부
    is_anchor_sample = np.zeros(n_sample, dtype=bool)
    for k in range(K):
        anchor_set = set(anchor_indices[k])
        for i, idx in enumerate(sample_indices):
            if idx in anchor_set:
                is_anchor_sample[i] = True
    
    # [1] PCA 2D
    print("  - PCA projection...")
    pca = PCA(n_components=2)
    f_pca = pca.fit_transform(f_sample)
    
    plt.figure(figsize=(12, 10))
    
    # Regular points (작게)
    colors = ['red', 'blue', 'green']
    for k in range(K):
        mask = (labels_sample == k) & (~is_anchor_sample)
        plt.scatter(f_pca[mask, 0], f_pca[mask, 1], 
                   c=colors[k], alpha=0.3, s=10, label=f'Class {k}')
    
    # Anchor points (크게 + 테두리)
    for k in range(K):
        mask = (labels_sample == k) & is_anchor_sample
        plt.scatter(f_pca[mask, 0], f_pca[mask, 1],
                   c=colors[k], alpha=0.8, s=100, 
                   edgecolors='black', linewidths=2,
                   marker='*', label=f'Anchor {k}')
    
    # Centers
    centers_pca = pca.transform(anchor_centers)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
               c='yellow', s=300, marker='X',
               edgecolors='black', linewidths=3,
               label='Anchor Centers', zorder=10)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('PCA Projection with Anchors', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'anchors_pca.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {output_dir / 'anchors_pca.png'}")
    
    # [2] t-SNE 2D
    print("  - t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    f_tsne = tsne.fit_transform(f_sample)
    
    plt.figure(figsize=(12, 10))
    
    for k in range(K):
        mask = (labels_sample == k) & (~is_anchor_sample)
        plt.scatter(f_tsne[mask, 0], f_tsne[mask, 1],
                   c=colors[k], alpha=0.3, s=10, label=f'Class {k}')
    
    for k in range(K):
        mask = (labels_sample == k) & is_anchor_sample
        plt.scatter(f_tsne[mask, 0], f_tsne[mask, 1],
                   c=colors[k], alpha=0.8, s=100,
                   edgecolors='black', linewidths=2,
                   marker='*', label=f'Anchor {k}')
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Projection with Anchors', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'anchors_tsne.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {output_dir / 'anchors_tsne.png'}")
    
    # [3] Distance distribution
    plt.figure(figsize=(12, 4))
    for k in range(K):
        class_mask = (pseudo_labels == k)
        f_k = all_f[class_mask]
        p_k = anchor_centers[k]
        distances = np.linalg.norm(f_k - p_k, axis=1)
        
        plt.subplot(1, K, k+1)
        plt.hist(distances, bins=50, color=colors[k], alpha=0.7)
        threshold = np.percentile(distances, percentile)
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Anchor threshold ({percentile}%)')
        plt.xlabel('Distance to Center')
        plt.ylabel('Count')
        plt.title(f'Class {k}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distance_distribution.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {output_dir / 'distance_distribution.png'}")
    
    print(f"\n✓ Anchor selection complete!")
    
    return anchor_indices, anchor_centers


# ============================================================
# Step 4: Self-Distillation
# ============================================================

def step4_self_distillation(crops, is_target, pseudo_labels, confidence,
                            anchor_indices, anchor_centers,
                            morphogenie, adapter, proto_head,
                            device, epochs=10, lr=1e-3, output_dir='./output'):
    """
    Step 4: Anchor 기준 Self-Distillation
    
    Returns:
        adapter: 학습된 adapter
        proto_head: 학습된 prototypical head
    """
    print("\n" + "="*70)
    print(f"Step 4: Self-Distillation (Epochs: {epochs})")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Anchor mask (전체 데이터 기준)
    N = len(crops)
    K = len(anchor_indices)
    is_anchor = np.zeros(N, dtype=bool)
    for k in range(K):
        is_anchor[anchor_indices[k]] = True
    
    print(f"\nAnchor statistics:")
    print(f"  Total anchors: {is_anchor.sum():,} ({is_anchor.mean()*100:.2f}%)")
    print(f"  Non-anchors: {(~is_anchor).sum():,}")
    
    # Teacher embeddings (frozen, from anchor_centers)
    teacher_embeddings = {}
    for k in range(K):
        teacher_embeddings[k] = torch.from_numpy(anchor_centers[k]).float().to(device)
    
    # DataLoader
    crops_tensor = torch.from_numpy(crops).float().unsqueeze(1) / 255.0
    labels_tensor = torch.from_numpy(pseudo_labels).long()
    is_anchor_tensor = torch.from_numpy(is_anchor)
    
    dataset = TensorDataset(crops_tensor, labels_tensor, is_anchor_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(adapter.parameters()) + list(proto_head.parameters()),
        lr=lr
    )
    
    # Loss
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    
    # Training loop
    adapter.train()
    proto_head.train()
    morphogenie.eval()
    
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
            
            # Forward
            with torch.no_grad():
                f, c = morphogenie(batch_crops)
            
            z, c_adapted = adapter(f, c)
            logits = proto_head(z)
            
            # Loss 1: CE (non-anchors)
            if (~batch_is_anchor).sum() > 0:
                loss_ce_batch = ce_loss(logits[~batch_is_anchor], 
                                       batch_labels[~batch_is_anchor])
            else:
                loss_ce_batch = torch.tensor(0.0).to(device)
            
            # Loss 2: Distillation (anchors)
            if batch_is_anchor.sum() > 0:
                anchor_mask = batch_is_anchor
                z_student = z[anchor_mask]
                labels_anchor = batch_labels[anchor_mask]
                
                # Teacher targets
                z_teacher_list = []
                for label in labels_anchor:
                    z_teacher_list.append(teacher_embeddings[label.item()])
                z_teacher = torch.stack(z_teacher_list)
                
                loss_distill_batch = mse_loss(z_student, z_teacher)
            else:
                loss_distill_batch = torch.tensor(0.0).to(device)
            
            # Total loss
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
                'ce': f'{loss_ce_batch.item():.4f}',
                'distill': f'{loss_distill_batch.item():.4f}',
                'acc': f'{correct/total:.3f}'
            })
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        avg_ce = np.mean(epoch_ce)
        avg_distill = np.mean(epoch_distill)
        accuracy = correct / total
        
        history['loss'].append(avg_loss)
        history['loss_ce'].append(avg_ce)
        history['loss_distill'].append(avg_distill)
        history['accuracy'].append(accuracy)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Distill: {avg_distill:.4f})")
        print(f"  Accuracy: {accuracy:.4f}")
    
    # Save
    torch.save({
        'adapter': adapter.state_dict(),
        'proto_head': proto_head.state_dict(),
        'history': history
    }, output_dir / 'step4_checkpoint.pth')
    print(f"\n✓ Saved: {output_dir / 'step4_checkpoint.pth'}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Total')
    plt.plot(history['loss_ce'], label='CE')
    plt.plot(history['loss_distill'], label='Distill')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['loss_distill'])
    plt.xlabel('Epoch')
    plt.ylabel('Distillation Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'step4_training.png', dpi=150)
    plt.close()
    print(f"✓ Saved: {output_dir / 'step4_training.png'}")
    
    return adapter, proto_head


# ============================================================
# Step 5-8: 간단한 구현
# ============================================================

def step5_update_prototypes(proto_head, anchor_centers):
    """Step 5: Prototype 업데이트"""
    with torch.no_grad():
        proto_head.prototypes.data = torch.from_numpy(anchor_centers).float()


def step6_contrastive_loss(z, labels, temperature=0.07):
    """Step 6: Supervised Contrastive Loss (간단)"""
    # Placeholder
    return torch.tensor(0.0)


def step7_mean_teacher_update(teacher, student, alpha=0.999):
    """Step 7: Mean-Teacher EMA (간단)"""
    with torch.no_grad():
        for t_param, s_param in zip(teacher.parameters(), student.parameters()):
            t_param.data = alpha * t_param.data + (1 - alpha) * s_param.data


def step8_reclustering(all_f, is_target, anchor_indices_old, K=3):
    """Step 8: Re-clustering (간단)"""
    # K-means 다시
    f_nontarget = all_f[~is_target]
    kmeans = KMeans(n_clusters=K-1, random_state=42)
    cluster_labels = kmeans.fit_predict(f_nontarget)
    
    pseudo_labels = np.zeros(len(all_f), dtype=int)
    pseudo_labels[is_target] = 0
    pseudo_labels[~is_target] = cluster_labels + 1
    
    # Anchor 재선정 (간단)
    anchor_indices = {}
    for k in range(K):
        class_mask = (pseudo_labels == k)
        indices_k = np.where(class_mask)[0]
        anchor_indices[k] = indices_k[:len(indices_k)//10]  # Top 10%
    
    return pseudo_labels, anchor_indices


# ============================================================
# Main Pipeline
# ============================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading data...")
    crops = np.load(args.data_dir + '/crops.npy')
    is_target = np.load(args.data_dir + '/is_target.npy')
    
    print(f"Data loaded:")
    print(f"  Crops: {crops.shape}")
    print(f"  Target: {is_target.sum()} / {len(is_target)}")
    
    # ===== Load Pretrained VAE =====
    morphogenie = load_morphogenie(args.vae_checkpoint, device)
    
    # Models
    adapter = Adapter().to(device)
    proto_head = PrototypicalHead(num_classes=args.K).to(device)
    
    # ===== Step 1-2 =====
    pseudo_labels, confidence, all_f, all_c = step1_initial_setup(
        crops, is_target, morphogenie, device
    )
    
    pseudo_labels, confidence, kmeans = step2_kmeans_clustering(
        all_f, is_target, pseudo_labels, confidence, K=args.K
    )
    
    # Step 3
    anchor_indices, anchor_centers = step3_anchor_selection(
        all_f, pseudo_labels, confidence,
        percentile=args.anchor_percentile,
        output_dir=args.output_dir + '/anchors'
    )
    
    # Step 4
    adapter, proto_head = step4_self_distillation(
        crops, is_target, pseudo_labels, confidence,
        anchor_indices, anchor_centers,
        morphogenie, adapter, proto_head,
        device, epochs=args.epochs_step4,
        lr=args.lr, output_dir=args.output_dir
    )
    
    # ===== Step 5-8: 간단한 구현 =====
    
    print("\n" + "="*70)
    print("Step 5-8: Refinement (Simple Implementation)")
    print("="*70)
    
    for epoch in range(args.epochs_final):
        print(f"\nEpoch {epoch+1}/{args.epochs_final}")
        
        # Step 5: Update prototypes
        step5_update_prototypes(proto_head, anchor_centers)
        
        # Step 8: Re-clustering (every 5 epochs)
        if epoch % 5 == 0:
            print("  Re-clustering...")
            # Extract new embeddings
            # ... (생략)
            # pseudo_labels, anchor_indices = step8_reclustering(...)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--vae-checkpoint', 
                       default='./checkpoint/CCy/VAE/chkpts/last')
    parser.add_argument('--output-dir', default='./train_result/')
    parser.add_argument('--K', type=int, default=3)
    
    args = parser.parse_args()
    main(args)
