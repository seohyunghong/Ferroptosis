"""
losses/contrastive.py

[파이프라인 단계 4: Contrastive = Metric learning]

미묘한 차이를 공간에서 벌려놓는 분리 원리
- SupCon: Supervised Contrastive Learning
- ArcFace: Angular margin for fine-grained separation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# Supervised Contrastive Loss
# ============================================================

class SupConLoss(nn.Module):
    """
    [4-2. SupCon 원리]
    
    동일 클래스는 positive, 다른 클래스는 negative
    
    L = -Σ (1/|P(i)|) Σ log[ exp(z_i·z_p/τ) / Σ exp(z_i·z_a/τ) ]
    
    핵심:
    - 같은 클래스끼리 반드시 가까워야 loss 감소
    - 앵커끼리 멀어져 있으면 penalty
    """
    
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, 
                features: torch.Tensor,
                labels: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Supervised Contrastive Loss
        
        Args:
            features: (B, D) - L2 normalized embeddings
            labels: (B,) - class labels
            mask: (B,) - which samples to use (optional)
            
        Returns:
            loss: scalar
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        # (B, B)
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Mask for same label
        labels = labels.contiguous().view(-1, 1)
        mask_same = torch.eq(labels, labels.T).float().to(device)
        
        # Mask out diagonal (자기 자신 제외)
        logits_mask = torch.ones_like(mask_same)
        logits_mask.fill_diagonal_(0)
        
        mask_same = mask_same * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive
        # (B,)
        mean_log_prob_pos = (mask_same * log_prob).sum(1) / (mask_same.sum(1) + 1e-10)
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        
        # If mask is provided, only use valid samples
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-10)
        else:
            loss = loss.mean()
        
        return loss


# ============================================================
# ArcFace Loss
# ============================================================

class ArcFaceLoss(nn.Module):
    """
    [4-3. ArcFace: 마진 기반 angular loss]
    
    각 클래스 방향과의 각도에 margin 추가
    → 미묘한 차이를 각도 마진으로 강하게 분리
    
    Fine-grained classification의 표준 (CVPR 2018)
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 scale: float = 30.0,
                 margin: float = 0.50,
                 easy_margin: bool = False):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale  # s
        self.margin = margin  # m
        self.easy_margin = easy_margin
        
        # Weight (클래스 방향)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Cos/sin margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, 
                features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        ArcFace forward
        
        Args:
            features: (B, in_features)
            labels: (B,) - ground truth labels
            
        Returns:
            logits: (B, out_features)
        """
        # Normalize features and weight
        features = F.normalize(features, dim=1)
        weight = F.normalize(self.weight, dim=1)
        
        # Cosine similarity
        # (B, out_features)
        cosine = F.linear(features, weight)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # cos(θ + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        
        # One-hot encoding
        one_hot = torch.zeros(cosine.size(), device=features.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin to ground truth class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale
        output = output * self.scale
        
        return output


# ============================================================
# Triplet Loss (대안)
# ============================================================

class TripletLoss(nn.Module):
    """
    [4. Contrastive - Triplet 버전]
    
    Anchor, Positive, Negative
    L = max(0, d(a,p) - d(a,n) + margin)
    """
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self,
                anchor: torch.Tensor,
                positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        """
        Triplet loss
        
        Args:
            anchor: (B, D)
            positive: (B, D)
            negative: (B, D)
        """
        dist_pos = torch.norm(anchor - positive, dim=1)
        dist_neg = torch.norm(anchor - negative, dim=1)
        
        loss = torch.relu(dist_pos - dist_neg + self.margin)
        
        return loss.mean()


# ============================================================
# Combined Contrastive Loss
# ============================================================

class CombinedContrastiveLoss(nn.Module):
    """
    SupCon + ArcFace 결합
    
    L_total = λ_supcon * L_supcon + λ_arcface * L_arcface
    """
    
    def __init__(self,
                 embedding_dim: int,
                 num_classes: int,
                 temperature: float = 0.07,
                 arcface_scale: float = 30.0,
                 arcface_margin: float = 0.50,
                 lambda_supcon: float = 1.0,
                 lambda_arcface: float = 1.0):
        super().__init__()
        
        self.lambda_supcon = lambda_supcon
        self.lambda_arcface = lambda_arcface
        
        # SupCon
        self.supcon = SupConLoss(temperature=temperature)
        
        # ArcFace
        self.arcface = ArcFaceLoss(
            in_features=embedding_dim,
            out_features=num_classes,
            scale=arcface_scale,
            margin=arcface_margin
        )
        
        # CE for ArcFace logits
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self,
                features: torch.Tensor,
                labels: torch.Tensor,
                mask: torch.Tensor = None) -> dict:
        """
        Combined loss
        
        Returns:
            {
                'total': total loss,
                'supcon': supcon loss,
                'arcface': arcface loss
            }
        """
        # SupCon
        loss_supcon = self.supcon(features, labels, mask)
        
        # ArcFace
        logits = self.arcface(features, labels)
        loss_arcface = self.ce(logits, labels)
        
        # Total
        loss_total = (self.lambda_supcon * loss_supcon + 
                     self.lambda_arcface * loss_arcface)
        
        return {
            'total': loss_total,
            'supcon': loss_supcon,
            'arcface': loss_arcface
        }


# ============================================================
# Usage Example
# ============================================================

if __name__ == '__main__':
    """
    Contrastive loss test
    """
    
    # Dummy data
    B = 32
    D = 256
    K = 3
    
    features = torch.randn(B, D)
    labels = torch.randint(0, K, (B,))
    
    # SupCon
    print("="*50)
    print("SupCon Loss")
    print("="*50)
    
    supcon = SupConLoss(temperature=0.07)
    loss_supcon = supcon(features, labels)
    print(f"Loss: {loss_supcon.item():.4f}")
    
    # ArcFace
    print("\n" + "="*50)
    print("ArcFace Loss")
    print("="*50)
    
    arcface = ArcFaceLoss(in_features=D, out_features=K)
    logits = arcface(features, labels)
    print(f"Logits shape: {logits.shape}")
    
    # Combined
    print("\n" + "="*50)
    print("Combined Loss")
    print("="*50)
    
    combined = CombinedContrastiveLoss(
        embedding_dim=D,
        num_classes=K
    )
    
    losses = combined(features, labels)
    print(f"Total: {losses['total'].item():.4f}")
    print(f"SupCon: {losses['supcon'].item():.4f}")
    print(f"ArcFace: {losses['arcface'].item():.4f}")