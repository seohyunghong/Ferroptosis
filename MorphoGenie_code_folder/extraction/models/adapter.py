"""
models/adapter.py

[파이프라인 단계 2: +Adapter]

도메인 편차만 흡수하는 얕은 보정 모듈
- MorphoGenie 출력 f_i → Adapter → z_i
- Residual connection으로 의미 유지
- Identity regularization으로 과도한 변화 방지
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Residual Adapter
# ============================================================

class ResidualAdapter(nn.Module):
    """
    [2) +Adapter: 도메인 편차만 흡수]
    
    z_i = f_i + A(f_i)  (residual)
    
    핵심:
    - 파라미터 수가 작음 (bottleneck)
    - 강한 정규화 (weight decay, dropout)
    - Identity regularization: ||A_c(c)||^2
    """
    
    def __init__(self,
                 input_dim=256,
                 bottleneck_dim=64,
                 dropout=0.5):
        super().__init__()
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        
        # Bottleneck MLP (작은 파라미터)
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, input_dim)
        )
        
        # Initialize to near-zero (identity 시작)
        self._init_near_zero()
    
    def _init_near_zero(self):
        """
        Adapter를 거의 identity로 초기화
        (초기에는 MorphoGenie를 그대로 사용)
        """
        for m in self.adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Residual adaptation
        
        Args:
            f: (B, input_dim) - MorphoGenie embedding
            
        Returns:
            z: (B, input_dim) - adapted embedding
        """
        delta = self.adapter(f)  # (B, input_dim)
        z = f + delta  # Residual
        
        return z


# ============================================================
# Concept-aware Adapter (개선 버전)
# ============================================================

class ConceptAwareAdapter(nn.Module):
    """
    [2) +Adapter: Concept 의미 유지 강화 버전]
    
    Embedding과 Concept를 모두 받아서
    Concept 의미가 과도하게 변하지 않도록 제약
    """
    
    def __init__(self,
                 embedding_dim=256,
                 concept_dim=64,
                 bottleneck_dim=64,
                 dropout=0.5,
                 identity_weight=0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.concept_dim = concept_dim
        self.identity_weight = identity_weight
        
        # Embedding adapter
        self.embedding_adapter = ResidualAdapter(
            input_dim=embedding_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout
        )
        
        # Concept adapter (더 작게)
        self.concept_adapter = nn.Sequential(
            nn.Linear(concept_dim, concept_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(concept_dim // 2, concept_dim)
        )
        
        # Initialize concept adapter near zero
        for m in self.concept_adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, 
                f: torch.Tensor,
                c: torch.Tensor) -> tuple:
        """
        Concept-aware adaptation
        
        Args:
            f: (B, embedding_dim)
            c: (B, concept_dim)
            
        Returns:
            z: (B, embedding_dim) - adapted embedding
            c_adapted: (B, concept_dim) - adapted concept
        """
        # Adapt embedding
        z = self.embedding_adapter(f)
        
        # Adapt concept (minimal change)
        c_delta = self.concept_adapter(c)
        c_adapted = c + c_delta
        
        return z, c_adapted
    
    def identity_loss(self, c_delta: torch.Tensor) -> torch.Tensor:
        """
        [2-3. Concept 의미 유지 제약]
        
        L_id = ||A_c(c)||^2
        
        Concept adapter가 과도하게 변화하지 않도록
        """
        return torch.mean(c_delta ** 2)


# ============================================================
# Adapter with Prototypes (통합 버전)
# ============================================================

class AdapterWithPrototypes(nn.Module):
    """
    [2 + 3 통합]
    
    Adapter + Prototypical network 결합
    """
    
    def __init__(self,
                 embedding_dim=256,
                 concept_dim=64,
                 num_prototypes=3,  # K cell types
                 bottleneck_dim=64,
                 dropout=0.5):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_prototypes = num_prototypes
        
        # Adapter
        self.adapter = ConceptAwareAdapter(
            embedding_dim=embedding_dim,
            concept_dim=concept_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout
        )
        
        # Prototypes (학습 가능한 파라미터)
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, embedding_dim)
        )
        
        # Normalize prototypes
        nn.init.normal_(self.prototypes, mean=0, std=0.01)
    
    def forward(self, f: torch.Tensor, c: torch.Tensor) -> dict:
        """
        Forward with adaptation
        
        Returns:
            {
                'z': adapted embedding,
                'c_adapted': adapted concept,
                'distances': distances to prototypes,
                'logits': classification logits
            }
        """
        # Adapt
        z, c_adapted = self.adapter(f, c)
        
        # Normalize z and prototypes
        z_norm = F.normalize(z, dim=1)
        p_norm = F.normalize(self.prototypes, dim=1)
        
        # Distances to prototypes
        # (B, K)
        distances = torch.cdist(z_norm.unsqueeze(1), p_norm.unsqueeze(0)).squeeze(1)
        
        # Logits (negative distance)
        logits = -distances
        
        return {
            'z': z,
            'c_adapted': c_adapted,
            'distances': distances,
            'logits': logits
        }
    
    def get_prototype_assignments(self, z: torch.Tensor) -> torch.Tensor:
        """
        가장 가까운 prototype 할당
        
        Returns:
            assignments: (B,) - prototype indices
        """
        z_norm = F.normalize(z, dim=1)
        p_norm = F.normalize(self.prototypes, dim=1)
        
        distances = torch.cdist(z_norm.unsqueeze(1), p_norm.unsqueeze(0)).squeeze(1)
        assignments = torch.argmin(distances, dim=1)
        
        return assignments


# ============================================================
# Usage Example
# ============================================================

if __name__ == '__main__':
    """
    Adapter test
    """
    
    # Dummy data
    B = 32
    f = torch.randn(B, 256)  # MorphoGenie embedding
    c = torch.randn(B, 64)   # Concept vector
    
    # Adapter
    adapter = ConceptAwareAdapter(
        embedding_dim=256,
        concept_dim=64,
        bottleneck_dim=64
    )
    
    # Forward
    z, c_adapted = adapter(f, c)
    
    print(f"Input f: {f.shape}")
    print(f"Adapted z: {z.shape}")
    print(f"Adapted concept: {c_adapted.shape}")
    
    # Identity loss
    c_delta = c_adapted - c
    id_loss = adapter.identity_loss(c_delta)
    print(f"\nIdentity loss: {id_loss.item():.6f}")
    
    # With prototypes
    model = AdapterWithPrototypes(
        embedding_dim=256,
        concept_dim=64,
        num_prototypes=3
    )
    
    outputs = model(f, c)
    print(f"\nPrototype outputs:")
    print(f"  z: {outputs['z'].shape}")
    print(f"  distances: {outputs['distances'].shape}")
    print(f"  logits: {outputs['logits'].shape}")