"""
models/morphogenie.py

[파이프라인 단계: 전처리 → 표현]

MorphoGenie: 해석 가능한 형태 표현 추출
- Concept vector c_i ∈ R^M (cylindrical index, granularity, ...)
- Embedding f_i ∈ R^d (연속 latent)

MorphoGenie는 고정 (해석가능성 유지)
      도메인 갭은 Adapter로 흡수
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# ============================================================
# MorphoGenie Base Model (고정)
# ============================================================

class MorphoGenieEncoder(nn.Module):
    """
    [1) Morphogenie: 해석 가능한 형태 표현]
    
    입력: 세포 crop (256, 256)
    출력:
      - concept vector c_i: (M,) - cylindrical_index, granularity, ...
      - embedding f_i: (d,) - latent representation
    
    중요: 이 모델은 freeze!
          해석가능성을 유지하기 위해 파라미터 고정
    """
    
    def __init__(self, 
                 latent_dim=256,
                 concept_dim=64,
                 pretrained_path: Optional[Path] = None):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.concept_dim = concept_dim
        
        # Encoder architecture (VAE-based)
        # 실제로는 MorphoGenie checkpoint 로드
        self.encoder = self._build_encoder()
        
        # Concept head (해석 가능한 특징)
        self.concept_head = nn.Linear(latent_dim, concept_dim)
        
        # Load pretrained
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)
        
        # Freeze all parameters
        self.freeze()
    
    def _build_encoder(self):
        """
        간단한 CNN encoder (실제로는 MorphoGenie 구조)
        """
        return nn.Sequential(
            # Conv blocks
            nn.Conv2d(1, 64, 3, stride=2, padding=1),  # 256 -> 128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Global pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # FC
            nn.Linear(512, self.latent_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: (B, 1, 256, 256) - cell crops
            
        Returns:
            f: (B, latent_dim) - embedding
            c: (B, concept_dim) - concept vector
        """
        # Ensure grayscale
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 256, 256) -> (B, 1, 256, 256)
        
        # Embedding
        f = self.encoder(x)  # (B, latent_dim)
        
        # Concept
        c = self.concept_head(f)  # (B, concept_dim)
        
        return f, c
    
    def load_pretrained(self, path: Path):
        """
        Load MorphoGenie pretrained weights
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self.load_state_dict(checkpoint, strict=False)
        
        print(f"✓ Loaded MorphoGenie from {path}")
    
    def freeze(self):
        """
        [1-2. 왜 고정하는가]
        
        해석가능한 concept의 의미가 흔들리지 않도록
        MorphoGenie는 freeze
        """
        for param in self.parameters():
            param.requires_grad = False
        
        print("✓ MorphoGenie frozen (해석가능성 유지)")
    
    def get_concept_names(self) -> list:
        """
        Concept dimension 이름
        
        Returns:
            ['cylindrical_index', 'granularity', 'eccentricity', ...]
        """
        # 실제로는 MorphoGenie metadata에서 로드
        names = []
        for i in range(self.concept_dim):
            if i == 0:
                names.append('cylindrical_index')
            elif i == 1:
                names.append('granularity')
            elif i == 2:
                names.append('eccentricity')
            elif i == 3:
                names.append('solidity')
            elif i == 4:
                names.append('circularity')
            else:
                names.append(f'concept_{i}')
        
        return names


# ============================================================
# Concept-aware Feature Extractor
# ============================================================

class ConceptAwareExtractor:
    """
    MorphoGenie wrapper for batch processing
    """
    
    def __init__(self, 
                 model: MorphoGenieEncoder,
                 device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def extract_features(self, 
                        crops: np.ndarray,
                        batch_size=32) -> Tuple[np.ndarray, np.ndarray]:
        """
        배치 feature extraction
        
        Args:
            crops: (N, 256, 256) numpy array
            
        Returns:
            embeddings: (N, latent_dim)
            concepts: (N, concept_dim)
        """
        N = len(crops)
        all_embeddings = []
        all_concepts = []
        
        for i in range(0, N, batch_size):
            batch = crops[i:i+batch_size]
            
            # To tensor
            x = torch.from_numpy(batch).float().unsqueeze(1)  # (B, 1, 256, 256)
            x = x.to(self.device)
            
            # Forward
            f, c = self.model(x)
            
            all_embeddings.append(f.cpu().numpy())
            all_concepts.append(c.cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
        concepts = np.vstack(all_concepts)
        
        return embeddings, concepts
    
    def get_important_concepts(self,
                              concepts_target: np.ndarray,
                              concepts_nontarget: np.ndarray,
                              top_k=10) -> list:
        """
        중요한 concept 찾기 (target vs non-target)
        
        Returns:
            List of (concept_idx, concept_name, t_statistic, p_value)
        """
        from scipy.stats import ttest_ind
        
        concept_names = self.model.get_concept_names()
        results = []
        
        for i in range(concepts_target.shape[1]):
            t_stat, p_val = ttest_ind(
                concepts_target[:, i],
                concepts_nontarget[:, i]
            )
            
            results.append({
                'idx': i,
                'name': concept_names[i] if i < len(concept_names) else f'concept_{i}',
                't_stat': abs(t_stat),
                'p_value': p_val,
                'target_mean': np.mean(concepts_target[:, i]),
                'nontarget_mean': np.mean(concepts_nontarget[:, i]),
                'fold_change': np.mean(concepts_target[:, i]) / (np.mean(concepts_nontarget[:, i]) + 1e-10)
            })
        
        # Sort by t-statistic
        results = sorted(results, key=lambda x: x['t_stat'], reverse=True)
        
        return results[:top_k]


# ============================================================
# Usage Example
# ============================================================

if __name__ == '__main__':
    """
    MorphoGenie feature extraction test
    """
    
    # Create model
    model = MorphoGenieEncoder(
        latent_dim=256,
        concept_dim=64,
        pretrained_path=None  # TODO: Add real checkpoint path
    )
    
    # Dummy data
    crops = np.random.randn(10, 256, 256).astype(np.float32)
    
    # Extract
    extractor = ConceptAwareExtractor(model, device='cpu')
    embeddings, concepts = extractor.extract_features(crops)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Concepts shape: {concepts.shape}")
    
    # Concept names
    print("\nConcept names:")
    for i, name in enumerate(model.get_concept_names()[:10]):
        print(f"  {i}: {name}")