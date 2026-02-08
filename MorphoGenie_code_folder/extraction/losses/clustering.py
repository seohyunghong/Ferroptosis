"""
utils/clustering.py

[파이프라인 단계 5: Constrained Clustering]

"K 고정 + 형광 seed" 제약을 클러스터링에 직접 반영
- Seeded k-means with constraints
- Must-link: 같은 타겟은 같은 클러스터
- Cannot-link: Living/Dead vs Target 분리
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from sklearn.metrics import pairwise_distances

# ============================================================
# Constrained K-means
# ============================================================

class ConstrainedKMeans:
    """
    [5-2. Constrained clustering]
    
    제약 조건:
    1. Seed (anchoring): 형광 양성 셀은 특정 클러스터로 고정
    2. Must-link: 같은 타겟 bag의 셀들은 같은 클러스터
    3. Cannot-link: Living vs Target 분리
    """
    
    def __init__(self,
                 n_clusters: int = 3,
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.cluster_centers_ = None
        self.labels_ = None
    
    def fit(self,
            X: np.ndarray,
            seed_labels: Optional[np.ndarray] = None,
            must_link: Optional[List[Tuple[int, int]]] = None,
            cannot_link: Optional[List[Tuple[int, int]]] = None) -> 'ConstrainedKMeans':
        """
        Constrained k-means clustering
        
        Args:
            X: (N, D) - features
            seed_labels: (N,) - -1 for unlabeled, 0,1,2,... for seeded
            must_link: List of (i, j) pairs that must be in same cluster
            cannot_link: List of (i, j) pairs that cannot be in same cluster
            
        Returns:
            self
        """
        np.random.seed(self.random_state)
        N, D = X.shape
        
        # Initialize cluster centers
        if seed_labels is not None:
            # Use seeds to initialize centers
            centers = []
            for k in range(self.n_clusters):
                seed_mask = (seed_labels == k)
                if np.sum(seed_mask) > 0:
                    # Use mean of seeded samples
                    centers.append(X[seed_mask].mean(axis=0))
                else:
                    # Random initialization
                    centers.append(X[np.random.randint(N)])
            self.cluster_centers_ = np.array(centers)
        else:
            # Random initialization
            indices = np.random.choice(N, self.n_clusters, replace=False)
            self.cluster_centers_ = X[indices].copy()
        
        # Iterative refinement
        for iteration in range(self.max_iter):
            old_centers = self.cluster_centers_.copy()
            
            # Assignment step (with constraints)
            self.labels_ = self._constrained_assignment(
                X, seed_labels, must_link, cannot_link
            )
            
            # Update step
            for k in range(self.n_clusters):
                mask = (self.labels_ == k)
                if np.sum(mask) > 0:
                    self.cluster_centers_[k] = X[mask].mean(axis=0)
            
            # Check convergence
            center_shift = np.sum((self.cluster_centers_ - old_centers) ** 2)
            if center_shift < self.tol:
                break
        
        return self
    
    def _constrained_assignment(self,
                               X: np.ndarray,
                               seed_labels: Optional[np.ndarray],
                               must_link: Optional[List[Tuple[int, int]]],
                               cannot_link: Optional[List[Tuple[int, int]]]) -> np.ndarray:
        """
        Assign samples to clusters with constraints
        """
        N = len(X)
        
        # Compute distances to centers
        # (N, K)
        distances = pairwise_distances(X, self.cluster_centers_)
        
        # Initial assignment (nearest center)
        labels = np.argmin(distances, axis=1)
        
        # Apply seed constraints (highest priority)
        if seed_labels is not None:
            seed_mask = (seed_labels >= 0)
            labels[seed_mask] = seed_labels[seed_mask]
        
        # Apply must-link constraints
        if must_link is not None:
            for i, j in must_link:
                # Make sure i and j have same label
                if seed_labels is not None and seed_labels[i] >= 0:
                    labels[j] = seed_labels[i]
                elif seed_labels is not None and seed_labels[j] >= 0:
                    labels[i] = seed_labels[j]
                else:
                    # Choose the one with smaller distance
                    if distances[i, labels[i]] < distances[j, labels[j]]:
                        labels[j] = labels[i]
                    else:
                        labels[i] = labels[j]
        
        # Apply cannot-link constraints
        if cannot_link is not None:
            for i, j in cannot_link:
                if labels[i] == labels[j]:
                    # Reassign j to next nearest cluster
                    sorted_clusters = np.argsort(distances[j])
                    for k in sorted_clusters:
                        if k != labels[i]:
                            labels[j] = k
                            break
        
        return labels
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels
        """
        distances = pairwise_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)


# ============================================================
# Seeded K-means (간단 버전)
# ============================================================

class SeededKMeans:
    """
    [5-2. Seeded k-means - 간단 버전]
    
    형광 seed만 고정, 나머지는 자동 할당
    """
    
    def __init__(self, n_clusters: int = 3, max_iter: int = 100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers_ = None
        self.labels_ = None
    
    def fit(self, 
            X: np.ndarray,
            seed_indices: dict) -> 'SeededKMeans':
        """
        Seeded k-means
        
        Args:
            X: (N, D)
            seed_indices: {cluster_id: [indices]}
                         예: {1: [10, 15, 20]} - cluster 1의 seed indices
        """
        N, D = X.shape
        
        # Initialize centers from seeds
        self.centers_ = np.zeros((self.n_clusters, D))
        for k, indices in seed_indices.items():
            if len(indices) > 0:
                self.centers_[k] = X[indices].mean(axis=0)
        
        # For clusters without seeds, random init
        for k in range(self.n_clusters):
            if k not in seed_indices or len(seed_indices[k]) == 0:
                self.centers_[k] = X[np.random.randint(N)]
        
        # K-means iterations
        for iteration in range(self.max_iter):
            # Assignment
            distances = pairwise_distances(X, self.centers_)
            self.labels_ = np.argmin(distances, axis=1)
            
            # Fix seed assignments
            for k, indices in seed_indices.items():
                self.labels_[indices] = k
            
            # Update centers
            old_centers = self.centers_.copy()
            for k in range(self.n_clusters):
                mask = (self.labels_ == k)
                if np.sum(mask) > 0:
                    self.centers_[k] = X[mask].mean(axis=0)
            
            # Check convergence
            if np.sum((self.centers_ - old_centers) ** 2) < 1e-4:
                break
        
        return self


# ============================================================
# Prototype-based Clustering (PyTorch)
# ============================================================

class PrototypeClusterer:
    """
    [5. Constrained clustering + 3. Prototypical 통합]
    
    PyTorch 버전 - gradient 기반 prototype 학습
    """
    
    def __init__(self,
                 n_clusters: int = 3,
                 embedding_dim: int = 256,
                 device='cuda'):
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Learnable prototypes
        self.prototypes = torch.randn(
            n_clusters, embedding_dim,
            device=device
        )
        self.prototypes.requires_grad = True
    
    def update_prototypes(self,
                         embeddings: torch.Tensor,
                         labels: torch.Tensor,
                         seed_mask: torch.Tensor = None):
        """
        Update prototypes with EMA
        
        Args:
            embeddings: (N, D)
            labels: (N,) - cluster assignments
            seed_mask: (N,) bool - which are seed samples
        """
        with torch.no_grad():
            for k in range(self.n_clusters):
                mask = (labels == k)
                
                # Prioritize seeds
                if seed_mask is not None:
                    seed_k = mask & seed_mask
                    if seed_k.sum() > 0:
                        # Use only seeds
                        self.prototypes[k] = embeddings[seed_k].mean(dim=0)
                        continue
                
                # Use all samples in cluster
                if mask.sum() > 0:
                    new_center = embeddings[mask].mean(dim=0)
                    # EMA update
                    alpha = 0.9
                    self.prototypes[k] = alpha * self.prototypes[k] + (1 - alpha) * new_center
    
    def assign_clusters(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Assign samples to nearest prototype
        
        Returns:
            labels: (N,)
        """
        # Normalize
        emb_norm = torch.nn.functional.normalize(embeddings, dim=1)
        proto_norm = torch.nn.functional.normalize(self.prototypes, dim=1)
        
        # Distances
        distances = torch.cdist(emb_norm.unsqueeze(0), proto_norm.unsqueeze(0)).squeeze(0)
        
        # Assign
        labels = torch.argmin(distances, dim=1)
        
        return labels


# ============================================================
# Usage Example
# ============================================================

if __name__ == '__main__':
    """
    Constrained clustering test
    """
    
    # Dummy data
    np.random.seed(42)
    N = 300
    D = 256
    K = 3
    
    X = np.random.randn(N, D)
    
    # Seeds: first 30 samples are cluster 1 (target)
    seed_labels = np.full(N, -1)
    seed_labels[:30] = 1  # Target cluster
    
    # Constrained k-means
    print("="*50)
    print("Constrained K-means")
    print("="*50)
    
    ckm = ConstrainedKMeans(n_clusters=K)
    ckm.fit(X, seed_labels=seed_labels)
    
    print(f"Cluster assignments: {np.bincount(ckm.labels_)}")
    print(f"Target cluster size: {np.sum(ckm.labels_ == 1)}")
    
    # Seeded k-means
    print("\n" + "="*50)
    print("Seeded K-means")
    print("="*50)
    
    skm = SeededKMeans(n_clusters=K)
    skm.fit(X, seed_indices={1: list(range(30))})
    
    print(f"Cluster assignments: {np.bincount(skm.labels_)}")
    print(f"Target cluster size: {np.sum(skm.labels_ == 1)}")