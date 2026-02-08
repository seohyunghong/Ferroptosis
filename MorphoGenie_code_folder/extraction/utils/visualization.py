"""
utils/visualization.py

세포 유형 분석을 위한 종합 시각화 모듈
- Cell type overlay
- Embedding space (PCA, t-SNE, UMAP)
- Concept importance
- Confusion matrix
- Training curves
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

# Seaborn style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

# ============================================================
# 1. Cell Type Overlay Visualization
# ============================================================

class CellTypeVisualizer:
    """
    세포 유형별 색상 overlay 시각화
    """
    
    def __init__(self):
        # K개 세포 유형을 위한 색상 팔레트 (최대 10개)
        self.colors = {
            1: [1.0, 0.0, 0.0],   # celltype_1 (target) - Red
            2: [0.0, 1.0, 0.0],   # celltype_2 - Green
            3: [0.0, 0.0, 1.0],   # celltype_3 - Blue
            4: [1.0, 1.0, 0.0],   # celltype_4 - Yellow
            5: [1.0, 0.0, 1.0],   # celltype_5 - Magenta
            6: [0.0, 1.0, 1.0],   # celltype_6 - Cyan
            7: [1.0, 0.5, 0.0],   # celltype_7 - Orange
            8: [0.5, 0.0, 1.0],   # celltype_8 - Purple
            9: [0.5, 1.0, 0.0],   # celltype_9 - Lime
            10: [0.0, 0.5, 1.0],  # celltype_10 - Sky blue
        }
        
        self.type_names = {
            1: 'Target (Ferroptosis)',
            2: 'Living',
            3: 'Dead',
        }
    
    def set_type_names(self, names: Dict[int, str]):
        """
        세포 유형 이름 설정
        
        Args:
            names: {1: 'Ferroptosis', 2: 'Living', 3: 'Dead'}
        """
        self.type_names = names
    
    def visualize_single_image(self,
                              image: np.ndarray,
                              seg_mask: np.ndarray,
                              cell_types: Dict[int, int],
                              save_path: Optional[Path] = None,
                              show_legend: bool = True) -> np.ndarray:
        """
        단일 이미지에 세포 유형 overlay
        
        Args:
            image: (H, W) grayscale
            seg_mask: (H, W) with labels 1, 2, ..., N
            cell_types: {seg_label: cell_type_id}
                       예: {1: 1, 2: 2, 3: 1, ...}
            save_path: 저장 경로 (optional)
            show_legend: 범례 표시 여부
            
        Returns:
            overlay: (H, W, 3) RGB overlay
        """
        H, W = image.shape
        
        # RGB overlay 생성
        overlay = np.zeros((H, W, 3), dtype=np.float32)
        
        # Background: grayscale (어둡게)
        gray_norm = image.astype(np.float32) / 255.0
        overlay[:, :, 0] = gray_norm * 0.3
        overlay[:, :, 1] = gray_norm * 0.3
        overlay[:, :, 2] = gray_norm * 0.3
        
        # 각 세포에 색상 적용
        for seg_label in np.unique(seg_mask):
            if seg_label == 0:  # Background
                continue
            
            cell_type_id = cell_types.get(seg_label, 0)
            if cell_type_id == 0:
                continue
            
            # 색상 가져오기
            color = self.colors.get(cell_type_id, [0.5, 0.5, 0.5])
            
            # Mask 적용
            mask = (seg_mask == seg_label)
            for c in range(3):
                overlay[mask, c] = overlay[mask, c] * 0.3 + color[c] * 0.7
        
        # Plot
        if save_path is not None or show_legend:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Original
            axes[0].imshow(image, cmap='gray')
            axes[0].set_title('Original Phase-Contrast', fontsize=14)
            axes[0].axis('off')
            
            # Overlay
            axes[1].imshow(overlay)
            axes[1].set_title('Cell Type Classification', fontsize=14)
            axes[1].axis('off')
            
            # Legend
            if show_legend:
                unique_types = sorted(set(cell_types.values()))
                legend_elements = []
                
                for type_id in unique_types:
                    if type_id == 0:
                        continue
                    
                    color = self.colors.get(type_id, [0.5, 0.5, 0.5])
                    name = self.type_names.get(type_id, f'Type {type_id}')
                    count = sum(1 for v in cell_types.values() if v == type_id)
                    
                    legend_elements.append(
                        Patch(facecolor=color, label=f'{name} (n={count})')
                    )
                
                axes[1].legend(
                    handles=legend_elements,
                    loc='upper right',
                    fontsize=10,
                    framealpha=0.9
                )
            
            plt.tight_layout()
            
            if save_path is not None:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"✓ Saved: {save_path}")
            
            plt.close()
        
        return overlay
    
    def visualize_batch(self,
                       images: List[np.ndarray],
                       seg_masks: List[np.ndarray],
                       cell_types_list: List[Dict[int, int]],
                       save_dir: Path,
                       max_display: int = 9):
        """
        여러 이미지를 grid로 시각화
        
        Args:
            images: List of (H, W)
            seg_masks: List of (H, W)
            cell_types_list: List of {seg_label: cell_type_id}
            save_dir: 저장 디렉토리
            max_display: 최대 표시 개수
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        n_images = min(len(images), max_display)
        n_cols = 3
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
        axes = axes.flatten() if n_images > 1 else [axes]
        
        for i in range(n_images):
            overlay = self.visualize_single_image(
                images[i],
                seg_masks[i],
                cell_types_list[i],
                save_path=None,
                show_legend=False
            )
            
            axes[i].imshow(overlay)
            axes[i].set_title(f'Image {i+1}', fontsize=12)
            axes[i].axis('off')
        
        # 빈 subplot 제거
        for i in range(n_images, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(save_dir / 'batch_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Batch visualization saved: {save_dir / 'batch_visualization.png'}")


# ============================================================
# 2. Embedding Space Visualization
# ============================================================

class EmbeddingVisualizer:
    """
    Embedding space 시각화 (PCA, t-SNE, UMAP)
    """
    
    def __init__(self):
        pass
    
    def plot_pca(self,
                embeddings: np.ndarray,
                labels: np.ndarray,
                label_names: Dict[int, str],
                save_path: Path,
                title: str = "PCA Projection"):
        """
        PCA 2D 시각화
        
        Args:
            embeddings: (N, D)
            labels: (N,) - 0, 1, 2, ...
            label_names: {0: 'Living', 1: 'Dead', ...}
            save_path: 저장 경로
        """
        # PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = (labels == label)
            name = label_names.get(label, f'Type {label}')
            
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=f'{name} (n={np.sum(mask)})',
                alpha=0.6,
                s=30,
                edgecolors='black',
                linewidth=0.5
            )
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ PCA visualization saved: {save_path}")
    
    def plot_tsne(self,
                 embeddings: np.ndarray,
                 labels: np.ndarray,
                 label_names: Dict[int, str],
                 save_path: Path,
                 perplexity: int = 30,
                 title: str = "t-SNE Projection"):
        """
        t-SNE 2D 시각화
        """
        print(f"  Running t-SNE (perplexity={perplexity})...")
        
        # t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(embeddings) // 2),
            random_state=42,
            n_iter=1000
        )
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = (labels == label)
            name = label_names.get(label, f'Type {label}')
            
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=f'{name} (n={np.sum(mask)})',
                alpha=0.6,
                s=30,
                edgecolors='black',
                linewidth=0.5
            )
        
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ t-SNE visualization saved: {save_path}")
    
    def plot_embedding_comparison(self,
                                 embeddings_before: np.ndarray,
                                 embeddings_after: np.ndarray,
                                 labels: np.ndarray,
                                 label_names: Dict[int, str],
                                 save_path: Path):
        """
        Adapter 전후 비교
        
        Args:
            embeddings_before: (N, D) - MorphoGenie 출력
            embeddings_after: (N, D) - Adapter 출력
        """
        # PCA
        pca = PCA(n_components=2)
        
        before_2d = pca.fit_transform(embeddings_before)
        after_2d = pca.transform(embeddings_after)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for ax, data, title in zip(
            axes,
            [before_2d, after_2d],
            ['Before Adapter', 'After Adapter']
        ):
            for i, label in enumerate(unique_labels):
                mask = (labels == label)
                name = label_names.get(label, f'Type {label}')
                
                ax.scatter(
                    data[mask, 0],
                    data[mask, 1],
                    c=[colors[i]],
                    label=f'{name} (n={np.sum(mask)})',
                    alpha=0.6,
                    s=30,
                    edgecolors='black',
                    linewidth=0.5
                )
            
            ax.set_xlabel('PC1', fontsize=12)
            ax.set_ylabel('PC2', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Embedding comparison saved: {save_path}")


# ============================================================
# 3. Concept Importance Visualization
# ============================================================

class ConceptVisualizer:
    """
    Morphological concept 중요도 시각화
    """
    
    def __init__(self):
        pass
    
    def plot_concept_importance(self,
                               concept_names: List[str],
                               importance_scores: np.ndarray,
                               save_path: Path,
                               top_k: int = 20):
        """
        Concept 중요도 bar plot
        
        Args:
            concept_names: ['cylindrical_index', 'granularity', ...]
            importance_scores: (M,) - importance values
            top_k: 상위 몇 개 표시
        """
        # Top-K
        top_indices = np.argsort(importance_scores)[::-1][:top_k]
        top_names = [concept_names[i] for i in top_indices]
        top_scores = importance_scores[top_indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(top_names))
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_names)))
        
        ax.barh(y_pos, top_scores, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, fontsize=10)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Top {top_k} Morphological Markers', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 값 표시
        for i, (name, score) in enumerate(zip(top_names, top_scores)):
            ax.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Concept importance saved: {save_path}")
    
    def plot_concept_distribution(self,
                                 concepts: np.ndarray,
                                 labels: np.ndarray,
                                 concept_names: List[str],
                                 label_names: Dict[int, str],
                                 save_path: Path,
                                 top_k: int = 12):
        """
        클래스별 concept 분포 비교
        
        Args:
            concepts: (N, M)
            labels: (N,)
            concept_names: List of concept names
            label_names: {0: 'Living', 1: 'Dead', ...}
            top_k: 상위 몇 개 concept
        """
        from scipy.stats import ttest_ind
        
        # Feature importance (t-test)
        unique_labels = np.unique(labels)
        if len(unique_labels) != 2:
            print("Warning: Concept distribution visualization works best with 2 classes")
            return
        
        label0, label1 = unique_labels[:2]
        concepts0 = concepts[labels == label0]
        concepts1 = concepts[labels == label1]
        
        t_stats = []
        for i in range(concepts.shape[1]):
            t_stat, _ = ttest_ind(concepts0[:, i], concepts1[:, i])
            t_stats.append(abs(t_stat))
        
        # Top-K
        top_indices = np.argsort(t_stats)[::-1][:top_k]
        
        # Plot
        n_cols = 4
        n_rows = (top_k + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5))
        axes = axes.flatten() if top_k > 1 else [axes]
        
        for i, idx in enumerate(top_indices):
            if i >= top_k:
                break
            
            name = concept_names[idx] if idx < len(concept_names) else f'Concept {idx}'
            
            axes[i].hist(
                concepts0[:, idx],
                bins=30,
                alpha=0.5,
                label=label_names.get(label0, f'Type {label0}'),
                color='blue',
                density=True
            )
            axes[i].hist(
                concepts1[:, idx],
                bins=30,
                alpha=0.5,
                label=label_names.get(label1, f'Type {label1}'),
                color='red',
                density=True
            )
            
            axes[i].set_title(name, fontsize=10)
            axes[i].set_xlabel('Value', fontsize=9)
            axes[i].set_ylabel('Density', fontsize=9)
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
        
        # 빈 subplot 제거
        for i in range(top_k, len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Morphological Concept Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Concept distribution saved: {save_path}")


# ============================================================
# 4. Training Curves
# ============================================================

class TrainingVisualizer:
    """
    학습 곡선 시각화
    """
    
    def __init__(self):
        pass
    
    def plot_training_curves(self,
                            history: Dict[str, List[float]],
                            save_path: Path):
        """
        Loss curves
        
        Args:
            history: {
                'train_loss': [...],
                'train_ce': [...],
                'train_contrastive': [...],
                'val_acc': [...]
            }
        """
        n_metrics = len(history)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for i, (key, values) in enumerate(history.items()):
            if i >= len(axes):
                break
            
            epochs = range(1, len(values) + 1)
            axes[i].plot(epochs, values, linewidth=2, marker='o', markersize=3)
            axes[i].set_xlabel('Epoch', fontsize=11)
            axes[i].set_ylabel(key.replace('_', ' ').title(), fontsize=11)
            axes[i].set_title(key.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        
        # 빈 subplot 제거
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training curves saved: {save_path}")


# ============================================================
# 5. Confusion Matrix
# ============================================================

def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         label_names: Dict[int, str],
                         save_path: Path,
                         normalize: bool = True):
    """
    Confusion matrix
    
    Args:
        y_true: (N,) ground truth
        y_pred: (N,) predictions
        label_names: {0: 'Living', 1: 'Dead', ...}
        normalize: True면 비율, False면 개수
    """
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Labels
    unique_labels = sorted(set(y_true) | set(y_pred))
    tick_labels = [label_names.get(i, f'Type {i}') for i in unique_labels]
    
    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_yticks(np.arange(len(tick_labels)))
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yticklabels(tick_labels, fontsize=10)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=11)
    
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved: {save_path}")


# ============================================================
# Usage Example
# ============================================================

if __name__ == '__main__':
    """
    Visualization test
    """
    
    # Dummy data
    np.random.seed(42)
    
    # [1] Cell type overlay
    print("="*50)
    print("Cell Type Overlay")
    print("="*50)
    
    visualizer = CellTypeVisualizer()
    
    image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    seg_mask = np.random.randint(0, 20, (512, 512))
    cell_types = {i: (i % 3) + 1 for i in range(1, 20)}
    
    overlay = visualizer.visualize_single_image(
        image, seg_mask, cell_types,
        save_path=Path('cell_overlay.png')
    )
    
    # [2] Embedding space
    print("\n" + "="*50)
    print("Embedding Space")
    print("="*50)
    
    emb_viz = EmbeddingVisualizer()
    
    embeddings = np.random.randn(300, 256)
    labels = np.random.randint(0, 3, 300)
    label_names = {0: 'Living', 1: 'Dead', 2: 'Ferroptosis'}
    
    emb_viz.plot_pca(
        embeddings, labels, label_names,
        save_path=Path('pca.png')
    )
    
    # [3] Concept importance
    print("\n" + "="*50)
    print("Concept Importance")
    print("="*50)
    
    concept_viz = ConceptVisualizer()
    
    concept_names = ['cylindrical_index', 'granularity', 'eccentricity'] + [f'concept_{i}' for i in range(3, 20)]
    importance = np.random.rand(20)
    
    concept_viz.plot_concept_importance(
        concept_names, importance,
        save_path=Path('concept_importance.png'),
        top_k=10
    )
    
    print("\n✓ All visualizations created!")