"""
Live vs Dead 세포 형태학적 분석
MorphoGenie VAE 없이 scikit-image의 형태학적 특징만 사용
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
from skimage import measure, feature
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Step 1: Cell Crop Extraction (256x256)
# ============================================================

class CellCropExtractor:
    """
    Cellpose mask → 개별 세포 256x256 crops
    """
    
    def __init__(self, crop_size=256):
        self.crop_size = crop_size
    
    def extract_single_cell(self, image, mask, cell_label):
        """
        개별 세포 crop 추출
        
        Args:
            image: 원본 이미지 (H, W) grayscale
            mask: Segmentation mask (H, W)
            cell_label: 추출할 세포 label
            
        Returns:
            crop: 256x256 crop (배경 제거, 중앙 정렬)
        """
        # Cell mask
        cell_mask = (mask == cell_label)
        
        # Region props
        labeled = measure.label(cell_mask)
        regions = measure.regionprops(labeled)
        
        if len(regions) == 0:
            return None
        
        region = regions[0]
        
        # 1. 배경 제거
        masked_image = image.copy()
        masked_image[~cell_mask] = 0
        
        # 2. Centroid 기준 crop
        cy, cx = int(region.centroid[0]), int(region.centroid[1])
        h, w = image.shape[:2]
        
        # Crop 크기 계산
        minr, minc, maxr, maxc = region.bbox
        cell_h = maxr - minr
        cell_w = maxc - minc
        crop_radius = max(cell_h, cell_w) // 2 + 20
        
        # Crop 좌표
        y1 = max(0, cy - crop_radius)
        y2 = min(h, cy + crop_radius)
        x1 = max(0, cx - crop_radius)
        x2 = min(w, cx + crop_radius)
        
        cropped = masked_image[y1:y2, x1:x2].copy()
        
        # 3. 정사각형 padding
        crop_h, crop_w = cropped.shape[:2]
        max_side = max(crop_h, crop_w)
        
        padded = np.zeros((max_side, max_side), dtype=image.dtype)
        pady = (max_side - crop_h) // 2
        padx = (max_side - crop_w) // 2
        padded[pady:pady+crop_h, padx:padx+crop_w] = cropped
        
        # 4. 256x256 리사이징
        resized = cv2.resize(padded, (self.crop_size, self.crop_size), 
                            interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def extract_all_cells(self, image, mask, cell_labels, cell_states):
        """
        모든 세포 crops 추출
        
        Args:
            image: 원본 이미지
            mask: Segmentation mask
            cell_labels: 추출할 세포 labels 리스트
            cell_states: 각 세포의 상태 ('living' or 'dead') 리스트
            
        Returns:
            crops: (N, 256, 256) numpy array
            labels: (N,) cell labels
            states: (N,) cell states
        """
        crops = []
        valid_labels = []
        valid_states = []
        
        for label, state in zip(cell_labels, cell_states):
            crop = self.extract_single_cell(image, mask, label)
            if crop is not None:
                crops.append(crop)
                valid_labels.append(label)
                valid_states.append(state)
        
        if len(crops) == 0:
            return None, None, None
        
        crops = np.array(crops)  # (N, 256, 256)
        labels = np.array(valid_labels)
        states = np.array(valid_states)
        
        return crops, labels, states

# ============================================================
# Step 2: Morphological Feature Extraction (VAE 대신)
# ============================================================

class MorphologicalFeatureExtractor:
    """
    형태학적 특징 추출 (VAE 없이)
    - Haralick texture features
    - Hu moments
    - LBP (Local Binary Patterns)
    - Intensity statistics
    """
    
    def __init__(self):
        pass
    
    def extract_haralick_features(self, image):
        """
        Haralick texture features (13차원)
        """
        from skimage.feature import graycomatrix, graycoprops
        
        # GLCM (Gray Level Co-occurrence Matrix)
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # 이미지를 8-bit로 변환
        img_8bit = (image / image.max() * 255).astype(np.uint8) if image.max() > 0 else image.astype(np.uint8)
        
        glcm = graycomatrix(img_8bit, distances=distances, angles=angles, 
                           levels=256, symmetric=True, normed=True)
        
        # Haralick features
        features = []
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        
        for prop in properties:
            feat = graycoprops(glcm, prop)
            features.append(feat.mean())  # 모든 각도의 평균
        
        return np.array(features)
    
    def extract_hu_moments(self, image):
        """
        Hu moments (7차원) - 회전/크기 불변
        """
        moments = cv2.moments(image)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Log transform (값이 너무 작아서)
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        return hu_moments
    
    def extract_lbp_features(self, image, num_points=24, radius=3):
        """
        Local Binary Patterns (히스토그램 사용)
        """
        from skimage.feature import local_binary_pattern
        
        lbp = local_binary_pattern(image, num_points, radius, method='uniform')
        
        # LBP 히스토그램 (59 bins for uniform LBP)
        n_bins = num_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        return hist
    
    def extract_intensity_stats(self, image):
        """
        강도 통계 특징
        """
        # 0이 아닌 픽셀만 (배경 제외)
        non_zero = image[image > 0]
        
        if len(non_zero) == 0:
            return np.zeros(10)
        
        features = [
            np.mean(non_zero),
            np.std(non_zero),
            np.median(non_zero),
            np.min(non_zero),
            np.max(non_zero),
            np.percentile(non_zero, 25),
            np.percentile(non_zero, 75),
            np.percentile(non_zero, 90),
            len(non_zero) / image.size,  # 세포 영역 비율
            np.sum(non_zero > np.mean(non_zero)) / len(non_zero)  # 밝은 픽셀 비율
        ]
        
        return np.array(features)
    
    def extract_shape_features(self, image):
        """
        형태 특징
        """
        # Binary mask
        binary = image > 0
        
        if np.sum(binary) == 0:
            return np.zeros(5)
        
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)
        
        if len(regions) == 0:
            return np.zeros(5)
        
        region = regions[0]
        
        features = [
            region.area,
            region.perimeter,
            region.eccentricity,
            region.solidity,
            (4 * np.pi * region.area) / (region.perimeter ** 2) if region.perimeter > 0 else 0
        ]
        
        return np.array(features)
    
    def extract_all_features(self, crops):
        """
        모든 crops에서 특징 추출
        
        Args:
            crops: (N, 256, 256) numpy array
            
        Returns:
            features: (N, feature_dim) numpy array
        """
        all_features = []
        
        for i, crop in enumerate(tqdm(crops, desc="Extracting features")):
            # 각 특징 추출
            haralick = self.extract_haralick_features(crop)
            hu = self.extract_hu_moments(crop)
            lbp = self.extract_lbp_features(crop)
            intensity = self.extract_intensity_stats(crop)
            shape = self.extract_shape_features(crop)
            
            # 통합
            features = np.concatenate([haralick, hu, lbp, intensity, shape])
            all_features.append(features)
        
        all_features = np.array(all_features)
        
        print(f"\n✓ Extracted {all_features.shape[1]} features per cell")
        print(f"  - Haralick: {len(haralick)}")
        print(f"  - Hu moments: {len(hu)}")
        print(f"  - LBP: {len(lbp)}")
        print(f"  - Intensity: {len(intensity)}")
        print(f"  - Shape: {len(shape)}")
        
        return all_features

# ============================================================
# Step 3: Feature Analysis & Visualization
# ============================================================

class MorphologicalAnalyzer:
    """
    Live vs Dead 형태학적 차이 분석
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def normalize_features(self, features):
        """Feature normalization"""
        return self.scaler.fit_transform(features)
    
    def compute_statistics(self, features, states):
        """
        Live vs Dead feature 통계
        """
        living_features = features[states == 'living']
        dead_features = features[states == 'dead']
        
        stats = {
            'living_count': len(living_features),
            'dead_count': len(dead_features),
            'living_mean': np.mean(living_features, axis=0),
            'living_std': np.std(living_features, axis=0),
            'dead_mean': np.mean(dead_features, axis=0),
            'dead_std': np.std(dead_features, axis=0),
        }
        
        # Feature importance (t-test)
        from scipy.stats import ttest_ind
        
        t_stats = []
        p_values = []
        
        for i in range(features.shape[1]):
            t_stat, p_val = ttest_ind(living_features[:, i], dead_features[:, i])
            t_stats.append(abs(t_stat))
            p_values.append(p_val)
        
        stats['t_stats'] = np.array(t_stats)
        stats['p_values'] = np.array(p_values)
        
        return stats
    
    def visualize_top_features(self, features, states, save_path, n_features=12):
        """
        가장 차이 나는 feature들 시각화
        """
        # Feature importance 계산
        stats = self.compute_statistics(features, states)
        top_indices = np.argsort(stats['t_stats'])[::-1][:n_features]
        
        living_mask = (states == 'living')
        dead_mask = (states == 'dead')
        
        # Plot
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5))
        axes = axes.flatten()
        
        for i, feat_idx in enumerate(top_indices):
            if i >= n_features:
                break
            
            living_vals = features[living_mask, feat_idx]
            dead_vals = features[dead_mask, feat_idx]
            
            axes[i].hist(living_vals, bins=30, alpha=0.5, 
                        label=f'Living (n={len(living_vals)})', color='green', density=True)
            axes[i].hist(dead_vals, bins=30, alpha=0.5, 
                        label=f'Dead (n={len(dead_vals)})', color='red', density=True)
            
            # p-value 표시
            p_val = stats['p_values'][feat_idx]
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            
            axes[i].set_title(f'Feature {feat_idx} ({sig})', fontsize=10)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Density')
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
        
        # 빈 subplot 제거
        for i in range(n_features, len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Top Discriminative Features (Live vs Dead)', fontsize=14, y=1.00)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Top features saved: {save_path}")
    
    def visualize_pca(self, features, states, save_path):
        """
        PCA로 2D 투영 시각화
        """
        # Normalize
        features_norm = self.normalize_features(features)
        
        # PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_norm)
        
        living_mask = (states == 'living')
        dead_mask = (states == 'dead')
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(features_2d[living_mask, 0], features_2d[living_mask, 1],
                  c='green', alpha=0.6, s=30, label=f'Living (n={np.sum(living_mask)})', edgecolors='darkgreen', linewidth=0.5)
        ax.scatter(features_2d[dead_mask, 0], features_2d[dead_mask, 1],
                  c='red', alpha=0.6, s=30, label=f'Dead (n={np.sum(dead_mask)})', edgecolors='darkred', linewidth=0.5)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title('Live vs Dead Cells - PCA Projection\n(Morphological Features)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ PCA visualization saved: {save_path}")
    
    def visualize_tsne(self, features, states, save_path):
        """
        t-SNE로 2D 투영 시각화
        """
        # Normalize
        features_norm = self.normalize_features(features)
        
        # t-SNE
        print("  Running t-SNE (this may take a while)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//2))
        features_2d = tsne.fit_transform(features_norm)
        
        living_mask = (states == 'living')
        dead_mask = (states == 'dead')
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(features_2d[living_mask, 0], features_2d[living_mask, 1],
                  c='green', alpha=0.6, s=30, label=f'Living (n={np.sum(living_mask)})', edgecolors='darkgreen', linewidth=0.5)
        ax.scatter(features_2d[dead_mask, 0], features_2d[dead_mask, 1],
                  c='red', alpha=0.6, s=30, label=f'Dead (n={np.sum(dead_mask)})', edgecolors='darkred', linewidth=0.5)
        
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title('Live vs Dead Cells - t-SNE Projection\n(Morphological Features)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ t-SNE visualization saved: {save_path}")

# ============================================================
# Step 4: Main Pipeline
# ============================================================

class SimpleMorphoPipeline:
    """
    간단한 형태학적 분석 파이프라인 (VAE 없이)
    """
    
    def __init__(self):
        self.crop_extractor = CellCropExtractor(crop_size=256)
        self.feature_extractor = MorphologicalFeatureExtractor()
        self.analyzer = MorphologicalAnalyzer()
    
    def process_cellpose_results(self, results_dir, output_dir):
        """
        Cellpose 결과 → 형태학적 특징 추출 → 시각화
        
        Args:
            results_dir: demo_imsang.py 출력 디렉토리
                - phase_XXX_seg_masks.npy
                - phase_XXX_labels.npy (dead_labels, living_labels 포함)
            output_dir: 출력 디렉토리
        """
        results_dir = Path(results_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("Morphological Analysis Pipeline (without VAE)")
        print("="*70)
        print(f"\nInput:  {results_dir}")
        print(f"Output: {output_dir}")
        
        # Cellpose 결과 파일 찾기
        seg_files = list(results_dir.glob("*_seg_masks.npy"))
        
        if len(seg_files) == 0:
            print(f"\n✗ No segmentation files found in {results_dir}")
            print("\nExpected files:")
            print("  - *_seg_masks.npy")
            print("  - *_labels.npy")
            print("\nPlease run demo_imsang.py first!")
            return
        
        print(f"\n✓ Found {len(seg_files)} segmentation files")
        
        all_crops = []
        all_labels = []
        all_states = []
        all_image_names = []
        
        # Phase 이미지 디렉토리
        # results_dir은 kanglab_data/results/run_XXXX 형태이므로
        # phase는 kanglab_data/phase에 있음 (parent.parent)
        phase_dir = results_dir.parent.parent / "phase"
        
        if not phase_dir.exists():
            print(f"\n✗ Phase directory not found: {phase_dir}")
            return
        
        # 각 이미지 처리
        for seg_file in tqdm(seg_files, desc="Extracting crops"):
            img_name = seg_file.stem.replace("_seg_masks", "")
            
            # 파일 로드
            mask = np.load(seg_file)
            labels_file = results_dir / f"{img_name}_labels.npy"
            
            if not labels_file.exists():
                print(f"Warning: No labels file for {img_name}")
                continue
            
            labels_dict = np.load(labels_file, allow_pickle=True).item()
            dead_labels = labels_dict['dead_labels']
            living_labels = labels_dict['living_labels']
            
            # 원본 phase 이미지 로드
            phase_file = None
            for ext in ['.tif', '.tiff', '.png']:
                potential_file = phase_dir / f"{img_name}{ext}"
                if potential_file.exists():
                    phase_file = potential_file
                    break
            
            if phase_file is None:
                print(f"Warning: No phase image for {img_name}")
                continue
            
            phase_img = cv2.imread(str(phase_file), cv2.IMREAD_GRAYSCALE)
            
            # Crops 추출
            cell_labels = dead_labels + living_labels
            cell_states = ['dead'] * len(dead_labels) + ['living'] * len(living_labels)
            
            crops, valid_labels, valid_states = self.crop_extractor.extract_all_cells(
                phase_img, mask, cell_labels, cell_states
            )
            
            if crops is not None:
                all_crops.append(crops)
                all_labels.append(valid_labels)
                all_states.append(valid_states)
                all_image_names.extend([img_name] * len(valid_labels))
        
        # 통합
        all_crops = np.vstack(all_crops)  # (N, 256, 256)
        all_labels = np.concatenate(all_labels)
        all_states = np.concatenate(all_states)
        
        print(f"\n✓ Extracted {len(all_crops)} cell crops")
        print(f"  Living: {np.sum(all_states == 'living')}")
        print(f"  Dead: {np.sum(all_states == 'dead')}")
        
        # 형태학적 특징 추출
        print("\n" + "="*70)
        print("Extracting Morphological Features")
        print("="*70)
        
        features = self.feature_extractor.extract_all_features(all_crops)
        
        # 저장
        np.save(output_dir / "cell_crops.npy", all_crops)
        np.save(output_dir / "cell_features.npy", features)
        np.save(output_dir / "cell_labels.npy", all_labels)
        np.save(output_dir / "cell_states.npy", all_states)
        
        # Metadata
        metadata = pd.DataFrame({
            'image_name': all_image_names,
            'cell_label': all_labels,
            'cell_state': all_states
        })
        metadata.to_csv(output_dir / "cell_metadata.csv", index=False)
        
        print(f"\n✓ Data saved to: {output_dir}")
        
        # 시각화
        print("\n" + "="*70)
        print("Visualizing Morphological Differences")
        print("="*70)
        
        self.analyzer.visualize_top_features(
            features, all_states, 
            output_dir / "top_features.png"
        )
        
        self.analyzer.visualize_pca(
            features, all_states,
            output_dir / "pca_projection.png"
        )
        
        self.analyzer.visualize_tsne(
            features, all_states,
            output_dir / "tsne_projection.png"
        )
        
        # 통계
        stats = self.analyzer.compute_statistics(features, all_states)
        
        stats_path = output_dir / "statistics.txt"
        with open(stats_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("Live vs Dead Morphological Analysis\n")
            f.write("="*70 + "\n")
            f.write(f"Total Cells: {len(features)}\n")
            f.write(f"Living Cells: {stats['living_count']}\n")
            f.write(f"Dead Cells: {stats['dead_count']}\n")
            f.write(f"Dead Ratio: {stats['dead_count']/len(features)*100:.1f}%\n")
            f.write("\n")
            f.write("Feature Dimensions:\n")
            f.write(f"  Total features: {features.shape[1]}\n")
            f.write("\n")
            f.write("Top 10 Discriminative Features (by t-statistic):\n")
            top_indices = np.argsort(stats['t_stats'])[::-1][:10]
            for i, idx in enumerate(top_indices, 1):
                f.write(f"  {i}. Feature {idx}: t={stats['t_stats'][idx]:.2f}, p={stats['p_values'][idx]:.4f}\n")
            f.write("="*70 + "\n")
        
        print(f"\n✓ Statistics saved: {stats_path}")
        print("\n" + "="*70)
        print("Pipeline Complete!")
        print("="*70)
        print(f"\nResults:")
        print(f"  {output_dir}/")
        print(f"  ├── cell_crops.npy")
        print(f"  ├── cell_features.npy")
        print(f"  ├── cell_metadata.csv")
        print(f"  ├── top_features.png")
        print(f"  ├── pca_projection.png")
        print(f"  ├── tsne_projection.png")
        print(f"  └── statistics.txt")

# ============================================================
# Main Execution
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Morphological Analysis Pipeline (no VAE)')
    parser.add_argument('--cellpose-results', type=str, required=True,
                       help='Cellpose results directory (from demo_imsang.py)')
    parser.add_argument('--output', type=str, default='./morphology_analysis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    pipeline = SimpleMorphoPipeline()
    pipeline.process_cellpose_results(
        results_dir=args.cellpose_results,
        output_dir=args.output
    )