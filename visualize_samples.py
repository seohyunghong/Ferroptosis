import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from cellpose import models, io, core
from skimage import measure, morphology, filters
from skimage.color import label2rgb
import sys

# demo_imsang.py에서 함수 import
sys.path.insert(0, os.path.dirname(__file__))
from demo_imsang import Config, detect_green_spots, match_masks_to_spots

def visualize_processing_pipeline(original_path, green_gt_path, output_path):
    """
    전체 처리 파이프라인을 한 이미지에 시각화
    """
    print(f"Processing: {Path(original_path).name}")

    # 1. 이미지 로드
    original_img = io.imread(original_path)
    green_gt_img = io.imread(green_gt_path)

    # Green GT가 3채널이면 green channel만 추출
    if len(green_gt_img.shape) == 3:
        green_gt_img = green_gt_img[:, :, 1]

    # 2. Cellpose Segmentation
    print("  Running Cellpose segmentation...")
    model = models.CellposeModel(gpu=Config.USE_GPU, model_type=Config.MODEL_TYPE)
    masks, flows, styles = model.eval(
        original_img,
        diameter=Config.DIAMETER,
        channels=[0, 0],
        flow_threshold=Config.FLOW_THRESHOLD,
        cellprob_threshold=Config.CELLPROB_THRESHOLD
    )

    # 3. Green spot 검출
    print("  Detecting green spots...")
    spot_centroids, spot_mask = detect_green_spots(green_gt_img, visualize=False)

    # 4. Mask-Spot 매칭
    print("  Matching spots to masks...")
    dead_cell_labels = match_masks_to_spots(masks, spot_centroids)
    all_cell_labels = set(range(1, masks.max() + 1))
    living_cell_labels = all_cell_labels - dead_cell_labels

    # 5. 시각화
    print("  Creating visualization...")
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Row 1: 원본 이미지들
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_img, cmap='gray')
    ax1.set_title('1. Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(green_gt_img, cmap='Greens')
    ax2.set_title('2. Green GT', fontsize=14, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(spot_mask, cmap='gray')
    ax3.set_title(f'3. Detected Spots (n={len(spot_centroids)})', fontsize=14, fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(green_gt_img, cmap='gray')
    for y, x in spot_centroids:
        ax4.plot(x, y, 'r+', markersize=12, markeredgewidth=2)
    ax4.set_title('4. Spot Centroids', fontsize=14, fontweight='bold')
    ax4.axis('off')

    # Row 2: Segmentation 결과
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(masks, cmap='nipy_spectral')
    ax5.set_title(f'5. Cell Masks (n={masks.max()})', fontsize=14, fontweight='bold')
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 1])
    overlay_all = label2rgb(masks, image=original_img, bg_label=0, alpha=0.4)
    ax6.imshow(overlay_all)
    ax6.set_title('6. Masks on Original', fontsize=14, fontweight='bold')
    ax6.axis('off')

    ax7 = fig.add_subplot(gs[1, 2])
    dead_mask = np.zeros_like(masks)
    for label in dead_cell_labels:
        dead_mask[masks == label] = label
    overlay_dead = label2rgb(dead_mask, image=original_img, bg_label=0, alpha=0.6, colors=[(1, 0, 0)])
    ax7.imshow(overlay_dead)
    ax7.set_title(f'7. Dead Cells (n={len(dead_cell_labels)})', fontsize=14, fontweight='bold', color='red')
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[1, 3])
    living_mask = np.zeros_like(masks)
    for label in living_cell_labels:
        living_mask[masks == label] = label
    overlay_living = label2rgb(living_mask, image=original_img, bg_label=0, alpha=0.6, colors=[(0, 1, 0)])
    ax8.imshow(overlay_living)
    ax8.set_title(f'8. Living Cells (n={len(living_cell_labels)})', fontsize=14, fontweight='bold', color='green')
    ax8.axis('off')

    # Row 3: 최종 결과
    ax9 = fig.add_subplot(gs[2, :2])
    ax9.imshow(original_img, cmap='gray', alpha=0.7)
    # Spot centroids 표시
    for y, x in spot_centroids:
        ax9.plot(x, y, 'y*', markersize=10, markeredgewidth=1, alpha=0.8)
    # Dead cells - red X
    from scipy import ndimage
    for label in dead_cell_labels:
        mask_region = masks == label
        y, x = ndimage.center_of_mass(mask_region)
        ax9.scatter(x, y, c='red', s=150, alpha=0.9, marker='x', linewidths=3)
    # Living cells - green O
    for label in living_cell_labels:
        mask_region = masks == label
        y, x = ndimage.center_of_mass(mask_region)
        ax9.scatter(x, y, c='lime', s=100, alpha=0.9, marker='o', linewidths=2, facecolors='none')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='yellow', label=f'Green Spots ({len(spot_centroids)})'),
        Patch(facecolor='red', label=f'Dead Cells ({len(dead_cell_labels)})'),
        Patch(facecolor='lime', label=f'Living Cells ({len(living_cell_labels)})')
    ]
    ax9.legend(handles=legend_elements, loc='upper right', fontsize=12)
    ax9.set_title('9. Final Classification Result', fontsize=14, fontweight='bold')
    ax9.axis('off')

    ax10 = fig.add_subplot(gs[2, 2:])
    # 통계 텍스트
    stats_text = f"""
    Processing Statistics
    ══════════════════════════════════

    Total Cells Detected:     {masks.max()}

    Green Spots Detected:     {len(spot_centroids)}

    Dead Cells:               {len(dead_cell_labels)}
    Living Cells:             {len(living_cell_labels)}

    Dead Cell Ratio:          {len(dead_cell_labels)/masks.max()*100:.1f}%
    Living Cell Ratio:        {len(living_cell_labels)/masks.max()*100:.1f}%

    ══════════════════════════════════

    Classification Method:
    - Cells with green spots → Dead
    - Cells without spots → Living
    """
    ax10.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
              verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax10.axis('off')

    # 전체 제목
    img_name = Path(original_path).stem
    fig.suptitle(f'Cell Death Analysis Pipeline - {img_name}',
                 fontsize=18, fontweight='bold', y=0.98)

    # 저장
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}\n")

def main():
    """3개 샘플 이미지 시각화"""
    print("="*70)
    print("Visualizing Sample Processing Pipeline")
    print("="*70)

    # 이미지 찾기
    img_dir = Path(Config.IMG_DIR)
    green_gt_dir = Path(Config.GREEN_GT_DIR)

    original_files = []
    for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']:
        original_files.extend(img_dir.glob(ext))

    # 매칭되는 쌍 찾기
    matched_pairs = []
    for original_file in original_files:
        original_name = original_file.stem
        if original_name.startswith('Era_'):
            green_gt_name = 'green_' + original_name[4:]
        else:
            green_gt_name = original_name

        green_gt_file = None
        for ext in [original_file.suffix, '.tif', '.tiff', '.png', '.jpg', '.jpeg']:
            potential_file = green_gt_dir / (green_gt_name + ext)
            if potential_file.exists():
                green_gt_file = potential_file
                break

        if green_gt_file:
            matched_pairs.append((str(original_file), str(green_gt_file)))

    print(f"Found {len(matched_pairs)} matched pairs")

    # 3개 샘플만 처리
    num_samples = min(3, len(matched_pairs))
    samples = matched_pairs[:num_samples]

    # 출력 디렉토리
    output_dir = "./data/ferroptosis/kanglab_data/sample_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # 각 샘플 처리
    for i, (original_path, green_gt_path) in enumerate(samples, 1):
        img_name = Path(original_path).stem
        output_path = os.path.join(output_dir, f"sample_{i}_{img_name}_pipeline.png")
        visualize_processing_pipeline(original_path, green_gt_path, output_path)

    print("="*70)
    print(f"All {num_samples} samples processed!")
    print(f"Results saved to: {output_dir}")
    print("="*70)

if __name__ == '__main__':
    main()
