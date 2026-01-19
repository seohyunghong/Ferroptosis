import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from cellpose import models, io, core
from skimage import measure, morphology, filters, segmentation
from scipy import ndimage
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
import cv2

# ===== 설정 =====
class Config:
    # 데이터 경로
    DATA_ROOT = "./data/ferroptosis/kanglab_data"
    IMG_DIR = os.path.join(DATA_ROOT, "phase")  # 원본 이미지
    GREEN_GT_DIR = os.path.join(DATA_ROOT, "green")  # GT 형광 이미지
    OUTPUT_DIR = os.path.join(DATA_ROOT, "results")
    
    # Cellpose 설정
    MODEL_TYPE = 'cyto3'
    DIAMETER = None
    FLOW_THRESHOLD = 0.3
    CELLPROB_THRESHOLD = 0.0
    
    # Green GT mask 생성 파라미터
    MIN_SPOT_AREA = 0  # 최소 spot 면적 (픽셀)
    MAX_SPOT_AREA = 100000  # 최대 spot 면적
    
    # Morphology 연산 파라미터
    GAUSSIAN_SIGMA = 2  # Gaussian blur
    MORPH_KERNEL_SIZE = 1  # Opening 연산 커널 크기
    
    USE_GPU = core.use_gpu()

# ===== Green GT Mask 생성 (Otsu) =====

def create_gt_mask_otsu(green_img, visualize=False):
    """
    Green 형광 이미지에서 Otsu threshold로 GT mask 생성

    Args:
        green_img: Green channel 이미지 (grayscale)
        visualize: 중간 과정 시각화 여부

    Returns:
        gt_mask: Binary GT mask
        spot_centroids: 검출된 spot의 중심 좌표 리스트 [(y, x), ...]
    """
    # 1. 8-bit 이미지로 변환 (smoothing 없이 바로 사용)
    if green_img.max() <= 1.0:
        img_uint8 = (green_img * 255).astype(np.uint8)
    else:
        img_uint8 = green_img.astype(np.uint8)

    # 2. OpenCV Otsu thresholding
    otsu_threshold, gt_mask_uint8 = cv2.threshold(
        img_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"    - Otsu threshold: {otsu_threshold:.2f} (0-255 scale)")

    # 3. Binary mask를 bool로 변환
    gt_mask = gt_mask_uint8 > 0

    # Opening 전 객체 수 계산
    labeled_before = measure.label(gt_mask)
    num_objects_before = labeled_before.max()

    # 4. Morphological opening (작은 노이즈 제거)
    kernel = morphology.disk(Config.MORPH_KERNEL_SIZE)
    gt_mask_clean = morphology.binary_opening(gt_mask, kernel)

    # Opening 후 객체 수 계산
    labeled_after = measure.label(gt_mask_clean)
    num_objects_after = labeled_after.max()

    # 5. Connected component labeling
    labeled = measure.label(gt_mask_clean)
    regions = measure.regionprops(labeled)

    # 6. Spot 필터링 (면적 기준만)
    valid_regions = []
    final_labeled_mask = np.zeros_like(green_img, dtype=np.int32)
    final_gt_mask = np.zeros_like(green_img, dtype=bool)

    new_label = 1
    for region in regions:
        area = region.area
        if Config.MIN_SPOT_AREA <= area <= Config.MAX_SPOT_AREA:
            valid_regions.append(region)
            final_gt_mask[labeled == region.label] = True
            final_labeled_mask[labeled == region.label] = new_label
            new_label += 1

    # 7. Centroid 추출
    spot_centroids = [(region.centroid[0], region.centroid[1]) for region in valid_regions]

    gt_pixels = np.sum(final_gt_mask)
    gt_percentage = gt_pixels / final_gt_mask.size * 100
    print(f"    - GT mask pixels: {gt_pixels} ({gt_percentage:.2f}%)")
    print(f"    - GT regions (spots): {len(spot_centroids)}")

    if visualize:
        from skimage.color import label2rgb

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # [0,0] Original
        axes[0, 0].imshow(green_img, cmap='Greens')
        axes[0, 0].set_title('Original Green GT', fontsize=12)
        axes[0, 0].axis('off')

        # [0,1] Histogram with Otsu threshold (0-255 스케일)
        axes[0, 1].hist(img_uint8.flatten(), bins=256, range=(0, 255), color='green', alpha=0.7)
        axes[0, 1].axvline(otsu_threshold, color='red', linestyle='--', linewidth=2, label=f'Otsu: {otsu_threshold:.0f}')
        axes[0, 1].set_title('Histogram with Otsu Threshold', fontsize=12)
        axes[0, 1].set_xlabel('Intensity (0-255)')
        axes[0, 1].set_ylabel('Count (log scale)')
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_xlim(0, 255)
        axes[0, 1].legend()

        # [1,0] Binary before vs after opening
        axes[1, 0].imshow(gt_mask, cmap='gray')
        axes[1, 0].set_title(f'Binary (Otsu): {num_objects_before} objects\n→ After Opening: {num_objects_after} objects', fontsize=12)
        axes[1, 0].axis('off')

        # [1,1] Final GT mask (객체별 색상)
        final_overlay = label2rgb(final_labeled_mask, bg_label=0)
        axes[1, 1].imshow(final_overlay)
        axes[1, 1].set_title(f'Final GT Mask\n{len(spot_centroids)} regions, {gt_percentage:.2f}% pixels', fontsize=12)
        axes[1, 1].axis('off')

        plt.tight_layout()
        return final_gt_mask, spot_centroids, fig

    return final_gt_mask, spot_centroids

# ===== Mask-Green Spot 매칭 =====
def match_masks_to_spots_robust(masks, spot_centroids, search_radius=10): 
    """
    Centroid 주변 영역을 검색해서 가장 가까운 세포 매칭
    """
    dead_cell_labels = set()
    
    for y, x in spot_centroids:
        yi, xi = int(round(y)), int(round(x))
        
        # 주변 영역 추출
        y_min = max(0, yi - search_radius)
        y_max = min(masks.shape[0], yi + search_radius)
        x_min = max(0, xi - search_radius)
        x_max = min(masks.shape[1], xi + search_radius)
        
        region = masks[y_min:y_max, x_min:x_max]
        
        # 가장 많이 나타나는 label (배경 제외)
        labels, counts = np.unique(region[region > 0], return_counts=True)
        
        if len(labels) > 0:
            most_common_label = labels[np.argmax(counts)]
            dead_cell_labels.add(most_common_label)
    
    return dead_cell_labels

def match_masks_to_spots_iou(seg_masks, gt_mask, iou_threshold=0.1):
    """
    각 spot region과 seg mask의 IoU로 매칭
    """
    from skimage import measure
    
    dead_cell_labels = set()
    
    # GT mask에서 각 spot region 추출
    labeled_gt = measure.label(gt_mask)
    gt_regions = measure.regionprops(labeled_gt)
    
    for gt_region in gt_regions:
        # 이 GT region의 bbox 영역에서 seg mask 찾기
        minr, minc, maxr, maxc = gt_region.bbox
        
        gt_region_mask = np.zeros_like(seg_masks, dtype=bool)
        gt_region_mask[labeled_gt == gt_region.label] = True
        
        # 각 seg mask와 IoU 계산
        for seg_label in range(1, seg_masks.max() + 1):
            seg_cell_mask = (seg_masks == seg_label)
            
            intersection = np.logical_and(seg_cell_mask, gt_region_mask).sum()
            union = np.logical_or(seg_cell_mask, gt_region_mask).sum()
            
            if union > 0:
                iou = intersection / union
                if iou > iou_threshold:
                    dead_cell_labels.add(seg_label)
    
    return dead_cell_labels
# ===== Feature Extraction =====

def extract_cell_features(mask, intensity_image, cell_labels=None):
    """
    세포별 형태학적 특징 추출
    """
    if cell_labels is not None:
        filtered_mask = np.zeros_like(mask)
        for label in cell_labels:
            filtered_mask[mask == label] = label
        regions = measure.regionprops(filtered_mask, intensity_image=intensity_image)
    else:
        regions = measure.regionprops(mask, intensity_image=intensity_image)
    
    features = []
    for region in regions:
        feature = {
            'label': region.label,
            'area': region.area,
            'perimeter': region.perimeter,
            'eccentricity': region.eccentricity,
            'solidity': region.solidity,
            'mean_intensity': region.mean_intensity,
            'max_intensity': region.max_intensity,
            'min_intensity': region.min_intensity,
            'centroid_y': region.centroid[0],
            'centroid_x': region.centroid[1],
            'bbox_min_y': region.bbox[0],
            'bbox_min_x': region.bbox[1],
            'bbox_max_y': region.bbox[2],
            'bbox_max_x': region.bbox[3],
        }
        
        if region.perimeter > 0:
            feature['circularity'] = (4 * np.pi * region.area) / (region.perimeter ** 2)
        else:
            feature['circularity'] = 0
        
        feature['aspect_ratio'] = region.major_axis_length / (region.minor_axis_length + 1e-10)
        
        features.append(feature)
    
    return pd.DataFrame(features)

# ===== 세포 분류 시각화 =====

def visualize_cell_classification(original_img, green_gt_img, masks, dead_labels, living_labels, save_path):
    """
    세포 분류 결과 시각화 (Dead/Living 구분)
    """
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # [0, 0] Original Image
    axes[0, 0].imshow(original_img, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')

    # [0, 1] Green GT
    axes[0, 1].imshow(green_gt_img, cmap='Greens')
    axes[0, 1].set_title('Green GT', fontsize=14)
    axes[0, 1].axis('off')

    # [1, 0] Dead Cells (red)
    dead_mask = np.zeros((*masks.shape, 3), dtype=np.float32)
    for label in dead_labels:
        dead_mask[masks == label] = [1, 0, 0]

    original_normalized = original_img.astype(np.float32) / original_img.max()
    if len(original_normalized.shape) == 2:
        original_rgb = np.stack([original_normalized] * 3, axis=-1)
    else:
        original_rgb = original_normalized

    dead_exists = np.any(dead_mask > 0, axis=-1)[:, :, np.newaxis]
    dead_overlay = np.where(dead_exists, dead_mask * 0.6 + original_rgb * 0.4, original_rgb)
    axes[1, 0].imshow(np.clip(dead_overlay, 0, 1))
    axes[1, 0].set_title(f'Dead Cells (n={len(dead_labels)})', fontsize=14, color='red')
    axes[1, 0].axis('off')

    # [1, 1] Living Cells (green)
    living_mask = np.zeros((*masks.shape, 3), dtype=np.float32)
    for label in living_labels:
        living_mask[masks == label] = [0, 1, 0]

    living_exists = np.any(living_mask > 0, axis=-1)[:, :, np.newaxis]
    living_overlay = np.where(living_exists, living_mask * 0.6 + original_rgb * 0.4, original_rgb)
    axes[1, 1].imshow(np.clip(living_overlay, 0, 1))
    axes[1, 1].set_title(f'Living Cells (n={len(living_labels)})', fontsize=14, color='green')
    axes[1, 1].axis('off')

    dead_ratio = len(dead_labels)/(len(dead_labels)+len(living_labels))*100 if (len(dead_labels)+len(living_labels)) > 0 else 0
    plt.suptitle(f'Cell Classification\nDead: {len(dead_labels)} | Living: {len(living_labels)} | Dead Ratio: {dead_ratio:.1f}%',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

# ===== 메인 파이프라인 =====

def process_image_pair(original_path, green_gt_path, output_dir, visualize_spots=False):
    """
    원본 + Green GT 이미지 쌍 처리
    """
    img_name = Path(original_path).stem
    print(f"\n  Processing: {img_name}")

    # 1. 이미지 로드
    original_img = io.imread(original_path)
    green_gt_img = io.imread(green_gt_path)

    if len(green_gt_img.shape) == 3:
        green_gt_img = green_gt_img[:, :, 1]

    print(f"    - Original image shape: {original_img.shape}")
    print(f"    - Green GT image shape: {green_gt_img.shape}")
    
    if green_gt_img.shape != original_img.shape[:2]:
        from skimage.transform import resize
        print(f"    - Resizing green image to match original image...")
        green_gt_img = resize(green_gt_img, original_img.shape[:2], preserve_range=True, anti_aliasing=True)
        green_gt_img = green_gt_img.astype(np.uint8)
        print(f"    - Resized green image shape: {green_gt_img.shape}")
    
    # 2. Cellpose Segmentation
    print(f"    - Running Cellpose segmentation...")
    model = models.CellposeModel(gpu=Config.USE_GPU, model_type=Config.MODEL_TYPE)
    masks, flows, styles = model.eval(
        original_img,
        diameter=Config.DIAMETER,
        channels=[0, 0],
        flow_threshold=Config.FLOW_THRESHOLD,
        cellprob_threshold=Config.CELLPROB_THRESHOLD
    )
    
    total_cells = masks.max()
    print(f"    - Total segmented cells: {total_cells}")
    
    # 3. GT Mask 생성 (Otsu)
    print(f"    - Creating GT mask with Otsu thresholding...")
    if visualize_spots:
        gt_mask, spot_centroids, gt_fig = create_gt_mask_otsu(green_gt_img, visualize=True)
        gt_vis_path = os.path.join(output_dir, f"{img_name}_gt_mask_creation.png")
        gt_fig.savefig(gt_vis_path, dpi=150, bbox_inches='tight')
        plt.close(gt_fig)
        print(f"    - GT mask creation visualization saved")
    else:
        gt_mask, spot_centroids = create_gt_mask_otsu(green_gt_img, visualize=False)
    
    # 4. Mask-Spot 매칭
    print(f"    - Matching spots to cell masks...")
    dead_cell_labels = match_masks_to_spots_iou(masks, gt_mask)
    all_cell_labels = set(range(1, total_cells + 1))
    living_cell_labels = all_cell_labels - dead_cell_labels
    
    print(f"    - Dead cells: {len(dead_cell_labels)}")
    print(f"    - Living cells: {len(living_cell_labels)}")
    if total_cells > 0:
        print(f"    - Dead ratio: {len(dead_cell_labels)/total_cells*100:.1f}%")
    
    # 5. Feature Extraction
    print(f"    - Extracting features...")
    dead_features = extract_cell_features(masks, original_img, cell_labels=dead_cell_labels)
    dead_features['cell_state'] = 'Dead'

    living_features = extract_cell_features(masks, original_img, cell_labels=living_cell_labels)
    living_features['cell_state'] = 'Living'
    
    all_features = pd.concat([dead_features, living_features], ignore_index=True)
    
    # 6. 저장
    csv_path = os.path.join(output_dir, f"{img_name}_features.csv")
    all_features.to_csv(csv_path, index=False)
    
    mask_path = os.path.join(output_dir, f"{img_name}_seg_masks.npy")
    np.save(mask_path, masks)
    
    gt_mask_path = os.path.join(output_dir, f"{img_name}_gt_mask.npy")
    np.save(gt_mask_path, gt_mask)
    
    labels_dict = {
        'dead_labels': list(dead_cell_labels),
        'living_labels': list(living_cell_labels)
    }
    labels_path = os.path.join(output_dir, f"{img_name}_labels.npy")
    np.save(labels_path, labels_dict)
    
    # 7. 시각화
    vis_path = os.path.join(output_dir, f"{img_name}_classification.png")
    visualize_cell_classification(original_img, green_gt_img, masks, dead_cell_labels, living_cell_labels, vis_path)
    
    # GT vs Segmentation 비교
    gt_vs_seg_path = os.path.join(output_dir, f"{img_name}_gt_vs_seg.png")
    iou, precision, recall, f1 = visualize_gt_vs_seg(original_img, green_gt_img, gt_mask, masks, gt_vs_seg_path)
    print(f"    - GT vs Seg - IoU: {iou:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    # 통계
    summary = {
        'image_name': img_name,
        'total_cells': int(total_cells),
        'dead_cells': int(len(dead_cell_labels)),
        'living_cells': int(len(living_cell_labels)),
        'dead_ratio': float(len(dead_cell_labels) / total_cells) if total_cells > 0 else 0,
        'gt_regions': len(spot_centroids),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
    
    return all_features, masks, summary

# ===== 배치 처리 =====

def batch_process(img_dir, green_gt_dir, base_output_dir):
    """
    배치 처리
    """
    print(f"{'='*70}")
    print(f">>> GPU Activated: {Config.USE_GPU}")
    print(f">>> Model: {Config.MODEL_TYPE}")
    print(f"{'='*70}")

    KST = timezone(timedelta(hours=9))
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f">>> Output directory: {output_dir}")

    original_files = []
    for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']:
        original_files.extend(Path(img_dir).glob(ext))

    print(f">>> Found {len(original_files)} original images")

    if len(original_files) == 0:
        print(f"!!! No images found in {img_dir}")
        return

    matched_pairs = []
    for original_file in original_files:
        original_name = original_file.stem
        original_ext = original_file.suffix

        if original_name.startswith('phase_'):
            green_gt_name = 'green_' + original_name[6:]
        else:
            green_gt_name = original_name

        green_gt_file = None
        for ext in [original_ext, '.tif', '.tiff', '.png', '.jpg', '.jpeg']:
            potential_file = Path(green_gt_dir) / (green_gt_name + ext)
            if potential_file.exists():
                green_gt_file = potential_file
                break

        if green_gt_file:
            matched_pairs.append((str(original_file), str(green_gt_file)))
        else:
            print(f"!!! No matching green GT for {original_file.name}")

    print(f">>> Matched {len(matched_pairs)} image pairs")
    
    if len(matched_pairs) == 0:
        print("!!! No matched pairs found.")
        return
    
    all_summaries = []

    for original_path, green_gt_path in tqdm(matched_pairs, desc="Processing pairs"):
        try:
            _, _, summary = process_image_pair(
                original_path, green_gt_path, output_dir,
                visualize_spots=True
            )
            all_summaries.append(summary)
        except Exception as e:
            print(f"\n  !!! Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_csv = os.path.join(output_dir, "batch_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        
        stats_path = os.path.join(output_dir, "statistics.txt")
        with open(stats_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CELL DEATH ANALYSIS - BATCH SUMMARY\n")
            f.write("="*70 + "\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write(f"Total Pairs Processed: {len(all_summaries)}\n")
            f.write(f"Total Cells: {summary_df['total_cells'].sum()}\n")
            f.write(f"Total Dead Cells: {summary_df['dead_cells'].sum()}\n")
            f.write(f"Total Living Cells: {summary_df['living_cells'].sum()}\n")
            f.write(f"Average Dead Ratio: {summary_df['dead_ratio'].mean():.2%}\n")
            f.write(f"Average GT Regions: {summary_df['gt_regions'].mean():.1f}\n")
            f.write(f"Average IoU: {summary_df['iou'].mean():.3f}\n")
            f.write(f"Average Precision: {summary_df['precision'].mean():.3f}\n")
            f.write(f"Average Recall: {summary_df['recall'].mean():.3f}\n")
            f.write(f"Average F1: {summary_df['f1'].mean():.3f}\n")
            f.write("="*70 + "\n")
        
        print(f"\n{'='*70}")
        print(">>> Batch Processing Complete!")
        print(f">>> Total Pairs: {len(all_summaries)}")
        print(f">>> Total Cells: {summary_df['total_cells'].sum()}")
        print(f">>> Total Dead: {summary_df['dead_cells'].sum()}")
        print(f">>> Avg Dead Ratio: {summary_df['dead_ratio'].mean():.1%}")
        print(f">>> Avg IoU: {summary_df['iou'].mean():.3f}")
        print(f">>> Results: {output_dir}")
        print(f"{'='*70}")

# ===== 실행 =====

if __name__ == '__main__':
    batch_process(
        img_dir=Config.IMG_DIR,
        green_gt_dir=Config.GREEN_GT_DIR,
        base_output_dir=Config.OUTPUT_DIR
    )