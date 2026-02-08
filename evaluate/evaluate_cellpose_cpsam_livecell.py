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
    # 1. 8-bit 이미지로 변환
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
        axes[0, 0].imshow(img_uint8, cmap='Greens')
        axes[0, 0].set_title('Original Green GT', fontsize=12)
        axes[0, 0].axis('off')

        # [0,1] Histogram with Otsu threshold (로그 스케일)
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
        return final_gt_mask, spot_centroids, fig, otsu_threshold

    return final_gt_mask, spot_centroids, otsu_threshold

# ===== 개선된 Mask-Green Spot 매칭 =====

def match_masks_to_spots_intersection(seg_masks, gt_mask):
    """
    GT mask의 각 region과 가장 많이 겹치는 seg mask 선택
    
    Args:
        seg_masks: Cellpose segmentation masks
        gt_mask: Binary GT mask
    
    Returns:
        dead_cell_labels: Dead cell로 분류된 seg mask label set
        matching_details: 매칭 상세 정보 (디버깅용)
    """
    from skimage import measure
    
    dead_cell_labels = set()
    matching_details = []
    
    # GT mask에서 각 spot region 추출
    labeled_gt = measure.label(gt_mask)
    gt_regions = measure.regionprops(labeled_gt)
    
    
    for i, gt_region in enumerate(gt_regions):
        # 이 GT region의 mask
        gt_region_mask = (labeled_gt == gt_region.label)
        
        # 각 seg mask와 intersection 계산
        best_intersection = 0
        best_label = None
        
        for seg_label in range(1, seg_masks.max() + 1):
            seg_cell_mask = (seg_masks == seg_label)
            
            # Intersection (겹치는 픽셀 수)
            intersection = np.logical_and(seg_cell_mask, gt_region_mask).sum()
            
            if intersection > best_intersection:
                best_intersection = intersection
                best_label = seg_label
        
        if best_label is not None:
            dead_cell_labels.add(best_label)
            
            cy, cx = gt_region.centroid
            matching_details.append({
                'gt_region_id': i + 1,
                'centroid': (int(cy), int(cx)),
                'matched_seg_label': best_label,
                'area': gt_region.area,
                'intersection': best_intersection
            })
        else:
            cy, cx = gt_region.centroid
            matching_details.append({
                'gt_region_id': i + 1,
                'centroid': (int(cy), int(cx)),
                'matched_seg_label': None,
                'area': gt_region.area,
                'intersection': 0
            })
        
    return dead_cell_labels, matching_details

def match_masks_to_spots_iou(seg_masks, gt_mask, iou_threshold=0.01):
    """
    각 spot region과 seg mask의 IoU로 매칭 (threshold 낮춤)
    """
    from skimage import measure
    
    dead_cell_labels = set()
    
    # GT mask에서 각 spot region 추출
    labeled_gt = measure.label(gt_mask)
    gt_regions = measure.regionprops(labeled_gt)
    
    for gt_region in gt_regions:
        gt_region_mask = np.zeros_like(seg_masks, dtype=bool)
        gt_region_mask[labeled_gt == gt_region.label] = True
        
        # 각 seg mask와 IoU 계산
        best_iou = 0
        best_label = None
        
        for seg_label in range(1, seg_masks.max() + 1):
            seg_cell_mask = (seg_masks == seg_label)
            
            intersection = np.logical_and(seg_cell_mask, gt_region_mask).sum()
            union = np.logical_or(seg_cell_mask, gt_region_mask).sum()
            
            if union > 0:
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
                    best_label = seg_label
        
        if best_iou > iou_threshold and best_label is not None:
            dead_cell_labels.add(best_label)
    
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

# ===== 세포 분류 시각화 (Outline 추가) =====

def get_cell_outlines(masks, labels, thickness=2):
    """
    특정 label들의 outline 추출
    
    Args:
        masks: Segmentation masks
        labels: Label set to extract outlines for
        thickness: Outline 두께 (픽셀)
    
    Returns:
        outline_mask: Boolean mask of outlines
    """
    outline_mask = np.zeros_like(masks, dtype=bool)
    
    for label in labels:
        cell_mask = (masks == label)
        
        # Erosion으로 outline 추출
        eroded = morphology.binary_erosion(cell_mask, morphology.disk(thickness))
        outline = cell_mask & ~eroded
        
        outline_mask |= outline
    
    return outline_mask

def visualize_cell_classification_with_outlines(original_img, green_gt_img, masks, dead_labels, living_labels, otsu_threshold, save_path):
    """
    세포 분류 결과를 8개의 이미지로 시각화 (Outline 추가)
    """
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    # uint8 변환 (histogram용)
    if green_gt_img.max() <= 1.0:
        green_gt_uint8 = (green_gt_img * 255).astype(np.uint8)
    else:
        green_gt_uint8 = green_gt_img.astype(np.uint8)

    # [0] Green GT with Otsu threshold
    axes[0].imshow(green_gt_uint8, cmap='Greens')
    axes[0].set_title(f'Green GT (Otsu: {otsu_threshold:.0f})', fontsize=12)
    axes[0].axis('off')

    # [1] Original Image
    axes[1].imshow(original_img, cmap='gray')
    axes[1].set_title('Original Image', fontsize=12)
    axes[1].axis('off')

    # [2] Green GT Image
    axes[2].imshow(green_gt_uint8, cmap='Greens')
    axes[2].set_title('Green GT', fontsize=12)
    axes[2].axis('off')

    # [3] Instance Segmentation
    segmented_image = np.zeros((*masks.shape, 3), dtype=np.float32)
    for label in np.unique(masks):
        if label > 0:
            segmented_image[masks == label] = np.random.rand(3)
    axes[3].imshow(segmented_image)
    axes[3].set_title(f'Instance Segmentation ({masks.max()} cells)', fontsize=12)
    axes[3].axis('off')

    # [4] Living Cells (Green with outline)
    living_mask = np.zeros((*masks.shape, 3), dtype=np.float32)
    for label in living_labels:
        living_mask[masks == label] = [0, 1, 0]
    
    # Living outline (dark green)
    living_outline = get_cell_outlines(masks, living_labels, thickness=2)
    living_mask[living_outline] = [0, 0.5, 0]
    
    axes[4].imshow(living_mask)
    axes[4].set_title(f'Living Cells ({len(living_labels)})', fontsize=12, color='green')
    axes[4].axis('off')

    # [5] Dead Cells (Red with outline)
    dead_mask = np.zeros((*masks.shape, 3), dtype=np.float32)
    for label in dead_labels:
        dead_mask[masks == label] = [1, 0, 0]
    
    # Dead outline (dark red)
    dead_outline = get_cell_outlines(masks, dead_labels, thickness=2)
    dead_mask[dead_outline] = [0.5, 0, 0]
    
    axes[5].imshow(dead_mask)
    axes[5].set_title(f'Dead Cells ({len(dead_labels)})', fontsize=12, color='red')
    axes[5].axis('off')

    # [6] Living + Dead Cells (Combined with outlines)
    combined_mask = np.zeros((*masks.shape, 3), dtype=np.float32)
    for label in living_labels:
        combined_mask[masks == label] = [0, 1, 0]
    for label in dead_labels:
        combined_mask[masks == label] = [1, 0, 0]
    
    # Outlines
    combined_mask[living_outline] = [0, 0.5, 0]
    combined_mask[dead_outline] = [0.5, 0, 0]
    
    axes[6].imshow(combined_mask)
    axes[6].set_title('Living + Dead', fontsize=12)
    axes[6].axis('off')

    # [7] Otsu Threshold Histogram (로그 스케일)
    axes[7].hist(green_gt_uint8.flatten(), bins=256, range=(0, 255), color='green', alpha=0.7)
    axes[7].axvline(otsu_threshold, color='red', linestyle='--', linewidth=2, label=f'Otsu: {otsu_threshold:.0f}')
    axes[7].set_title('Histogram (log scale)', fontsize=12)
    axes[7].set_xlabel('Intensity (0-255)')
    axes[7].set_ylabel('Count (log scale)')
    axes[7].set_yscale('log')
    axes[7].set_xlim(0, 255)
    axes[7].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def visualize_gt_vs_seg(original_img, green_gt_img, gt_mask, masks, dead_labels, save_path):
    """
    Dead cell에 대해서만 GT mask vs segmentation 비교
    
    Args:
        original_img: Original image
        green_gt_img: Green GT image
        gt_mask: Binary GT mask (전체)
        masks: Segmentation masks (전체)
        dead_labels: Dead cell로 분류된 seg mask labels
        save_path: Save path
    
    Returns:
        iou, precision, recall, f1: Dead cell에 대한 metrics
    
    Metrics:
    - IoU: GT와 Dead seg의 겹침 정도 (0~1)
    - Precision: Dead seg 중 GT와 일치하는 비율 (TP / (TP+FP))
    - Recall: GT 중 Dead seg로 검출된 비율 (TP / (TP+FN))
    - F1: Precision과 Recall의 조화평균
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Dead cell에 해당하는 seg mask만 추출
    dead_seg_mask = np.zeros_like(masks, dtype=bool)
    for label in dead_labels:
        dead_seg_mask |= (masks == label)
    
    # GT mask와 Dead seg mask를 flatten
    gt_mask_flat = gt_mask.flatten()
    dead_seg_flat = dead_seg_mask.flatten()
    
    # IoU (Dead cell 영역만)
    intersection = np.sum(gt_mask_flat & dead_seg_flat)
    union = np.sum(gt_mask_flat | dead_seg_flat)
    iou = intersection / float(union) if union != 0 else 0
    
    # Precision, Recall, F1 (Dead cell 영역만)
    if np.sum(dead_seg_flat) > 0 or np.sum(gt_mask_flat) > 0:
        precision = precision_score(gt_mask_flat, dead_seg_flat, zero_division=0)
        recall = recall_score(gt_mask_flat, dead_seg_flat, zero_division=0)
        f1 = f1_score(gt_mask_flat, dead_seg_flat, zero_division=0)
    else:
        precision = recall = f1 = 0.0
    
    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # [0] Original
    if len(original_img.shape) == 3:
        axes[0].imshow(original_img)
    else:
        axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # [1] GT mask (green spots only)
    axes[1].imshow(gt_mask, cmap='Greens', alpha=0.7)
    axes[1].set_title(f'GT Mask\n({np.sum(gt_mask)} pixels)', fontsize=12)
    axes[1].axis('off')
    
    # [2] Dead cell segmentation
    axes[2].imshow(dead_seg_mask, cmap='Reds', alpha=0.7)
    axes[2].set_title(f'Dead Cell Seg\n({len(dead_labels)} cells, {np.sum(dead_seg_mask)} pixels)', fontsize=12)
    axes[2].axis('off')
    
    # [3] GT (green) vs Dead Seg (red) overlay
    if len(original_img.shape) == 3:
        gray_base = np.mean(original_img, axis=2).astype(np.uint8)
    else:
        gray_base = original_img.astype(np.uint8)
    
    gray_base_norm = gray_base / 255.0
    overlay = np.stack([gray_base_norm, gray_base_norm, gray_base_norm], axis=-1)
    
    # GT only (green), Seg only (red), Overlap (yellow)
    gt_only = gt_mask & ~dead_seg_mask
    seg_only = ~gt_mask & dead_seg_mask
    overlap = gt_mask & dead_seg_mask
    
    alpha = 0.6
    overlay[gt_only] = overlay[gt_only] * (1 - alpha) + np.array([0, 1, 0]) * alpha
    overlay[seg_only] = overlay[seg_only] * (1 - alpha) + np.array([1, 0, 0]) * alpha
    overlay[overlap] = overlay[overlap] * (1 - alpha) + np.array([1, 1, 0]) * alpha
    
    axes[3].imshow(overlay)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label=f'GT only'),
        Patch(facecolor='red', label=f'Seg only'),
        Patch(facecolor='yellow', label=f'Overlap')
    ]
    axes[3].legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    metrics_text = f'IoU: {iou:.3f} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}'
    axes[3].set_title(f'GT vs Dead Seg\n{metrics_text}', fontsize=11)
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # 상세 통계 출력
    print(f"    - Dead Cell GT vs Seg Comparison:")
    print(f"      GT pixels: {np.sum(gt_mask)}")
    print(f"      Dead seg pixels: {np.sum(dead_seg_mask)}")
    print(f"      Intersection: {intersection}")
    print(f"      Union: {union}")
    print(f"      IoU: {iou:.3f}")
    print(f"      Precision: {precision:.3f} (dead seg 중 GT와 일치하는 비율)")
    print(f"      Recall: {recall:.3f} (GT 중 dead seg로 검출된 비율)")
    print(f"      F1: {f1:.3f}")
    
    return iou, precision, recall, f1

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
        gt_mask, spot_centroids, gt_fig, otsu_threshold = create_gt_mask_otsu(green_gt_img, visualize=True)
        gt_vis_path = os.path.join(output_dir, f"{img_name}_gt_mask_creation.png")
        gt_fig.savefig(gt_vis_path, dpi=150, bbox_inches='tight')
        plt.close(gt_fig)
        print(f"    - GT mask creation saved")
    else:
        gt_mask, spot_centroids, otsu_threshold = create_gt_mask_otsu(green_gt_img, visualize=False)
    
    # 4. Mask-Spot 매칭 (Intersection 기반)
    print(f"    - Matching GT spots to cell masks (intersection-based)...")
    dead_cell_labels, matching_details = match_masks_to_spots_intersection(masks, gt_mask)
    
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
        'living_labels': list(living_cell_labels),
        'matching_details': matching_details
    }
    labels_path = os.path.join(output_dir, f"{img_name}_labels.npy")
    np.save(labels_path, labels_dict, allow_pickle=True)
    
    # Visualization with outlines
    save_path = os.path.join(output_dir, f"{img_name}_classification.png")
    visualize_cell_classification_with_outlines(original_img, green_gt_img, masks, dead_cell_labels, living_cell_labels, otsu_threshold, save_path)

    # GT vs Segmentation (Dead cell만)
    gt_vs_seg_path = os.path.join(output_dir, f"{img_name}_gt_vs_seg.png")
    iou, precision, recall, f1 = visualize_gt_vs_seg(original_img, green_gt_img, gt_mask, masks, dead_cell_labels, gt_vs_seg_path)
    print(f"    - GT vs Seg - IoU: {iou:.3f}, P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}")
    
    # 통계
    summary = {
        'image_name': img_name,
        'total_cells': int(total_cells),
        'dead_cells': int(len(dead_cell_labels)),
        'living_cells': int(len(living_cell_labels)),
        'dead_ratio': float(len(dead_cell_labels) / total_cells) if total_cells > 0 else 0,
        'gt_regions': len(spot_centroids),
        'matched_regions': len([d for d in matching_details if d['matched_seg_label'] is not None]),
        'unmatched_regions': len([d for d in matching_details if d['matched_seg_label'] is None]),
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
                visualize_spots=False
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
            f.write(f"Average Matched Regions: {summary_df['matched_regions'].mean():.1f}\n")
            f.write(f"Average Unmatched Regions: {summary_df['unmatched_regions'].mean():.1f}\n")
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
        print(f">>> Avg Matching Rate: {summary_df['matched_regions'].sum() / summary_df['gt_regions'].sum() * 100:.1f}%")
        print(f">>> Results: {output_dir}")
        print(f"{'='*70}")

# ===== 실행 =====

if __name__ == '__main__':
    batch_process(
        img_dir=Config.IMG_DIR,
        green_gt_dir=Config.GREEN_GT_DIR,
        base_output_dir=Config.OUTPUT_DIR
    )