"""
preprocessing_final.py

demo_imsang.py의 green 전처리 방법 적용
- 배경 형광 제거
- 아티팩트 필터링
- Otsu threshold
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from skimage import measure, morphology
from cellpose import models

# ============================================================
# Green 전처리 (demo_imsang.py 방식)
# ============================================================

def preprocess_green_image(green_img, background_threshold=20, min_spot_area=10, max_artifact_area=300):
    """
    배경 형광과 작은 아티팩트를 제거하여 명확한 spot만 남김
    
    Args:
        green_img: Green channel 이미지
        background_threshold: 평균 밝기가 이 값보다 높으면 전처리 수행
        min_spot_area: 최소 spot 면적
        max_artifact_area: 최대 아티팩트 면적
    
    Returns:
        processed_img: 전처리된 이미지
        is_preprocessed: 전처리 수행 여부
    """
    # 8-bit 변환
    if green_img.max() <= 1.0:
        img_uint8 = (green_img * 255).astype(np.uint8)
    else:
        img_uint8 = green_img.astype(np.uint8)
    
    # 통계
    mean_intensity = np.mean(img_uint8)
    median_intensity = np.median(img_uint8)
    
    # 평균 밝기가 높으면 전처리
    if mean_intensity > background_threshold:
        # 1. 배경 제거 (median-based)
        background_level = median_intensity
        processed = img_uint8.astype(np.float32) - background_level
        processed = np.clip(processed, 0, 255).astype(np.uint8)
        
        # 2. 작은 아티팩트 제거
        otsu_thresh, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Connected components 분석
        labeled = measure.label(binary > 0)
        regions = measure.regionprops(labeled, intensity_image=processed)
        
        # 아티팩트 필터링
        artifact_mask = np.zeros_like(binary, dtype=bool)
        
        for region in regions:
            area = region.area
            mean_intensity_region = region.mean_intensity
            
            # 너무 작거나 어두운 영역 제거
            is_artifact = (
                area < min_spot_area or
                (area < max_artifact_area and mean_intensity_region < otsu_thresh * 2)
            )
            
            if is_artifact:
                artifact_mask[labeled == region.label] = True
        
        # 아티팩트 제거
        processed[artifact_mask] = 0
        
        return processed, True
    else:
        return img_uint8, False


def create_gt_mask_otsu(green_img, min_spot_area=0, max_spot_area=100000, morph_kernel_size=1):
    """
    Green 형광 이미지에서 Otsu threshold로 GT mask 생성
    
    Args:
        green_img: Green channel 이미지
        min_spot_area: 최소 spot 면적
        max_spot_area: 최대 spot 면적
        morph_kernel_size: Opening 커널 크기
    
    Returns:
        gt_mask: Binary GT mask
        otsu_threshold: 사용된 threshold
    """
    # 전처리
    img_uint8, _ = preprocess_green_image(green_img)
    
    # Otsu thresholding
    otsu_threshold, gt_mask_uint8 = cv2.threshold(
        img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    gt_mask = gt_mask_uint8 > 0
    
    # Morphological opening (노이즈 제거)
    kernel = morphology.disk(morph_kernel_size)
    gt_mask_clean = morphology.binary_opening(gt_mask, kernel)
    
    # Connected component labeling
    labeled = measure.label(gt_mask_clean)
    regions = measure.regionprops(labeled)
    
    # Spot 필터링 (면적 기준)
    final_gt_mask = np.zeros_like(green_img, dtype=bool)
    
    for region in regions:
        area = region.area
        if min_spot_area <= area <= max_spot_area:
            final_gt_mask[labeled == region.label] = True
    
    return final_gt_mask, otsu_threshold


# ============================================================
# Preprocessor
# ============================================================

class Preprocessor:
    """
    demo_imsang.py 방식의 green 전처리 적용
    """
    
    def __init__(self, phase_dir, green_dir, output_dir):
        self.phase_dir = Path(phase_dir)
        self.green_dir = Path(green_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cellpose
        self.model = models.CellposeModel(gpu=True, model_type='cyto3')
    
    def extract_green_channel(self, green_path: Path):
        """Green channel 추출"""
        if not green_path.exists():
            return None
        
        img = cv2.imread(str(green_path))
        if img is None:
            return None
        
        if len(img.shape) == 3:
            green_channel = img[:, :, 1]  # G channel
        else:
            green_channel = img
        
        return green_channel
    
    def process_single_image(self, phase_path: Path, green_path: Path):
        """단일 이미지 처리"""
        
        # Load phase
        phase_img = cv2.imread(str(phase_path), cv2.IMREAD_GRAYSCALE)
        
        # Load green (RGB → G channel)
        green_img = self.extract_green_channel(green_path)
        
        # Segmentation
        masks, _, _ = self.model.eval(
            phase_img,
            diameter=None,
            channels=[0, 0],
            flow_threshold=0.3,
            cellprob_threshold=0.0
        )
        
        # Extract crops
        crops = []
        is_target = []
        
        unique_labels = np.unique(masks)
        unique_labels = unique_labels[unique_labels > 0]
        
        for label in unique_labels:
            cell_mask = (masks == label)
            labeled = measure.label(cell_mask)
            regions = measure.regionprops(labeled)
            
            if len(regions) == 0:
                continue
            
            region = regions[0]
            
            # Bbox
            minr, minc, maxr, maxc = region.bbox
            padding = 20
            minr = max(0, minr - padding)
            minc = max(0, minc - padding)
            maxr = min(phase_img.shape[0], maxr + padding)
            maxc = min(phase_img.shape[1], maxc + padding)
            
            # Masked crop
            masked = phase_img.copy()
            masked[~cell_mask] = 0
            cropped = masked[minr:maxr, minc:maxc]
            
            # Square padding
            h, w = cropped.shape
            max_side = max(h, w)
            padded = np.zeros((max_side, max_side), dtype=phase_img.dtype)
            pady = (max_side - h) // 2
            padx = (max_side - w) // 2
            padded[pady:pady+h, padx:padx+w] = cropped
            
            # Resize
            resized = cv2.resize(padded, (256, 256), interpolation=cv2.INTER_LINEAR)
            
            # ===== Target 판정 (개선된 방법!) =====
            target = False
            if green_img is not None:
                # Green GT mask 생성 (Otsu + 전처리)
                gt_mask, _ = create_gt_mask_otsu(green_img)
                
                # Cell mask와 GT mask 겹침 확인
                cell_region = cell_mask[minr:maxr, minc:maxc]
                gt_region = gt_mask[minr:maxr, minc:maxc]
                
                # Intersection
                intersection = np.sum(cell_region & gt_region)
                
                # Threshold: intersection이 일정 이상이면 target
                if intersection > 50:  # 픽셀 개수 기준
                    target = True
            
            crops.append(resized)
            is_target.append(target)
        
        return {
            'crops': np.array(crops),
            'is_target': np.array(is_target),
            'image_name': phase_path.stem
        }
    
    def build_dataset(self):
        """전체 데이터셋 빌드"""
        print("="*70)
        print("Preprocessing (demo_imsang.py method)")
        print("="*70)
        
        phase_files = sorted(self.phase_dir.glob("*.tif"))
        phase_files = phase_files[:3]
        print(f"\nFound {len(phase_files)} phase images")
        
        all_crops = []
        all_is_target = []
        all_image_names = []
        
        for phase_file in tqdm(phase_files, desc="Processing"):
            green_name = phase_file.name.replace('phase_', 'green_')
            green_file = self.green_dir / green_name
            
            try:
                result = self.process_single_image(phase_file, green_file)
                
                if len(result['crops']) == 0:
                    continue
                
                all_crops.append(result['crops'])
                all_is_target.append(result['is_target'])
                all_image_names.extend([result['image_name']] * len(result['crops']))
                
            except Exception as e:
                print(f"\n❌ Error: {phase_file.name}: {e}")
                continue
        
        # Concatenate
        crops = np.vstack(all_crops)
        is_target = np.concatenate(all_is_target)
        image_names = np.array(all_image_names)
        
        # Save
        np.save(self.output_dir / 'crops.npy', crops)
        np.save(self.output_dir / 'is_target.npy', is_target)
        np.save(self.output_dir / 'image_names.npy', image_names)
        
        # Statistics
        print("\n" + "="*70)
        print("Dataset Built!")
        print("="*70)
        print(f"Total cells: {len(crops)}")
        print(f"Target cells: {np.sum(is_target)} ({np.mean(is_target)*100:.1f}%)")
        print(f"Non-target cells: {np.sum(~is_target)}")
        
        return {
            'crops': crops,
            'is_target': is_target,
            'image_names': image_names
        }


# ============================================================
# 실행
# ============================================================

if __name__ == '__main__':
    preprocessor = Preprocessor(
        phase_dir='/mnt/nas5/AIBL-Research/shhong/Workspace/ferroptosis/data/ferroptosis/kanglab_data/phase',
        green_dir='/mnt/nas5/AIBL-Research/shhong/Workspace/ferroptosis/data/ferroptosis/kanglab_data/green',
        output_dir='./processed_3sample_0203'
    )
    
    dataset = preprocessor.build_dataset()