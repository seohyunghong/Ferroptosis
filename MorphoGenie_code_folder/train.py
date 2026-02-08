"""
preprocessing_fast.py

빠른 전처리 (최적화 버전)
- GPU 최적화
- Batch processing
- Sleep 방지
- Progress 저장
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from skimage import measure, morphology
from cellpose import models, core
import torch
import os
import signal
import time

# ============================================================
# Sleep 방지
# ============================================================

def prevent_sleep():
    """
    시스템 sleep 방지
    """
    # Linux caffeine 설치
    os.system('caffeinate -d &')  # macOS
    # 또는
    os.system('systemd-inhibit --what=idle --who="preprocessing" --why="Running cellpose" sleep infinity &')  # Linux

# ============================================================
# Cellpose 최적화 설정
# ============================================================

class FastPreprocessor:
    """
    최적화된 전처리
    """
    
    def __init__(self, phase_dir, green_dir, output_dir):
        self.phase_dir = Path(phase_dir)
        self.green_dir = Path(green_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress 파일
        self.progress_file = self.output_dir / 'progress.txt'
        
        # ===== Cellpose 최적화 =====
        print("Initializing Cellpose...")
        print(f"GPU available: {core.use_gpu()}")
        
        self.model = models.CellposeModel(
            gpu=True, 
            model_type='cyto3'
        )
        
        # ===== 최적화 파라미터 =====
        self.diameter = 30  # 고정 diameter (더 빠름)
        self.flow_threshold = 0.4  # 높이면 더 빠름
        self.cellprob_threshold = 0.0
        
        # Batch size
        self.batch_size = 4  # GPU 메모리에 맞게
    
    def get_processed_images(self):
        """이미 처리된 이미지"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return set(line.strip() for line in f)
        return set()
    
    def mark_processed(self, image_name):
        """처리 완료 표시"""
        with open(self.progress_file, 'a') as f:
            f.write(f"{image_name}\n")
    
    def extract_green_channel(self, green_path):
        """Green channel 추출"""
        if not green_path.exists():
            return None
        
        img = cv2.imread(str(green_path))
        if img is None:
            return None
        
        if len(img.shape) == 3:
            return img[:, :, 1]
        return img
    
    def preprocess_green(self, green_img):
        """Green 전처리 (빠른 버전)"""
        if green_img is None:
            return None
        
        # 8-bit
        if green_img.max() <= 1.0:
            img_uint8 = (green_img * 255).astype(np.uint8)
        else:
            img_uint8 = green_img.astype(np.uint8)
        
        # 빠른 전처리: Gaussian blur + Otsu
        blurred = cv2.GaussianBlur(img_uint8, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary > 0
    
    def process_batch(self, phase_paths, green_paths):
        """
        배치 처리 (여러 이미지 동시에)
        """
        # Load images
        phase_imgs = []
        for p in phase_paths:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            phase_imgs.append(img)
        
        # Cellpose batch inference (더 빠름!)
        masks_batch = self.model.eval(
            phase_imgs,
            diameter=self.diameter,
            channels=[0, 0],
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            batch_size=len(phase_imgs)
        )[0]
        
        # Process each
        results = []
        for i, (phase_path, green_path) in enumerate(zip(phase_paths, green_paths)):
            phase_img = phase_imgs[i]
            masks = masks_batch[i]
            
            # Green GT
            green_img = self.extract_green_channel(green_path)
            gt_mask = self.preprocess_green(green_img)
            
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
                
                # Crop
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
                
                # Target 판정
                target = False
                if gt_mask is not None:
                    cell_region = cell_mask[minr:maxr, minc:maxc]
                    gt_region = gt_mask[minr:maxr, minc:maxc]
                    intersection = np.sum(cell_region & gt_region)
                    if intersection > 50:
                        target = True
                
                crops.append(resized)
                is_target.append(target)
            
            results.append({
                'crops': np.array(crops),
                'is_target': np.array(is_target),
                'image_name': phase_path.stem
            })
        
        return results
    
    def build_dataset(self):
        """전체 데이터셋 빌드"""
        print("="*70)
        print("Fast Preprocessing")
        print("="*70)
        
        # Sleep 방지
        prevent_sleep()
        
        # 파일 목록
        phase_files = sorted(self.phase_dir.glob("*.tif"))
        print(f"\nFound {len(phase_files)} phase images")
        
        # 이미 처리된 것 제외
        processed = self.get_processed_images()
        remaining = [f for f in phase_files if f.stem not in processed]
        
        if len(processed) > 0:
            print(f"✓ Already processed: {len(processed)}")
            print(f"⏳ Remaining: {len(remaining)}")
        
        # 배치 단위로 처리
        all_crops = []
        all_is_target = []
        all_image_names = []
        
        # 배치 생성
        batches = []
        for i in range(0, len(remaining), self.batch_size):
            batch_phase = remaining[i:i+self.batch_size]
            batch_green = []
            
            for phase_file in batch_phase:
                green_name = phase_file.name.replace('phase_', 'green_')
                green_file = self.green_dir / green_name
                batch_green.append(green_file)
            
            batches.append((batch_phase, batch_green))
        
        # Progress bar
        pbar = tqdm(batches, desc="Processing batches")
        
        for phase_batch, green_batch in pbar:
            try:
                # 배치 처리
                results = self.process_batch(phase_batch, green_batch)
                
                for result in results:
                    if len(result['crops']) > 0:
                        all_crops.append(result['crops'])
                        all_is_target.append(result['is_target'])
                        all_image_names.extend([result['image_name']] * len(result['crops']))
                    
                    # Progress 저장
                    self.mark_processed(result['image_name'])
                
                # 진행 상황 표시
                pbar.set_postfix({
                    'cells': sum(len(c) for c in all_crops),
                    'target': sum(sum(t) for t in all_is_target)
                })
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
        
        # Concatenate
        if len(all_crops) == 0:
            print("❌ No crops extracted!")
            return None
        
        crops = np.vstack(all_crops)
        is_target = np.concatenate(all_is_target)
        image_names = np.array(all_image_names)
        
        # Save
        np.save(self.output_dir / 'crops.npy', crops)
        np.save(self.output_dir / 'is_target.npy', is_target)
        np.save(self.output_dir / 'image_names.npy', image_names)
        
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
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase-dir', default='/mnt/nas5/AIBL-Research/shhong/Workspace/ferroptosis/data/ferroptosis/kanglab_data/phase')
    parser.add_argument('--green-dir', default='/mnt/nas5/AIBL-Research/shhong/Workspace/ferroptosis/data/ferroptosis/kanglab_data/green')
    parser.add_argument('--output-dir', default='./processed_fast')
    parser.add_argument('--batch-size', type=int, default=4)
    
    args = parser.parse_args()
    
    preprocessor = FastPreprocessor(
        phase_dir=args.phase_dir,
        green_dir=args.green_dir,
        output_dir=args.output_dir
    )
    
    preprocessor.batch_size = args.batch_size
    
    dataset = preprocessor.build_dataset()