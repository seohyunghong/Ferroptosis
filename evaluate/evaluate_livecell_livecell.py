import sys
import os
import time
import numpy as np
from pycocotools.coco import COCO
from cellpose import models, io, core, metrics, utils
from tqdm import tqdm

# =========================================================
# 1. 설정 및 경로
# =========================================================
DATA_ROOT = "/mnt/nas5/AIBL-Research/shhong/Workspace/ferroptosis/data/livecell"
IMG_DIR = os.path.join(DATA_ROOT, "images/livecell_test_images") 
ANNOT_FILE = os.path.join(DATA_ROOT, "livecell_coco_test.json")
MODEL_DIR = "/home/shhong/.cellpose/models"

# 평가할 모델 목록
# LIVECell은 phase-contrast 이미지이므로 cyto3 계열 모델이 적합
MODEL_PATHS = {
    'cyto3':        'cyto3',  # 내장 모델 (자동 다운로드)
    'livecell_cp3': os.path.join(MODEL_DIR, 'livecell_cp3'),  # LIVECell fine-tuned
    'livecell':     os.path.join(MODEL_DIR, 'livecell'),      # LIVECell 전용 모델
}

# 테스트 이미지 제한 (None이면 전체, 디버깅 시 10-20 추천)
TEST_LIMIT = None  # None = 전체

use_GPU = core.use_gpu()
print(f">>> GPU Activated: {use_GPU}")

# =========================================================
# 2. 유틸리티 함수
# =========================================================

def get_ground_truth_mask(coco, img_id, img_shape):
    """
    COCO JSON annotation을 instance segmentation mask로 변환
    
    Returns:
        mask: (H, W) uint16 array, 각 instance는 1, 2, 3, ... 으로 라벨링
    """
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    mask = np.zeros(img_shape[:2], dtype=np.uint16)
    
    for i, ann in enumerate(anns):
        m = coco.annToMask(ann)
        mask[m > 0] = i + 1 
    
    return mask

def calc_diameter_from_mask(mask):
    """
    Ground Truth mask에서 세포들의 평균 diameter 계산
    (Cellpose 공식 벤치마크 방식)
    
    Returns:
        diameter: 평균 세포 지름 (픽셀)
    """
    if mask.max() == 0:
        return 30.0  # 기본값
    
    diam, _ = utils.diameters(mask)
    return diam

# =========================================================
# 3. 평가 로직
# =========================================================

def evaluate_livecell_dataset():
    """
    LIVECell 데이터셋에서 Cellpose 모델들을 평가
    
    Metrics:
    - mAP@0.5: IoU threshold 0.5에서의 Average Precision
    - mAP@0.5:0.95: IoU 0.5~0.95 범위의 평균 AP (COCO 방식)
    - Processing Time: 이미지당 평균 처리 시간
    """
    print(f"\n{'='*70}")
    print(f"LIVECell Dataset Benchmark Evaluation")
    print(f"{'='*70}")
    print(f">>> Annotation file: {ANNOT_FILE}")
    print(f">>> Image directory: {IMG_DIR}")
    
    # COCO 데이터 로드
    coco = COCO(ANNOT_FILE)
    
    img_ids = coco.getImgIds()
    if TEST_LIMIT:
        img_ids = img_ids[:TEST_LIMIT]
        print(f">>> Test mode: Limited to {TEST_LIMIT} images")
    
    print(f">>> Total images to evaluate: {len(img_ids)}")
    
    # IoU thresholds (COCO 방식: 0.5, 0.55, 0.6, ..., 0.95)
    thresholds = np.arange(0.5, 1.0, 0.05)
    
    # 결과 저장용 딕셔너리
    results = {
        m: {
            'ap50': [],      # AP at IoU=0.5
            'ap_avg': [],    # Average AP across all thresholds
            'time': [],      # Processing time per image
            'num_cells_gt': [],   # Ground truth cell count
            'num_cells_pred': []  # Predicted cell count
        } 
        for m in MODEL_PATHS.keys()
    }
    
    # 각 모델에 대해 평가
    for model_key, model_path in MODEL_PATHS.items():
        print(f"\n{'='*70}")
        print(f">>> Evaluating Model: {model_key}")
        print(f"    Model path: {model_path}")
        print(f"{'='*70}")
        
        # 모델 로드
        try:
            # 내장 모델 (문자열) vs 커스텀 모델 (파일 경로)
            if model_key == 'cyto3':
                model = models.CellposeModel(gpu=use_GPU, model_type=model_path)
                print(f"    ✓ Loaded built-in model: {model_path}")
            else:
                # 커스텀 모델 파일 확인
                if not os.path.exists(model_path):
                    print(f"    ✗ Model file not found: {model_path}")
                    print(f"    → Skipping {model_key}")
                    continue
                
                model = models.CellposeModel(gpu=use_GPU, pretrained_model=model_path)
                print(f"    ✓ Loaded custom model: {model_path}")
        
        except Exception as e:
            print(f"    ✗ Error loading model: {e}")
            continue
        
        # 각 이미지에 대해 평가
        for img_id in tqdm(img_ids, desc=f"Evaluating {model_key}"):
            # 이미지 정보 로드
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(IMG_DIR, img_info['file_name'])
            
            if not os.path.exists(img_path):
                print(f"    ! Image not found: {img_info['file_name']}")
                continue
            
            # 이미지 및 GT mask 로드
            img = io.imread(img_path)
            mask_true = get_ground_truth_mask(coco, img_id, img.shape)
            
            # GT diameter 계산 (공식 벤치마크 방식)
            optimal_diam = calc_diameter_from_mask(mask_true)
            
            # Cellpose 실행
            start_time = time.time()
            
            try:
                mask_pred, flows, styles = model.eval(
                    img, 
                    diameter=optimal_diam,      # GT 기반 최적 diameter
                    channels=[0, 0],            # Grayscale (phase-contrast)
                    flow_threshold=0.4,         # LIVECell 권장값
                    cellprob_threshold=0.0,     # 기본값
                    rescale=True,               # Diameter에 맞춰 리스케일
                    resample=True               # 고해상도 처리
                )
            except Exception as e:
                print(f"    ! Prediction error on {img_info['file_name']}: {e}")
                continue
            
            proc_time = time.time() - start_time
            
            # Metrics 계산 (AP at multiple IoU thresholds)
            ap, tp, fp, fn = metrics.average_precision(
                mask_true, 
                mask_pred, 
                threshold=thresholds
            )
            
            # 결과 저장
            results[model_key]['ap50'].append(ap[0])           # AP @ IoU=0.5
            results[model_key]['ap_avg'].append(ap.mean())     # mAP @ 0.5:0.95
            results[model_key]['time'].append(proc_time)
            results[model_key]['num_cells_gt'].append(mask_true.max())
            results[model_key]['num_cells_pred'].append(mask_pred.max())
    
    # =========================================================
    # 4. 결과 출력
    # =========================================================
    print(f"\n{'='*80}")
    print(f"LIVECell Benchmark Results")
    print(f"{'='*80}")
    print(f"{'Model':<15} | {'mAP@0.5':<10} | {'mAP@0.5:0.95':<15} | {'Avg Time(s)':<12} | {'Cell Acc':<10}")
    print(f"{'-'*80}")
    
    for model_key in MODEL_PATHS.keys():
        if not results[model_key]['ap50']:
            print(f"{model_key:<15} | {'N/A':<10} | {'N/A':<15} | {'N/A':<12} | {'N/A':<10}")
            continue
        
        # 평균 계산
        mean_ap50 = np.mean(results[model_key]['ap50'])
        mean_ap_avg = np.mean(results[model_key]['ap_avg'])
        mean_time = np.mean(results[model_key]['time'])
        
        # Cell count accuracy (GT vs Pred)
        total_gt = np.sum(results[model_key]['num_cells_gt'])
        total_pred = np.sum(results[model_key]['num_cells_pred'])
        cell_acc = (1 - abs(total_gt - total_pred) / total_gt) * 100 if total_gt > 0 else 0
        
        print(f"{model_key:<15} | {mean_ap50:.4f}     | {mean_ap_avg:.4f}          | {mean_time:.3f}s       | {cell_acc:.1f}%")
    
    print(f"{'='*80}")
    
    # 상세 통계 출력
    print(f"\nDetailed Statistics:")
    print(f"{'-'*80}")
    for model_key in MODEL_PATHS.keys():
        if not results[model_key]['ap50']:
            continue
        
        print(f"\n{model_key}:")
        print(f"  - Images evaluated: {len(results[model_key]['ap50'])}")
        print(f"  - Total GT cells: {np.sum(results[model_key]['num_cells_gt'])}")
        print(f"  - Total predicted cells: {np.sum(results[model_key]['num_cells_pred'])}")
        print(f"  - Average cells per image (GT): {np.mean(results[model_key]['num_cells_gt']):.1f}")
        print(f"  - Average cells per image (Pred): {np.mean(results[model_key]['num_cells_pred']):.1f}")
        print(f"  - mAP@0.5: {np.mean(results[model_key]['ap50']):.4f} (±{np.std(results[model_key]['ap50']):.4f})")
        print(f"  - mAP@0.5:0.95: {np.mean(results[model_key]['ap_avg']):.4f} (±{np.std(results[model_key]['ap_avg']):.4f})")
        print(f"  - Avg time/image: {np.mean(results[model_key]['time']):.3f}s (±{np.std(results[model_key]['time']):.3f}s)")
    
    print(f"\n{'='*80}")
    
    return results

# =========================================================
# 5. 실행
# =========================================================

if __name__ == '__main__':
    results = evaluate_livecell_dataset()
    
    # 결과를 CSV로 저장 (선택사항)
    try:
        import pandas as pd
        
        summary_data = []
        for model_key in MODEL_PATHS.keys():
            if not results[model_key]['ap50']:
                continue
            
            summary_data.append({
                'model': model_key,
                'mAP@0.5': np.mean(results[model_key]['ap50']),
                'mAP@0.5:0.95': np.mean(results[model_key]['ap_avg']),
                'avg_time': np.mean(results[model_key]['time']),
                'total_gt_cells': np.sum(results[model_key]['num_cells_gt']),
                'total_pred_cells': np.sum(results[model_key]['num_cells_pred']),
                'images_evaluated': len(results[model_key]['ap50'])
            })
        
        df = pd.DataFrame(summary_data)
        output_csv = os.path.join(DATA_ROOT, "livecell_benchmark_results.csv")
        df.to_csv(output_csv, index=False)
        print(f"\n>>> Results saved to: {output_csv}")
    
    except ImportError:
        print("\n>>> pandas not available, skipping CSV export")