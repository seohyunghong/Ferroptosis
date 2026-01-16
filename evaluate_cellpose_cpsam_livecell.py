import os
import time
import numpy as np
from pycocotools.coco import COCO
from cellpose import models, io, core, metrics
from tqdm import tqdm

# ===== 설정 =====
DATA_ROOT = "./data/livecell"
IMG_DIR = os.path.join(DATA_ROOT, "images/livecell_test_images")
ANNOT_FILE = os.path.join(DATA_ROOT, "livecell_coco_test.json")

TEST_LIMIT = None  # None으로 설정하면 전체 테스트

use_GPU = core.use_gpu()
print(f">>> GPU Activated: {use_GPU}")

# ===== 함수 =====

def get_ground_truth_mask(coco, img_id, img_shape):
    """COCO annotation을 mask로 변환"""
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    mask = np.zeros(img_shape[:2], dtype=np.uint16)
    
    for i, ann in enumerate(anns):
        m = coco.annToMask(ann)
        mask[m > 0] = i + 1 
        
    return mask

def evaluate_cpsam():
    """CPSAM 모델로 LIVECell 평가"""
    
    # COCO 데이터 로드
    print(f">>> Loading annotations from: {ANNOT_FILE}")
    coco = COCO(ANNOT_FILE)
    
    img_ids = coco.getImgIds()
    if TEST_LIMIT:
        img_ids = img_ids[:TEST_LIMIT]
    
    print(f">>> Total images to evaluate: {len(img_ids)}")
    
    # CPSAM 모델 로드
    print(f"\n{'='*40}")
    print(">>> Loading CPSAM model...")
    print(f"{'='*40}")
    
    model = models.CellposeModel(gpu=use_GPU, pretrained_model='cpsam')
    
    # 결과 저장
    results = {
        'ap50': [],
        'ap_avg': [],
        'time': [],
        'cell_counts_pred': [],
        'cell_counts_true': []
    }
    
    thresholds = np.arange(0.5, 1.0, 0.05)
    
    # 이미지별 평가
    print("\n>>> Starting evaluation...")
    for img_id in tqdm(img_ids, desc="Evaluating CPSAM"):
        # 이미지 로드
        img_info = coco.loadImgs(img_id)[0]
        fpath = os.path.join(IMG_DIR, img_info['file_name'])
        img = io.imread(fpath)
        
        # Ground truth
        mask_true = get_ground_truth_mask(coco, img_id, img.shape)
        n_cells_true = len(np.unique(mask_true)) - 1  # 0 제외
        
        # CPSAM 예측
        try:
            start_t = time.time()
            mask_pred, flows, styles = model.eval(
                img, 
                diameter=None,  # 자동 계산
                flow_threshold=0.4,
                cellprob_threshold=0.0
            )
            proc_time = time.time() - start_t
            
            n_cells_pred = len(np.unique(mask_pred)) - 1  # 0 제외
            
            # 성능 지표 계산
            ap, tp, fp, fn = metrics.average_precision(
                mask_true, 
                mask_pred, 
                threshold=thresholds
            )
            
            # 결과 저장
            results['ap50'].append(ap[0])  # AP@0.5
            results['ap_avg'].append(ap.mean())  # AP@0.5:0.95
            results['time'].append(proc_time)
            results['cell_counts_pred'].append(n_cells_pred)
            results['cell_counts_true'].append(n_cells_true)
            
        except Exception as e:
            print(f"\n!!! Error on {img_info['file_name']}: {e}")
            continue
    
    # 최종 결과 출력
    print(f"\n{'='*60}")
    print("CPSAM EVALUATION RESULTS ON LIVECELL")
    print(f"{'='*60}")
    
    if len(results['ap50']) > 0:
        mean_ap50 = np.mean(results['ap50'])
        std_ap50 = np.std(results['ap50'])
        mean_ap_avg = np.mean(results['ap_avg'])
        std_ap_avg = np.std(results['ap_avg'])
        mean_time = np.mean(results['time'])
        
        mean_cells_pred = np.mean(results['cell_counts_pred'])
        mean_cells_true = np.mean(results['cell_counts_true'])
        
        print(f"Images evaluated: {len(results['ap50'])}/{len(img_ids)}")
        print(f"-"*60)
        print(f"mAP@0.5      : {mean_ap50:.4f} ± {std_ap50:.4f}")
        print(f"mAP@0.5:0.95 : {mean_ap_avg:.4f} ± {std_ap_avg:.4f}")
        print(f"Avg Time     : {mean_time:.2f}s per image")
        print(f"-"*60)
        print(f"Avg Cells (GT)   : {mean_cells_true:.1f}")
        print(f"Avg Cells (Pred) : {mean_cells_pred:.1f}")
        print(f"{'='*60}")
        
        # 상세 통계
        print(f"\nDetailed Statistics:")
        print(f"  AP@0.5 - Min: {np.min(results['ap50']):.4f}, Max: {np.max(results['ap50']):.4f}")
        print(f"  AP@0.5:0.95 - Min: {np.min(results['ap_avg']):.4f}, Max: {np.max(results['ap_avg']):.4f}")
        print(f"  Time - Min: {np.min(results['time']):.2f}s, Max: {np.max(results['time']):.2f}s")
        
    else:
        print("!!! No successful evaluations")
        print(f"{'='*60}")
    
    return results

if __name__ == '__main__':
    results = evaluate_cpsam()
    
    # 결과를 numpy로 저장 (선택사항)
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "cpsam_livecell_results.npy"), results)
    print(f"\n>>> Results saved to {output_dir}/cpsam_livecell_results.npy")