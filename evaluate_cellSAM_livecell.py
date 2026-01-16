import os
import time
import numpy as np
from pycocotools.coco import COCO
from cellSAM import segment_cellular_image, get_model
from cellpose import metrics
from skimage import io
import torch
from tqdm import tqdm

# ===== 설정 =====
DATA_ROOT = "./data/livecell"
IMG_DIR = os.path.join(DATA_ROOT, "images/livecell_test_images")
ANNOT_FILE = os.path.join(DATA_ROOT, "livecell_coco_test.json")

MODEL_LIST = ['cellsam']
TEST_LIMIT = 20

# GPU 설정
use_GPU = torch.cuda.is_available()
device = 'cuda' if use_GPU else 'cpu'

print(f">>> GPU Available: {use_GPU}")
if use_GPU:
    print(f">>> GPU Device: {torch.cuda.get_device_name(0)}")
    print(f">>> Device: {device}")

# ===== 함수 정의 =====

def get_ground_truth_mask(coco, img_id, img_shape):
    """COCO JSON annotation을 Binary Mask로 변환"""
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    mask = np.zeros(img_shape[:2], dtype=np.uint16)
    
    for i, ann in enumerate(anns):
        m = coco.annToMask(ann)
        mask[m > 0] = i + 1 
        
    return mask

def evaluate_cellsam():
    print(f"\n>>> Loading annotations from: {ANNOT_FILE}")
    coco = COCO(ANNOT_FILE)
    
    img_ids = coco.getImgIds()
    if TEST_LIMIT:
        img_ids = img_ids[:TEST_LIMIT]
    
    print(f">>> Total images to evaluate: {len(img_ids)}")
    
    results = {m: {'ap50': [], 'ap_avg': [], 'time': []} for m in MODEL_LIST}
    thresholds = np.arange(0.5, 1.0, 0.05)
    
    print(f"\n{'='*40}")
    print(f">>> Evaluating Model: CellSAM")
    print(f">>> Using device: {device}")
    print(f"{'='*40}\n")
    
    # CellSAM 모델 로드
    print(">>> Loading CellSAM model...")
    try:
        # get_model()로 pretrained 모델 다운로드/로드
        cellsam_model = get_model(device=device)
        print(">>> CellSAM model loaded successfully!")
    except Exception as e:
        print(f"!!! Error loading CellSAM model: {e}")
        import traceback
        traceback.print_exc()
        return results
    
    for img_id in tqdm(img_ids, desc="Running CellSAM"):
        img_info = coco.loadImgs(img_id)[0]
        fpath = os.path.join(IMG_DIR, img_info['file_name'])
        
        # 이미지 로드
        img = io.imread(fpath)
        mask_true = get_ground_truth_mask(coco, img_id, img.shape)
        
        # CellSAM 실행
        start_t = time.time()
        try:
            # segment_cellular_image(img, model, device=device)
            mask_pred = segment_cellular_image(
                img, 
                cellsam_model,
                normalize=True,
                postprocess=False,
                device=device
            )
            proc_time = time.time() - start_t
        except Exception as e:
            print(f"\n!!! Error on image {img_info['file_name']}: {e}")
            continue
        
        # 평가 지표 계산
        try:
            ap, tp, fp, fn = metrics.average_precision(
                mask_true, 
                mask_pred, 
                threshold=thresholds
            )
            
            ap_50 = ap[0]
            ap_mean = ap.mean()
            
            results['cellsam']['ap50'].append(ap_50)
            results['cellsam']['ap_avg'].append(ap_mean)
            results['cellsam']['time'].append(proc_time)
        except Exception as e:
            print(f"\n!!! Metric error on {img_info['file_name']}: {e}")
            continue
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"{'Model':<15} | {'mAP@0.5':<10} | {'mAP@0.5:0.95':<15} | {'Avg Time(s)':<10}")
    print(f"{'-'*60}")
    
    if len(results['cellsam']['ap50']) > 0:
        mean_ap50 = np.mean(results['cellsam']['ap50'])
        mean_ap_avg = np.mean(results['cellsam']['ap_avg'])
        mean_time = np.mean(results['cellsam']['time'])
        
        print(f"{'CellSAM':<15} | {mean_ap50:.4f}     | {mean_ap_avg:.4f}          | {mean_time:.2f}")
        
        # 추가 통계
        print(f"\nDetailed Statistics:")
        print(f"  Successful evaluations: {len(results['cellsam']['ap50'])}/{len(img_ids)}")
        print(f"  AP@0.5 - Min: {np.min(results['cellsam']['ap50']):.4f}, Max: {np.max(results['cellsam']['ap50']):.4f}")
        print(f"  Time - Min: {np.min(results['cellsam']['time']):.2f}s, Max: {np.max(results['cellsam']['time']):.2f}s")
    else:
        print("!!! No successful evaluations")
    
    print(f"{'='*60}")
    
    return results

if __name__ == '__main__':
    results = evaluate_cellsam()
    
    # 결과 저장
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "cellsam_livecell_results.npy"), results)
    print(f"\n>>> Results saved to {output_dir}/cellsam_livecell_results.npy")