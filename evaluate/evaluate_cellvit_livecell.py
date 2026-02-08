import os
import time
import numpy as np
from pycocotools.coco import COCO
from cellpose import metrics
from skimage import io
import torch
from tqdm import tqdm

# CellViT import (여러 가능성 대응)
try:
    from cell_segmentation.inference.inference_cellvit_experiment_pannuke import CellViT
    CELLVIT_AVAILABLE = True
except ImportError:
    try:
        from cellvit.inference import CellViT
        CELLVIT_AVAILABLE = True
    except ImportError:
        print("Warning: CellViT not properly installed")
        CELLVIT_AVAILABLE = False

# ===== 설정 =====
DATA_ROOT = "./data/livecell"
IMG_DIR = os.path.join(DATA_ROOT, "images/livecell_test_images")
ANNOT_FILE = os.path.join(DATA_ROOT, "livecell_coco_test.json")

MODEL_LIST = ['cellvit']
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

def load_cellvit_model(device='cuda'):
    """CellViT 모델 로드"""
    if not CELLVIT_AVAILABLE:
        raise ImportError("CellViT is not installed. Please install it first.")
    
    # CellViT 모델 초기화 (pretrained weights 자동 다운로드)
    try:
        model = CellViT(
            model_path=None,  # None이면 pretrained 자동 다운로드
            device=device
        )
    except Exception as e:
        print(f"Error loading CellViT: {e}")
        print("Trying alternative loading method...")
        # 대체 방법
        from cellvit.inference import CellViTInference
        model = CellViTInference(device=device)
    
    return model

def segment_with_cellvit(model, img):
    """CellViT로 segmentation 수행"""
    # CellViT는 보통 inference 메서드를 제공
    try:
        # 방법 1: predict 메서드
        result = model.predict(img)
        if isinstance(result, dict):
            mask = result.get('instance_map', result.get('masks', None))
        else:
            mask = result
    except AttributeError:
        try:
            # 방법 2: __call__ 메서드
            result = model(img)
            mask = result.get('instance_map', result.get('masks', result))
        except Exception as e:
            print(f"Segmentation error: {e}")
            return None
    
    return mask

def evaluate_cellvit():
    print(f"\n>>> Loading annotations from: {ANNOT_FILE}")
    coco = COCO(ANNOT_FILE)
    
    img_ids = coco.getImgIds()
    if TEST_LIMIT:
        img_ids = img_ids[:TEST_LIMIT]
    
    print(f">>> Total images to evaluate: {len(img_ids)}")
    
    results = {m: {'ap50': [], 'ap_avg': [], 'time': []} for m in MODEL_LIST}
    thresholds = np.arange(0.5, 1.0, 0.05)
    
    print(f"\n{'='*40}")
    print(f">>> Evaluating Model: CellViT")
    print(f">>> Using device: {device}")
    print(f"{'='*40}\n")
    
    # CellViT 모델 로드
    print("Loading CellViT model...")
    try:
        model = load_cellvit_model(device=device)
        print("CellViT model loaded successfully!")
    except Exception as e:
        print(f"!!! Failed to load CellViT: {e}")
        return results
    
    for img_id in tqdm(img_ids, desc="Running CellViT"):
        img_info = coco.loadImgs(img_id)[0]
        fpath = os.path.join(IMG_DIR, img_info['file_name'])
        
        # 이미지 로드
        img = io.imread(fpath)
        
        # Grayscale를 RGB로 변환 (CellViT는 RGB 필요할 수 있음)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        
        mask_true = get_ground_truth_mask(coco, img_id, img.shape)
        
        # CellViT 실행
        start_t = time.time()
        try:
            mask_pred = segment_with_cellvit(model, img)
            
            if mask_pred is None:
                print(f"\n!!! Segmentation failed on {img_info['file_name']}")
                continue
                
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
            
            results['cellvit']['ap50'].append(ap_50)
            results['cellvit']['ap_avg'].append(ap_mean)
            results['cellvit']['time'].append(proc_time)
        except Exception as e:
            print(f"\n!!! Metric error on {img_info['file_name']}: {e}")
            continue
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"{'Model':<15} | {'mAP@0.5':<10} | {'mAP@0.5:0.95':<15} | {'Avg Time(s)':<10}")
    print(f"{'-'*60}")
    
    if len(results['cellvit']['ap50']) > 0:
        mean_ap50 = np.mean(results['cellvit']['ap50'])
        mean_ap_avg = np.mean(results['cellvit']['ap_avg'])
        mean_time = np.mean(results['cellvit']['time'])
        
        print(f"{'CellViT':<15} | {mean_ap50:.4f}     | {mean_ap_avg:.4f}          | {mean_time:.2f}")
        
        # 추가 통계
        print(f"\nDetailed Statistics:")
        print(f"  Successful evaluations: {len(results['cellvit']['ap50'])}/{len(img_ids)}")
        print(f"  AP@0.5 - Min: {np.min(results['cellvit']['ap50']):.4f}, Max: {np.max(results['cellvit']['ap50']):.4f}")
        print(f"  Time - Min: {np.min(results['cellvit']['time']):.2f}s, Max: {np.max(results['cellvit']['time']):.2f}s")
    else:
        print("!!! No successful evaluations")
    
    print(f"{'='*60}")
    
    return results

if __name__ == '__main__':
    if not CELLVIT_AVAILABLE:
        print("\n" + "="*60)
        print("ERROR: CellViT is not installed!")
        print("="*60)
        print("\nPlease install CellViT first:")
        print("  Option 1: pip install cell-segmentation-models")
        print("  Option 2: git clone https://github.com/TIO-IKIM/CellViT.git && cd CellViT && pip install -e .")
        print("="*60)
    else:
        results = evaluate_cellvit()
        
        # 결과 저장
        output_dir = "./results"
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "cellvit_livecell_results.npy"), results)
        print(f"\n>>> Results saved to {output_dir}/cellvit_livecell_results.npy")