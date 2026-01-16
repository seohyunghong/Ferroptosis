import sys
import os
import time
import numpy as np
from pycocotools.coco import COCO
from cellpose import models, io, core, metrics, utils
from tqdm import tqdm

# =========================================================
# 1. 설정 및 경로 (사용자 환경에 맞게 수정)
# =========================================================
DATA_ROOT = "/mnt/nas5/AIBL-Research/shhong/Workspace/ferroptosis/data/livecell"
IMG_DIR = os.path.join(DATA_ROOT, "images/livecell_test_images") 
ANNOT_FILE = os.path.join(DATA_ROOT, "livecell_coco_test.json")
MODEL_DIR = "/home/shhong/.cellpose/models"  # 모델 파일들이 있는 폴더

# [중요] 모델 이름과 실제 파일명을 매핑합니다.
# 다운로드 받은 zip 파일 내용물에 따라 파일명이 다를 수 있으니 확인해주세요.
# 예: 'livecell' 모델의 실제 파일명이 'livecell' 인지 'livecell_0' 인지 등
MODEL_PATHS = {
    'livecell_cp3': os.path.join(MODEL_DIR, 'livecell_cp3'), 
    'livecell':     os.path.join(MODEL_DIR, 'livecell'),      # 파일명이 정확해야 합니다!
    'cyto3':        os.path.join(MODEL_DIR, 'cyto3')
}

TEST_LIMIT = None  # 전체 다 돌리려면 None

use_GPU = core.use_gpu()
print(f">>> GPU Activated: {use_GPU}")

# =========================================================
# 2. 유틸리티 함수
# =========================================================

def get_ground_truth_mask(coco, img_id, img_shape):
    """COCO JSON -> Mask 변환"""
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    mask = np.zeros(img_shape[:2], dtype=np.uint16)
    for i, ann in enumerate(anns):
        m = coco.annToMask(ann)
        mask[m > 0] = i + 1 
    return mask

def calc_diameter_from_mask(mask):
    """
    [공식 벤치마크 방식]
    Ground Truth 마스크에서 세포들의 평균 지름을 계산합니다.
    """
    if mask.max() == 0:
        return 30.0
    diam, _ = utils.diameters(mask)
    return diam

# =========================================================
# 3. 평가 로직
# =========================================================

def evaluate_models():
    print(f">>> Loading annotations from: {ANNOT_FILE}")
    coco = COCO(ANNOT_FILE)
    
    img_ids = coco.getImgIds()
    if TEST_LIMIT:
        img_ids = img_ids[:TEST_LIMIT]
    
    print(f">>> Total images to evaluate: {len(img_ids)}")
    
    results = {m: {'ap50': [], 'ap_avg': [], 'time': []} for m in MODEL_PATHS.keys()}
    thresholds = np.arange(0.5, 1.0, 0.05)
    
    for model_key, model_path in MODEL_PATHS.items():
        print(f"\n{'='*60}")
        print(f">>> Evaluating Model: {model_key}")
        print(f"    Path: {model_path}")
        print(f"{'='*60}")
        
        # 1. 파일 존재 여부 확인
        if not os.path.exists(model_path):
            print(f"!!! Error: 모델 파일을 찾을 수 없습니다: {model_path}")
            print("!!! 경로를 확인하거나 파일명을 확인해주세요.")
            continue

        # 2. 모델 로드 (pretrained_model에 경로를 직접 넣으면 다운로드 시도 안 함)
        try:
            model = models.CellposeModel(gpu=use_GPU, pretrained_model=model_path)
        except Exception as e:
            print(f"!!! 모델 로드 중 에러 발생: {e}")
            continue
        
        # 3. 이미지 평가
        for img_id in tqdm(img_ids, desc=f"Running {model_key}"):
            img_info = coco.loadImgs(img_id)[0]
            fpath = os.path.join(IMG_DIR, img_info['file_name'])
            
            if not os.path.exists(fpath):
                continue

            img = io.imread(fpath)
            mask_true = get_ground_truth_mask(coco, img_id, img.shape)
            
            # [핵심] GT Diameter 사용 (공식 논문 방식)
            optimal_diam = calc_diameter_from_mask(mask_true)
            
            start_t = time.time()
            
            # CellposeModel.eval 사용 (pretrained_model로 로드했으므로)
            mask_pred, flows, styles = model.eval(
                img, 
                diameter=optimal_diam,  # GT 지름 입력
                channels=[0, 0],       # Grayscale
                flow_threshold=0.4,    # LiveCell은 복잡하므로 0.4 권장
                cellprob_threshold=0.0,
                rescale=True,          # Diameter에 맞춰 리스케일링
                resample=True
            )
            proc_time = time.time() - start_t
            
            ap, tp, fp, fn = metrics.average_precision(
                mask_true, 
                mask_pred, 
                threshold=thresholds
            )
            
            results[model_key]['ap50'].append(ap[0])
            results[model_key]['ap_avg'].append(ap.mean())
            results[model_key]['time'].append(proc_time)

    # 결과 출력
    print(f"\n{'='*70}")
    print(f"{'Model':<15} | {'mAP@0.5':<10} | {'mAP@0.5:0.95':<15} | {'Avg Time(s)':<10}")
    print(f"{'-'*70}")
    
    for model_key in MODEL_PATHS.keys():
        if not results[model_key]['ap50']:
            print(f"{model_key:<15} | No Data (Check paths)")
            continue
            
        mean_ap50 = np.mean(results[model_key]['ap50'])
        mean_ap_avg = np.mean(results[model_key]['ap_avg'])
        mean_time = np.mean(results[model_key]['time'])
        
        print(f"{model_key:<15} | {mean_ap50:.4f}     | {mean_ap_avg:.4f}          | {mean_time:.2f}")
    print(f"{'='*70}")

if __name__ == '__main__':
    evaluate_models()