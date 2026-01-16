import sys
import os

# ===== 로컬 cellpose 사용 설정 (최우선) =====
CELLPOSE_PATH = '/mnt/nas5/AIBL-Research/shhong/Workspace/ferroptosis/cellpose'
sys.path.insert(0, CELLPOSE_PATH)
print(f">>> Using local cellpose from: {CELLPOSE_PATH}")

import time
import numpy as np
import matplotlib.pyplot as plt

# cellpose 로컬 모듈 import
from cellpose import models, io, core, denoise
from cellpose.utils import outlines_list

# 1. GPU 및 환경 설정
use_GPU = core.use_gpu()
print(f">>> GPU Activated: {use_GPU}")

image_path = '/mnt/nas5/AIBL-Research/shhong/Workspace/ferroptosis/data/kanglab_data/images/Era_B2_2_00d03h00m.jpg' 
img_name = image_path.split('/')[-1].split('.')[0]

# 2. 이미지 로드
print(f">>> Loading Image: {img_name} ...")
img = io.imread(image_path)
print(f">>> Image shape: {img.shape}")

# 3. 노이즈 제거 모델 로드
print(">>> Loading Denoise Model (denoise_cyto3)...")
try:
    denoise_model = denoise.CellposeDenoiseModel(
        gpu=use_GPU, 
        model_type='denoise_cyto3', 
        restore_type='denoise_cyto3'
    )
    print(">>> Denoise model loaded successfully!")
    
    # 디노이즈 수행
    print(">>> Starting denoising process...")
    denoised_img = denoise_model.dn.eval(img, diameter=30)
    print(">>> Denoising complete!")
    
    DENOISE_SUCCESS = True
    
except Exception as e:
    print(f"!!! Denoise failed: {e}")
    print("!!! Proceeding without denoising...")
    denoised_img = img
    DENOISE_SUCCESS = False

# 4. 저장 디렉토리 생성
save_dir = "./result_denoise_comparison"
os.makedirs(save_dir, exist_ok=True)

# 5. 모델별 Segmentation 루프
model_list = ['cyto2', 'nuclei', 'cyto3','livecell']

for model_name in model_list:
    try:
        print(f"\n{'='*60}")
        print(f">>> Processing Model: {model_name}")
        print(f"{'='*60}")
        
        # 모델 로드
        print(f">>> Loading segmentation model: {model_name}...")
        model = models.CellposeModel(gpu=use_GPU, model_type=model_name)
        
        # (1) 원본 이미지로 Segmentation
        print(f">>> [Original] Segmentation Start...")
        start = time.time()
        masks_orig, flows_orig, styles_orig = model.eval(
            img,
            diameter=None,
            channels=[0, 0],
            flow_threshold=0.8,
            cellprob_threshold=-1.5
        )
        time_orig = time.time() - start
        cells_orig = masks_orig.max()
        print(f">>> [Original] Time: {time_orig:.2f}s | Cells: {cells_orig}")
        
        # (2) 디노이즈된 이미지로 Segmentation
        print(f">>> [Denoised] Segmentation Start...")
        start = time.time()
        masks_denoised, flows_denoised, styles_denoised = model.eval(
            denoised_img,
            diameter=None,
            channels=[0, 0],
            flow_threshold=0.8,
            cellprob_threshold=-1.5
        )
        time_denoised = time.time() - start
        cells_denoised = masks_denoised.max()
        print(f">>> [Denoised] Time: {time_denoised:.2f}s | Cells: {cells_denoised}")
        
        # 6. 결과 시각화 (2x2 그리드)
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        # [0, 0] 원본 이미지
        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].set_title(f'Original Image\n{img_name}', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # [0, 1] 원본 + Segmentation
        outlines_orig = outlines_list(masks_orig)
        axes[0, 1].imshow(img, cmap='gray')
        for o in outlines_orig:
            axes[0, 1].plot(o[:, 0], o[:, 1], color='red', linewidth=1.5)
        axes[0, 1].set_title(
            f'Original + {model_name}\nCells: {cells_orig} | Time: {time_orig:.2f}s',
            fontsize=14,
            fontweight='bold',
            color='red'
        )
        axes[0, 1].axis('off')
        
        # [1, 0] 디노이즈된 이미지
        axes[1, 0].imshow(denoised_img, cmap='gray')
        denoise_title = 'Denoised Image (denoise_cyto3)' if DENOISE_SUCCESS else 'Original (Denoise Failed)'
        axes[1, 0].set_title(denoise_title, fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # [1, 1] 디노이즈 + Segmentation
        outlines_denoised = outlines_list(masks_denoised)
        axes[1, 1].imshow(denoised_img, cmap='gray')
        for o in outlines_denoised:
            axes[1, 1].plot(o[:, 0], o[:, 1], color='lime', linewidth=1.5)
        axes[1, 1].set_title(
            f'Denoised + {model_name}\nCells: {cells_denoised} | Time: {time_denoised:.2f}s',
            fontsize=14,
            fontweight='bold',
            color='green'
        )
        axes[1, 1].axis('off')
        
        # 전체 타이틀
        diff = cells_denoised - cells_orig
        fig.suptitle(
            f'Denoise Comparison - Model: {model_name}\n'
            f'Cell Count Difference: {diff:+d} ({diff/cells_orig*100:+.1f}%)',
            fontsize=16,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        # 저장
        output_filename = os.path.join(save_dir, f"{img_name}_{model_name}_comparison.png")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f">>> Saved: {output_filename}")
        plt.close()
        
    except Exception as e:
        print(f"!!! Error with model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*60}")
print(">>> All models processed!")
print(f">>> Results saved in: {save_dir}")
print(f"{'='*60}")