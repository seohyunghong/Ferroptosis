"""
validate_preprocessing.py
after preprocessing.py  그 다음 시각화단계. 
전처리 결과 검증 시각화
"""
"""
validate_preprocessing_fixed.py

전처리 결과 검증 시각화 (수정 버전)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# ============================================================
# 설정
# ============================================================

PROCESSED_DIR = './processed'  # 전처리 결과 경로
OUTPUT_DIR = './validation_results'   # 시각화 저장 경로
Path(OUTPUT_DIR).mkdir(exist_ok=True)

PHASE_DIR = '/mnt/nas5/AIBL-Research/shhong/Workspace/ferroptosis/data/ferroptosis/kanglab_data/phase'
GREEN_DIR = '/mnt/nas5/AIBL-Research/shhong/Workspace/ferroptosis/data/ferroptosis/kanglab_data/green'

# ============================================================
# 1. 데이터 로드
# ============================================================

print("Loading data...")
crops = np.load(f'{PROCESSED_DIR}/crops.npy')
is_target = np.load(f'{PROCESSED_DIR}/is_target.npy')
image_names = np.load(f'{PROCESSED_DIR}/image_names.npy')

print(f"✓ Total cells: {len(crops)}")
print(f"✓ Target cells: {is_target.sum()} ({is_target.mean()*100:.1f}%)")
print(f"✓ Non-target cells: {(~is_target).sum()}")

# ============================================================
# 2. Target vs Non-target 비교
# ============================================================

print("\n[1] Target vs Non-target comparison...")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Target cells
target_indices = np.where(is_target)[0]
if len(target_indices) >= 5:
    for i, idx in enumerate(target_indices[:5]):
        axes[0, i].imshow(crops[idx], cmap='gray')
        axes[0, i].set_title(f'Target #{idx}', color='red', fontsize=10)
        axes[0, i].axis('off')
else:
    # Target이 5개 미만이면
    for i, idx in enumerate(target_indices):
        axes[0, i].imshow(crops[idx], cmap='gray')
        axes[0, i].set_title(f'Target #{idx}', color='red', fontsize=10)
        axes[0, i].axis('off')
    for i in range(len(target_indices), 5):
        axes[0, i].axis('off')

# Non-target cells
nontarget_indices = np.where(~is_target)[0]
for i, idx in enumerate(nontarget_indices[:5]):
    axes[1, i].imshow(crops[idx], cmap='gray')
    axes[1, i].set_title(f'Non-target #{idx}', color='blue', fontsize=10)
    axes[1, i].axis('off')

plt.suptitle('Target vs Non-target Cells', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/1_target_vs_nontarget.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR}/1_target_vs_nontarget.png")

# ============================================================
# 3. 원본 이미지 + Green + Overlay
# ============================================================

print("\n[2] Original + Green + Overlay...")

# 첫 번째 이미지
first_image_name = image_names[0]
if first_image_name.startswith('phase_'):
    base_name = first_image_name[6:]
else:
    base_name = first_image_name

phase_path = f'{PHASE_DIR}/phase_{base_name}.tif'
green_path = f'{GREEN_DIR}/green_{base_name}.tif'

phase_img = cv2.imread(phase_path, cv2.IMREAD_GRAYSCALE)
green_img = cv2.imread(green_path)

if green_img is not None and len(green_img.shape) == 3:
    green_channel = green_img[:, :, 1]  # Green channel
else:
    green_channel = green_img

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Phase
axes[0].imshow(phase_img, cmap='gray')
axes[0].set_title('Phase-Contrast', fontsize=12)
axes[0].axis('off')

# Green channel (Greens colormap으로 수정!)
if green_channel is not None:
    axes[1].imshow(green_channel, cmap='Greens', vmin=0, vmax=100)
    axes[1].set_title('Green Channel', fontsize=12)
    axes[1].axis('off')

# Overlay
if phase_img is not None and green_channel is not None:
    overlay = cv2.cvtColor(phase_img, cv2.COLOR_GRAY2RGB)
    # Green을 빨간색으로 overlay
    mask = green_channel > 10
    overlay[mask, 0] = np.minimum(overlay[mask, 0] + green_channel[mask], 255)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Red = High Green)', fontsize=12)
    axes[2].axis('off')

plt.suptitle(f'Example: {base_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/2_original_green_overlay.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR}/2_original_green_overlay.png")

# ============================================================
# 4. Crop quality 확인
# ============================================================

print("\n[3] Crop quality check...")

fig, axes = plt.subplots(3, 6, figsize=(18, 9))

# Random 18개
random_indices = np.random.choice(len(crops), min(18, len(crops)), replace=False)

for i, idx in enumerate(random_indices):
    row = i // 6
    col = i % 6
    
    axes[row, col].imshow(crops[idx], cmap='gray')
    
    if is_target[idx]:
        axes[row, col].set_title(f'#{idx} (T)', color='red', fontsize=9)
        for spine in axes[row, col].spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)
    else:
        axes[row, col].set_title(f'#{idx}', fontsize=9)
    
    axes[row, col].axis('off')

plt.suptitle('Random Crop Samples (Red border = Target)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/3_crop_quality.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR}/3_crop_quality.png")

# ============================================================
# 5. 통계 그래프
# ============================================================

print("\n[4] Statistics...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Target ratio
target_ratio = is_target.mean() * 100
axes[0].bar(['Non-target', 'Target'], 
           [(~is_target).sum(), is_target.sum()],
           color=['blue', 'red'])
axes[0].set_ylabel('Count', fontsize=11)
axes[0].set_title(f'Target Ratio: {target_ratio:.1f}%', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Pixel intensity distribution
if is_target.sum() > 0 and (~is_target).sum() > 0:
    target_intensities = crops[is_target].flatten()
    nontarget_intensities = crops[~is_target].flatten()

    axes[1].hist(nontarget_intensities, bins=50, alpha=0.5, label='Non-target', 
                color='blue', density=True)
    axes[1].hist(target_intensities, bins=50, alpha=0.5, label='Target', 
                color='red', density=True)
    axes[1].set_xlabel('Pixel Intensity', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title('Intensity Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

# Cells per image
from collections import Counter
image_counts = Counter(image_names)
axes[2].hist(list(image_counts.values()), bins=30, color='green', alpha=0.7)
axes[2].set_xlabel('Cells per Image', fontsize=11)
axes[2].set_ylabel('Frequency', fontsize=11)
axes[2].set_title(f'Avg: {np.mean(list(image_counts.values())):.1f} cells/image', 
                 fontsize=12, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/4_statistics.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR}/4_statistics.png")

# ============================================================
# 6. 시계열 확인
# ============================================================

print("\n[5] Time-series check...")

# B10_1 well
well_name = 'B10_1'
well_images = sorted([name for name in np.unique(image_names) if well_name in name])[:6]

if len(well_images) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, img_name in enumerate(well_images):
        # 해당 이미지의 세포들
        if img_name.startswith('phase_'):
            search_name = img_name
        else:
            search_name = f'phase_{img_name}'
        
        mask = (image_names == search_name)
        img_target_ratio = is_target[mask].mean() * 100 if mask.sum() > 0 else 0
        
        # 첫 번째 세포
        cell_idx = np.where(mask)[0][0] if mask.sum() > 0 else 0
        
        axes[i].imshow(crops[cell_idx], cmap='gray')
        axes[i].set_title(f'{img_name}\nTarget: {img_target_ratio:.1f}%', fontsize=10)
        axes[i].axis('off')
    
    # 빈 subplot 제거
    for j in range(len(well_images), 6):
        fig.delaxes(axes[j])
    
    plt.suptitle(f'Time-series: Well {well_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/5_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR}/5_timeseries.png")

# ============================================================
# 요약
# ============================================================

print("\n" + "="*70)
print("Validation Complete!")
print("="*70)
print(f"\nResults saved in: {OUTPUT_DIR}/")
print("\nGenerated files:")
print("  1_target_vs_nontarget.png")
print("  2_original_green_overlay.png")
print("  3_crop_quality.png")
print("  4_statistics.png")
print("  5_timeseries.png")

print("\n⚠️ 확인 사항:")
if target_ratio < 1.0:
    print(f"  ⚠️  Target ratio가 너무 낮음: {target_ratio:.1f}%")
    print("      - Green threshold 확인 필요")
elif target_ratio > 50.0:
    print(f"  ⚠️  Target ratio가 너무 높음: {target_ratio:.1f}%")
    print("      - Green threshold 너무 낮음")
else:
    print(f"  ✓ Target ratio 정상: {target_ratio:.1f}%")

print(f"  ✓ Total: {len(crops)} cells")
print(f"  ✓ Target: {is_target.sum()} cells")