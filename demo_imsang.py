import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI 없이 이미지 저장
import matplotlib.pyplot as plt
from pathlib import Path
from cellpose import models, io, core
from skimage import measure
from tqdm import tqdm
from datetime import datetime

# ===== 설정 =====
class Config:
    # 데이터 경로
    DATA_ROOT = "./data/ferroptosis/kanglab_data"
    IMG_DIR = os.path.join(DATA_ROOT, "images")
    OUTPUT_DIR = os.path.join(DATA_ROOT, "results")
    
    # Cellpose 설정
    MODEL_TYPE = 'cyto3'
    DIAMETER = None
    FLOW_THRESHOLD = 0.4
    CELLPROB_THRESHOLD = 0.0
    
    # Feature Extraction 임계값
    FERROPTOSIS_AREA_MIN = 1500
    FERROPTOSIS_CIRCULARITY_MIN = 0.7
    DEAD_AREA_MAX = 500
    
    USE_GPU = core.use_gpu()

# ===== 유틸리티 함수 =====

def get_timestamp():
    """현재 시간 기반 타임스탬프 생성"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_run_number(output_dir):
    """실행 번호 자동 증가"""
    if not os.path.exists(output_dir):
        return 1
    
    existing = [d for d in os.listdir(output_dir) if d.startswith('run_')]
    if not existing:
        return 1
    
    numbers = []
    for d in existing:
        try:
            num = int(d.split('_')[1])
            numbers.append(num)
        except:
            continue
    
    return max(numbers) + 1 if numbers else 1

def create_run_directory(base_dir):
    """실행 디렉토리 생성"""
    run_num = get_run_number(base_dir)
    timestamp = get_timestamp()
    run_dir = os.path.join(base_dir, f"run_{run_num:03d}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_num

def extract_cell_features(mask, intensity_image):
    """세포별 형태학적 특징 추출"""
    props = measure.regionprops_table(
        mask, 
        intensity_image=intensity_image,
        properties=[
            'label',
            'area',
            'perimeter',
            'eccentricity',
            'solidity',
            'mean_intensity',
            'max_intensity',
            'min_intensity',
            'centroid',
            'bbox'
        ]
    )
    
    df = pd.DataFrame(props)
    
    # 원형도 계산
    df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
    
    # Aspect Ratio
    df['aspect_ratio'] = 1 / (1 - df['eccentricity']**2 + 1e-10)
    
    return df

def classify_cell_state(features_df):
    """형태학적 특징 기반 세포 상태 분류"""
    def classify_single(row):
        area = row['area']
        circ = row['circularity']
        
        if area > Config.FERROPTOSIS_AREA_MIN and circ > Config.FERROPTOSIS_CIRCULARITY_MIN:
            return 'Ferroptosis'
        elif area < Config.DEAD_AREA_MAX:
            return 'Dead'
        else:
            return 'Living'
    
    features_df['cell_state'] = features_df.apply(classify_single, axis=1)
    return features_df

def visualize_segmentation(image, mask, features_df, save_path):
    """Segmentation 결과 시각화 (파일로 저장)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Phase Contrast')
    axes[0].axis('off')
    
    # Mask overlay
    from skimage.color import label2rgb
    overlay = label2rgb(mask, image=image, bg_label=0, alpha=0.3)
    axes[1].imshow(overlay)
    axes[1].set_title(f'Segmentation ({len(features_df)} cells)')
    axes[1].axis('off')
    
    # Classification overlay
    axes[2].imshow(image, cmap='gray', alpha=0.5)
    
    colors = {'Ferroptosis': 'red', 'Dead': 'blue', 'Living': 'green'}
    
    for _, row in features_df.iterrows():
        y, x = row['centroid-0'], row['centroid-1']
        state = row['cell_state']
        axes[2].scatter(x, y, c=colors[state], s=50, alpha=0.7, 
                       edgecolors='white', linewidth=1)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[k], label=k) for k in colors.keys()]
    axes[2].legend(handles=legend_elements, loc='upper right')
    axes[2].set_title('Cell State Classification')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_text_report(summary, save_path):
    """텍스트 리포트 저장"""
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FERROPTOSIS ANALYSIS REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Image: {summary['image_name']}\n")
        f.write(f"Total Cells: {summary['total_cells']}\n")
        f.write(f"Ferroptosis: {summary['ferroptosis']} ({summary['ferroptosis_ratio']:.1%})\n")
        f.write(f"Dead: {summary['dead']}\n")
        f.write(f"Living: {summary['living']}\n")
        f.write("="*60 + "\n")

# ===== 메인 파이프라인 =====

def process_single_image(image_path, model, output_dir):
    """단일 이미지 처리"""
    img = io.imread(image_path)
    img_name = Path(image_path).stem
    
    print(f"  Processing: {img_name}")
    
    # Segmentation
    masks, flows, styles, diams = model.eval(
        img,
        diameter=Config.DIAMETER,
        channels=[0, 0],
        flow_threshold=Config.FLOW_THRESHOLD,
        cellprob_threshold=Config.CELLPROB_THRESHOLD
    )
    
    # Feature Extraction
    features_df = extract_cell_features(masks, img)
    features_df = classify_cell_state(features_df)
    
    # 통계
    state_counts = features_df['cell_state'].value_counts()
    print(f"    - Total cells: {len(features_df)}")
    print(f"    - Ferroptosis: {state_counts.get('Ferroptosis', 0)}")
    print(f"    - Dead: {state_counts.get('Dead', 0)}")
    print(f"    - Living: {state_counts.get('Living', 0)}")
    
    # 저장
    csv_path = os.path.join(output_dir, f"{img_name}_features.csv")
    features_df.to_csv(csv_path, index=False)
    
    mask_path = os.path.join(output_dir, f"{img_name}_masks.npy")
    np.save(mask_path, masks)
    
    vis_path = os.path.join(output_dir, f"{img_name}_visualization.png")
    visualize_segmentation(img, masks, features_df, save_path=vis_path)
    
    summary = {
        'image_name': img_name,
        'total_cells': int(len(features_df)),
        'ferroptosis': int(state_counts.get('Ferroptosis', 0)),
        'dead': int(state_counts.get('Dead', 0)),
        'living': int(state_counts.get('Living', 0)),
        'ferroptosis_ratio': float(state_counts.get('Ferroptosis', 0) / len(features_df)) if len(features_df) > 0 else 0
    }
    
    txt_path = os.path.join(output_dir, f"{img_name}_report.txt")
    save_text_report(summary, txt_path)
    
    return features_df, masks, summary

def batch_process(image_dir, base_output_dir):
    """배치 처리"""
    print(f">>> GPU Activated: {Config.USE_GPU}")
    print(f">>> Loading Cellpose model: {Config.MODEL_TYPE}")
    
    # 실행 디렉토리 생성
    output_dir, run_num = create_run_directory(base_output_dir)
    print(f">>> Output directory: {output_dir} (Run #{run_num})")
    
    # 모델 로드
    model = models.Cellpose(gpu=Config.USE_GPU, model_type=Config.MODEL_TYPE)
    
    # 이미지 파일 찾기
    image_files = []
    for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']:
        image_files.extend(Path(image_dir).glob(ext))
    
    print(f">>> Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print(f"!!! No images found in {image_dir}")
        print(f"!!! Please check the path and image extensions")
        return
    
    # 배치 처리
    all_summaries = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            _, _, summary = process_single_image(str(img_path), model, output_dir)
            all_summaries.append(summary)
        except Exception as e:
            print(f"  !!! Error processing {img_path}: {e}")
            continue
    
    # 전체 통계 저장
    if all_summaries:
        total_summary_df = pd.DataFrame(all_summaries)
        summary_csv_path = os.path.join(output_dir, "batch_summary.csv")
        total_summary_df.to_csv(summary_csv_path, index=False)
        
        # 전체 통계 텍스트 저장
        stats_path = os.path.join(output_dir, "batch_statistics.txt")
        with open(stats_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("BATCH PROCESSING SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Run Number: {run_num}\n\n")
            f.write(f"Total Images Processed: {len(all_summaries)}\n")
            f.write(f"Total Cells: {total_summary_df['total_cells'].sum()}\n")
            f.write(f"Total Ferroptosis: {total_summary_df['ferroptosis'].sum()}\n")
            f.write(f"Total Dead: {total_summary_df['dead'].sum()}\n")
            f.write(f"Total Living: {total_summary_df['living'].sum()}\n")
            f.write(f"Average Ferroptosis Ratio: {total_summary_df['ferroptosis_ratio'].mean():.1%}\n")
            f.write("="*60 + "\n")
        
        print(f"\n{'='*60}")
        print("Batch Processing Complete!")
        print(f"{'='*60}")
        print(f"Run Number: {run_num}")
        print(f"Total Images: {len(all_summaries)}")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")
    else:
        print("\n!!! No images were successfully processed")

# ===== 실행 =====

if __name__ == '__main__':
    batch_process(
        image_dir=Config.IMG_DIR,
        base_output_dir=Config.OUTPUT_DIR
    )