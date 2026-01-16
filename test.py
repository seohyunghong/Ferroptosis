import numpy as np
from cellpose import models
import time

def test_cellpose():
    print(">>> 1. Cellpose 모델 로딩 중... (처음이면 다운로드에 시간이 걸립니다)")
    # GPU 사용 설정, 'cyto' 모델 사용
    model = models.Cellpose(gpu=True, model_type='cyto')

    print(">>> 2. 더미 이미지 생성 중...")
    # 512x512 크기의 랜덤 노이즈 이미지 생성 (테스트용)
    img = np.random.randint(0, 255, (512, 512), dtype='uint8')

    print(">>> 3. 세그멘테이션 실행 중...")
    start_time = time.time()
    
    # eval 실행 (지름 30픽셀 가정)
    masks, flows, styles, diams = model.eval(img, diameter=30, channels=[0,0])
    
    end_time = time.time()
    
    print(f">>> 4. 완료! 소요 시간: {end_time - start_time:.4f}초")
    print(f"    생성된 마스크 형태: {masks.shape}")
    print(">>> 테스트 성공!")

if __name__ == '__main__':
    test_cellpose()