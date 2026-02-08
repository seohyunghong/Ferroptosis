"""
Ferroptosis Detection Pipeline
===============================

Label-free, Interpretable, Semi-supervised Ferroptosis Cell Detection

전체 파이프라인: UPC + Weak Label → K-way Cell Typing + Morphological Markers
"""
python train.py \
    --phase-dir ./data/ferroptosis/kanglab_data/phase \
    --green-dir ./data/ferroptosis/kanglab_data/green \
    --processed-dir ./processed \
    --output-dir ./output \
    --K 3 \
    --epochs 100 \
    --batch-size 32 2>&1 | tail -50

# ============================================================
# 📦 프로젝트 구조
# ============================================================

"""
ferroptosis_detection/
├── data/
│   ├── __init__.py
│   ├── preprocessing_before_cellpose.py       # [단계 0] Cellpose segmentation, crop extraction
│   └── augmentation.py        # Augmentation for consistency
│
├── models/
│   ├── __init__.py
│   ├── morphogenie.py         # [단계 1] MorphoGenie (고정)
│   ├── adapter.py             # [단계 2] Domain adaptation
│   ├── prototype.py           # [단계 3] Prototypical network
│   └── mean_teacher.py        # [단계 6] Mean-Teacher SSL
│
├── losses/
│   ├── __init__.py
│   ├── contrastive.py         # [단계 4] SupCon, ArcFace
│   ├── consistency.py         # Consistency regularization
│   └── clustering.py          # Constrained clustering loss
│
├── utils/
│   ├── __init__.py
│   ├── anchors.py             # Anchor selection
│   ├── clustering.py          # [단계 5] Constrained k-means
│   └── visualization.py       # Cell type visualization
│
├── train.py                   # 전체 통합 학습
└── inference.py               # 추론 파이프라인
"""

# ============================================================
# 🎯 각 파일의 역할 및 파이프라인 단계
# ============================================================

PIPELINE_MAPPING = {
    
    # ─────────────────────────────────────────────────────────
    # 입력 → 전처리
    # ─────────────────────────────────────────────────────────
    
    "data/preprocessing.py": {
        "단계": "입력 → 전처리",
        "역할": [
            "Cellpose3 segmentation",
            "256x256 crop extraction (MorphoGenie 표준)",
            "Green GT bbox detection",
            "Target cell assignment (weak label → seed)"
        ],
        "핵심 클래스": [
            "CellposeSegmenter",
            "GreenGTProcessor",
            "CellCropExtractor",
            "TargetCellAssigner",
            "FerroptosisDataset"
        ],
        "입력": "UPC images + Green GT (일부 프레임)",
        "출력": {
            "crops": "(N, 256, 256) - 세포 crops",
            "is_target": "(N,) bool - target cell seed"
        },
        "사용법": """
        from data.preprocessing import FerroptosisDataset
        
        builder = FerroptosisDataset(
            phase_dir='./data/phase',
            green_dir='./data/green',
            output_dir='./processed'
        )
        dataset = builder.build_dataset()
        """
    },
    
    # ─────────────────────────────────────────────────────────
    # 단계 1: MorphoGenie (고정)
    # ─────────────────────────────────────────────────────────
    
    "models/morphogenie.py": {
        "단계": "1) Morphogenie: 해석 가능한 형태 표현",
        "역할": [
            "Concept vector c_i ∈ R^M 생성",
            "  - cylindrical_index, granularity, ...",
            "Embedding f_i ∈ R^d 생성",
            "파라미터 freeze (해석가능성 유지)"
        ],
        "핵심 원칙": "도메인 갭은 Adapter로 흡수, MorphoGenie는 고정",
        "핵심 클래스": [
            "MorphoGenieEncoder - VAE 기반 인코더",
            "ConceptAwareExtractor - Batch processing"
        ],
        "입력": "(N, 1, 256, 256) - cell crops",
        "출력": {
            "f": "(N, latent_dim) - embedding",
            "c": "(N, concept_dim) - concept vector"
        },
        "사용법": """
        from models.morphogenie import MorphoGenieEncoder
        
        morphogenie = MorphoGenieEncoder(
            latent_dim=256,
            concept_dim=64,
            pretrained_path='morphogenie.pth'
        )
        
        f, c = morphogenie(crops)
        # f: embedding for downstream
        # c: interpretable concepts for marker discovery
        """
    },
    
    # ─────────────────────────────────────────────────────────
    # 단계 2: Adapter (도메인 편차 흡수)
    # ─────────────────────────────────────────────────────────
    
    "models/adapter.py": {
        "단계": "2) +Adapter: 도메인 편차만 흡수",
        "역할": [
            "z_i = f_i + A(f_i) - Residual adaptation",
            "Bottleneck MLP (작은 파라미터)",
            "Identity regularization: ||A_c(c)||^2"
        ],
        "핵심 원칙": "표현의 의미는 유지, 분포 shift만 보정",
        "핵심 클래스": [
            "ResidualAdapter",
            "ConceptAwareAdapter - Concept 의미 유지 강화",
            "AdapterWithPrototypes - Adapter + Prototypical 통합"
        ],
        "입력": {
            "f": "(B, embedding_dim)",
            "c": "(B, concept_dim)"
        },
        "출력": {
            "z": "(B, embedding_dim) - adapted",
            "c_adapted": "(B, concept_dim)",
            "logits": "(B, K) - classification"
        },
        "Loss": "L_id = ||A_c(c)||^2 (concept 과도한 변화 방지)",
        "사용법": """
        from models.adapter import AdapterWithPrototypes
        
        adapter = AdapterWithPrototypes(
            embedding_dim=256,
            concept_dim=64,
            num_prototypes=3,  # K cell types
            bottleneck_dim=64
        )
        
        outputs = adapter(f, c)
        # z: adapted embedding
        # logits: classification logits
        """
    },
    
    # ─────────────────────────────────────────────────────────
    # 단계 3: Prototypical Network
    # ─────────────────────────────────────────────────────────
    
    "models/adapter.py (AdapterWithPrototypes)": {
        "단계": "3) Prototypical: K개 프로토타입으로 분류",
        "역할": [
            "각 클래스마다 대표점 p_k",
            "P(y=k|z_i) = softmax(-||z_i - p_k||^2)",
            "Unlabeled도 가까운 프로토타입으로 자연스럽게 할당"
        ],
        "핵심 원칙": "Cluster 구조와 분류가 같은 언어",
        "왜 프로토타입인가": [
            "K가 고정, 라벨 제한적",
            "클래스 간 미묘한 차이",
            "Linear classifier는 경계만 그림 → 분포 변화에 약함",
            "Prototypical은 중심점 기반 → 안정적"
        ],
        "프로토타입 업데이트": "EMA + Seed constraint",
        "사용법": "AdapterWithPrototypes에 통합됨"
    },
    
    # ─────────────────────────────────────────────────────────
    # 단계 4: Contrastive Loss
    # ─────────────────────────────────────────────────────────
    
    "losses/contrastive.py": {
        "단계": "4) Contrastive: 미묘한 차이를 공간에서 분리",
        "역할": [
            "SupCon: 같은 클래스는 가까이, 다른 클래스는 멀리",
            "ArcFace: 각도 margin으로 fine-grained separation",
            "Triplet: Anchor-Positive-Negative"
        ],
        "핵심 원칙": "CE는 확률만 올림, Contrastive는 공간 구조 형성",
        "왜 필요한가": [
            "미묘한 형태 차이는 결정경계 하나로 안 잡힘",
            "Intra-class 분산 감소, Inter-class 거리 증가"
        ],
        "앵커 선택": [
            "모든 샘플 사용하면 망가짐",
            "현재 representation에서 클러스터 중심부 샘플만",
            "예: 거리 하위 10% = 매우 확실한 샘플"
        ],
        "핵심 클래스": [
            "SupConLoss",
            "ArcFaceLoss - CVPR 2018 표준",
            "TripletLoss",
            "CombinedContrastiveLoss"
        ],
        "사용법": """
        from losses.contrastive import CombinedContrastiveLoss
        
        contrastive = CombinedContrastiveLoss(
            embedding_dim=256,
            num_classes=3,
            lambda_supcon=1.0,
            lambda_arcface=1.0
        )
        
        losses = contrastive(features, labels, anchor_mask)
        """
    },
    
    # ─────────────────────────────────────────────────────────
    # 단계 5: Constrained Clustering
    # ─────────────────────────────────────────────────────────
    
    "utils/clustering.py": {
        "단계": "5) Constrained Clustering: K 고정 + 형광 seed 반영",
        "역할": [
            "Seeded k-means with constraints",
            "Must-link: 같은 타겟은 같은 클러스터",
            "Cannot-link: Living/Dead vs Target 분리",
            "Seed anchoring: 형광 양성 셀은 특정 클러스터 고정"
        ],
        "핵심 원칙": "관측 가능한 생물학적 앵커(형광)를 제약으로 주입",
        "왜 필요한가": [
            "Unlabeled 대부분, 타겟은 소량 형광",
            "K 고정",
            "그냥 k-means는 타겟이 흡수되거나 군집 수 흔들림"
        ],
        "핵심 클래스": [
            "ConstrainedKMeans",
            "SeededKMeans - 간단 버전",
            "PrototypeClusterer - PyTorch 버전"
        ],
        "사용법": """
        from utils.clustering import SeededKMeans
        
        clusterer = SeededKMeans(n_clusters=3)
        clusterer.fit(
            X=features,
            seed_indices={1: target_indices}  # Cluster 1 = target
        )
        """
    },
    
    # ─────────────────────────────────────────────────────────
    # 단계 6: Mean-Teacher
    # ─────────────────────────────────────────────────────────
    
    "models/mean_teacher.py": {
        "단계": "6) Mean-Teacher: Pseudo label 안전장치",
        "역할": [
            "Student θ: gradient 업데이트",
            "Teacher φ: EMA 업데이트 (φ ← αφ + (1-α)θ)",
            "Teacher 예측을 pseudo-label로 사용",
            "Consistency regularization"
        ],
        "핵심 원칙": "Teacher는 평균적 상태라 더 안정적",
        "왜 필요한가": [
            "Pseudo-label 노이즈가 self-training 붕괴 유발",
            "Teacher는 노이즈가 적은 pseudo-label 제공"
        ],
        "Consistency Loss": "L_cons = KL(Teacher || Student)",
        "핵심 클래스": [
            "MeanTeacher",
            "ConsistencyLoss",
            "ConsistencyRampUp - Weight scheduler",
            "MeanTeacherTrainer - 통합 모듈"
        ],
        "사용법": """
        from models.mean_teacher import MeanTeacherTrainer
        
        trainer = MeanTeacherTrainer(
            student_model=model,
            ema_decay=0.999,
            consistency_weight=1.0
        )
        
        losses = trainer.train_step(labeled_data, unlabeled_data, epoch)
        trainer.update_teacher()
        """
    },
    
    # ─────────────────────────────────────────────────────────
    # 전체 통합
    # ─────────────────────────────────────────────────────────
    
    "train.py": {
        "단계": "전체 파이프라인 통합",
        "역할": [
            "0. 데이터 전처리 (preprocessing.py)",
            "1. MorphoGenie (frozen)",
            "2. Adapter + Prototypes",
            "3. Contrastive loss",
            "4. Constrained clustering",
            "5. Mean-Teacher"
        ],
        "Loss 구성": """
        Total Loss = L_CE + L_Contrastive + λ_id * L_identity
        
        - L_CE: Target vs Non-target classification
        - L_Contrastive: SupCon + ArcFace
        - L_identity: ||A_c(c)||^2 (concept 의미 유지)
        """,
        "학습 순서": [
            "1. Forward: MorphoGenie (frozen) → Adapter",
            "2. Loss 계산: CE + Contrastive (anchor) + Identity",
            "3. Backward: Adapter 파라미터만 업데이트",
            "4. Mean-Teacher EMA 업데이트",
            "5. (매 5 epoch) Constrained clustering"
        ],
        "사용법": """
        python train.py \\
            --phase-dir ./data/phase \\
            --green-dir ./data/green \\
            --K 3 \\
            --epochs 100 \\
            --batch-size 32
        """
    }
}

# ============================================================
# 🚀 사용 방법
# ============================================================

USAGE_GUIDE = """
╔══════════════════════════════════════════════════════════════════╗
║  전체 파이프라인 실행                                             ║
╚══════════════════════════════════════════════════════════════════╝

Step 1: 데이터 준비
────────────────────
data/
├── phase/
│   ├── B10_1_00d00h00m.tif
│   ├── B10_1_00d03h00m.tif
│   └── ...
└── green/
    ├── B10_1_00d00h00m.tif
    ├── B10_1_00d03h00m.tif
    └── ...

Step 2: 전체 학습
────────────────────
python train.py \\
    --phase-dir ./data/phase \\
    --green-dir ./data/green \\
    --processed-dir ./processed \\
    --output-dir ./output \\
    --K 3 \\
    --epochs 100 \\
    --batch-size 32 \\
    --lr 1e-3

Step 3: 추론
────────────────────
python inference.py \\
    --checkpoint ./output/checkpoint.pth \\
    --input-dir ./test/phase \\
    --output-dir ./results

╔══════════════════════════════════════════════════════════════════╗
║  모듈별 독립 사용                                                 ║
╚══════════════════════════════════════════════════════════════════╝

[1] 전처리만 실행
──────────────────
from data.preprocessing import FerroptosisDataset

builder = FerroptosisDataset(
    phase_dir='./data/phase',
    green_dir='./data/green',
    output_dir='./processed'
)
dataset = builder.build_dataset()

[2] MorphoGenie feature 추출만
───────────────────────────────
from models.morphogenie import MorphoGenieEncoder, ConceptAwareExtractor

model = MorphoGenieEncoder(latent_dim=256, concept_dim=64)
extractor = ConceptAwareExtractor(model)

embeddings, concepts = extractor.extract_features(crops)

[3] Contrastive loss만 사용
──────────────────────────────
from losses.contrastive import SupConLoss

supcon = SupConLoss(temperature=0.07)
loss = supcon(features, labels)

[4] Constrained clustering만
────────────────────────────────
from utils.clustering import SeededKMeans

clusterer = SeededKMeans(n_clusters=3)
clusterer.fit(X, seed_indices={1: target_indices})
"""

# ============================================================
# 📊 Loss 구성 및 Hyperparameters
# ============================================================

LOSS_COMPOSITION = """
╔══════════════════════════════════════════════════════════════════╗
║  Loss 구성                                                        ║
╚══════════════════════════════════════════════════════════════════╝

Total Loss = α * L_CE + β * L_Contrastive + γ * L_identity + δ * L_consistency

1. L_CE (Classification)
   - CrossEntropy(logits, labels)
   - Target vs Non-target 구분

2. L_Contrastive (SupCon + ArcFace)
   - SupCon: 같은 클래스 가까이, 다른 클래스 멀리
   - ArcFace: 각도 margin으로 fine-grained separation
   - Anchor만 사용 (확실한 샘플)

3. L_identity (Concept 의미 유지)
   - ||A_c(c)||^2
   - Adapter가 concept을 과도하게 변경 방지

4. L_consistency (Mean-Teacher)
   - KL(Teacher || Student)
   - Consistency regularization

╔══════════════════════════════════════════════════════════════════╗
║  권장 Hyperparameters                                            ║
╚══════════════════════════════════════════════════════════════════╝

Loss weights:
  α (CE): 1.0
  β (Contrastive): 1.0
  γ (Identity): 0.1
  δ (Consistency): 1.0 (ramp-up)

Contrastive:
  SupCon temperature: 0.07
  ArcFace scale: 30.0
  ArcFace margin: 0.50

Adapter:
  Bottleneck dim: 64
  Dropout: 0.5
  Weight decay: 1e-4

Mean-Teacher:
  EMA decay: 0.999
  Consistency rampup: 50 epochs

Training:
  Epochs: 100
  Batch size: 32
  Learning rate: 1e-3
  Optimizer: Adam
"""

# ============================================================
# 🎓 논문 작성 가이드
# ============================================================

PAPER_GUIDE = """
╔══════════════════════════════════════════════════════════════════╗
║  논문 Method Section 구성                                         ║
╚══════════════════════════════════════════════════════════════════╝

3.1 Problem Formulation
  - UPC + weak label (형광)
  - K-way cell typing
  - Semi-supervised setting

3.2 MorphoGenie: Interpretable Morphological Representation
  - Concept vector c_i (cylindrical_index, granularity, ...)
  - Embedding f_i
  - Frozen to preserve interpretability

3.3 Domain Adaptation via Residual Adapter
  - z_i = f_i + A(f_i)
  - Identity regularization: ||A_c(c)||^2
  - Small parameters (bottleneck)

3.4 Prototypical Metric Space
  - K prototypes p_k
  - P(y=k|z_i) = softmax(-||z_i - p_k||^2)
  - Why: cluster structure = classification

3.5 Contrastive Learning for Fine-grained Separation
  - SupCon: same class close, different class far
  - ArcFace: angular margin
  - Anchor selection (cluster core)

3.6 Constrained Clustering with Fluorescence Seeds
  - Seeded k-means
  - Must-link, Cannot-link
  - Biological anchor injection

3.7 Mean-Teacher for Pseudo-label Stability
  - Teacher φ: EMA of student θ
  - Consistency regularization
  - Unlabeled data expansion

╔══════════════════════════════════════════════════════════════════╗
║  Ablation Study 설계                                              ║
╚══════════════════════════════════════════════════════════════════╝

Baseline vs Proposed:

| Method                        | F1 Score |
|-------------------------------|----------|
| Frozen MorphoGenie + Classifier | 0.75   |
| + Adapter                     | 0.80     |
| + Contrastive                 | 0.85     |
| + Constrained Clustering      | 0.87     |
| + Mean-Teacher (Full)         | 0.89     |

각 component의 기여도 검증
"""

# ============================================================
# 출력
# ============================================================

if __name__ == '__main__':
    print("="*70)
    print("Ferroptosis Detection Pipeline")
    print("="*70)
    
    print("\n📦 프로젝트 구조:")
    print(__doc__)
    
    print("\n🎯 파이프라인 단계별 매핑:")
    for filename, info in PIPELINE_MAPPING.items():
        print(f"\n{filename}")
        print(f"  단계: {info['단계']}")
        print(f"  역할: {', '.join(info['역할'][:2])}")
    
    print(USAGE_GUIDE)
    print(LOSS_COMPOSITION)
    print(PAPER_GUIDE)