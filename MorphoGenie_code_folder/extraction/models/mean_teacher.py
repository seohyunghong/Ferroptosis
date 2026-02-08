"""
models/mean_teacher.py

[파이프라인 단계 6: Mean-Teacher]

Pseudo label 안정장치 - Teacher-Student SSL
- Student: gradient로 업데이트
- Teacher: EMA로 안정화
- Consistency regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# ============================================================
# Mean-Teacher Framework
# ============================================================

class MeanTeacher(nn.Module):
    """
    [6. Mean-Teacher: Pseudo label 안전장치]
    
    핵심 원리:
    - Student θ: gradient 업데이트
    - Teacher φ: EMA 업데이트 (φ ← αφ + (1-α)θ)
    - Teacher 예측을 pseudo-label로 사용
    - Consistency regularization
    """
    
    def __init__(self,
                 student_model: nn.Module,
                 ema_decay: float = 0.999):
        super().__init__()
        
        self.student = student_model
        self.ema_decay = ema_decay
        
        # Teacher: copy of student
        self.teacher = deepcopy(student_model)
        
        # Freeze teacher (no gradient)
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor, mode='student'):
        """
        Forward pass
        
        Args:
            x: input
            mode: 'student' or 'teacher'
        """
        if mode == 'student':
            return self.student(x)
        else:
            with torch.no_grad():
                return self.teacher(x)
    
    @torch.no_grad()
    def update_teacher(self):
        """
        [6-2. EMA 업데이트]
        
        φ ← α·φ + (1-α)·θ
        
        Teacher는 최근 student의 평균적 상태
        → 더 안정적
        """
        for teacher_param, student_param in zip(
            self.teacher.parameters(),
            self.student.parameters()
        ):
            teacher_param.data.mul_(self.ema_decay).add_(
                student_param.data, alpha=1 - self.ema_decay
            )


# ============================================================
# Consistency Loss
# ============================================================

class ConsistencyLoss(nn.Module):
    """
    [6-3. Consistency regularization]
    
    같은 샘플에 다른 augmentation을 넣었을 때
    student와 teacher 예측이 일치하도록
    
    L_cons = KL(q_teacher(x_aug1) || q_student(x_aug2))
    """
    
    def __init__(self, consistency_type='kl'):
        super().__init__()
        self.consistency_type = consistency_type
    
    def forward(self,
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Consistency loss
        
        Args:
            student_logits: (B, K)
            teacher_logits: (B, K)
            mask: (B,) - which samples to use
        """
        if self.consistency_type == 'kl':
            # KL divergence
            student_probs = F.log_softmax(student_logits, dim=1)
            teacher_probs = F.softmax(teacher_logits, dim=1)
            
            loss = F.kl_div(
                student_probs,
                teacher_probs,
                reduction='none'
            ).sum(dim=1)
        
        elif self.consistency_type == 'mse':
            # MSE
            loss = F.mse_loss(
                student_logits,
                teacher_logits,
                reduction='none'
            ).mean(dim=1)
        
        else:
            raise ValueError(f"Unknown consistency type: {self.consistency_type}")
        
        # Apply mask
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-10)
        else:
            loss = loss.mean()
        
        return loss


# ============================================================
# Ramp-up Scheduler
# ============================================================

class ConsistencyRampUp:
    """
    Consistency weight ramp-up
    
    초기에는 consistency weight를 0으로
    점진적으로 증가시켜 안정적 학습
    """
    
    def __init__(self,
                 max_weight: float = 1.0,
                 rampup_length: int = 50):
        self.max_weight = max_weight
        self.rampup_length = rampup_length
    
    def __call__(self, epoch: int) -> float:
        """
        Get consistency weight for current epoch
        """
        if epoch < self.rampup_length:
            # Sigmoid rampup
            phase = 1.0 - epoch / self.rampup_length
            return self.max_weight * float(np.exp(-5.0 * phase * phase))
        else:
            return self.max_weight


# ============================================================
# Complete Mean-Teacher Training Module
# ============================================================

class MeanTeacherTrainer:
    """
    [6. Mean-Teacher 전체 학습 모듈]
    
    통합:
    - Labeled data: CE loss
    - Unlabeled data: consistency loss (teacher pseudo-label)
    - EMA update for teacher
    """
    
    def __init__(self,
                 student_model: nn.Module,
                 ema_decay: float = 0.999,
                 consistency_weight: float = 1.0,
                 consistency_rampup: int = 50,
                 device='cuda'):
        
        self.device = device
        
        # Mean-Teacher
        self.mean_teacher = MeanTeacher(
            student_model=student_model,
            ema_decay=ema_decay
        ).to(device)
        
        # Losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.consistency_loss = ConsistencyLoss(consistency_type='kl')
        
        # Rampup scheduler
        self.rampup = ConsistencyRampUp(
            max_weight=consistency_weight,
            rampup_length=consistency_rampup
        )
    
    def train_step(self,
                   labeled_data: dict,
                   unlabeled_data: dict,
                   epoch: int) -> dict:
        """
        Single training step
        
        Args:
            labeled_data: {
                'x': (B_l, ...),
                'y': (B_l,)
            }
            unlabeled_data: {
                'x1': (B_u, ...) - augmented view 1,
                'x2': (B_u, ...) - augmented view 2
            }
            epoch: current epoch
            
        Returns:
            losses: dict
        """
        # Labeled loss (supervised)
        student_logits_labeled = self.mean_teacher(
            labeled_data['x'], mode='student'
        )['logits']
        
        loss_labeled = self.ce_loss(
            student_logits_labeled,
            labeled_data['y']
        )
        
        # Unlabeled loss (consistency)
        student_logits_unlabeled = self.mean_teacher(
            unlabeled_data['x1'], mode='student'
        )['logits']
        
        with torch.no_grad():
            teacher_logits_unlabeled = self.mean_teacher(
                unlabeled_data['x2'], mode='teacher'
            )['logits']
        
        loss_consistency = self.consistency_loss(
            student_logits_unlabeled,
            teacher_logits_unlabeled
        )
        
        # Consistency weight (ramp-up)
        cons_weight = self.rampup(epoch)
        
        # Total loss
        loss_total = loss_labeled + cons_weight * loss_consistency
        
        return {
            'total': loss_total,
            'labeled': loss_labeled,
            'consistency': loss_consistency,
            'cons_weight': cons_weight
        }
    
    def update_teacher(self):
        """
        Update teacher with EMA
        """
        self.mean_teacher.update_teacher()
    
    @torch.no_grad()
    def get_pseudo_labels(self,
                         unlabeled_x: torch.Tensor,
                         confidence_threshold: float = 0.95) -> tuple:
        """
        [6. Teacher pseudo-label 생성]
        
        Teacher로 pseudo-label 예측
        High-confidence만 사용
        
        Returns:
            pseudo_labels: (B,)
            confidence_mask: (B,) bool
        """
        teacher_logits = self.mean_teacher(unlabeled_x, mode='teacher')['logits']
        teacher_probs = F.softmax(teacher_logits, dim=1)
        
        # Max probability
        max_probs, pseudo_labels = torch.max(teacher_probs, dim=1)
        
        # Confidence mask
        confidence_mask = (max_probs > confidence_threshold)
        
        return pseudo_labels, confidence_mask


# ============================================================
# Usage Example
# ============================================================

if __name__ == '__main__':
    """
    Mean-Teacher test
    """
    import numpy as np
    
    # Dummy student model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(256, 3)
        
        def forward(self, x):
            return {'logits': self.fc(x)}
    
    student = DummyModel()
    
    # Mean-Teacher trainer
    trainer = MeanTeacherTrainer(
        student_model=student,
        ema_decay=0.999,
        consistency_weight=1.0,
        device='cpu'
    )
    
    # Dummy data
    labeled_data = {
        'x': torch.randn(16, 256),
        'y': torch.randint(0, 3, (16,))
    }
    
    unlabeled_data = {
        'x1': torch.randn(64, 256),
        'x2': torch.randn(64, 256)
    }
    
    # Training step
    losses = trainer.train_step(labeled_data, unlabeled_data, epoch=0)
    
    print("Losses:")
    print(f"  Total: {losses['total'].item():.4f}")
    print(f"  Labeled: {losses['labeled'].item():.4f}")
    print(f"  Consistency: {losses['consistency'].item():.4f}")
    print(f"  Cons weight: {losses['cons_weight']:.4f}")
    
    # Update teacher
    trainer.update_teacher()
    
    # Pseudo-labels
    pseudo_labels, conf_mask = trainer.get_pseudo_labels(unlabeled_data['x1'])
    print(f"\nPseudo-labels: {pseudo_labels[:10]}")
    print(f"High-confidence: {conf_mask.sum()}/{len(conf_mask)}")