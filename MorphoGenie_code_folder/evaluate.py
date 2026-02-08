"""
evaluate.py

학습된 모델 평가
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys

# ============================================================
# Model Definition (train.py와 동일)
# ============================================================

class MorphoGenieEncoder(nn.Module):
    """MorphoGenie (simplified)"""
    def __init__(self, latent_dim=256, concept_dim=64):
        super().__init__()
        # Placeholder - 실제는 pretrained model 로드
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc_embedding = nn.Linear(128, latent_dim)
        self.fc_concept = nn.Linear(128, concept_dim)
    
    def forward(self, x):
        features = self.encoder(x).flatten(1)
        f = self.fc_embedding(features)
        c = self.fc_concept(features)
        return f, c


class AdapterModule(nn.Module):
    """Single Adapter"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.adapter(x)


class AdapterWithPrototypes(nn.Module):
    """Adapter + Prototypical Network"""
    def __init__(self, embedding_dim=256, concept_dim=64, num_prototypes=3):
        super().__init__()
        
        # Adapter (nested structure!)
        self.adapter = nn.ModuleDict({
            'embedding_adapter': AdapterModule(embedding_dim, 64, embedding_dim),
            'concept_adapter': nn.Sequential(
                nn.Linear(concept_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(32, concept_dim)
            )
        })
        
        # Prototypes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embedding_dim))
    
    def forward(self, f, c):
        # Adapter
        delta_f = self.adapter['embedding_adapter'](f)
        z = f + delta_f
        
        delta_c = self.adapter['concept_adapter'](c)
        c_adapted = c + delta_c
        
        # Prototypical
        z_expanded = z.unsqueeze(1)  # (B, 1, D)
        p_expanded = self.prototypes.unsqueeze(0)  # (1, K, D)
        distances = torch.sqrt(torch.sum((z_expanded - p_expanded) ** 2, dim=2))
        logits = -distances
        
        return {
            'z': z,
            'c_adapted': c_adapted,
            'logits': logits
        }


# ============================================================
# Evaluation
# ============================================================

def evaluate(checkpoint_path, data_dir, output_dir, test_ratio=0.2):
    """
    모델 평가
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ===== 데이터 로드 =====
    print("\n" + "="*70)
    print("Loading data...")
    print("="*70)
    
    crops = np.load(f'{data_dir}/crops.npy')
    is_target = np.load(f'{data_dir}/is_target.npy')
    
    print(f"Total cells: {len(crops)}")
    print(f"Target: {is_target.sum()} ({is_target.mean()*100:.1f}%)")
    
    # Train/Test split
    np.random.seed(42)
    n_total = len(crops)
    indices = np.random.permutation(n_total)
    n_train = int((1 - test_ratio) * n_total)
    
    test_indices = indices[n_train:]
    crops_test = crops[test_indices]
    is_target_test = is_target[test_indices]
    
    print(f"\nTest set: {len(crops_test)} cells")
    print(f"  Target: {is_target_test.sum()}")
    print(f"  Non-target: {(~is_target_test).sum()}")
    
    # Tensor 변환
    crops_test = torch.from_numpy(crops_test).float().unsqueeze(1) / 255.0
    is_target_test = torch.from_numpy(is_target_test)
    
    # DataLoader
    dataset = TensorDataset(crops_test, is_target_test)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # ===== 모델 로드 =====
    print("\n" + "="*70)
    print("Loading model...")
    print("="*70)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    morphogenie = MorphoGenieEncoder().to(device)
    adapter = AdapterWithPrototypes(
        embedding_dim=256,
        concept_dim=64,
        num_prototypes=3
    ).to(device)
    
    adapter.load_state_dict(checkpoint['adapter_state_dict'])
    adapter.eval()
    morphogenie.eval()
    
    print("✓ Model loaded")
    
    # ===== 추론 =====
    print("\n" + "="*70)
    print("Running inference...")
    print("="*70)
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_embeddings = []
    
    with torch.no_grad():
        for crops_batch, labels_batch in dataloader:
            crops_batch = crops_batch.to(device)
            
            # Forward
            f, c = morphogenie(crops_batch)
            outputs = adapter(f, c)
            
            # Predictions
            probs = torch.softmax(outputs['logits'], dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels_batch.numpy())
            all_probs.append(probs.cpu().numpy())
            all_embeddings.append(outputs['z'].cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_probs = np.concatenate(all_probs)
    z_all = np.concatenate(all_embeddings)
    
    print(f"✓ Inference complete")
    
    # ===== Metrics =====
    print("\n" + "="*70)
    print("Calculating metrics...")
    print("="*70)
    
    # Binary (target vs non-target)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ===== 출력 =====
    
    # Console
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"\nTarget Detection:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Pred_Non  Pred_Target")
    print(f"True_Non      {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"True_Target   {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # Text file
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("Model Evaluation Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Test Set Size: {len(y_true)}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        
        f.write("Target Detection Metrics:\n")
        f.write("-"*70 + "\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"Support:   {support}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write("-"*70 + "\n")
        f.write(f"              Predicted_Non  Predicted_Target\n")
        f.write(f"True_Non            {cm[0,0]:6d}         {cm[0,1]:6d}\n")
        f.write(f"True_Target         {cm[1,0]:6d}         {cm[1,1]:6d}\n")
    
    print(f"\n✓ Saved: {output_dir / 'metrics.txt'}")
    
    # CSV
    import pandas as pd
    df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    df.to_csv(output_dir / 'metrics.csv', index=False)
    print(f"✓ Saved: {output_dir / 'metrics.csv'}")
    
    # ===== Confusion Matrix Plot =====
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-target', 'Target'],
                yticklabels=['Non-target', 'Target'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'confusion_matrix.png'}")
    
    # ===== Save predictions =====
    
    np.save(output_dir / 'predictions.npy', {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': y_probs,
        'embeddings': z_all
    }, allow_pickle=True)
    print(f"✓ Saved: {output_dir / 'predictions.npy'}")
    
    # ===== Summary =====
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    print(f"\nResults saved in: {output_dir}/")
    print("\nGenerated files:")
    print("  - metrics.txt")
    print("  - metrics.csv")
    print("  - confusion_matrix.png")
    print("  - predictions.npy")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--data-dir', type=str, default='./processed_final',
                       help='Path to processed data directory')
    parser.add_argument('--output-dir', type=str, default='./evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Test set ratio (default: 0.2)')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if not Path(args.data_dir).exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Run evaluation
    results = evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio
    )