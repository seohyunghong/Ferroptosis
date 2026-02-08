"""
train_fixed.py

수정사항:
1. VAE checkpoint 제대로 로드 (shape 기반 matching)
2. GPU 최적화 (batch size↑, num_workers, pin_memory)
3. Feature quality 분석
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
import argparse
from pathlib import Path
from tqdm import tqdm

# Model 정의 (train_semisupervised.py와 동일)
class MorphoGenieEncoder(nn.Module):
    def __init__(self, latent_dim=256, concept_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_concept = nn.Linear(512 * 4 * 4, concept_dim)
    
    def encode_deterministic(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_concept(h)

def load_vae(checkpoint_path, device='cuda'):
    """Shape 기반 layer matching"""
    print("="*70)
    print("Loading VAE")
    print("="*70)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vae_state = checkpoint['model_states']['VAE']
    
    encoder = MorphoGenieEncoder().to(device)
    our_state = encoder.state_dict()
    
    loaded = 0
    used = set()
    
    for our_key in sorted(our_state.keys()):
        our_shape = our_state[our_key].shape
        for vae_key, vae_param in vae_state.items():
            if vae_key in used or not isinstance(vae_param, torch.Tensor):
                continue
            if vae_param.shape == our_shape:
                our_state[our_key] = vae_param.to(device)
                used.add(vae_key)
                loaded += 1
                print(f"  {our_key:40s} ← {vae_key:35s} {str(our_shape)}")
                break
    
    encoder.load_state_dict(our_state)
    print(f"\n✓ Loaded {loaded}/{len(our_state)} layers ({loaded/len(our_state)*100:.0f}%)")
    
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    return encoder

def extract_features(crops, model, device):
    crops_tensor = torch.from_numpy(crops).float().unsqueeze(1) / 255.0
    dataloader = DataLoader(
        TensorDataset(crops_tensor),
        batch_size=512, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    all_f, all_c = [], []
    with torch.no_grad():
        for (batch,) in tqdm(dataloader, desc="Extracting"):
            batch = batch.to(device, non_blocking=True)
            f, c = model.encode_deterministic(batch)
            all_f.append(f.cpu().numpy())
            all_c.append(c.cpu().numpy())
    
    return np.vstack(all_f), np.vstack(all_c)

def analyze(all_f, is_target):
    f_t = all_f[is_target]
    f_nt = all_f[~is_target]
    
    print("\nTarget:    ", f_t.mean(axis=0)[:5])
    print("Non-target:", f_nt.mean(axis=0)[:5])
    
    dist = np.linalg.norm(f_t.mean(axis=0) - f_nt.mean(axis=0))
    print(f"\nDistance: {dist:.6f}")
    
    if dist > 0.5:
        print("✓ GOOD!")
    elif dist > 0.1:
        print("○ Moderate")
    else:
        print("✗ WEAK - check loading!")
    
    return dist

def main(args):
    device = torch.device('cuda')
    
    crops = np.load(args.data_dir + '/crops.npy')
    is_target = np.load(args.data_dir + '/is_target.npy')
    
    print(f"Data: {crops.shape}, Target: {is_target.sum()}/{len(is_target)}")
    
    model = load_vae(args.vae_checkpoint, device)
    all_f, all_c = extract_features(crops, model, device)
    
    dist = analyze(all_f, is_target)
    
    # K-means
    pseudo_labels = np.zeros(len(all_f), dtype=int)
    pseudo_labels[is_target] = 0
    if is_target.sum() < len(is_target):
        kmeans = KMeans(n_clusters=2, random_state=42)
        pseudo_labels[~is_target] = kmeans.fit_predict(all_f[~is_target]) + 1
    
    # Save
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    np.save(f'{args.output_dir}/features_f.npy', all_f)
    np.save(f'{args.output_dir}/pseudo_labels.npy', pseudo_labels)
    
    print(f"\n✓ Saved to {args.output_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--vae-checkpoint', default='./checkpoint/CCy/VAE/chkpts/last')
    parser.add_argument('--output-dir', default='./vae_features')
    args = parser.parse_args()
    main(args)