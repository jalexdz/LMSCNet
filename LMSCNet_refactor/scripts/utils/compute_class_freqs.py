import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))  # one level up
sys.path.insert(0, PROJECT_ROOT)


from data.loader import SynthSSCDataset  # Adjust if different

def compute_class_frequencies(dataset, num_classes, ignore_index=255):
    class_counts = np.zeros(num_classes, dtype=np.int64)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for data in tqdm(loader, desc="Scanning voxel labels"):
        label = data['3D_LABEL']  # Should be [B, D, H, W]
        label = label.squeeze().cpu().numpy().flatten().astype(np.int64)
        label = label[label != ignore_index]
        counts = np.bincount(label, minlength=num_classes)
        class_counts += counts

    frequencies = class_counts 
    return frequencies

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--out', type=str, default='class_frequencies.npy')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset = SynthSSCDataset(cfg, split='train', mode='default')
    freqs = compute_class_frequencies(dataset, num_classes=cfg['model']['num_classes'])

    np.save(args.out, freqs)
    print(f"Saved class frequencies to {args.out}")
    print("Frequencies:", freqs)