#!/usr/bin/env python3

# ==============================================================================
# Inference script for LMSCNet
# Copyright© The University of Texas at Austin, 2025.
# Author: Alex Diaz (j.a.diaz@utexas.edu)
# ==============================================================================

import os
import sys
import yaml
import argparse
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.lmscnet import LMSCNetModel
from data.loader import SynthSSCDataset

# Hardcoded color map (SemanticKITTI 0–19)
COLOR_MAP = np.array([
    [0, 0, 0],       # 0: empty
    [255, 0, 0],     # 1
    [0, 255, 0],     # 2
    [0, 0, 255],     # 3
    [255, 255, 0],   # 4
    [255, 0, 255],   # 5
    [0, 255, 255],   # 6
    [128, 128, 0],   # 7
    [128, 0, 128],   # 8
    [0, 128, 128],   # 9
    [200, 100, 100], # 10
    [100, 200, 100], # 11
    [100, 100, 200], # 12
    [50, 50, 50],    # 13
], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def apply_color_map(label_grid):
    label_grid = np.clip(label_grid, 0, COLOR_MAP.shape[0] - 1).astype(np.int32)
    return COLOR_MAP[label_grid]


def voxelgrid_to_colored_pointcloud(label_grid, color_grid):
    coords = np.stack(np.nonzero(label_grid > 0), axis=-1)
    colors = color_grid[coords[:, 0], coords[:, 1], coords[:, 2]] / 255.0
    points = coords.astype(float)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, [2, 0, 1]])  # D,H,W → x,y,z
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def test_and_save(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LMSCNetModel(cfg).to(device)
    model.load_state_dict(torch.load(cfg['eval']['checkpoint'], map_location=device)['model_state_dict'])
    model.eval()

    test_set = SynthSSCDataset(cfg, split='test')
    subset = Subset(test_set, [12, 56, 199, 3000, 1000])

    test_loader = DataLoader(subset, batch_size=1, shuffle=False)

    save_dir = cfg['eval']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            pred_logits = model(batch)['pred']  # [B, C, D, H, W]
            print(pred_logits)

            pred = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()  # [D,H,W]
            label = batch["3D_LABEL"].squeeze().cpu().numpy()                 # [D,H,W]
            print("Unique pred labels:", np.unique(pred))
            print("Unique gt labels:", np.unique(label))

            pred_rgb = apply_color_map(pred)
            gt_rgb = apply_color_map(label)

            pred_pcd = voxelgrid_to_colored_pointcloud(pred, pred_rgb)
            gt_pcd = voxelgrid_to_colored_pointcloud(label, gt_rgb)

            o3d.io.write_point_cloud(os.path.join(save_dir, f"pred_{i}.pcd"), pred_pcd)
            o3d.io.write_point_cloud(os.path.join(save_dir, f"gt_{i}.pcd"), gt_pcd)

def main():
    args = parse_args()
    cfg = load_config(args.config)
    test_and_save(cfg)

if __name__ == '__main__':
    main()
