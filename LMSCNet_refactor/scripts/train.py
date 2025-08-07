#! /usr/bin/env python3

# ==============================================================================
# Training script for LMSCNet
# Copyright© The University of Texas at Austin, 2025. All rights reserved.
# Author: Alex Diaz (j.a.diaz@utexas.edu)
# ==============================================================================
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import yaml
import torch
import argparse
from torch.utils.data import DataLoader, Subset
from models.lmscnet import LMSCNetModel
from data.loader import SynthSSCDataset
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import numpy as np
import multiprocessing as mp
from pytorch3d.loss import chamfer_distance

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def convert_to_pointcloud(voxel_grid, voxel_size, voxel_origin):
    """
    Convert a voxel grid to a point cloud.

    Args:
        voxel_grid (np.ndarray): Binary or label voxel grid of shape [D, H, W] or [W, H, D].
        voxel_size (float): Size of each voxel in meters.
        voxel_origin (list or np.ndarray): 3D origin of the voxel grid in world coordinates.

    Returns:
        np.ndarray: Nx3 point cloud of occupied voxel centers.
    """
    # Binary mask of occupied voxels
    occ_mask = voxel_grid != 0  # e.g., could use voxel_grid > threshold for soft preds
    voxel_indices = np.argwhere(occ_mask)  # Shape: [N, 3]

    # Convert voxel indices to world coordinates (center of voxel)
    points = (voxel_indices + 0.5) * voxel_size + np.asarray(voxel_origin)

    # Optionally reorder to [X, Y, Z] if voxel grid was in [Z, X, Y]
    # Common in SSC pipelines: [W, H, D] => reorder to [X, Y, Z]
    # This is only needed if your voxel indexing is nonstandard
    points = points[:, [2, 0, 1]].astype(np.float32)

    return points



def compute_metrics(preds, targets, num_classes):
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    ious = []
    precisions = []
    recalls = []
    F1s = []

    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        denom_iou = TP + FP + FN
        denom_prec = TP + FP
        denom_rec = TP + FN

        iou = TP / denom_iou if denom_iou > 0 else 0
        prec = TP / denom_prec if denom_prec > 0 else 0
        rec = TP / denom_rec if denom_rec > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0


        ious.append(iou)
        precisions.append(prec)
        recalls.append(rec)
        F1s.append(f1)

    print(f"Per class IoUs: {ious}")
    miou = np.mean(ious)
    print(f"mIoU: {miou: .4f}")
    return miou, ious, precisions, recalls, F1s

def main():
    mp.set_start_method('spawn', force=True)

    args = parse_args()
    cfg = load_config(args.config)

    # Datasets and loaders
    train_set = SynthSSCDataset(cfg, split='train')
    val_set = SynthSSCDataset(cfg, split='val')
    train_subset = Subset(train_set, [12, 56, 199, 3000, 1000])
    val_subset = train_subset
    print(f'Length: {len(train_set)}')
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    # Model and optimizer

    model = LMSCNetModel(cfg).cuda()
    model.weights_init()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'])

    # TensorBoard logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", f"lmscnet_{timestamp}")
    writer = SummaryWriter(log_dir)

    os.makedirs(cfg['train']['checkpoint_dir'], exist_ok=True)
    best_miou = 0.0
    global_step = 0
    for epoch in range(cfg['train']['epochs']):
        model.train()
        for i, batch in enumerate(train_loader):
            print(f'Epoch {epoch}, batch {i}')
            optimizer.zero_grad()
            batch = {k: v.cuda() for k, v in batch.items()} 
            scores = model(batch)
            loss_dict = model.compute_loss(scores, batch)
            loss = loss_dict['total']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            target = batch['3D_LABEL']
            valid_mask = target != 255
            valid_labels = target[valid_mask]
            valid_labels = valid_labels.to(torch.int64)

            # Also fix these:
            print("Target label distribution:", torch.bincount(batch['3D_LABEL'][batch['3D_LABEL'] != 255].to(torch.int64).flatten(), minlength=cfg['model']['num_classes']))
            #print(f"Raw scores: {scores['pred']}")
            pred_labels = torch.argmax(scores['pred'], dim=1)
            print("Pred label distribution:", torch.bincount(pred_labels.to(torch.int64).flatten(), minlength=cfg['model']['num_classes']))
            if global_step % cfg['train']['log_every'] == 0:
                print(f"[Epoch {epoch} | Step {i}] Loss: {loss.item():.4f}")
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)

            global_step += 1

        # Validation loop
        model.eval()
        val_loss_total = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = {k: v.cuda() for k, v in val_batch.items()}
                val_scores = model(val_batch)
                val_loss = model.compute_loss(val_scores, val_batch)
                val_loss_total += val_loss['total'].item()

                preds = torch.argmax(val_scores['pred'], dim=1)
                targets = val_batch['3D_LABEL']

                mask = (targets != 255)
                preds = preds[mask].cpu().numpy()
                targets = targets[mask].cpu().numpy()

                all_preds.append(preds)
                all_targets.append(targets)

        avg_val_loss = val_loss_total / len(val_loader)
        writer.add_scalar("val/loss", avg_val_loss, epoch)
        print(f"[VAL] Epoch {epoch}: Avg Loss = {avg_val_loss:.4f}")

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        miou, ious, precisions, recalls, F1s = compute_metrics(all_preds, all_targets, num_classes=cfg['model']['num_classes'])
        writer.add_scalar("val/mIoU", miou, epoch)
        print(f"IoUs, {ious}")
        for i, (iou, prec, rec, f1) in enumerate(zip(ious, precisions, recalls, F1s)):
            writer.add_scalar(f"val/class_{i}/IoU", iou, epoch)
            writer.add_scalar(f"val/class_{i}/Precision", prec, epoch)
            writer.add_scalar(f"val/class_{i}/Recall", rec, epoch)
            writer.add_scalar(f"val/class_{i}/F1", f1, epoch)

        writer.add_scalar(f"val/mIoU", miou, epoch)
        writer.add_scalar(f"val/mF1", np.mean(F1s), epoch)

        # Compute chamfer
        voxel_size = cfg['data']['voxel_size'] 
        voxel_origin = cfg['data']['volume_size_min']

        pred_points = torch.from_numpy(convert_to_pointcloud(preds, voxel_size, voxel_origin)).unsqueeze(0).cuda()
        gt_points = torch.from_numpy(convert_to_pointcloud(targets, voxel_size, voxel_origin)).unsqueeze(0).cuda()

        cd, _ = chamfer_distance(pred_points, gt_points)
        writer.add_scalar(f"val/CD", cd, epoch)

        # Save checkpoint
        # Save best model if mIoU improves
        if miou > best_miou:
            best_miou = miou
            best_ckpt_path = os.path.join(cfg['train']['checkpoint_dir'], 'best_model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, best_ckpt_path)
            print(f"[VAL] Saved new best model with mIoU {miou:.4f}")

        ckpt_path = os.path.join(cfg['train']['checkpoint_dir'], f'model_epoch_{epoch}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, ckpt_path)

if __name__ == '__main__':
    main()