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
from torch.utils.data import DataLoader
from models.lmscnet import LMSCNetModel
from data.loader import SynthSSCDataset
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import numpy as np
import multiprocessing as mp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def compute_metrics(preds, targets, num_classes):
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    ious = []
    precisions = []
    recalls = []

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

        ious.append(iou)
        precisions.append(prec)
        recalls.append(rec)

    miou = np.mean(ious)
    return miou, ious, precisions, recalls

def main():
    mp.set_start_method('spawn', force=True)

    args = parse_args()
    cfg = load_config(args.config)

    # Datasets and loaders
    train_set = SynthSSCDataset(cfg, split='train')
    val_set = SynthSSCDataset(cfg, split='val')

    train_loader = DataLoader(train_set, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # Model and optimizer
    model = LMSCNetModel(cfg).cuda()
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
            optimizer.zero_grad()
            scores = model(batch)
            loss_dict = model.compute_loss(scores, batch)
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()

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

        miou, ious, precisions, recalls = compute_metrics(all_preds, all_targets, num_classes=cfg['model']['num_classes'])
        writer.add_scalar("val/mIoU", miou, epoch)
        for i, (iou, prec, rec) in enumerate(zip(ious, precisions, recalls)):
            writer.add_scalar(f"val/class_{i}/IoU", iou, epoch)
            writer.add_scalar(f"val/class_{i}/Precision", prec, epoch)
            writer.add_scalar(f"val/class_{i}/Recall", rec, epoch)

        # Save checkpoint
        # Save best model if mIoU improves
        if miou > best_miou:
            best_miou = miou
            best_ckpt_path = os.path.join(cfg['train']['checkpoint_dir'], 'best_model.pt')
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"[VAL] Saved new best model with mIoU {miou:.4f}")

        ckpt_path = os.path.join(cfg['train']['checkpoint_dir'], f'model_epoch_{epoch}.pt')
        torch.save(model.state_dict(), ckpt_path)

if __name__ == '__main__':
    main()