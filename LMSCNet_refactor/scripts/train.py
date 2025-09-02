# #! /usr/bin/env python3

# # ==============================================================================
# # Training script for LMSCNet
# # Copyright© The University of Texas at Austin, 2025. All rights reserved.
# # Author: Alex Diaz (j.a.diaz@utexas.edu)
# # ==============================================================================
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import os
# import yaml
# import torch
# import argparse
# from torch.utils.data import DataLoader, Subset, DistributedSampler
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from models.lmscnet import LMSCNetModel
# from data.loader import SynthSSCDataset
# from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
# from sklearn.metrics import confusion_matrix
# import numpy as np
# import multiprocessing as mp
# from pytorch3d.loss import chamfer_distance

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
#     return parser.parse_args()

# def load_config(path):
#     with open(path, 'r') as f:
#         return yaml.safe_load(f)

# def convert_to_pointcloud(voxel_grid, voxel_size, voxel_origin):
#     """
#     Convert a voxel grid to a point cloud.

#     Args:
#         voxel_grid (np.ndarray): Binary or label voxel grid of shape [D, H, W] or [W, H, D].
#         voxel_size (float): Size of each voxel in meters.
#         voxel_origin (list or np.ndarray): 3D origin of the voxel grid in world coordinates.

#     Returns:
#         np.ndarray: Nx3 point cloud of occupied voxel centers.
#     """
#     # Binary mask of occupied voxels
#     occ_mask = voxel_grid != 0  # e.g., could use voxel_grid > threshold for soft preds
#     voxel_indices = np.argwhere(occ_mask)  # Shape: [N, 3]

#     # Convert voxel indices to world coordinates (center of voxel)
#     points = (voxel_indices + 0.5) * voxel_size + np.asarray(voxel_origin)

#     # Optionally reorder to [X, Y, Z] if voxel grid was in [Z, X, Y]
#     # Common in SSC pipelines: [W, H, D] => reorder to [X, Y, Z]
#     # This is only needed if your voxel indexing is nonstandard
#     points = points[:, [2, 0, 1]].astype(np.float32)

#     return points

# def compute_metrics(preds, targets, num_classes):
#     cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
#     ious = []
#     precisions = []
#     recalls = []
#     F1s = []

#     for i in range(num_classes):
#         TP = cm[i, i]
#         FP = cm[:, i].sum() - TP
#         FN = cm[i, :].sum() - TP
#         denom_iou = TP + FP + FN
#         denom_prec = TP + FP
#         denom_rec = TP + FN 

#         iou = TP / denom_iou if denom_iou > 0 else 0
#         prec = TP / denom_prec if denom_prec > 0 else 0
#         rec = TP / denom_rec if denom_rec > 0 else 0
#         f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0


#         ious.append(iou)
#         precisions.append(prec)
#         recalls.append(rec)
#         F1s.append(f1)

#     print(f"Per class IoUs: {ious}")
#     miou = np.mean(ious)
#     print(f"mIoU: {miou: .4f}")
#     return miou, ious, precisions, recalls, F1s

# def setup_distributed():
#     dist.init_process_group(backend='nccl', init_method='env://')
#     local_rank = int(os.environ['LOCAL_RANK'])
#     torch.cuda.set_device(local_rank)
#     return local_rank

# def main():
#     local_rank = setup_distributed()

#     mp.set_start_method('spawn', force=True)

#     args = parse_args()
#     cfg = load_config(args.config)

#     # Datasets and loaders
#     train_set = SynthSSCDataset(cfg, split='train')
#     val_set = SynthSSCDataset(cfg, split='val')
#     #train_subset = Subset(train_set, [12, 56, 199, 3000, 1000])
#     #val_subset = train_subset
#     print(f'Length: {len(train_set)}')
#     train_sampler = DistributedSampler(train_set, shuffle=True)
#     train_loader = DataLoader(train_set, batch_size=1, sampler=train_sampler, num_workers=0, pin_memory=True)
#     val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

#     # Model and optimizer

#     model = LMSCNetModel(cfg)
#     model.weights_init()
#     model = model.to(local_rank)
#     ddp_model = DDP(model, device_ids=[local_rank])
#     optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'])

#     # TensorBoard logging
#     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#     log_dir = os.path.join("runs", f"lmscnet_{timestamp}")
#     writer = SummaryWriter(log_dir)

#     os.makedirs(cfg['train']['checkpoint_dir'], exist_ok=True)
#     best_miou = 0.0
#     global_step = 0
#     for epoch in range(cfg['train']['epochs']):
#         ddp_model.train()
#         train_sampler.set_epoch(epoch)
#         for i, batch in enumerate(train_loader):
#             print(f'Epoch {epoch}, batch {i}')
#             optimizer.zero_grad()
#             batch = {k: v.to(local_rank) for k, v in batch.items()} 
#             scores = ddp_model(batch)
#             loss_dict = ddp_model.compute_loss(scores, batch)
#             loss = loss_dict['total']
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
#             optimizer.step()
#             target = batch['3D_LABEL']
#             valid_mask = target != 255
#             valid_labels = target[valid_mask]
#             valid_labels = valid_labels.to(torch.int64)

#             # Also fix these:
#             print("Target label distribution:", torch.bincount(batch['3D_LABEL'][batch['3D_LABEL'] != 255].to(torch.int64).flatten(), minlength=cfg['model']['num_classes']))
#             #print(f"Raw scores: {scores['pred']}")
#             pred_labels = torch.argmax(scores['pred'], dim=1)
#             print("Pred label distribution:", torch.bincount(pred_labels.to(torch.int64).flatten(), minlength=cfg['model']['num_classes']))
#             if global_step % cfg['train']['log_every'] == 0:
#                 print(f"[Epoch {epoch} | Step {i}] Loss: {loss.item():.4f}")
#                 writer.add_scalar("train/loss", loss.item(), global_step)
#                 writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)

#             global_step += 1

#         # Validation loop
#         ddp_model.eval()
#         val_loss_total = 0.0
#         all_preds, all_targets = [], []
#         with torch.no_grad():
#             for val_batch in val_loader:
#                 val_batch = {k: v.to(local_rank) for k, v in val_batch.items()}
#                 val_scores = ddp_model(val_batch)
#                 val_loss = ddp_model.compute_loss(val_scores, val_batch)
#                 val_loss_total += val_loss['total'].item()

#                 preds = torch.argmax(val_scores['pred'], dim=1)
#                 targets = val_batch['3D_LABEL']

#                 mask = (targets != 255)
#                 preds = preds[mask].cpu().numpy()
#                 targets = targets[mask].cpu().numpy()

#                 all_preds.append(preds)
#                 all_targets.append(targets)

#         avg_val_loss = val_loss_total / len(val_loader)
#         writer.add_scalar("val/loss", avg_val_loss, epoch)
#         print(f"[VAL] Epoch {epoch}: Avg Loss = {avg_val_loss:.4f}")

#         all_preds = np.concatenate(all_preds)
#         all_targets = np.concatenate(all_targets)

#         miou, ious, precisions, recalls, F1s = compute_metrics(all_preds, all_targets, num_classes=cfg['model']['num_classes'])
#         writer.add_scalar("val/mIoU", miou, epoch)
#         print(f"IoUs, {ious}")
#         for i, (iou, prec, rec, f1) in enumerate(zip(ious, precisions, recalls, F1s)):
#             writer.add_scalar(f"val/class_{i}/IoU", iou, epoch)
#             writer.add_scalar(f"val/class_{i}/Precision", prec, epoch)
#             writer.add_scalar(f"val/class_{i}/Recall", rec, epoch)
#             writer.add_scalar(f"val/class_{i}/F1", f1, epoch)

#         writer.add_scalar(f"val/mIoU", miou, epoch)
#         writer.add_scalar(f"val/mF1", np.mean(F1s), epoch)

#         # Compute chamfer
#         voxel_size = cfg['data']['voxel_size'] 
#         voxel_origin = cfg['data']['volume_size_min']

#         pred_points = torch.from_numpy(convert_to_pointcloud(preds, voxel_size, voxel_origin)).unsqueeze(0).to(local_rank)
#         gt_points = torch.from_numpy(convert_to_pointcloud(targets, voxel_size, voxel_origin)).unsqueeze(0).to(local_rank)

#         cd, _ = chamfer_distance(pred_points, gt_points)
#         writer.add_scalar(f"val/CD", cd, epoch)

#         # Save checkpoint
#         # Save best model if mIoU improves
#         if miou > best_miou:
#             best_miou = miou
#             best_ckpt_path = os.path.join(cfg['train']['checkpoint_dir'], 'best_model.pt')
#             torch.save({
#                 'model_state_dict': ddp_model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'epoch': epoch
#             }, best_ckpt_path)
#             print(f"[VAL] Saved new best model with mIoU {miou:.4f}")

#         ckpt_path = os.path.join(cfg['train']['checkpoint_dir'], f'model_epoch_{epoch}.pt')
#         torch.save({
#             'model_state_dict': ddp_model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'epoch': epoch
#         }, ckpt_path)

# if __name__ == '__main__':
#     main()

#! /usr/bin/env python3
# ==============================================================================
# Training script for LMSCNet (DDP-ready)
# Copyright© The University of Texas at Austin, 2025.
# Author: Alex Diaz (j.a.diaz@utexas.edu)
# ==============================================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
import argparse
import numpy as np
from datetime import datetime

from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models.lmscnet import LMSCNetModel
from data.loader import SynthSSCDataset
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.loss import chamfer_distance

# ----------------------------- Args & Config -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True, help="Path to config YAML")
    return p.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ----------------------------- DDP Helpers ------------------------------

def setup_distributed():
    """Assumes torchrun with env://."""
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

# ------------------------------ Metrics ---------------------------------

@torch.no_grad()
def fast_confusion(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = 255):
    """
    preds/targets: [N,...] integer tensors on same device.
    Returns a [C,C] confusion matrix on the same device (long).
    """
    mask = (targets != ignore_index) & (targets >= 0) & (targets < num_classes)
    if mask.numel() == 0 or mask.sum() == 0:
        return torch.zeros(num_classes, num_classes, device=preds.device, dtype=torch.long)
    t = targets[mask].view(-1)
    p = preds[mask].view(-1)
    idx = t * num_classes + p
    cm = torch.bincount(idx, minlength=num_classes * num_classes)
    return cm.view(num_classes, num_classes)

def cm_to_prf1_iou(cm: torch.Tensor, eps: float = 1e-8):
    """
    cm: [C, C] on CPU (preferred) or GPU.
    Returns dict with per-class and mean metrics.
    """
    TP = cm.diag().to(torch.float64)
    FP = cm.sum(0) - TP
    FN = cm.sum(1) - TP

    iou = TP / (TP + FP + FN + eps)
    prec = TP / (TP + FP + eps)
    rec  = TP / (TP + FN + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)

    return {
        "per_class_iou": iou,
        "per_class_prec": prec,
        "per_class_rec": rec,
        "per_class_f1": f1,
        "miou": iou.mean(),
        "mf1": f1.mean(),
    }

# ------------------------ Chamfer Utilities -----------------------------

@torch.no_grad()
def voxel_labels_to_points(vox_labels: torch.Tensor,
                           voxel_size: float,
                           voxel_origin: torch.Tensor,
                           empty_label: int = 0,
                           ignore_index: int = 255):
    """
    vox_labels: [B, W, H, D] (int). Returns list of B point clouds as float tensors [Ni, 3].
    World coords: center-of-voxel; assumes (W,H,D) index order maps to (X,Y,Z).
    """
    B, W, H, D = vox_labels.shape
    clouds = []
    device = vox_labels.device

    for b in range(B):
        lbl = vox_labels[b]
        occ = (lbl != empty_label) & (lbl != ignore_index)
        if occ.any():
            idx = occ.nonzero(as_tuple=False).to(torch.float32)  # [N,3] in (w,h,d)
            # convert to centers & world coords
            pts = (idx + 0.5) * voxel_size
            # reorder (w,h,d)->(x,y,z) is identity by assumption; if your pipeline uses different
            # convention, permute here accordingly, e.g.: pts = pts[:, [2,0,1]]
            pts = pts + voxel_origin  # broadcast (3,)
            clouds.append(pts)
        else:
            clouds.append(torch.empty(0, 3, device=device))
    return clouds

@torch.no_grad()
def mean_chamfer_for_batch(pred_labels_3d: torch.Tensor, gt_labels_3d: torch.Tensor,
                           voxel_size: float, voxel_origin: torch.Tensor):
    """
    Computes CD for each sample, returns (sum_cd, count_nonempty)
    """
    pred_clouds = voxel_labels_to_points(pred_labels_3d, voxel_size, voxel_origin)
    gt_clouds   = voxel_labels_to_points(gt_labels_3d,   voxel_size, voxel_origin)

    total_cd = pred_labels_3d.new_tensor(0.0, dtype=torch.float32)
    count = 0
    for P, G in zip(pred_clouds, gt_clouds):
        # Skip if either is empty; CD is ill-defined and PyTorch3D will error.
        if P.numel() == 0 or G.numel() == 0:
            continue
        # shape: [1, N, 3]
        P1 = P.unsqueeze(0).to(torch.float32)
        G1 = G.unsqueeze(0).to(torch.float32)
        cd, _ = chamfer_distance(P1, G1)  # scalar
        total_cd += cd.detach()
        count += 1
    return total_cd, count

# ------------------------------ Main ------------------------------------

def main():
    local_rank = setup_distributed()
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    cfg = load_config(args.config)

    # Datasets & samplers
    train_set = SynthSSCDataset(cfg, split='train')
    val_set   = SynthSSCDataset(cfg, split='val')

    train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=False)
    val_sampler   = DistributedSampler(val_set,   shuffle=False, drop_last=False)

    # DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=cfg['train'].get('batch_size', 1),
        sampler=train_sampler,
        num_workers=cfg['train'].get('num_workers', 4),
        pin_memory=True,
        persistent_workers=(cfg['train'].get('num_workers', 4) > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg['train'].get('val_batch_size', 1),
        sampler=val_sampler,
        num_workers=cfg['train'].get('val_num_workers', 2),
        pin_memory=True,
        persistent_workers=(cfg['train'].get('val_num_workers', 2) > 0),
    )

    # Model, optimizer, DDP
    device = torch.device(f'cuda:{local_rank}')
    model = LMSCNetModel(cfg)
    model.weights_init()
    model = model.to(device)

    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    optimizer = torch.optim.Adam(ddp_model.module.parameters(), lr=cfg['train']['learning_rate'])

    # TensorBoard (rank-0 only)
    if is_main_process():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("runs", f"lmscnet_{timestamp}")
        writer = SummaryWriter(log_dir)
        os.makedirs(cfg['train']['checkpoint_dir'], exist_ok=True)
    else:
        writer = None

    best_miou = 0.0
    global_step = 0
    num_classes = cfg['model']['num_classes']
    ignore_index = cfg['data'].get('ignore_index', 255)

    voxel_size   = float(cfg['data']['voxel_size'])
    voxel_origin = torch.tensor(cfg['data']['volume_size_min'], device=device, dtype=torch.float32)  # [3]

    epochs = cfg['train']['epochs']
    log_every = cfg['train'].get('log_every', 50)

    for epoch in range(epochs):
        # ------------------------- Train -------------------------
        ddp_model.train()
        train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            # Move batch to device
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Forward & loss
            scores = ddp_model(batch)
            # Prefer calling on .module explicitly
            loss_dict = ddp_model.module.compute_loss(scores, batch)
            loss = loss_dict['total']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.module.parameters(), max_norm=1.0)
            optimizer.step()

            if is_main_process() and (global_step % log_every == 0):
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)

            global_step += 1

        # ------------------------- Validate -------------------------
        ddp_model.eval()
        with torch.no_grad():
            # local accumulators (tensors on device to all_reduce)
            val_loss_sum_local = torch.zeros(1, device=device, dtype=torch.float32)
            val_batches_local  = torch.zeros(1, device=device, dtype=torch.float32)
            cm_local = torch.zeros(num_classes, num_classes, device=device, dtype=torch.long)

            cd_sum_local = torch.zeros(1, device=device, dtype=torch.float32)
            cd_count_local = torch.zeros(1, device=device, dtype=torch.float32)

            for val_batch in val_loader:
                val_batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in val_batch.items()}

                val_scores = ddp_model(val_batch)
                val_loss = ddp_model.module.compute_loss(val_scores, val_batch)['total']
                val_loss_sum_local += val_loss.detach()
                val_batches_local  += 1

                # Predictions & labels
                # scores['pred']: [B, C, W, H, D]
                pred_labels = torch.argmax(val_scores['pred'], dim=1).to(torch.int64)      # [B, W, H, D]
                targets     = val_batch['3D_LABEL'].to(torch.int64)                        # [B, W, H, D]

                # Confusion aggregation
                cm_local += fast_confusion(pred_labels.view(-1), targets.view(-1), num_classes, ignore_index)

                # Chamfer (mean over non-empty pairs)
                cd_sum_b, cd_cnt_b = mean_chamfer_for_batch(pred_labels, targets, voxel_size, voxel_origin)
                cd_sum_local   += cd_sum_b
                cd_count_local += torch.tensor(float(cd_cnt_b), device=device)

            # Reduce across ranks
            dist.all_reduce(val_loss_sum_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_batches_local,  op=dist.ReduceOp.SUM)
            dist.all_reduce(cm_local,            op=dist.ReduceOp.SUM)
            dist.all_reduce(cd_sum_local,        op=dist.ReduceOp.SUM)
            dist.all_reduce(cd_count_local,      op=dist.ReduceOp.SUM)

            avg_val_loss = (val_loss_sum_local / (val_batches_local + 1e-8)).item()

            # Compute metrics on rank-0
            if is_main_process():
                if writer:
                    writer.add_scalar("val/loss", avg_val_loss, epoch)

                cm_cpu = cm_local.cpu()
                metrics = cm_to_prf1_iou(cm_cpu)
                miou = metrics["miou"].item()
                mf1  = metrics["mf1"].item()

                # per-class logs
                if writer:
                    writer.add_scalar("val/mIoU", miou, epoch)
                    writer.add_scalar("val/mF1",  mf1,  epoch)
                    for c in range(num_classes):
                        writer.add_scalar(f"val/class_{c}/IoU",      metrics["per_class_iou"][c].item(),  epoch)
                        writer.add_scalar(f"val/class_{c}/Precision", metrics["per_class_prec"][c].item(), epoch)
                        writer.add_scalar(f"val/class_{c}/Recall",    metrics["per_class_rec"][c].item(),  epoch)
                        writer.add_scalar(f"val/class_{c}/F1",        metrics["per_class_f1"][c].item(),   epoch)

                # Chamfer
                cd_mean = (cd_sum_local / (cd_count_local + 1e-8)).item()
                if writer:
                    writer.add_scalar("val/Chamfer", cd_mean, epoch)

                # Checkpointing (best + per-epoch)
                ckpt_dir = cfg['train']['checkpoint_dir']
                # Save the wrapped state dict (works for single/DP load via strict=False if needed)
                to_save = {
                    'model_state_dict': ddp_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'miou': miou,
                    'val_loss': avg_val_loss,
                }

                # Best by mIoU
                if miou > best_miou:
                    best_miou = miou
                    best_ckpt_path = os.path.join(ckpt_dir, 'best_model.pt')
                    torch.save(to_save, best_ckpt_path)
                    print(f"[VAL][Epoch {epoch}] New best mIoU={miou:.4f}. Saved {best_ckpt_path}")

                ckpt_path = os.path.join(ckpt_dir, f'model_epoch_{epoch}.pt')
                torch.save(to_save, ckpt_path)
                print(f"[VAL][Epoch {epoch}] loss={avg_val_loss:.4f}, mIoU={miou:.4f}, Chamfer={cd_mean:.6f}")

    if is_main_process() and writer:
        writer.close()
    cleanup_distributed()

if __name__ == '__main__':
    main()
