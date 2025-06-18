#! /usr/bin/env python3

# ==============================================================================
# Copyright© The University of Texas at Austin, 2025. All rights reserved.
# Author: Alex Diaz (j.a.diaz@utexas.edu)
# 
# About
# --------------
# Single-scale LMSCNet implementation
# ==============================================================================

import torch
import torch.nn as nn
import torch.functional as F
from models.base_model import BaseSSCModel

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dilations=(1, 2, 3)):
        super().__init__()
        self.init_conv = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)

        self.aspp_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm3d(mid_channels),
            )
            for d in dilations
        ])

        self.relu = nn.ReLU(inplace=True)
        self.output_conv = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = x[:, None, :, :, :]  # expand 3D grid: [B, D, H, W] → [B, 1, D, H, W]
        x = self.relu(self.init_conv(x))

        aspp_out = sum(block(x) for block in self.aspp_blocks)
        x = self.relu(aspp_out + x)

        return self.output_conv(x)
    
class LMSCNetModel(BaseSSCModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        input_dim = cfg['model']['input_dim'] # Voxel dimensions (W, H, D)
        f = input_dim[1] # Base dimension

        # 2D Encoder (on H x W slices of the voxel grid)
        self.encoder = nn.Sequential(
            nn.Conv2d(f, f, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(f, f, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(f, f * 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(f * 2, f * 2, kernel_size=3, padding=1), nn.ReLU()
        )

        # 3D Decoder head
        self.head = SegmentationHead(in_channels=1, mid_channels=8, out_channels=self.num_class)

    def forward(self, batch):
        # Input: {'3D_OCCUPANCY': [B, 1, W, H, D]}
        x = batch['3D_OCCUPANCY'].squeeze(1).permute(0, 2, 1, 3)  # → [B, H, W, D]
        feat = self.encoder(x)  # 2D encoder → [B, C, H', D']
        pred = self.head(feat)  # 3D segmentation head → [B, num_class, W, H, D]
        return {'pred': pred}
    
    def compute_loss(self, prediction, batch):
        pred_logits = prediction['pred']           # [B, C, W, H, D]
        target = batch['3D_LABEL'].long()          # [B, W, H, D]

        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(target.device) if self.class_weights is not None else None,
            ignore_index=255
        )
        loss = loss_fn(pred_logits, target)
        return {'total': loss}
    
    def get_target(self, batch):
        return batch['3D_LABEL'] 