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
import numpy as np
from models.base_model import BaseSSCModel

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, weight=None, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logpt = nn.functional.cross_entropy(input, target, weight=self.weight,
                                            ignore_index=255,
                                            reduction='none')
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        return loss.mean()

class SegmentationHead(nn.Module):
    '''
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D -> class logits
    '''
    def __init__(self, in_channels, mid_channels, out_channels, dilations=(1, 2, 3)):
        super().__init__()
        # Initial convolution
        self.init_conv = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)

        # ASPP block
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
        x = x[:, None, :, :, :]  # Dim expand 3D grid: [B, D, H, W] → [B, 1, D, H, W]
        x = self.relu(self.init_conv(x))

        aspp_out = sum(block(x) for block in self.aspp_blocks)
        x = self.relu(aspp_out + x)

        return self.output_conv(x) # (B, C_out, D, H, W)
    
class UNetEncoder(nn.Module):
    def __init__(self, f):
        super().__init__()

        # First skip connection
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(f, f, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, f, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Second skip connection
        self.encoder_block2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(f, int(f * 1.5), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(f * 1.5), int(f * 1.5), kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Third skip connection
        self.encoder_block3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(int(f * 1.5), int(f * 2), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(f * 2), int(f * 2), kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Fourth skip connection
        self.encoder_block4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(int(f * 2), int(f * 2.5), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(f * 2.5), int(f * 2.5), kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.encoder_block1(x)  # H × W
        x2 = self.encoder_block2(x1)  # H/2 × W/2
        x3 = self.encoder_block3(x2)  # H/4 × W/4
        x4 = self.encoder_block4(x3)  # H/8 × W/8 # feed directly into decoder
        return x1, x2, x3, x4  # for skip connections
    
class UNetDecoder(nn.Module):
    def __init__(self, f, num_classes):
        super().__init__()

        # First block
        self.conv_out_1_8 = nn.Conv2d(int(f * 2.5), int(f / 8), kernel_size=3, padding=1)

        # Dilations blocks
        self.deconv_1_8_to_1_4 = nn.ConvTranspose2d(int(f / 8), int(f / 8), kernel_size=6, padding=2, stride=2)
        self.deconv_1_8_to_1_2 = nn.ConvTranspose2d(int(f / 8), int(f / 8), kernel_size=4, padding=0, stride=4)
        self.deconv_1_8_to_1_1 = nn.ConvTranspose2d(int(f / 8), int(f / 8), kernel_size=8, padding=0, stride=8)

        # Concat here

        # Second block (input is concatenated)
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(int(f * 2) + int(f / 8), int(f * 2), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_out_1_4 = nn.Conv2d(int(f * 2), int(f / 4), kernel_size=3, padding=1)

        # Dilations blocks
        self.deconv_1_4_to_1_2 = nn.ConvTranspose2d(int(f / 4), int(f / 4), kernel_size=6, padding=2, stride=2)
        self.deconv_1_4_to_1_1 = nn.ConvTranspose2d(int(f / 4), int(f / 4), kernel_size=4, padding=0, stride=4)

        # Concat here

        # Third block
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(int(f * 1.5) + int(f / 4) + int(f / 8), int(f * 1.5), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_out_1_2 = nn.Conv2d(int(f * 1.5), int(f / 2), kernel_size=3, padding=1)
        self.deconv_1_2_to_1_1 = nn.ConvTranspose2d(int(f / 2), int(f / 2), kernel_size=6, padding=2, stride=2)

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(f + int(f / 8) + int(f / 4) + int(f / 2), f, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.seg_head_1_1 = SegmentationHead(1, 8, num_classes, dilations=(1, 2, 3))
    
    def forward(self, x1_1, x1_2, x1_4, x1_8):
        # 1:8
        out_1_8 = self.conv_out_1_8(x1_8)

        # 1:4
        out_1_4 = self.deconv_1_8_to_1_4(out_1_8)
        out_1_4 = self.conv1_4(torch.cat([out_1_4, x1_4], dim=1))
        out_1_4_out = self.conv_out_1_4(out_1_4)

        # 1:2
        d8_1_2 = self.deconv_1_8_to_1_2(out_1_8)
        out_1_2 = self.deconv_1_4_to_1_2(out_1_4_out)
        out_1_2 = self.conv1_2(torch.cat([out_1_2, x1_2, d8_1_2], dim=1))
        out_1_2_out = self.conv_out_1_2(out_1_2)

        # 1:1
        d4_1_1 = self.deconv_1_4_to_1_1(out_1_4_out)
        d8_1_1 = self.deconv_1_8_to_1_1(out_1_8)
        out_1_1 = self.deconv_1_2_to_1_1(out_1_2_out)

        out_1_1 = self.conv1_1(torch.cat([
            out_1_1,
            x1_1,
            d4_1_1,
            d8_1_1
        ], dim=1))

        out_3d = self.seg_head_1_1(out_1_1)
        return out_3d
    
class LMSCNetModel(BaseSSCModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        vmin = np.array(cfg['data']['volume_size_min'])
        vmax = np.array(cfg['data']['volume_size_max'])
        voxel_size = cfg['data']['voxel_size']
 
        input_dim = np.round((vmax - vmin) / voxel_size).astype(int)  # (W, H, D)  

        num_classes = cfg['model']['num_classes']
        f = input_dim[1] 

        # 2D Encoder (on H x W slices of the voxel grid)
        self.encoder = UNetEncoder(f)
       
        # 3D Decoder 
        self.decoder = UNetDecoder(f, num_classes)

        freq_path = cfg['data']['class_frequencies']
        self.class_frequencies = np.load(freq_path, allow_pickle=False)

    def forward(self, batch):
        # Input: {'3D_OCCUPANCY': [B, 1, W, H, D]}
        x = batch['3D_OCCUPANCY']
        x = torch.squeeze(x, dim=1).permute(0, 2, 1, 3)  # → [B, H, W, D]
        x1, x2, x3, x4 = self.encoder(x)
        
        out = self.decoder(x1, x2, x3, x4)  # 3D segmentation head → [B, num_class, W, H, D]
        
        return {'pred': out.permute(0, 1, 3, 2, 4)}  # B, C, D, H, W → B, C, W, H, D
    
    def compute_loss(self, scores, data):
        target = data['3D_LABEL']
        pred = scores['pred']

        device, dtype = target.device, target.dtype

        class_weights = self.get_effective_class_weights().to(device=device, dtype=torch.float32)
        # criterion = nn.CrossEntropyLoss(
        #     weight=class_weights,
        #     ignore_index=255,
        #     reduction='mean'
        # ).to(device=device)

        criterion = FocalLoss(gamma=2.0)

        # Compute loss between predicted logits and labels
        loss_main = criterion(scores['pred'], target.long())
    
        return {'total': loss_main, 'semantic_1_1': loss_main}
   
    def get_class_weights(self):
        if not hasattr(self, 'class_frequencies'):
            raise ValueError("class_frequencies must be set in cfg to compute class weights.")
        epsilon = 1
        weights = torch.from_numpy(1 / np.log(self.class_frequencies + epsilon))
        weights = weights / weights.max()
        weights[0] = 0.1 * weights[1:].min()
        return weights
   
    def get_effective_class_weights(self, beta=0.9999):
        if not hasattr(self, 'class_frequencies'):
            raise ValueError("class_frequencies must be set")

        counts = self.class_frequencies.astype(np.float32)
        scaled_counts = counts / 1e6  # Avoid overflow from large raw values

        effective_num = 1.0 - np.power(beta, scaled_counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)  # stabilize div
        weights = weights / weights.max()  # normalize to [0, 1]

        # Rescale to [0.5, 2.0]
        min_val, max_val = 1.0, 5.0
        weights = min_val + (max_val - min_val) * (weights - weights.min()) / (weights.max() - weights.min())

        # Manually downscale class 0 to prevent collapse
        weights[0] = 0.05  # empirically safe — tune as needed

        return torch.tensor(weights, dtype=torch.float32)

    def weights_initializer(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def weights_init(self):
        self.apply(self.weights_initializer)

    def get_target(self, batch):
        return batch['3D_LABEL'] 