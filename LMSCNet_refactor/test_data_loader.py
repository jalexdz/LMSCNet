from torch.utils.data import DataLoader
import torch
import os
import yaml
from data.loader import SynthSSCDataset

with open(os.getcwd() + '/configs/default.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
    
train_ds = SynthSSCDataset(cfg)

test_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

for idx, batch in enumerate(test_loader):
    print(f"Batch {idx + 1}:", batch['3D_OCCUPANCY'].max(), batch['3D_LABEL'].max())
    print(f"sparse shape: {batch['3D_OCCUPANCY'].shape}")
    print(f"dense shape: {batch['3D_LABEL'].shape}")