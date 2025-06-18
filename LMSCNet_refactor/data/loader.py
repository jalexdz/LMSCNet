import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import yaml
import glob

from scripts.utils.voxelizer import Voxelizer

class SSCDataset(Dataset):
    def __init__(self, split: str, cfg: dict, seed=42):
        self.data_dir = cfg['data']['data_dir']
        self.split = split
        self.seed = seed
        self.split_ratio = cfg['train']['split_ratio']

        # Get all sequence paths
        self.sequence_paths = list(sorted(glob(os.path.join(self.data_dir, 'dataset', 'sequences', '*'))))

        random.seed(seed)
        random.shuffle(self.sequence_paths)

        # Train/validate/test split
        n_total = len(self.sequence_paths)
        n_train = int(self.split_ratio[0] * n_total)
        n_valid = int(self.split_ratio[1] * n_total)


        split_map = {'train': self.sequence_paths[: n_train], 
                      'val': self.sequence_paths[n_train : n_train + n_valid],
                      'test': self.sequence_paths[n_train + n_valid : ] }
        
        selected_sequences = split_map[split]

        # Collect all scan file paths for the selected split
        self.occ_files = []
        self.label_files = []
        for seq_path in selected_sequences: 
            voxel_dir = os.path.join(seq_path, 'voxels')
            bin_files = sorted(glob(os.path.join(voxel_dir, '*.bin')))
            label_files = [f.replace('.bin', '.label') for f in bin_files]
            self.occ_files.extend(bin_files)
            self.label_files.extend(label_files)

    def __len__(self):
        return len(self.occ_files)
    
    def _unpack(compressed):
        ''' Given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
        uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
        uncompressed[::8] = compressed[:] >> 7 & 1
        uncompressed[1::8] = compressed[:] >> 6 & 1
        uncompressed[2::8] = compressed[:] >> 5 & 1
        uncompressed[3::8] = compressed[:] >> 4 & 1
        uncompressed[4::8] = compressed[:] >> 3 & 1
        uncompressed[5::8] = compressed[:] >> 2 & 1
        uncompressed[6::8] = compressed[:] >> 1 & 1
        uncompressed[7::8] = compressed[:] & 1

        return uncompressed

    def __getitem__(self, idx):
        occ_path = self._unpack(self.occ_files[idx])
        label_path = self.label_files[idx]

        data = np.fromfile(occ_path, dtype=np.uint8).reshape(1, *self.voxel_dims)
        label = np.fromfile(label_path, dtype=np.uint16).reshape(*self.voxel_dims)
        label = label.astype(np.int32)

        return data, label
