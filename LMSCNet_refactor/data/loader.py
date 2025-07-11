import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import yaml
import glob
import open3d as o3d
import matplotlib.pyplot as plt

from scripts.utils.voxelizer import Voxelizer
from scripts.utils.random_transform import RandomTransform

class SynthSSCDataset(Dataset):
    def __init__(self, cfg, split='train', mode='default', seed=42):
        self.root_dir = cfg['data']['data_dir']
        self.split_ratio = cfg['train']['split_ratio']
        self.voxelizer = Voxelizer(cfg['data']['volume_size_max'], cfg['data']['volume_size_min'], cfg['data']['voxel_size'])
        self.transform = RandomTransform() if cfg['data'].get('transform', False) else None
        self.split = split
        self.mode = mode
        with open(cfg['data']['label_map'], 'r') as f:
            label_config = yaml.safe_load(f)

        self.raw_to_remap = label_config['raw_to_remap']

        # Store data samples
        self.data_files = [] # (sparse_path, dense_path, label_path)
        
        for path in sorted(glob.glob(os.path.join(self.root_dir, 'sequences', '*'))):
            sparse_dir = os.path.join(path, 'sparse')
            dense_dir = os.path.join(path, 'dense')

            for sparse_path in sorted(glob.glob(os.path.join(sparse_dir, '*.pcd'))):
                fname = os.path.basename(sparse_path).replace('.pcd', '')
                dense_path = os.path.join(dense_dir, fname + '.pcd')
                label_path = os.path.join(dense_dir, fname + '.labels')

                if os.path.exists(dense_path) and os.path.exists(label_path):
                    self.data_files.append((sparse_path, dense_path, label_path))

        random.seed(seed)
        random.shuffle(self.data_files)
        N = len(self.data_files)
        n_train = int(self.split_ratio[0] * N)
        n_valid = int(self.split_ratio[1] * N)

        self.split_data = {
            'train': self.data_files[: n_train],
            'val': self.data_files[n_train : n_train + n_valid],
            'test': self.data_files[n_train + n_valid :]
        }

    def __getitem__(self, index):
        # Load sparse data
        sparse_path, dense_path, label_path = self.split_data[self.split][index]
        sparse_points = o3d.io.read_point_cloud(sparse_path)

        # Load dense data 
        dense_points = o3d.io.read_point_cloud(dense_path)
        with open(label_path, 'r') as f:
            raw_labels = np.array([line.lower().strip().replace(' ', '_') for line in f])

        # Remap
        labels = np.array([
            self.raw_to_remap.get(name.lower().strip().replace(' ', '_')) for name in raw_labels
        ], dtype=np.uint8).copy()

        sparse_pts = np.asarray(sparse_points.points)
        dense_pts = np.asarray(dense_points.points)

        if self.transform:
            R, t = self.transform.sample()
            sparse_pts = self.transform(sparse_pts, R=R, t=t)
            dense_pts, labels = self.transform(dense_pts, labels, R=R, t=t)

        # Voxelize
        sparse_vox = self.voxelizer.voxelize(np.asarray(sparse_pts))
        label_vox = self.voxelizer.voxelize_with_labels(np.asarray(dense_pts), labels)
        #self.voxelizer.visualize_alignment(sparse_pts, dense_pts, sparse_vox, label_vox)

        if self.mode == 'scan_labels':
            return label_vox.squeeze(0)
        
        return {
            '3D_OCCUPANCY': torch.from_numpy(sparse_vox).unsqueeze(0).float().to('cuda'), 
            '3D_LABEL': torch.from_numpy(label_vox).float().to('cuda')
        }
        
    def __len__(self):
        return len(self.split_data[self.split])

   
# class SSCDataset(Dataset):
#     def __init__(self, split: str, cfg: dict, seed=42):
#         self.data_dir = cfg['data']['data_dir']
#         self.split = split
#         self.seed = seed
#         self.split_ratio = cfg['train']['split_ratio']

#         # Get all sequence paths
#         self.sequence_paths = list(sorted(glob(os.path.join(self.data_dir, 'dataset', 'sequences', '*'))))

#         random.seed(seed)
#         random.shuffle(self.sequence_paths)

#         # Train/validate/test split
#         n_total = len(self.sequence_paths)
#         n_train = int(self.split_ratio[0] * n_total)
#         n_valid = int(self.split_ratio[1] * n_total)


#         split_map = {'train': self.sequence_paths[: n_train], 
#                       'val': self.sequence_paths[n_train : n_train + n_valid],
#                       'test': self.sequence_paths[n_train + n_valid : ] }
        
#         selected_sequences = split_map[split]

#         # Collect all scan file paths for the selected split
#         self.occ_files = []
#         self.label_files = []
#         for seq_path in selected_sequences: 
#             voxel_dir = os.path.join(seq_path, 'voxels')
#             bin_files = sorted(glob(os.path.join(voxel_dir, '*.bin')))
#             label_files = [f.replace('.bin', '.label') for f in bin_files]
#             self.occ_files.extend(bin_files)
#             self.label_files.extend(label_files)

#     def __len__(self):
#         return len(self.occ_files)
    
#     def _unpack(compressed):
#         ''' Given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
#         uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
#         uncompressed[::8] = compressed[:] >> 7 & 1
#         uncompressed[1::8] = compressed[:] >> 6 & 1
#         uncompressed[2::8] = compressed[:] >> 5 & 1
#         uncompressed[3::8] = compressed[:] >> 4 & 1
#         uncompressed[4::8] = compressed[:] >> 3 & 1
#         uncompressed[5::8] = compressed[:] >> 2 & 1
#         uncompressed[6::8] = compressed[:] >> 1 & 1
#         uncompressed[7::8] = compressed[:] & 1

#         return uncompressed

#     def __getitem__(self, idx):
#         occ_path = self._unpack(self.occ_files[idx])
#         label_path = self.label_files[idx]

#         data = np.fromfile(occ_path, dtype=np.uint8).reshape(1, *self.voxel_dims)
#         label = np.fromfile(label_path, dtype=np.uint16).reshape(*self.voxel_dims)
#         label = label.astype(np.int32)

#         return data, label
