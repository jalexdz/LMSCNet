import numpy as np


class Voxelizer():
    def __init__(self, volume_dims, voxel_size):
        self.volume_dims = volume_dims # (W, H, D) m
        self.voxel_size = voxel_size # m

    def voxelize(self, point_cloud):
        ''' 
        Convert point cloud to voxel grid format to pass to PyTorch model input
        Input: 
            point_cloud: np.ndarray of shape (N, 3)
                         -N: number of points
                         -Each row respresents (x, y, z) coordinates in m
        Output: 
            voxel_grid
        '''

        # Reorder [x, y, z] -> [y, z, x] to match voxel conventions
        point_cloud = point_cloud[:, [1, 2, 0]]
 
        # Initialize empty voxel grid of shape (W, H, D)
        voxel_grid = np.zeros(self.grid_dims, dtype=np.uint8)

        # Convert to voxel indices
        voxel_origin = np.asarray(self.voxel_origin)[[1, 2, 0]] # Reorient from [x, y, z] -> [y, z, x] to match voxel conventions
        pc_voxel = ((point_cloud - np.asarray(voxel_origin)) / self.voxel_size).astype(np.int32)

        # Filter inside grid bounds
        mask = (
            (pc_voxel[:, 0] >= 0) & (pc_voxel[:, 0] < self.grid_dims[0]) &
            (pc_voxel[:, 1] >= 0) & (pc_voxel[:, 1] < self.grid_dims[1]) &
            (pc_voxel[:, 2] >= 0) & (pc_voxel[:, 2] < self.grid_dims[2])
        )

        pc_voxel = pc_voxel[mask]

        # Mark occupied voxels
        voxel_grid[pc_voxel[:, 0], pc_voxel[:, 1], pc_voxel[:, 2]] = 1

        return voxel_grid
        
    def voxelize_with_labels(self):
        pass

