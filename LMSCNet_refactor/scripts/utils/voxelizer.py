import numpy as np
from collections import defaultdict, Counter

class Voxelizer():
    def __init__(self, volume_size_max, volume_size_min, voxel_size):
        self.voxel_size = voxel_size # m
        self.voxel_origin = volume_size_min.copy()

        self.volume_dims = np.asarray(volume_size_max) - np.asarray(volume_size_min)

        self.grid_dims = tuple(np.round(np.asarray(self.volume_dims) / self.voxel_size).astype(np.int32))

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
        voxel_origin = np.asarray(self.voxel_origin) 
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
        
    def voxelize_with_labels(self, point_cloud, labels):
        ''' 
        Convert point cloud to voxel grid format to pass to PyTorch model input
        Input: 
            point_cloud: np.ndarray of shape (N, 3)
            labels: np.ndarray of shape (N,)
        Output: 
            voxel_grid: np.ndarray of shape (Y, Z, X)
        '''
        # Reorder [x, y, z] -> [y, z, x] to match voxel conventions
        point_cloud = point_cloud[:, [1, 2, 0]]
    
        # Initialize voxel grid in (Y, Z, X)
        voxel_grid = np.zeros(self.grid_dims, dtype=np.uint8)
        
        # Compute voxel indices
        voxel_origin = np.asarray(self.voxel_origin)  # already in [y, z, x]
        pc_voxel = np.floor((point_cloud - voxel_origin + 1e-4) / self.voxel_size).astype(int)

        # Filter inside bounds
        mask = (
            (pc_voxel[:, 0] >= 0) & (pc_voxel[:, 0] < self.grid_dims[0]) &
            (pc_voxel[:, 1] >= 0) & (pc_voxel[:, 1] < self.grid_dims[1]) &
            (pc_voxel[:, 2] >= 0) & (pc_voxel[:, 2] < self.grid_dims[2])
        )
        pc_voxel = pc_voxel[mask]
        filtered_labels = labels[mask]

        # Accumulate per-voxel labels
        voxel_label_map = defaultdict(list)
        for idx, label in zip(pc_voxel, filtered_labels):
            key = tuple(idx)
            voxel_label_map[key].append(label)

        # Assign majority label per voxel (including 0)
        for idx, label_list in voxel_label_map.items():
            label_counter = Counter(label_list)
            majority_label = label_counter.most_common(1)[0][0]
            voxel_grid[idx] = majority_label 

        return voxel_grid


    def visualize_voxelgrid_with_pointcloud(self, point_cloud_xyz, voxel_grid, voxel_origin, voxel_size):
        import open3d as o3d

        # 1. Original point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_xyz)
        pcd.paint_uniform_color([0, 1, 0])  # Green

        # 2. Get occupied voxel indices
        occupied_indices = np.argwhere(voxel_grid > 0)
        print(occupied_indices)

        if occupied_indices.shape[0] == 0:
            print("No occupied voxels to visualize.")
            return

        # 3. Convert voxel indices to voxel center coordinates
        voxel_centers_yzx = (occupied_indices + 0.5) * voxel_size + voxel_origin
        voxel_centers_xyz = voxel_centers_yzx[:, [2, 0, 1]]  # reorder to [x, y, z]

        # 4. Create a point cloud of voxel centers
        voxel_pcd = o3d.geometry.PointCloud()
        voxel_pcd.points = o3d.utility.Vector3dVector(voxel_centers_xyz)

        # 5. Generate voxel grid from the voxel center cloud
        voxel_grid_o3d = o3d.geometry.VoxelGrid.create_from_point_cloud(
            voxel_pcd,
            voxel_size=voxel_size
        )

        # 6. Visualize
        o3d.visualization.draw_geometries([pcd, voxel_grid_o3d])

    def visualize_alignment(self, sparse_pts, dense_pts, sparse_vox, label_vox):
            import open3d as o3d
            from matplotlib import cm

            # Convert to Open3D point clouds
            sparse_pcd = o3d.geometry.PointCloud()
            sparse_pcd.points = o3d.utility.Vector3dVector(sparse_pts)
            sparse_pcd.paint_uniform_color([0, 1, 0])  # green

            dense_pcd = o3d.geometry.PointCloud()
            dense_pcd.points = o3d.utility.Vector3dVector(dense_pts)
            dense_pcd.paint_uniform_color([0, 0, 0])  # white

            # Get occupied voxel indices for both
            sparse_indices = np.argwhere(sparse_vox > 0)
            dense_indices = np.argwhere(label_vox > 0)

            # Convert to coordinates in [x, y, z]
            origin = np.asarray(self.voxel_origin)  # in [x, y, z]
            sparse_centers = (sparse_indices + 0.5) * self.voxel_size + origin
            dense_centers = (dense_indices + 0.5) * self.voxel_size + origin

            sparse_centers = sparse_centers[:, [2, 0, 1]]  # yzx → xyz
            dense_centers = dense_centers[:, [2, 0, 1]]

            # Create Open3D point clouds from voxel centers
            sparse_voxels = o3d.geometry.PointCloud()
            sparse_voxels.points = o3d.utility.Vector3dVector(sparse_centers)
            sparse_voxels.paint_uniform_color([1, 0, 0])  # red

            dense_voxels = o3d.geometry.PointCloud()
            dense_voxels.points = o3d.utility.Vector3dVector(dense_centers)
            dense_voxels.paint_uniform_color([0, 0, 1])  # blue

            print(f"sparse voxel shape: {sparse_voxels}")
            print(f"dense voxel shape: {dense_voxels}")
            
            # Visualize all together
            o3d.visualization.draw_geometries([sparse_voxels])
            o3d.visualization.draw_geometries([dense_voxels])
            o3d.visualization.draw_geometries([sparse_pcd])
            o3d.visualization.draw_geometries([dense_pcd])

    def visualize_voxel_labels(self, voxel_grid):
        import open3d as o3d
        from matplotlib import cm

        occupied_indices = np.argwhere(voxel_grid > 0)
        voxel_labels = voxel_grid[voxel_grid >0]

        voxel_centers = (occupied_indices + 0.5) * self.voxel_size + np.asarray(self.voxel_origin)
        voxel_centers = voxel_centers[:, [2, 0, 1]]  # YZX -> XYZ

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(voxel_centers)

        max_label = voxel_labels.max()
        colors = cm.tab20(voxel_labels / (max_label if max_label > 0 else 1))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])
