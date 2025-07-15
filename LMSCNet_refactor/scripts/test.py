import os
import torch
import open3d as o3d
from tqdm import tqdm
from scripts.utils.color_map import apply_color_map  # write this to map label -> RGB

def test_and_save(model, test_loader, save_dir, device='cuda'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            occ = batch["3D_OCCUPANCY"].to(device)  # [B,1,D,H,W]
            pred = model(occ).argmax(dim=1).squeeze(0).cpu().numpy()  # [D,H,W]
            label = batch["3D_LABEL"].squeeze().cpu().numpy()         # [D,H,W]

            # Optional: Apply color maps
            pred_rgb = apply_color_map(pred)
            gt_rgb = apply_color_map(label)

            # Convert to point cloud (non-zero voxels only)
            pred_pcd = voxelgrid_to_colored_pointcloud(pred, pred_rgb)
            gt_pcd = voxelgrid_to_colored_pointcloud(label, gt_rgb)

            # Save
            o3d.io.write_point_cloud(os.path.join(save_dir, f"pred_{i}.ply"), pred_pcd)
            o3d.io.write_point_cloud(os.path.join(save_dir, f"gt_{i}.ply"), gt_pcd)

def show_comparison(pred_pcd, gt_pcd):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    vis.add_geometry(pred_pcd)

    def toggle(vis):
        vis.clear_geometries()
        vis.add_geometry(gt_pcd if toggle.state else pred_pcd)
        toggle.state = not toggle.state

    toggle.state = False
    vis.register_key_callback(ord("T"), toggle)
    vis.run()
    vis.destroy_window()

def voxelgrid_to_colored_pointcloud(label_grid, color_grid):
    coords = (label_grid > 0).nonzero()
    colors = color_grid[coords[:, 0], coords[:, 1], coords[:, 2]] / 255.0
    points = coords.astype(float)  # optional: scale by voxel size, shift origin
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, [2, 0, 1]])  # back to x,y,z
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

