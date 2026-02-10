"""📐 Point cloud utilities for object scale computation."""
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycocotools.mask as mask_util
import trimesh
from sklearn.cluster import DBSCAN

from utils.camera import CameraIntrinsics, dynamic_intrinsics


def decode_mask_rle(mask_rle: dict) -> np.ndarray:
    """🎭 Decode COCO RLE mask to binary array.

    Args:
        mask_rle: COCO RLE dict {"size": [h, w], "counts": str}.

    Returns:
        Binary mask (H, W) uint8.
    """
    rle = mask_rle.copy()
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return mask_util.decode(rle)


def depth_to_pointcloud(
    depth_path: Path,
    mask: np.ndarray,
    camera: CameraIntrinsics,
    depth_scale: float = 0.001,
    max_depth_m: float = 3.0,
) -> np.ndarray:
    """🌊 Unproject masked depth to 3D point cloud via pinhole model.

    Args:
        depth_path: Path to depth.png (uint16, mm).
        mask: Binary object mask (H, W).
        camera: Camera intrinsic parameters.
        depth_scale: Depth unit conversion factor (default 0.001 = mm→meters).
        max_depth_m: Maximum valid depth in meters.

    Returns:
        Point cloud (N, 3) float32 in meters.
    """
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    H, W = depth_raw.shape[:2]

    # 🔄 Resize mask if resolution mismatch
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    depth_m = depth_raw.astype(np.float32) * depth_scale

    # 📐 Reuse dynamic_intrinsics() to scale camera to depth resolution
    cam = dynamic_intrinsics(camera, W, H)

    # 🎯 Valid mask: in-range depth AND object mask
    valid = (depth_m > 0) & (depth_m < max_depth_m) & np.isfinite(depth_m) & (mask > 0)

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_m[valid]
    x = (u[valid] - cam.ppx) * z / cam.fx
    y = (v[valid] - cam.ppy) * z / cam.fy

    return np.column_stack([x, y, z]).astype(np.float32)


def denoise_pointcloud(
    points: np.ndarray,
    dbscan_eps: float = 0.005,
    dbscan_min_samples: int = 10,
    stat_nb_neighbors: int = 20,
    stat_std_ratio: float = 2.0,
) -> np.ndarray:
    """🧹 Denoise point cloud: DBSCAN clustering + statistical outlier removal.

    Args:
        points: Input point cloud (N, 3).
        dbscan_eps: DBSCAN neighborhood radius.
        dbscan_min_samples: DBSCAN minimum cluster size.
        stat_nb_neighbors: Statistical outlier removal neighbor count.
        stat_std_ratio: Statistical outlier removal std ratio.

    Returns:
        Cleaned point cloud (M, 3).
    """
    if points.shape[0] < dbscan_min_samples:
        return points

    # 🔬 DBSCAN: keep largest cluster + nearby clusters
    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(points)
    unique = set(labels) - {-1}
    if not unique:
        return points

    clusters = {l: points[labels == l] for l in unique}
    main_label = max(clusters, key=lambda l: len(clusters[l]))
    main_pts = clusters[main_label]

    pcd_main = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(main_pts))
    kdtree = o3d.geometry.KDTreeFlann(pcd_main)
    bbox_extent = np.max(pcd_main.get_max_bound() - pcd_main.get_min_bound())
    dist_thresh = bbox_extent * 0.5

    keep_labels = {main_label}
    for l, cpts in clusters.items():
        if l != main_label:
            centroid = cpts.mean(axis=0)
            _, _, dist_sq = kdtree.search_knn_vector_3d(centroid, 1)
            if np.sqrt(dist_sq[0]) <= dist_thresh:
                keep_labels.add(l)

    points = points[np.isin(labels, list(keep_labels))]

    # 📊 Statistical outlier removal
    if points.shape[0] == 0:
        return points
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    _, ind = pcd.remove_statistical_outlier(stat_nb_neighbors, stat_std_ratio)
    return points[ind]


def compute_obj_scale(
    points: np.ndarray, mesh_path: Path
) -> tuple[float, float, float]:
    """📏 Compute object scale factor: pcd_max_extent / mesh_max_extent.

    Uses Minimal Oriented Bounding Box (OBB) for both point cloud and mesh.

    Args:
        points: Cleaned point cloud (M, 3).
        mesh_path: Path to base.obj mesh file.

    Returns:
        (scale_factor, pcd_max_extent, mesh_max_extent).
    """
    # 📐 Point cloud OBB
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd_obb = pcd.get_minimal_oriented_bounding_box()
    pcd_max_extent = float(np.max(pcd_obb.extent))

    # 📐 Mesh OBB
    mesh = trimesh.load(str(mesh_path), process=False)
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
    )
    mesh_obb = o3d_mesh.get_minimal_oriented_bounding_box()
    mesh_max_extent = float(np.max(mesh_obb.extent))

    scale_factor = 1.0 if mesh_max_extent == 0 else pcd_max_extent / mesh_max_extent
    return scale_factor, pcd_max_extent, mesh_max_extent
