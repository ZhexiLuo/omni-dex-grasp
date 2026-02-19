"""🚀 Reconstruction pipeline entry point.

Usage: python -m recons.run
"""
import base64
import io
import json
from pathlib import Path

import cv2
import hydra
import numpy as np
import requests
import torch
import trimesh
from omegaconf import DictConfig
from PIL import Image

from recons.data import (
    DA3Result,
    FDPoseResult,
    GSAMResult,
    HaMeRResult,
    ScaleResult,
    TaskInput,
    TaskOutput,
    load_tasks,
)
from utils.camera import CameraIntrinsics, compute_focal, dynamic_intrinsics, intrinsics_to_3x3, pose_to_quat_json
from utils.pointcloud import (
    compute_obj_scale,
    decode_mask_rle,
    denoise_pointcloud,
    depth_to_pointcloud,
)


# ══════════════════════════════════════════════════════════════════════════════
# 🔧 Helpers
# ══════════════════════════════════════════════════════════════════════════════


def decode_image_b64(b64_str: str) -> np.ndarray:
    """🖼️ Decode base64 PNG/JPG to BGR image array."""
    img_bytes = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def decode_array_b64(b64_str: str) -> np.ndarray:
    """📐 Decode base64 npy string to numpy array."""
    arr_bytes = base64.b64decode(b64_str)
    return np.load(io.BytesIO(arr_bytes))


def encode_array_b64(arr: np.ndarray) -> str:
    """📐 Encode numpy array to base64 npy string."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_obj_mask_rle(gsam_result: GSAMResult) -> dict:
    """🎭 Extract highest-confidence non-hand object mask_rle from GSAM result."""
    obj_dets = [d for d in gsam_result.detections if not d.get("is_hand", False)]
    if not obj_dets:
        raise ValueError("❌ No object detection in GSAM scene result")
    best = max(obj_dets, key=lambda d: d["score"])
    return best["mask_rle"]


def extract_obj_bbox(gsam_result: GSAMResult) -> list[float]:
    """📦 Extract highest-confidence non-hand object bbox [x1,y1,x2,y2]."""
    obj_dets = [d for d in gsam_result.detections if not d.get("is_hand", False)]
    if not obj_dets:
        raise ValueError("❌ No object detection in GSAM result")
    best = max(obj_dets, key=lambda d: d["score"])
    return best["bbox"]


def load_depth_as_b64(depth_path: Path, depth_scale: float) -> str:
    """🌊 Load depth.png (uint16 mm) → base64 npy (float32 meters)."""
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"❌ Failed to load depth: {depth_path}")
    depth_m = depth_raw.astype(np.float32) * depth_scale
    return encode_array_b64(depth_m)


def scale_and_center_mesh(mesh_path: Path, scale_factor: float) -> "trimesh.Trimesh":
    """📏 Scale + center mesh, return in-memory trimesh object."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    mesh.vertices *= scale_factor
    mesh.vertices -= mesh.bounding_box.centroid
    return mesh


def encode_mesh_b64(mesh: "trimesh.Trimesh") -> str:
    """🧊 Encode trimesh to base64 OBJ string.

    Args:
        mesh: trimesh.Trimesh object with optional materials

    Returns:
        Base64-encoded OBJ binary string
    """
    # 📦 Export to OBJ format (returns str, not bytes)
    obj_str = mesh.export(file_type="obj")
    obj_bytes = obj_str.encode("utf-8")
    return base64.b64encode(obj_bytes).decode("utf-8")


def encode_mtl_file(task_dir: Path) -> str | None:
    """📄 Encode MTL file to base64 string.

    Args:
        task_dir: Task directory containing material.mtl

    Returns:
        Base64-encoded MTL string, or None if file doesn't exist
    """
    mtl_path = task_dir / "material.mtl"
    if not mtl_path.exists():
        return None

    # 📄 Read MTL file as text
    mtl_content = mtl_path.read_text(encoding='utf-8')
    return base64.b64encode(mtl_content.encode('utf-8')).decode('utf-8')


def encode_texture_file(task_dir: Path, texture_filename: str) -> str | None:
    """🖼️ Encode texture image to base64 string.

    Args:
        task_dir: Task directory containing texture file
        texture_filename: Texture filename (e.g., 'shaded.png')

    Returns:
        Base64-encoded texture bytes, or None if file doesn't exist
    """
    texture_path = task_dir / texture_filename
    if not texture_path.exists():
        return None

    # 🖼️ Read texture as binary
    texture_bytes = texture_path.read_bytes()
    return base64.b64encode(texture_bytes).decode('utf-8')


def extract_texture_filename_from_mtl(task_dir: Path) -> str | None:
    """🔍 Extract texture filename from MTL file.

    Args:
        task_dir: Task directory containing material.mtl

    Returns:
        Texture filename (e.g., 'shaded.png'), or None if not found
    """
    mtl_path = task_dir / "material.mtl"
    if not mtl_path.exists():
        return None

    # 📄 Parse MTL to find map_Kd line
    mtl_content = mtl_path.read_text(encoding='utf-8')
    for line in mtl_content.splitlines():
        line = line.strip()
        if line.startswith('map_Kd '):
            # Extract filename after 'map_Kd '
            texture_filename = line.split('map_Kd ', 1)[1].strip()
            return texture_filename

    return None


def encode_image_file_b64(image_path: Path) -> str:
    """🖼️ Encode image file to base64 (preserves original PNG/JPG format)."""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def rescale_da3_depth(
    scene_depth: np.ndarray,
    scene_mask_rle: dict,
    grasp_depth_b64: str,
    window_size: int = 20,
) -> tuple[str, float, float]:
    """🔄 Rescale DA3 relative depth to metric using scene depth as reference.

    Args:
        scene_depth: Real depth (H_s, W_s) float32 in meters
        scene_mask_rle: COCO RLE mask of object in scene image
        grasp_depth_b64: DA3 predicted depth (base64 npy) for grasp image
        grasp_mask_rle: COCO RLE mask of object in grasp image
        window_size: Window size for center region sampling

    Returns:
        (rescaled_depth_b64, alpha, beta) where rescaled_depth_b64 is base64 npy
    """
    from utils.depth import (
        compute_depth_affine,
        find_mask_center_coords,
        rescale_depth,
    )

    # 🎭 Decode masks
    scene_mask = decode_mask_rle(scene_mask_rle)

    # 🌊 Decode DA3 depth
    grasp_depth = decode_array_b64(grasp_depth_b64)

    # 🎯 Find center coords for both images
    scene_coords = find_mask_center_coords(scene_mask, window_size)

    # 📐 Compute affine from grasp (pred) to scene (gt) depth
    # Resize grasp depth to scene resolution for matching
    h_s, w_s = scene_depth.shape
    h_g, w_g = grasp_depth.shape

    if (h_g, w_g) != (h_s, w_s):
        grasp_depth_resized = cv2.resize(
            grasp_depth, (w_s, h_s), interpolation=cv2.INTER_CUBIC
        )
    else:
        grasp_depth_resized = grasp_depth

    # Use scene coords (both should be similar since same object)
    alpha, beta = compute_depth_affine(
        depth_pred=grasp_depth_resized,
        depth_gt=scene_depth,
        pixel_coords=scene_coords,
        mask_invalid=True,
    )

    # 🔄 Apply rescale to original grasp depth (not resized)
    rescaled = rescale_depth(grasp_depth, alpha, beta)

    # 📦 Encode to b64
    rescaled_b64 = encode_array_b64(rescaled)

    return rescaled_b64, alpha, beta


# ══════════════════════════════════════════════════════════════════════════════
# 🌐 Server Calls
# ══════════════════════════════════════════════════════════════════════════════


def call_gsam(
    url: str, image_b64: str, text_prompt: str, include_hand: bool, timeout: int
) -> GSAMResult:
    """🦖 Call GSAM2 server for detection and segmentation."""
    resp = requests.post(
        url,
        json={
            "image_b64": image_b64,
            "text_prompt": text_prompt,
            "include_hand": include_hand,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return GSAMResult(
        status=data["status"],
        message=data["message"],
        detections=data.get("detections", []),
        img_size=data.get("img_size", []),
        annotated_b64=data.get("annotated_image_b64", ""),
        mask_b64=data.get("mask_image_b64", ""),
    )


def call_hamer(
    url: str, image_b64: str, focal_length: float, timeout: int
) -> HaMeRResult:
    """🤚 Call HaMeR server for hand reconstruction."""
    resp = requests.post(
        url,
        json={
            "image_b64": image_b64,
            "focal_length": focal_length,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return HaMeRResult(
        status=data["status"],
        message=data["message"],
        mano_params=data.get("mano_params", {}),
        vertices_b64=data.get("vertices_b64", ""),
        cam_transl=data.get("cam_transl", []),
        is_right=data.get("is_right", False),
        mask_b64=data.get("mask_b64", ""),
    )


def call_da3(
    url: str, image_b64: str, intrinsics_3x3: list[list[float]], timeout: int
) -> DA3Result:
    """🌊 Call DA3 server for metric depth estimation."""
    resp = requests.post(
        url,
        json={
            "image_b64": image_b64,
            "intrinsics": intrinsics_3x3,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return DA3Result(
        status=data["status"],
        message=data["message"],
        depth_b64=data.get("depth_b64", ""),
        conf_b64=data.get("conf_b64", ""),
        depth_vis_b64=data.get("depth_vis_b64", ""),
    )


def call_fdpose(
    url: str,
    image_b64: str,
    depth_b64: str,
    mesh_b64: str,
    mtl_b64: str | None,
    texture_b64: str | None,
    texture_filename: str | None,
    bbox: list[float],
    intrinsics_3x3: list[list[float]],
    timeout: int,
) -> FDPoseResult:
    """📐 Call FDPose server for 6D pose estimation."""
    payload = {
        "image_b64": image_b64,
        "depth_b64": depth_b64,
        "obj_mesh_b64": mesh_b64,
        "bbox": bbox,
        "intrinsics": intrinsics_3x3,
        "obj_mtl_b64": mtl_b64,
        "obj_texture_b64": texture_b64,
        "texture_filename": texture_filename,
    }

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return FDPoseResult(
        status=data["status"],
        message=data["message"],
        pose=data.get("pose", []),
        pose_vis_b64=data.get("pose_vis_b64", ""),
        obj_mask_b64=data.get("obj_mask_b64", ""),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 🎯 Task Processing
# ══════════════════════════════════════════════════════════════════════════════


def process_task(task: TaskInput, cfg: DictConfig) -> TaskOutput:
    """🎯 Process a single task through the reconstruction pipeline.

    """
    output = TaskOutput(name=task.name)
    timeout = cfg.servers.timeout

    # 📷 Pre-encode images (encode once, reuse for all servers)
    scene_b64 = encode_image_file_b64(task.scene_image)
    grasp_b64 = encode_image_file_b64(task.generated_grasp)

    # 📐 Pre-compute intrinsics (grasp image resolution)
    grasp_img = Image.open(task.generated_grasp)
    grasp_cam = dynamic_intrinsics(task.camera, grasp_img.width, grasp_img.height)
    grasp_K_3x3 = intrinsics_to_3x3(grasp_cam)
    grasp_focal = compute_focal(grasp_cam)

    # 📐 Pre-compute intrinsics (scene image resolution)
    scene_img = Image.open(task.scene_image)
    scene_cam = dynamic_intrinsics(task.camera, scene_img.width, scene_img.height)
    scene_K_3x3 = intrinsics_to_3x3(scene_cam)

    # 1️⃣ GSAM: scene_image (no hand)
    print(f"  🎭 GSAM: scene_image (no hand)")
    output.gsam_scene = call_gsam(
        cfg.servers.gsam,
        scene_b64,
        task.obj_description,
        include_hand=False,
        timeout=timeout,
    )
    print(f"     └─ {output.gsam_scene.status}: {output.gsam_scene.message}")

    # 2️⃣ Scale: compute object real-world scale + save scaled mesh
    print(f"  📏 Scale: computing object real-world scale")
    obj_mask_rle = extract_obj_mask_rle(output.gsam_scene)
    obj_mask = decode_mask_rle(obj_mask_rle)

    pcd_raw = depth_to_pointcloud(
        task.depth, obj_mask, task.camera,
        depth_scale=cfg.scale.depth_scale,
        max_depth_m=cfg.scale.max_depth_m,
    )
    print(f"     └─ raw points: {pcd_raw.shape[0]}")

    pcd_clean = denoise_pointcloud(
        pcd_raw,
        dbscan_eps=cfg.scale.dbscan_eps,
        dbscan_min_samples=cfg.scale.dbscan_min_samples,
        stat_nb_neighbors=cfg.scale.stat_nb_neighbors,
        stat_std_ratio=cfg.scale.stat_std_ratio,
    )
    print(f"     └─ cleaned points: {pcd_clean.shape[0]}")

    scale_factor, pcd_ext, mesh_ext = compute_obj_scale(pcd_clean, task.obj_mesh)
    scaled_mesh = scale_and_center_mesh(task.obj_mesh, scale_factor)
    output.scale = ScaleResult(
        scale_factor=scale_factor,
        pcd_num_points=pcd_clean.shape[0],
        pcd_max_extent=pcd_ext,
        mesh_max_extent=mesh_ext,
        scaled_mesh=scaled_mesh,
    )
    print(f"     └─ scale_factor: {scale_factor:.6f}")
    print(f"     └─ pcd_extent: {pcd_ext:.4f}m, mesh_extent: {mesh_ext:.4f}")

    # 3️⃣ GSAM: generated_grasp (with hand)
    print(f"  🎭 GSAM: generated_grasp (with hand)")
    output.gsam_grasp = call_gsam(
        cfg.servers.gsam,
        grasp_b64,
        task.obj_description,
        include_hand=True,
        timeout=timeout,
    )
    print(f"     └─ {output.gsam_grasp.status}: {output.gsam_grasp.message}")

    # 4️⃣ HaMeR: hand reconstruction
    print(f"  🤚 HaMeR: hand reconstruction")
    print(f"     └─ focal_length: {grasp_focal:.2f}")

    output.hamer = call_hamer(
        cfg.servers.hamer, grasp_b64, grasp_focal, timeout=timeout,
    )
    print(f"     └─ {output.hamer.status}: {output.hamer.message}")

    # 5️⃣ DA3: metric depth from grasp image
    print(f"  🌊 DA3: metric depth estimation (grasp)")
    output.da3_grasp = call_da3(
        cfg.servers.da3, grasp_b64, grasp_K_3x3, timeout=timeout,
    )
    print(f"     └─ {output.da3_grasp.status}: {output.da3_grasp.message}")

    # 🔄 Rescale DA3 depth to metric using scene depth
    print(f"  🔄 Rescaling DA3 depth to metric scale")
    scene_depth = cv2.imread(str(task.depth), cv2.IMREAD_UNCHANGED).astype(np.float32) * cfg.scale.depth_scale
    scene_mask_rle = extract_obj_mask_rle(output.gsam_scene)

    rescaled_depth_b64, alpha, beta = rescale_da3_depth(
        scene_depth=scene_depth,
        scene_mask_rle=scene_mask_rle,
        grasp_depth_b64=output.da3_grasp.depth_b64,
        window_size=cfg.get("rescale_window_size", 20),
    )

    # Update DA3Result with rescaled depth
    output.da3_grasp.rescaled_depth_b64 = rescaled_depth_b64
    output.da3_grasp.alpha = alpha
    output.da3_grasp.beta = beta
    print(f"     └─ alpha={alpha:.4f}, beta={beta:.4f}")

    # 6️⃣ FDPose: encode mesh once, reuse for both calls
    mesh_b64 = encode_mesh_b64(output.scale.scaled_mesh)

    # 📄 Encode MTL if present
    mtl_b64 = encode_mtl_file(task.task_dir)

    # 🖼️ Encode texture if MTL exists
    texture_filename = None
    texture_b64 = None
    texture_filename = extract_texture_filename_from_mtl(task.task_dir)
    texture_b64 = encode_texture_file(task.task_dir, texture_filename)
    print(f"  🖼️ Texture: {texture_filename} encoded")

    # 6️⃣ FDPose: scene pose (real depth)
    print(f"  📐 FDPose: scene pose (real depth)")
    scene_depth_b64 = load_depth_as_b64(task.depth, cfg.scale.depth_scale)
    scene_bbox = extract_obj_bbox(output.gsam_scene)
    output.fdpose_scene = call_fdpose(
        cfg.servers.fdpose,
        scene_b64,
        scene_depth_b64,
        mesh_b64,
        mtl_b64,           # 📄 MTL
        texture_b64,       # 🖼️ Texture
        texture_filename,  # 📝 Filename
        scene_bbox,
        scene_K_3x3,
        timeout=timeout,
    )
    print(f"     └─ {output.fdpose_scene.status}: {output.fdpose_scene.message}")

    # 7️⃣ FDPose: grasp pose (RESCALED DA3 depth) 🆕
    print(f"  📐 FDPose: grasp pose (rescaled DA3 depth)")
    grasp_bbox = extract_obj_bbox(output.gsam_grasp)
    output.fdpose_grasp = call_fdpose(
        cfg.servers.fdpose,
        grasp_b64,
        output.da3_grasp.rescaled_depth_b64,  # 🆕 Use rescaled instead of raw
        mesh_b64,
        mtl_b64,           # 📄 MTL
        texture_b64,       # 🖼️ Texture
        texture_filename,  # 📝 Filename
        grasp_bbox,
        grasp_K_3x3,
        timeout=timeout,
    )
    print(f"     └─ {output.fdpose_grasp.status}: {output.fdpose_grasp.message}")

    output.grasp_cam = grasp_cam
    return output


# ══════════════════════════════════════════════════════════════════════════════
# 💾 Output Saving
# ══════════════════════════════════════════════════════════════════════════════


def save_output(
    result: TaskOutput, output_dir: Path, cfg: DictConfig
) -> None:
    """💾 Save task output to directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.out.vis:
        save_visualizations(result, output_dir)
    if cfg.out.inter_out:
        save_intermediate(result, output_dir)

    save_out(result, output_dir, result.grasp_cam)
    print(f"  💾 Saved to: {output_dir}")


def save_visualizations(result: TaskOutput, output_dir: Path) -> None:
    """📊 Save visualization images."""
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(exist_ok=True)

    # 🎭 GSAM scene
    if result.gsam_scene and result.gsam_scene.annotated_b64:
        img = decode_image_b64(result.gsam_scene.annotated_b64)
        cv2.imwrite(str(vis_dir / "gsam_scene_annotated.jpg"), img)
    if result.gsam_scene and result.gsam_scene.mask_b64:
        img = decode_image_b64(result.gsam_scene.mask_b64)
        cv2.imwrite(str(vis_dir / "gsam_scene_mask.png"), img)

    # 🎭 GSAM grasp
    if result.gsam_grasp and result.gsam_grasp.annotated_b64:
        img = decode_image_b64(result.gsam_grasp.annotated_b64)
        cv2.imwrite(str(vis_dir / "gsam_grasp_annotated.jpg"), img)
    if result.gsam_grasp and result.gsam_grasp.mask_b64:
        img = decode_image_b64(result.gsam_grasp.mask_b64)
        cv2.imwrite(str(vis_dir / "gsam_grasp_mask.png"), img)

    # 🤚 HaMeR mask
    if result.hamer and result.hamer.mask_b64:
        img = decode_image_b64(result.hamer.mask_b64)
        cv2.imwrite(str(vis_dir / "hamer_mask.png"), img)

    # 🌊 DA3 depth visualization
    if result.da3_grasp and result.da3_grasp.depth_vis_b64:
        img = decode_image_b64(result.da3_grasp.depth_vis_b64)
        cv2.imwrite(str(vis_dir / "da3_grasp_depth.png"), img)

        # 🆕 Visualize rescaled depth
        if result.da3_grasp.rescaled_depth_b64:
            rescaled = decode_array_b64(result.da3_grasp.rescaled_depth_b64)

            # Normalize to 0-255 for visualization
            rescaled_vis = rescaled.copy()
            rescaled_vis[rescaled_vis < 0] = 0
            max_depth = np.percentile(rescaled_vis[rescaled_vis > 0], 99) if np.any(rescaled_vis > 0) else 3.0
            rescaled_vis = np.clip(rescaled_vis / max_depth * 255, 0, 255).astype(np.uint8)
            rescaled_vis = cv2.applyColorMap(rescaled_vis, cv2.COLORMAP_TURBO)

            cv2.imwrite(str(vis_dir / "da3_grasp_depth_rescaled.png"), rescaled_vis)

    # 📐 FDPose visualizations
    if result.fdpose_scene and result.fdpose_scene.pose_vis_b64:
        img = decode_image_b64(result.fdpose_scene.pose_vis_b64)
        cv2.imwrite(str(vis_dir / "fdpose_scene_pose.jpg"), img)
    if result.fdpose_grasp and result.fdpose_grasp.pose_vis_b64:
        img = decode_image_b64(result.fdpose_grasp.pose_vis_b64)
        cv2.imwrite(str(vis_dir / "fdpose_grasp_pose.jpg"), img)


def save_intermediate(result: TaskOutput, output_dir: Path) -> None:
    """💾 Save intermediate data (detections, vertices, params)."""
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # 🎭 GSAM detections
    if result.gsam_scene:
        with open(data_dir / "gsam_scene.json", "w") as f:
            json.dump(
                {
                    "status": result.gsam_scene.status,
                    "message": result.gsam_scene.message,
                    "detections": result.gsam_scene.detections,
                    "img_size": result.gsam_scene.img_size,
                },
                f, indent=2,
            )
    if result.gsam_grasp:
        with open(data_dir / "gsam_grasp.json", "w") as f:
            json.dump(
                {
                    "status": result.gsam_grasp.status,
                    "message": result.gsam_grasp.message,
                    "detections": result.gsam_grasp.detections,
                    "img_size": result.gsam_grasp.img_size,
                },
                f, indent=2,
            )

    # 🤚 HaMeR data
    if result.hamer:
        with open(data_dir / "hamer.json", "w") as f:
            json.dump(
                {
                    "status": result.hamer.status,
                    "message": result.hamer.message,
                    "mano_params": result.hamer.mano_params,
                    "cam_transl": result.hamer.cam_transl,
                    "is_right": result.hamer.is_right,
                },
                f, indent=2,
            )
        if result.hamer.vertices_b64:
            vertices = decode_array_b64(result.hamer.vertices_b64)
            np.save(data_dir / "hamer_vertices.npy", vertices)

    # 📏 Scale data
    if result.scale:
        with open(data_dir / "scale.json", "w") as f:
            json.dump(
                {
                    "scale_factor": result.scale.scale_factor,
                    "pcd_num_points": result.scale.pcd_num_points,
                    "pcd_max_extent": result.scale.pcd_max_extent,
                    "mesh_max_extent": result.scale.mesh_max_extent,
                },
                f, indent=2,
            )

    # 🌊 DA3 depth
    if result.da3_grasp and result.da3_grasp.depth_b64:
        depth = decode_array_b64(result.da3_grasp.depth_b64)
        np.save(data_dir / "da3_grasp_depth.npy", depth)

        # 🆕 Save rescaled depth
        if result.da3_grasp.rescaled_depth_b64:
            rescaled = decode_array_b64(result.da3_grasp.rescaled_depth_b64)
            np.save(data_dir / "da3_grasp_depth_rescaled.npy", rescaled)

            # Save affine params
            with open(data_dir / "da3_rescale_params.json", "w") as f:
                json.dump({
                    "alpha": result.da3_grasp.alpha,
                    "beta": result.da3_grasp.beta,
                }, f, indent=2)

    # 📐 FDPose poses
    if result.fdpose_scene and result.fdpose_scene.pose:
        with open(data_dir / "fdpose_scene.json", "w") as f:
            json.dump({"pose_4x4": result.fdpose_scene.pose}, f, indent=2)
    if result.fdpose_grasp and result.fdpose_grasp.pose:
        with open(data_dir / "fdpose_grasp.json", "w") as f:
            json.dump({"pose_4x4": result.fdpose_grasp.pose}, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# 📦 Flat Output for Downstream Optim
# ══════════════════════════════════════════════════════════════════════════════


def save_out(
    result: TaskOutput, output_dir: Path, grasp_cam: CameraIntrinsics
) -> None:
    """📦 Save output in flat structure for downstream optim."""
    print(f"  📦 Saving output...")
    output_dir.mkdir(parents=True, exist_ok=True)

    img_w, img_h = grasp_cam.width, grasp_cam.height

    # 📷 intrinsics.json (grasp image dynamic intrinsics)
    with open(output_dir / "intrinsics.json", "w") as f:
        json.dump({
            "fx": grasp_cam.fx, "fy": grasp_cam.fy,
            "ppx": grasp_cam.ppx, "ppy": grasp_cam.ppy,
            "width": grasp_cam.width, "height": grasp_cam.height,
            "depth_scale": 0.001,
        }, f, indent=2)

    # 🧊 scaled_mesh.obj
    if result.scale and result.scale.scaled_mesh is not None:
        result.scale.scaled_mesh.export(str(output_dir / "scaled_mesh.obj"))

    # 🎭 seg_mask.png (0=bg, 128=hand, 255=obj)
    if result.gsam_grasp:
        dets = result.gsam_grasp.detections
        h, w = result.gsam_grasp.img_size
        seg = np.zeros((h, w), dtype=np.uint8)
        hand_dets = [d for d in dets if d.get("is_hand", False)]
        obj_dets = [d for d in dets if not d.get("is_hand", False)]
        if hand_dets:
            hand_rle = max(hand_dets, key=lambda d: d["score"])["mask_rle"]
            seg[decode_mask_rle(hand_rle).astype(bool)] = 128
        if obj_dets:
            obj_rle = max(obj_dets, key=lambda d: d["score"])["mask_rle"]
            seg[decode_mask_rle(obj_rle).astype(bool)] = 255
        cv2.imwrite(str(output_dir / "seg_mask.png"), seg)

    # 🎭 inpaint_mask.png (FDPose rendered object mask)
    if result.fdpose_grasp and result.fdpose_grasp.obj_mask_b64:
        mask_bytes = base64.b64decode(result.fdpose_grasp.obj_mask_b64)
        nparr = np.frombuffer(mask_bytes, np.uint8)
        mask_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(output_dir / "inpaint_mask.png"), mask_img)

    # 📐 scene_pose.json
    if result.fdpose_scene and result.fdpose_scene.pose:
        with open(output_dir / "scene_pose.json", "w") as f:
            json.dump(pose_to_quat_json(result.fdpose_scene.pose), f, indent=2)

    # 📐 grasp_pose.json
    if result.fdpose_grasp and result.fdpose_grasp.pose:
        with open(output_dir / "grasp_pose.json", "w") as f:
            json.dump(pose_to_quat_json(result.fdpose_grasp.pose), f, indent=2)

    # 🤚 hand_params.pt
    if result.hamer:
        torch.save({
            "mano_params": result.hamer.mano_params,
            "cam_transl": [result.hamer.cam_transl],
            "is_right": [result.hamer.is_right],
            "batch_size": 1,
        }, output_dir / "hand_params.pt")

    # 📷 camera_params.json (normalized intrinsics + identity extrinsics)
    if result.hamer:
        with open(output_dir / "camera_params.json", "w") as f:
            json.dump({
                "extrinsics": np.eye(3, 4).tolist(),
                "fx": grasp_cam.fx / img_w,
                "fy": grasp_cam.fy / img_h,
                "cx": grasp_cam.ppx / img_w,
                "cy": grasp_cam.ppy / img_h,
            }, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# 🚀 Entry Point
# ══════════════════════════════════════════════════════════════════════════════


@hydra.main(config_path="../cfg", config_name="recons", version_base=None)
def main(cfg: DictConfig) -> None:
    """🚀 Run reconstruction pipeline."""
    datasets_dir = Path(cfg.datasets)
    output_dir = Path(cfg.output)

    print(f"🚀 Starting reconstruction pipeline")
    print(f"📂 Datasets: {datasets_dir.resolve()}")
    print(f"📁 Output: {output_dir.resolve()}")
    print(f"🖥️  GSAM: {cfg.servers.gsam}")
    print(f"🖥️  HaMeR: {cfg.servers.hamer}")
    print(f"🖥️  DA3: {cfg.servers.da3}")
    print(f"🖥️  FDPose: {cfg.servers.fdpose}")

    task_count = 0
    failed = []
    for task in load_tasks(datasets_dir):
        print(f"\n{'='*60}")
        print(f"🎯 Processing: {task.name}")

        try:
            result = process_task(task, cfg)
            save_output(result, output_dir / task.name, cfg)
            task_count += 1
            print(f"✅ Done: {task.name}")
        except Exception as e:
            failed.append(task.name)
            print(f"  ❌ Failed: {task.name} — {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"🎉 Completed: {task_count}, Failed: {len(failed)}")
    if failed:
        print(f"❌ Failed tasks: {', '.join(failed)}")


if __name__ == "__main__":
    main()
