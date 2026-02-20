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
    GSAMResult,
    HaMeRResult,
    ScaleResult,
    TaskInput,
    TaskOutput,
    load_tasks,
)
from utils.camera import CameraIntrinsics, compute_focal, dynamic_intrinsics
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


def extract_obj_mask_rle(gsam_result: GSAMResult) -> dict:
    """🎭 Extract highest-confidence non-hand object mask_rle from GSAM result."""
    obj_dets = [d for d in gsam_result.detections if not d.get("is_hand", False)]
    if not obj_dets:
        raise ValueError("❌ No object detection in GSAM scene result")
    best = max(obj_dets, key=lambda d: d["score"])
    return best["mask_rle"]


def scale_and_center_mesh(mesh_path: Path, scale_factor: float) -> "trimesh.Trimesh":
    """📏 Scale + center mesh, return in-memory trimesh object."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    mesh.vertices *= scale_factor
    mesh.vertices -= mesh.bounding_box.centroid
    return mesh


def encode_image_file_b64(image_path: Path) -> str:
    """🖼️ Encode image file to base64 (preserves original PNG/JPG format)."""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


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


# ══════════════════════════════════════════════════════════════════════════════
# 🎯 Task Processing
# ══════════════════════════════════════════════════════════════════════════════


def process_task(task: TaskInput, cfg: DictConfig) -> TaskOutput:
    """🎯 Process a single task through the reconstruction pipeline."""
    output = TaskOutput(name=task.name)
    timeout = cfg.servers.timeout

    # 📷 Pre-encode images (encode once, reuse for all servers)
    scene_b64 = encode_image_file_b64(task.scene_image)
    grasp_b64 = encode_image_file_b64(task.generated_grasp)

    # 📐 Pre-compute intrinsics (grasp image resolution)
    grasp_img = Image.open(task.generated_grasp)
    grasp_cam = dynamic_intrinsics(task.camera, grasp_img.width, grasp_img.height)
    grasp_focal = compute_focal(grasp_cam)

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
        edge_erode_px=cfg.scale.edge_erode_px,
    )
    print(f"     └─ raw points: {pcd_raw.shape[0]}")

    pcd_clean = denoise_pointcloud(pcd_raw, cfg.scale.stat_nb_neighbors, cfg.scale.stat_std_ratio)
    print(f"     └─ cleaned: {pcd_clean.shape[0]}")

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
