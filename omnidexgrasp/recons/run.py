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
from omegaconf import DictConfig
from PIL import Image

from recons.data import (
    GSAMResult,
    HaMeRResult,
    TaskInput,
    TaskOutput,
    load_tasks,
)
from utils.camera import compute_focal, dynamic_intrinsics


# ══════════════════════════════════════════════════════════════════════════════
# 🌐 Server Calls
# ══════════════════════════════════════════════════════════════════════════════


def call_gsam(
    url: str, image_path: Path, text_prompt: str, include_hand: bool, timeout: int
) -> GSAMResult:
    """🦖 Call GSAM2 server for detection and segmentation.

    Args:
        url: GSAM2 server URL.
        image_path: Path to input image.
        text_prompt: Object description for detection.
        include_hand: Whether to include hand detection.
        timeout: Request timeout in seconds.

    Returns:
        GSAMResult with detection and segmentation data.
    """
    resp = requests.post(
        url,
        json={
            "image_path": str(image_path.resolve()),
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
    url: str, image_path: Path, focal_length: float, timeout: int
) -> HaMeRResult:
    """🤚 Call HaMeR server for hand reconstruction.

    Args:
        url: HaMeR server URL.
        image_path: Path to input image.
        focal_length: Camera focal length.
        timeout: Request timeout in seconds.

    Returns:
        HaMeRResult with hand reconstruction data.
    """
    resp = requests.post(
        url,
        json={
            "image_path": str(image_path.resolve()),
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
    """🎯 Process a single task through the reconstruction pipeline.

    Args:
        task: Input task data.
        cfg: Pipeline configuration.

    Returns:
        TaskOutput with all processing results.
    """
    output = TaskOutput(name=task.name)
    timeout = cfg.servers.timeout

    # 1️⃣ GSAM: scene_image (no hand)
    print(f"  🎭 GSAM: scene_image (no hand)")
    output.gsam_scene = call_gsam(
        cfg.servers.gsam,
        task.scene_image,
        task.obj_description,
        include_hand=False,
        timeout=timeout,
    )
    print(f"     └─ {output.gsam_scene.status}: {output.gsam_scene.message}")

    # 2️⃣ GSAM: generated_grasp (with hand)
    print(f"  🎭 GSAM: generated_grasp (with hand)")
    output.gsam_grasp = call_gsam(
        cfg.servers.gsam,
        task.generated_grasp,
        task.obj_description,
        include_hand=True,
        timeout=timeout,
    )
    print(f"     └─ {output.gsam_grasp.status}: {output.gsam_grasp.message}")

    # 3️⃣ HaMeR: hand reconstruction
    print(f"  🤚 HaMeR: hand reconstruction")
    grasp_img = Image.open(task.generated_grasp)
    scaled_cam = dynamic_intrinsics(task.camera, grasp_img.width, grasp_img.height)
    focal = compute_focal(scaled_cam)
    print(f"     └─ focal_length: {focal:.2f}")

    output.hamer = call_hamer(
        cfg.servers.hamer,
        task.generated_grasp,
        focal,
        timeout=timeout,
    )
    print(f"     └─ {output.hamer.status}: {output.hamer.message}")

    return output


# ══════════════════════════════════════════════════════════════════════════════
# 💾 Output Saving
# ══════════════════════════════════════════════════════════════════════════════


def decode_image_b64(b64_str: str) -> np.ndarray:
    """🖼️ Decode base64 string to image array."""
    img_bytes = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def decode_array_b64(b64_str: str) -> np.ndarray:
    """📐 Decode base64 string to numpy array."""
    arr_bytes = base64.b64decode(b64_str)
    return np.load(io.BytesIO(arr_bytes))


def save_output(result: TaskOutput, output_dir: Path, cfg: DictConfig) -> None:
    """💾 Save task output to directory.

    Args:
        result: Task processing results.
        output_dir: Output directory for this task.
        cfg: Pipeline configuration.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 📊 Save visualizations
    if cfg.out.vis:
        save_visualizations(result, output_dir)

    # 💾 Save intermediate data
    if cfg.out.inter_out:
        save_intermediate(result, output_dir)

    print(f"  💾 Saved to: {output_dir}")


def save_visualizations(result: TaskOutput, output_dir: Path) -> None:
    """📊 Save visualization images."""
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(exist_ok=True)

    # GSAM scene
    if result.gsam_scene and result.gsam_scene.annotated_b64:
        img = decode_image_b64(result.gsam_scene.annotated_b64)
        cv2.imwrite(str(vis_dir / "gsam_scene_annotated.jpg"), img)
    if result.gsam_scene and result.gsam_scene.mask_b64:
        img = decode_image_b64(result.gsam_scene.mask_b64)
        cv2.imwrite(str(vis_dir / "gsam_scene_mask.png"), img)

    # GSAM grasp
    if result.gsam_grasp and result.gsam_grasp.annotated_b64:
        img = decode_image_b64(result.gsam_grasp.annotated_b64)
        cv2.imwrite(str(vis_dir / "gsam_grasp_annotated.jpg"), img)
    if result.gsam_grasp and result.gsam_grasp.mask_b64:
        img = decode_image_b64(result.gsam_grasp.mask_b64)
        cv2.imwrite(str(vis_dir / "gsam_grasp_mask.png"), img)

    # HaMeR mask
    if result.hamer and result.hamer.mask_b64:
        img = decode_image_b64(result.hamer.mask_b64)
        cv2.imwrite(str(vis_dir / "hamer_mask.png"), img)


def save_intermediate(result: TaskOutput, output_dir: Path) -> None:
    """💾 Save intermediate data (detections, vertices, params)."""
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # GSAM detections
    if result.gsam_scene:
        with open(data_dir / "gsam_scene.json", "w") as f:
            json.dump(
                {
                    "status": result.gsam_scene.status,
                    "message": result.gsam_scene.message,
                    "detections": result.gsam_scene.detections,
                    "img_size": result.gsam_scene.img_size,
                },
                f,
                indent=2,
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
                f,
                indent=2,
            )

    # HaMeR data
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
                f,
                indent=2,
            )
        # FIXME: save as .obj
        if result.hamer.vertices_b64:
            vertices = decode_array_b64(result.hamer.vertices_b64)
            np.save(data_dir / "hamer_vertices.npy", vertices)


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
    for task in load_tasks(datasets_dir):
        print(f"\n{'='*60}")
        print(f"🎯 Processing: {task.name}")

        result = process_task(task, cfg)
        save_output(result, output_dir / task.name, cfg)

        task_count += 1
        print(f"✅ Done: {task.name}")

    print(f"\n{'='*60}")
    print(f"🎉 All {task_count} tasks completed!")


if __name__ == "__main__":
    main()
