"""📦 Data classes and loader for reconstruction pipeline."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import yaml

from utils.camera import CameraIntrinsics


# ══════════════════════════════════════════════════════════════════════════════
# 📋 Input DataClasses
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TaskInput:
    """🎯 Input data for a single reconstruction task."""

    name: str
    task_dir: Path
    scene_image: Path
    generated_grasp: Path
    camera: CameraIntrinsics
    obj_description: str
    obj_mesh: Path
    depth: Path


# ══════════════════════════════════════════════════════════════════════════════
# 📤 Output DataClasses
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class GSAMResult:
    """🎭 GSAM2 detection and segmentation result."""

    status: str
    message: str
    detections: list[dict] = field(default_factory=list)
    img_size: list[int] = field(default_factory=list)
    annotated_b64: str = ""
    mask_b64: str = ""


@dataclass
class HaMeRResult:
    """🤚 HaMeR hand reconstruction result."""

    status: str
    message: str
    mano_params: dict = field(default_factory=dict)
    vertices_b64: str = ""
    cam_transl: list[float] = field(default_factory=list)
    is_right: bool = False
    mask_b64: str = ""


@dataclass
class ScaleResult:
    """📏 Object real-world scale computation result."""

    scale_factor: float
    pcd_num_points: int
    pcd_max_extent: float
    mesh_max_extent: float


@dataclass
class TaskOutput:
    """📤 Complete output data for a single task."""

    name: str
    gsam_scene: GSAMResult | None = None
    gsam_grasp: GSAMResult | None = None
    hamer: HaMeRResult | None = None
    scale: ScaleResult | None = None


# ══════════════════════════════════════════════════════════════════════════════
# 📂 DataLoader
# ══════════════════════════════════════════════════════════════════════════════


def load_tasks(datasets_dir: Path) -> Iterator[TaskInput]:
    """🔄 Load all tasks from datasets directory.

    Args:
        datasets_dir: Path to datasets directory containing task folders.

    Yields:
        TaskInput for each valid task directory.
    """
    for task_dir in sorted(datasets_dir.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        yield load_single_task(task_dir)


def load_single_task(task_dir: Path) -> TaskInput:
    """📁 Load a single task from directory.

    Args:
        task_dir: Path to task directory.

    Returns:
        TaskInput with all task data loaded.
    """
    camera_yaml = task_dir / "camera.yaml"
    with open(camera_yaml) as f:
        cam_data = yaml.safe_load(f)

    camera = CameraIntrinsics(
        fx=cam_data["fx"],
        fy=cam_data["fy"],
        ppx=cam_data["ppx"],
        ppy=cam_data["ppy"],
        width=cam_data["width"],
        height=cam_data["height"],
    )

    return TaskInput(
        name=task_dir.name,
        task_dir=task_dir,
        scene_image=task_dir / "scene_image.png",
        generated_grasp=task_dir / "generated_human_grasp.png",
        camera=camera,
        obj_description=cam_data["obj_description"],
        obj_mesh=task_dir / "base.obj",
        depth=task_dir / "depth.png",
    )
