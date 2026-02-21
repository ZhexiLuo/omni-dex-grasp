"""🎯 6DoF object pose estimation using MegaPose6D.

Usage:
    conda activate megapose && python -m recons.pose_est
    conda activate megapose && python -m recons.pose_est megapose6d.bsz_images=32
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import hydra
import numpy as np
import pandas as pd
import torch
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import ObservationTensor
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import load_named_model
from megapose.utils.tensor_collection import PandasTensorCollection
from omegaconf import DictConfig
from PIL import Image

import recons.panda3d_batch_renderer_wrapper  # apply sequential renderer patch (megapose6d issue #66)
from recons.data import PoseEstInput, PoseEstResult
from utils.camera import load_k_from_json, load_k_from_yaml


# ── Helper functions ───────────────────────────────────────────────────────────


def _build_detections(label: str, bbox: np.ndarray) -> PandasTensorCollection:
    """Build MegaPose DetectionsType from a single bbox (XYXY format)."""
    return PandasTensorCollection(
        infos=pd.DataFrame({"label": [label], "batch_im_id": [0], "instance_id": [0]}),
        bboxes=torch.from_numpy(bbox[np.newaxis].astype(np.float32)),
    )


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay green mask region onto RGB image for visualization."""
    overlay = rgb.copy()
    overlay[mask > 0] = (
        (1 - alpha) * rgb[mask > 0] + alpha * np.array([0, 200, 0])
    ).astype(np.uint8)
    return overlay


# ── MegaPoseEstimator: load once, run for all tasks ───────────────────────────


class MegaPoseEstimator:
    """Load-once MegaPose6D estimator shared across all tasks.

    Loads scene_model (with ICP refiner) once at startup.
    - Scene: run_depth_refiner=True (ICP enabled, depth provided)
    - Grasp: run_depth_refiner=False (ICP disabled, no depth)
    Both share the same neural network checkpoint — no repeated loading.
    """

    def __init__(self, object_dataset: RigidObjectDataset, cfg: DictConfig) -> None:
        # Load neural network once for the full object dataset
        print(f"  🔄 Loading model: {cfg.scene_model}")
        self.estimator = load_named_model(
            cfg.scene_model,
            object_dataset=object_dataset,
            n_workers=cfg.get("n_workers", 1),
            bsz_images=cfg.bsz_images,
        ).cuda()
        # Defer renderer construction until first render_mask call to avoid EGL context exhaustion
        self._object_dataset = object_dataset   # stored for lazy renderer creation
        self._renderer: Panda3dSceneRenderer | None = None  # init deferred to first render_mask call
        self.cfg = cfg

    def estimate(self, inp: PoseEstInput) -> PoseEstResult:
        """Run pose estimation; ICP enabled only when inp.depth is not None.

        Args:
            inp: PoseEstInput — scene passes depth, grasp passes depth=None.

        Returns:
            PoseEstResult with pose [4x4] T_CO and confidence score.
        """
        obs = ObservationTensor.from_numpy(inp.rgb, inp.depth, inp.K).cuda()
        dets = _build_detections(inp.label, inp.bbox).cuda()
        use_icp = inp.depth is not None

        output, _ = self.estimator.run_inference_pipeline(
            obs,
            detections=dets,
            run_detector=False,
            n_refiner_iterations=self.cfg.n_refiner_iterations,
            n_pose_hypotheses=self.cfg.n_pose_hypotheses,
            run_depth_refiner=use_icp,  # True for scene, False for grasp
        )

        pose = output.poses[0].cpu().numpy()           # [4, 4] T_CO float32
        score = float(output.infos["pose_score"].iloc[0])
        image_key = "scene" if use_icp else "grasp"
        return PoseEstResult(label=inp.label, pose=pose.tolist(), score=score, image_key=image_key)

    def render_mask(self, label: str, pose_4x4: np.ndarray, K: np.ndarray, img_hw: tuple[int, int]) -> np.ndarray:
        """Render binary object mask from an estimated pose.

        Args:
            label: object label matching a RigidObject in the dataset.
            pose_4x4: T_CO [4,4] float32.
            K: camera intrinsics [3,3] float32.
            img_hw: (height, width) of the target image.

        Returns:
            [H, W] uint8 mask (0=background, 255=object).
        """
        if self._renderer is None:
            self._renderer = Panda3dSceneRenderer(self._object_dataset)  # deferred: workers already init'd
        h, w = img_hw
        camera = CameraData(K=K.astype(np.float64), resolution=(w, h))  # resolution=(W, H)
        obj_data = ObjectData(label=label, TWO=Transform(pose_4x4))
        cam_p3d, obj_p3d = convert_scene_observation_to_panda3d(camera, [obj_data])
        lights = [Panda3dLightData(light_type="ambient", color=(1.0, 1.0, 1.0, 1))]
        renderings = self._renderer.render_scene(obj_p3d, [cam_p3d], lights, render_depth=True, render_binary_mask=True)
        return renderings[0].binary_mask.astype(np.uint8) * 255


# ── Input builders ─────────────────────────────────────────────────────────────


def build_scene_input(task_dir: Path, task_out: Path, det: dict) -> PoseEstInput:
    """Build PoseEstInput for scene image (with depth, K from camera.yaml)."""
    rgb = np.array(Image.open(task_dir / "scene_image.png"), dtype=np.uint8)
    K = load_k_from_yaml(task_dir / "camera.yaml")
    depth = np.array(Image.open(task_dir / "depth.png"), dtype=np.uint16).astype(np.float32) * 0.001  # mm → meters
    return PoseEstInput(
        rgb=rgb, K=K,
        bbox=np.array(det["bbox"], dtype=np.float32),
        label=det["label"],
        mesh_path=task_out / "scaled_mesh.obj",
        depth=depth,
    )


def build_grasp_input(task_dir: Path, task_out: Path, det: dict) -> PoseEstInput:
    """Build PoseEstInput for grasp image (no depth, K from intrinsics.json)."""
    rgb = np.array(Image.open(task_dir / "generated_human_grasp.png"), dtype=np.uint8)
    K = load_k_from_json(task_out / "intrinsics.json")
    return PoseEstInput(
        rgb=rgb, K=K,
        bbox=np.array(det["bbox"], dtype=np.float32),
        label=det["label"],
        mesh_path=task_out / "scaled_mesh.obj",
        depth=None,  # grasp has no depth map
    )


# ── Output saving ──────────────────────────────────────────────────────────────


def save_pose_output(
    task_out: Path,
    scene_result: PoseEstResult,
    scene_inp: PoseEstInput,
    grasp_result: PoseEstResult,
    grasp_inp: PoseEstInput,
    estimator: MegaPoseEstimator,
) -> None:
    """Save pose_est.json, obj_mask.png, and visualization overlays."""
    vis_dir = task_out / "vis"
    vis_dir.mkdir(exist_ok=True)

    # Write scene + grasp poses to JSON
    (task_out / "pose_est.json").write_text(json.dumps({
        "scene": {"pose": scene_result.pose, "score": scene_result.score},
        "grasp": {"pose": grasp_result.pose, "score": grasp_result.score},
    }, indent=2))

    # Render grasp object mask (saved to out/ for downstream use)
    grasp_pose = np.array(grasp_result.pose, dtype=np.float32)
    grasp_mask = estimator.render_mask(grasp_inp.label, grasp_pose, grasp_inp.K, grasp_inp.rgb.shape[:2])
    Image.fromarray(grasp_mask).save(task_out / "obj_mask.png")

    # Render scene mask for visualization overlay only
    scene_pose = np.array(scene_result.pose, dtype=np.float32)
    scene_mask = estimator.render_mask(scene_inp.label, scene_pose, scene_inp.K, scene_inp.rgb.shape[:2])

    cv2.imwrite(str(vis_dir / "pose_scene.jpg"),
                cv2.cvtColor(_overlay_mask(scene_inp.rgb, scene_mask), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(vis_dir / "pose_grasp.jpg"),
                cv2.cvtColor(_overlay_mask(grasp_inp.rgb, grasp_mask), cv2.COLOR_RGB2BGR))


# ── Per-task processing ────────────────────────────────────────────────────────


def process_task(task_dir: Path, out_dir: Path, estimator: MegaPoseEstimator) -> None:
    """Run scene + grasp pose estimation for one task using shared estimator."""
    task_out = out_dir / task_dir.name
    det_data = json.loads((task_out / "detection.json").read_text())

    scene_inp = build_scene_input(task_dir, task_out, det_data["scene"])
    grasp_inp = build_grasp_input(task_dir, task_out, det_data["grasp"])

    print("  🎯 Scene pose estimation (RGBD + ICP)...")
    scene_result = estimator.estimate(scene_inp)
    print(f"     └─ score: {scene_result.score:.4f}")

    print("  🤚 Grasp pose estimation (RGB, no ICP)...")
    grasp_result = estimator.estimate(grasp_inp)
    print(f"     └─ score: {grasp_result.score:.4f}")

    save_pose_output(task_out, scene_result, scene_inp, grasp_result, grasp_inp, estimator)
    print("  💾 Saved: pose_est.json + obj_mask.png + vis/")


# ── Entry point ────────────────────────────────────────────────────────────────


@hydra.main(config_path="../cfg", config_name="recons", version_base=None)
def main(cfg: DictConfig) -> None:
    """🚀 Run MegaPose6D pose estimation for all tasks (model loaded once)."""
    datasets_dir = Path(cfg.datasets).resolve()
    output_dir = Path(cfg.output).resolve()

    all_task_dirs = sorted(d for d in datasets_dir.iterdir() if d.is_dir() and not d.name.startswith("."))

    # Pre-collect all valid task objects into a single RigidObjectDataset
    # This allows load_named_model() to be called just once for all tasks
    rigid_objects: list[RigidObject] = []
    valid_tasks: list[Path] = []
    for task_dir in all_task_dirs:
        task_out = output_dir / task_dir.name
        if not (task_out / "scaled_mesh.obj").exists():
            print(f"⏭️  Skip {task_dir.name}: run client.py first")
            continue
        if not (task_out / "detection.json").exists():
            print(f"⏭️  Skip {task_dir.name}: detection.json missing")
            continue
        rigid_objects.append(
            RigidObject(label=task_dir.name, mesh_path=task_out / "scaled_mesh.obj", mesh_units="m")
        )
        valid_tasks.append(task_dir)

    if not valid_tasks:
        print("❌ No valid tasks found.")
        return

    print(f"🚀 Pose estimation: {len(valid_tasks)} tasks | model={cfg.megapose6d.scene_model}")

    # Load neural network and renderer ONCE for all tasks
    object_dataset = RigidObjectDataset(rigid_objects)
    estimator = MegaPoseEstimator(object_dataset, cfg.megapose6d)

    # Iterate tasks; estimator is reused without reloading
    failed: list[str] = []
    for task_dir in valid_tasks:
        print(f"\n🎯 {task_dir.name}")
        try:
            process_task(task_dir, output_dir, estimator)
            print("  ✅ Done")
        except Exception as e:
            import traceback
            failed.append(task_dir.name)
            print(f"  ❌ {type(e).__name__}: {e}")
            traceback.print_exc()

    print(f"\n🎉 Done: {len(valid_tasks) - len(failed)} OK, {len(failed)} failed")
    if failed:
        print(f"❌ Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
