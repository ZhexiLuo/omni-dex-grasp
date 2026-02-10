"""📐 FoundationPose 6D pose estimation server.

POST /predict - Estimate 6D object pose from RGB + depth + mesh.

Usage: conda activate fdpose && python -m recons.server.fdpose
"""
import base64
import io
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

_FP_DIR = Path(__file__).resolve().parents[2] / "thirdparty" / "FoundationPose"
sys.path.insert(0, str(_FP_DIR))

import cv2
import hydra
import numpy as np
import torch
import trimesh
import uvicorn
from fastapi import FastAPI, Request
from omegaconf import DictConfig
from PIL import Image
from pydantic import BaseModel

from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
from Utils import draw_posed_3d_box, draw_xyz_axis
import nvdiffrast.torch as dr


# ══════════════════════════════════════════════════════════════════════════════
# 🧠 Model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FDPoseModel:
    """📐 FoundationPose model wrapper."""
    estimator: FoundationPose
    device: str
    cfg: DictConfig

    def __post_init__(self):
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "FDPoseModel":
        """🚀 Initialize FoundationPose (scorer/refiner/glctx loaded once)."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading FoundationPose on {device}...")

        print("  📊 Loading ScorePredictor...")
        scorer = ScorePredictor()
        print("  🔧 Loading PoseRefinePredictor...")
        refiner = PoseRefinePredictor()
        print("  🎨 Creating CUDA rasterize context...")
        glctx = dr.RasterizeCudaContext()

        # 📦 Initialize with dummy mesh (first request will reset_object)
        dummy_mesh = trimesh.primitives.Box()
        debug_dir = str(_FP_DIR / "debug")
        est = FoundationPose(
            model_pts=dummy_mesh.vertices.copy(),
            model_normals=dummy_mesh.vertex_normals.copy(),
            mesh=dummy_mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=cfg.inference.debug,
            debug_dir=debug_dir,
        )

        print("✅ FoundationPose loaded!")
        return cls(estimator=est, device=device, cfg=cfg)

    def estimate_pose(
        self,
        image_path: str,
        depth: np.ndarray,
        obj_mesh_path: str,
        bbox: list[float],
        intrinsics: np.ndarray,
    ) -> dict:
        """📐 Estimate 6D pose for a single object."""
        with self._lock:
            # 1️⃣ Load RGB
            rgb = np.array(Image.open(image_path).convert("RGB"))
            H, W = rgb.shape[:2]

            # 📏 Validate depth-RGB shape match
            if depth.shape != (H, W):
                raise ValueError(f"Depth shape {depth.shape} != RGB shape ({H}, {W})")

            # 2️⃣ Load mesh & reset estimator (float32 required by FoundationPose)
            mesh = trimesh.load(obj_mesh_path, force="mesh")
            mesh.vertices = mesh.vertices.astype(np.float32)
            self.estimator.reset_object(
                model_pts=mesh.vertices.copy(),
                model_normals=mesh.vertex_normals.copy().astype(np.float32),
                mesh=mesh,
            )

            # 3️⃣ Build mask from bbox (clip to image bounds)
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, x2 = max(0, x1), min(W, x2)
            y1, y2 = max(0, y1), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Degenerate bbox after clipping: [{x1},{y1},{x2},{y2}] for image ({H},{W})")
            mask = np.zeros((H, W), dtype=bool)
            mask[y1:y2, x1:x2] = True

            # 4️⃣ Register (single-frame pose estimation)
            pose = self.estimator.register(
                K=intrinsics,
                rgb=rgb,
                depth=depth,
                ob_mask=mask,
                iteration=self.cfg.inference.est_refine_iter,
            )
            # pose: (4,4) numpy array, object-in-camera

            # ⚠️ Detect degenerate pose (identity rotation = insufficient valid depth)
            is_degenerate = np.allclose(pose[:3, :3], np.eye(3), atol=1e-6)

            # 5️⃣ Generate visualization
            vis = self._generate_vis(rgb, pose, mesh, intrinsics)

            return {"pose": pose, "vis": vis, "is_degenerate": is_degenerate}

    def _generate_vis(
        self,
        rgb: np.ndarray,
        pose: np.ndarray,
        mesh: trimesh.Trimesh,
        K: np.ndarray,
    ) -> np.ndarray:
        """🎨 Generate pose visualization with 3D bbox and axes. Returns BGR image."""
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox_3d = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        center_pose = pose @ np.linalg.inv(to_origin)

        vis_bgr = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
        vis_bgr = draw_posed_3d_box(
            K, img=vis_bgr, ob_in_cam=center_pose, bbox=bbox_3d,
        )
        vis_bgr = draw_xyz_axis(
            vis_bgr, ob_in_cam=center_pose, scale=0.1,
            K=K, thickness=3, transparency=0, is_input_rgb=False,
        )
        return vis_bgr


# ══════════════════════════════════════════════════════════════════════════════
# 🌐 API
# ══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    image_path: str
    depth_b64: str                     # 📏 base64 encoded depth npy (H,W) float32, meters
    obj_mesh_path: str
    bbox: list[float]                  # [x1, y1, x2, y2]
    intrinsics: list[list[float]]      # 3x3 camera intrinsics matrix


class PredictResponse(BaseModel):
    status: str
    message: str
    pose: list[list[float]] = []       # 📐 4x4 pose matrix (object-in-camera)
    pose_vis_b64: str = ""             # 🎨 base64 encoded visualization


# ══════════════════════════════════════════════════════════════════════════════
# 🔧 Helpers
# ══════════════════════════════════════════════════════════════════════════════

def decode_array_b64(b64_str: str) -> np.ndarray:
    """📐 Decode base64 npy string to numpy array."""
    arr_bytes = base64.b64decode(b64_str)
    return np.load(io.BytesIO(arr_bytes))


def encode_image_b64(img_bgr: np.ndarray) -> str:
    """🖼️ Encode BGR image to base64 PNG string."""
    _, buffer = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buffer).decode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# 🚀 Server
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="📐 FoundationPose Server")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    """📐 Estimate 6D object pose."""
    model: FDPoseModel = request.app.state.model
    print(f"\n{'='*60}")
    print(f"📨 New request: {req.image_path}")

    # 📋 Validate inputs
    image_path = Path(req.image_path)
    if not image_path.exists():
        return PredictResponse(status="error", message=f"Image not found: {req.image_path}")

    mesh_path = Path(req.obj_mesh_path)
    if not mesh_path.exists():
        return PredictResponse(status="error", message=f"Mesh not found: {req.obj_mesh_path}")

    K = np.array(req.intrinsics, dtype=np.float64)  # FoundationPose expects float64 K
    if K.shape != (3, 3):
        return PredictResponse(status="error", message=f"Intrinsics must be 3x3, got {K.shape}")

    if len(req.bbox) != 4:
        return PredictResponse(status="error", message=f"Bbox must be [x1,y1,x2,y2], got len={len(req.bbox)}")

    # 📏 Decode depth
    depth = decode_array_b64(req.depth_b64)
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)

    print(f"  📏 Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}] m")
    print(f"  📦 Bbox: {req.bbox}")
    print(f"  🧊 Mesh: {req.obj_mesh_path}")

    # 📐 Estimate pose
    try:
        result = model.estimate_pose(
            image_path=str(image_path),
            depth=depth,
            obj_mesh_path=str(mesh_path),
            bbox=req.bbox,
            intrinsics=K,
        )
    except Exception as e:
        print(f"  ❌ Pose estimation failed: {e}")
        return PredictResponse(status="error", message=f"Pose estimation failed: {e}")

    # 📦 Encode outputs
    pose_list = result["pose"].tolist()
    vis_b64 = encode_image_b64(result["vis"])

    status = "warning" if result["is_degenerate"] else "success"
    msg_suffix = " (⚠️ degenerate pose, may be unreliable)" if result["is_degenerate"] else ""

    print(f"  📐 Pose estimated! {'⚠️ DEGENERATE' if result['is_degenerate'] else '✅'}")
    print(f"  📍 Translation: [{result['pose'][0,3]:.4f}, {result['pose'][1,3]:.4f}, {result['pose'][2,3]:.4f}]")
    print(f"🎉 Done!")

    return PredictResponse(
        status=status,
        message=f"Pose estimated for {mesh_path.stem}{msg_suffix}",
        pose=pose_list,
        pose_vis_b64=vis_b64,
    )


@hydra.main(config_path="../../cfg/model", config_name="fdpose", version_base=None)
def main(cfg: DictConfig) -> None:
    """🚀 Start FDPose server with Hydra config."""
    app.state.model = FDPoseModel.from_config(cfg)
    print(f"🌐 Server starting at http://{cfg.server.host}:{cfg.server.port}")
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)


if __name__ == "__main__":
    main()
