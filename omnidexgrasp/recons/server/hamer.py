"""🤚 HaMeR hand reconstruction server.

POST /predict - Reconstruct hand mesh from image.

Usage: python -m omnidexgrasp.recons.server.server_hamer
"""
import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from omegaconf import DictConfig
from PIL import Image
from pydantic import BaseModel

# ViTPose wholebody keypoint indices (COCO-WholeBody format)
VITPOSE_LEFT_HAND_START = -42
VITPOSE_LEFT_HAND_END = -21
VITPOSE_RIGHT_HAND_START = -21

# ══════════════════════════════════════════════════════════════════════════════
# 🧠 Model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HaMeRModel:
    """🤚 HaMeR model wrapper for hand reconstruction."""
    hamer_model: "HAMER"  # type: ignore
    model_cfg: DictConfig
    body_detector: "DefaultPredictor_Lazy"  # type: ignore
    keypoint_detector: "ViTPoseModel"  # type: ignore
    renderer: "Renderer"  # type: ignore
    device: str
    cfg: DictConfig

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "HaMeRModel":
        """🚀 Load HaMeR models from config."""
        import hamer
        import hamer.configs
        from hamer.models import load_hamer
        from hamer.utils.renderer import Renderer
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        from vitpose_model import ViTPoseModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading HaMeR models on {device}...")

        # 📁 Configure paths from yaml
        hamer.configs.CACHE_DIR_HAMER = str(Path(cfg.model.checkpoint).parent)
        ViTPoseModel.MODEL_DICT = {
            "ViTPose+-G (multi-task train, COCO)": {
                "config": cfg.model.vitpose_config,
                "model": cfg.model.vitpose_ckpt,
            },
        }

        # 🧠 Load HaMeR
        print("  🤚 Loading HaMeR...")
        model, model_cfg = load_hamer(str(cfg.model.checkpoint))
        model = model.to(device)
        model.eval()

        # 🔍 Load body detector
        print("  🔍 Loading body detector...")
        if cfg.model.body_detector == "vitdet":
            from detectron2.config import LazyConfig
            det_cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
            det_cfg = LazyConfig.load(str(det_cfg_path))
            det_cfg.train.init_checkpoint = str(cfg.model.vitdet_checkpoint)
            for i in range(3):
                det_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            body_detector = DefaultPredictor_Lazy(det_cfg)
        else:
            from detectron2 import model_zoo
            det_cfg = model_zoo.get_config("new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True)
            det_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            det_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            body_detector = DefaultPredictor_Lazy(det_cfg)

        # 🦴 Load keypoint detector
        print("  🦴 Loading keypoint detector...")
        keypoint_detector = ViTPoseModel(device)

        # 🎨 Setup renderer
        renderer = Renderer(model_cfg, faces=model.mano.faces)

        print("✅ All HaMeR models loaded!")
        return cls(model, model_cfg, body_detector, keypoint_detector, renderer, device, cfg)

    def detect_best_hand(self, img_cv2: np.ndarray) -> dict | None:
        """🔍 Detect the best hand in image."""
        from hamer.utils import recursive_to

        img_rgb = img_cv2.copy()[:, :, ::-1]
        det_out = self.body_detector(img_cv2)
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        if len(pred_bboxes) == 0:
            return None

        vitposes_out = self.keypoint_detector.predict_pose(
            img_rgb,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        # 🏆 Find best hand by confidence
        max_conf = 0
        best_hand = None
        for vitposes in vitposes_out:
            left_kp = vitposes["keypoints"][VITPOSE_LEFT_HAND_START:VITPOSE_LEFT_HAND_END]
            right_kp = vitposes["keypoints"][VITPOSE_RIGHT_HAND_START:]
            for is_right, keyp in enumerate([left_kp, right_kp]):
                valid = keyp[:, 2] > 0.5
                if sum(valid) > 3:
                    conf = keyp[valid, 2].mean()
                    if conf > max_conf:
                        max_conf = conf
                        bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(),
                                keyp[valid, 0].max(), keyp[valid, 1].max()]
                        best_hand = {
                            "bboxes": np.array([bbox]),
                            "is_right": np.array([is_right]),
                            "keypts": np.array([keyp[:, :2]]),
                        }
        return best_hand

    def reconstruct(
        self, img_cv2: np.ndarray, hand_data: dict, focal_length: float
    ) -> dict:
        """🤚 Run HaMeR reconstruction."""
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.utils import recursive_to
        from hamer.utils.renderer import cam_crop_to_full

        dataset = ViTDetDataset(
            self.model_cfg, img_cv2, hand_data["bboxes"], hand_data["is_right"],
            rescale_factor=self.cfg.inference.rescale_factor,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.cfg.inference.batch_size, shuffle=False, num_workers=0
        )

        batch = next(iter(dataloader))
        batch = recursive_to(batch, self.device)
        with torch.no_grad():
            out = self.hamer_model(batch)

        # 📐 Process camera
        multiplier = 2 * batch["right"] - 1
        pred_cam = out["pred_cam"]
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        pred_cam_t = cam_crop_to_full(
            pred_cam, box_center, box_size, img_size, focal_length
        ).detach().cpu().numpy()

        # 📦 Extract single result
        n = 0
        verts = out["pred_vertices"][n].detach().cpu().numpy()
        is_right = batch["right"][n].cpu().numpy()
        verts[:, 0] = (2 * is_right - 1) * verts[:, 0]

        return {
            "vertices": verts,
            "cam_transl": pred_cam_t[n],
            "is_right": bool(is_right),
            "mano_params": {k: v.detach().cpu().numpy().tolist() for k, v in out["pred_mano_params"].items()},
            "img_size": img_size[n].cpu().numpy().tolist(),
        }

    def render_mask(
        self, recon_data: dict, focal_length: float
    ) -> np.ndarray:
        """🎨 Render hand mask."""
        LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
        img_size = recon_data["img_size"]
        cam_view = self.renderer.render_rgba_multiple(
            [recon_data["vertices"]],
            cam_t=[recon_data["cam_transl"]],
            render_res=img_size,
            is_right=[recon_data["is_right"]],
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            focal_length=focal_length,
        )
        # 🎭 Binary mask: 255 where no hand, 0 where hand
        hand_mask = np.ones((int(img_size[1]), int(img_size[0]), 3), dtype=np.uint8) * 255
        hand_mask[cam_view[:, :, 3] > 0] = 0
        return hand_mask


# ══════════════════════════════════════════════════════════════════════════════
# 🌐 API
# ══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    image_path: str
    focal_length: float


class PredictResponse(BaseModel):
    status: str  # success | warning | error
    message: str
    mano_params: dict = {}
    vertices_b64: str = ""  # base64 numpy (778, 3)
    cam_transl: list[float] = []
    is_right: bool = False
    mask_b64: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# 🔧 Helpers
# ══════════════════════════════════════════════════════════════════════════════

def encode_array_b64(arr: np.ndarray) -> str:
    """Convert numpy array to base64 string."""
    buffer = io.BytesIO()
    np.save(buffer, arr)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_image_b64(img: np.ndarray) -> str:
    """Convert image to base64 PNG string."""
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# 🚀 Server
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="🤚 HaMeR Hand Reconstruction Server")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    """🤚 Reconstruct hand mesh from image."""
    model = request.app.state.model
    print(f"\n{'='*60}")
    print(f"📨 New request: {req.image_path}")

    image_path = Path(req.image_path)
    if not image_path.exists():
        return PredictResponse(status="error", message=f"Image not found: {req.image_path}")

    img_cv2 = cv2.imread(str(image_path))
    if img_cv2 is None:
        return PredictResponse(status="error", message=f"Failed to read image: {req.image_path}")

    # 🔍 Detect hand
    hand_data = model.detect_best_hand(img_cv2)
    if hand_data is None:
        return PredictResponse(status="warning", message="No hands detected")

    # 🤚 Reconstruct
    print(f"🤚 Reconstructing with focal_length={req.focal_length:.2f}")
    recon_data = model.reconstruct(img_cv2, hand_data, req.focal_length)

    # 🎨 Render mask
    mask = model.render_mask(recon_data, req.focal_length)
    mask_b64 = encode_image_b64(mask)

    # 📦 Encode vertices
    vertices_b64 = encode_array_b64(recon_data["vertices"])

    print(f"🎉 Done! is_right={recon_data['is_right']}")
    return PredictResponse(
        status="success",
        message="Hand reconstructed successfully",
        mano_params=recon_data["mano_params"],
        vertices_b64=vertices_b64,
        cam_transl=recon_data["cam_transl"].tolist(),
        is_right=recon_data["is_right"],
        mask_b64=mask_b64,
    )


@hydra.main(config_path="../../cfg/model", config_name="hamer", version_base=None)
def main(cfg: DictConfig) -> None:
    """🚀 Start HaMeR server with Hydra config."""
    app.state.model = HaMeRModel.from_config(cfg)
    print(f"🌐 Server starting at http://{cfg.server.host}:{cfg.server.port}")
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)


if __name__ == "__main__":
    main()
