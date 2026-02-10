"""🌊 Depth-Anything-3 metric depth estimation server.

POST /predict - Estimate metric depth from single RGB image.

Usage: conda activate da3 && python -m recons.server.da3
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

from depth_anything_3.api import DepthAnything3


# ══════════════════════════════════════════════════════════════════════════════
# 🧠 Model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DA3Model:
    """🌊 Depth-Anything-3 model wrapper."""
    model: DepthAnything3
    device: str
    cfg: DictConfig

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "DA3Model":
        """🚀 Load DA3 model from config."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading Depth-Anything-3 on {device}...")

        model = DepthAnything3.from_pretrained(cfg.model.pretrained).to(device)
        print(f"✅ DA3 model loaded: {cfg.model.pretrained}")
        return cls(model=model, device=device, cfg=cfg)

    def predict_depth(self, image_path: str, intrinsics: np.ndarray) -> dict:
        """🌊 Run metric depth estimation on a single image.

        Uses original image resolution for inference, then resizes back.
        """
        img = Image.open(image_path)
        original_h, original_w = img.height, img.width
        process_res = max(original_h, original_w)

        prediction = self.model.inference(
            image=[image_path],
            intrinsics=intrinsics[None, ...],  # (1, 3, 3)
            process_res=process_res,
        )
        depth = prediction.depth[0]  # (H, W) float32
        conf = prediction.conf[0] if prediction.conf is not None else None

        # 📐 Resize to original resolution (DA3 rounds internally to multiples of 14)
        if depth.shape != (original_h, original_w):
            depth = cv2.resize(depth, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            if conf is not None:
                conf = cv2.resize(conf, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        return {
            "depth": depth.astype(np.float32),
            "conf": conf.astype(np.float32) if conf is not None else None,
            "is_metric": bool(prediction.is_metric),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 🌐 API
# ══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    image_path: str
    intrinsics: list[list[float]]  # 3x3 camera intrinsics matrix


class PredictResponse(BaseModel):
    status: str
    message: str
    depth_b64: str = ""      # 📏 base64 encoded depth npy (H,W) float32, meters
    conf_b64: str = ""       # 📊 base64 encoded confidence npy (H,W) float32
    is_metric: bool = False
    depth_vis_b64: str = ""  # 🎨 base64 encoded colormap visualization


# ══════════════════════════════════════════════════════════════════════════════
# 🔧 Helpers
# ══════════════════════════════════════════════════════════════════════════════

def encode_array_b64(arr: np.ndarray) -> str:
    """📐 Encode numpy array to base64 npy string."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_depth_vis(depth: np.ndarray) -> np.ndarray:
    """🎨 Generate colorized depth visualization."""
    d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
    d_uint8 = d_norm.astype(np.uint8)
    return cv2.applyColorMap(d_uint8, cv2.COLORMAP_INFERNO)


# ══════════════════════════════════════════════════════════════════════════════
# 🚀 Server
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="🌊 Depth-Anything-3 Server")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    """🌊 Estimate metric depth from RGB image."""
    model: DA3Model = request.app.state.model
    print(f"\n{'='*60}")
    print(f"📨 New request: {req.image_path}")

    image_path = Path(req.image_path)
    if not image_path.exists():
        return PredictResponse(status="error", message=f"Image not found: {req.image_path}")

    K = np.array(req.intrinsics, dtype=np.float32)
    if K.shape != (3, 3):
        return PredictResponse(status="error", message=f"Intrinsics must be 3x3, got {K.shape}")

    result = model.predict_depth(str(image_path), K)

    # 📦 Encode outputs to base64
    depth_b64 = encode_array_b64(result["depth"])
    conf_b64 = encode_array_b64(result["conf"]) if result["conf"] is not None else ""

    # 🎨 Depth visualization
    vis = generate_depth_vis(result["depth"])
    _, vis_buffer = cv2.imencode(".png", vis)
    depth_vis_b64 = base64.b64encode(vis_buffer).decode("utf-8")

    h, w = result["depth"].shape
    print(f"  📏 Depth shape: ({h}, {w}), metric: {result['is_metric']}")
    print(f"  📊 Depth range: [{result['depth'].min():.3f}, {result['depth'].max():.3f}] m")
    print(f"🎉 Done!")

    return PredictResponse(
        status="success",
        message=f"Depth estimated ({h}x{w}), metric={result['is_metric']}",
        depth_b64=depth_b64,
        conf_b64=conf_b64,
        is_metric=result["is_metric"],
        depth_vis_b64=depth_vis_b64,
    )


@hydra.main(config_path="../../cfg/model", config_name="da3", version_base=None)
def main(cfg: DictConfig) -> None:
    """🚀 Start DA3 server with Hydra config."""
    app.state.model = DA3Model.from_config(cfg)
    print(f"🌐 Server starting at http://{cfg.server.host}:{cfg.server.port}")
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)


if __name__ == "__main__":
    main()
