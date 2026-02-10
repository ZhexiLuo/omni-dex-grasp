"""🎯 Grounded-SAM2 segmentation server.

POST /predict - Detect and segment objects/hands in image.

Usage: python -m recons.server.server_gsam2
"""
import os

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import hydra
import numpy as np
import pycocotools.mask as mask_util
import supervision as sv
import torch
import uvicorn
from fastapi import FastAPI, Request
from omegaconf import DictConfig
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


# ══════════════════════════════════════════════════════════════════════════════
# 🧠 Model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GSAM2Model:
    """🎯 Grounded-SAM2 model wrapper."""
    sam2_predictor: "SAM2ImagePredictor" 
    grounding_processor: AutoProcessor
    grounding_model: AutoModelForZeroShotObjectDetection
    device: str
    cfg: DictConfig

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "GSAM2Model":
        """🚀 Load models from config."""

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading Grounded-SAM2 models on {device}...")

        print("  🎭 Loading SAM2...")
        sam2_model = build_sam2(cfg.model.sam2_config, cfg.model.sam2_checkpoint, device=device)
        print("  🦖 Loading Grounding DINO...")
        processor = AutoProcessor.from_pretrained(cfg.model.grounding_dino_path)
        grounding = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.model.grounding_dino_path).to(device)

        print("✅ All models loaded!")
        return cls(SAM2ImagePredictor(sam2_model), processor, grounding, device, cfg)

    def detect(self, image: Image.Image, text_prompt: str) -> list[dict]:
        """🦖 Run Grounding DINO detection."""
        print(f"🦖 Detecting with prompt: '{text_prompt}'")
        inputs = self.grounding_processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.grounding_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=self.cfg.inference.box_threshold,
            text_threshold=self.cfg.inference.text_threshold,
            target_sizes=[image.size[::-1]],
        )

        if not results or len(results[0]["scores"]) == 0:
            print("  ⚠️ No detections found")
            return []

        r = results[0]
        dets = [
            {"score": float(s), "label": l, "box": b.cpu().numpy(), "is_hand": "hand" in l.lower()}
            for s, l, b in zip(r["scores"], r["labels"], r["boxes"])
        ]
        print(f"  📦 Found {len(dets)} raw detections")
        return dets

    def segment(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """🎭 Run SAM2 segmentation."""
        print(f"🎭 Segmenting {len(boxes)} regions...")
        self.sam2_predictor.set_image(image)
        masks, _, _ = self.sam2_predictor.predict(box=boxes, multimask_output=False)
        return masks.squeeze(1) if masks.ndim == 4 else masks

    def filter_detections(self, dets: list[dict], include_hand: bool) -> list[dict]:
        """🤚 Filter detections by confidence threshold."""
        filtered = []
        for d in dets:
            threshold = self.cfg.inference.hand_confidence_threshold if d["is_hand"] else self.cfg.inference.obj_confidence_threshold
            if d["score"] >= threshold and (include_hand or not d["is_hand"]):
                filtered.append(d)
        print(f"  🔍 Filtered: {len(dets)} → {len(filtered)} detections")
        return filtered


# ══════════════════════════════════════════════════════════════════════════════
# 🌐 API
# ══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    image_path: str
    text_prompt: str
    include_hand: bool = True


class Detection(BaseModel):
    class_name: str
    bbox: list[float]
    score: float
    mask_rle: dict
    is_hand: bool


class PredictResponse(BaseModel):
    status: str
    message: str
    detections: list[Detection] = []
    img_size: list[int] = []
    annotated_image_b64: str = ""  # 🖼️ base64 detection visualization
    mask_image_b64: str = ""  # 🎨 base64 combined mask


# ══════════════════════════════════════════════════════════════════════════════
# 🔧 Helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_response_detections(dets: list[dict], masks: np.ndarray) -> list[Detection]:
    """📦 Convert to API response format with RLE masks."""
    result = []
    for det, mask in zip(dets, masks):
        rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        result.append(Detection(
            class_name=det["label"],
            bbox=det["box"].tolist(),
            score=det["score"],
            mask_rle=rle,
            is_hand=det["is_hand"],
        ))
    return result


def generate_visuals(
    image_rgb: np.ndarray, masks: np.ndarray, dets: list[dict], cfg: DictConfig
) -> tuple[str, str]:
    """🎨 Generate base64 encoded visualization images.

    Returns:
        annotated_b64: Detection visualization (JPEG base64)
        mask_b64: Combined mask (PNG base64)
    """
    h, w = masks.shape[1], masks.shape[2]
    print(f"🎨 Generating visualizations...")

    # 🖼️ Detection visualization
    sv_dets = sv.Detections(
        xyxy=np.array([d["box"] for d in dets]),
        mask=masks.astype(bool),
        class_id=np.arange(len(dets)),
    )
    labels = [f"{d['label']} {d['score']:.2f}" for d in dets]
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    annotated = sv.BoxAnnotator().annotate(scene=img_bgr.copy(), detections=sv_dets)
    annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=sv_dets, labels=labels)
    annotated = sv.MaskAnnotator().annotate(scene=annotated, detections=sv_dets)

    _, annotated_buffer = cv2.imencode(".jpg", annotated)
    annotated_b64 = base64.b64encode(annotated_buffer).decode("utf-8")

    # 🎨 Combined mask with colors
    hand_mask = np.zeros((h, w), dtype=bool)
    obj_mask = np.zeros((h, w), dtype=bool)
    for mask, det in zip(masks, dets):
        if det["is_hand"]:
            hand_mask |= mask.astype(bool)
        else:
            obj_mask |= mask.astype(bool)

    combined = np.zeros((h, w, 3), dtype=np.uint8)
    combined[obj_mask & ~hand_mask] = cfg.visualization.obj_color
    combined[hand_mask] = cfg.visualization.hand_color
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

    _, mask_buffer = cv2.imencode(".png", combined_bgr)
    mask_b64 = base64.b64encode(mask_buffer).decode("utf-8")

    return annotated_b64, mask_b64


# ══════════════════════════════════════════════════════════════════════════════
# 🚀 Server
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="🎯 Grounded-SAM2 Server")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    """🔍 Detect and segment objects/hands."""
    model = request.app.state.model
    print(f"\n{'='*60}")
    print(f"📨 New request: {req.image_path}")
    image_path = Path(req.image_path)
    image = Image.open(image_path).convert("RGB")

    text_prompt = req.text_prompt
    if req.include_hand:
        text_prompt = f"{text_prompt.rstrip('. ')}. hand."
        print(f"  🤚 Auto-appended 'hand.' to prompt: '{text_prompt}'")

    dets = model.detect(image, text_prompt)
    dets = model.filter_detections(dets, req.include_hand)
    if not dets:
        return PredictResponse(status="warning", message="No detections above threshold")

    boxes = np.array([d["box"] for d in dets])
    masks = model.segment(np.array(image), boxes)

    h, w = masks.shape[1], masks.shape[2]
    detections = build_response_detections(dets, masks)
    annotated_b64, mask_b64 = generate_visuals(np.array(image), masks, dets, model.cfg)
    print(f"🎉 Done! {len(detections)} objects detected")

    return PredictResponse(
        status="success",
        message=f"Detected {len(detections)} objects",
        detections=detections,
        img_size=[h, w],
        annotated_image_b64=annotated_b64,
        mask_image_b64=mask_b64,
    )


@hydra.main(config_path="../../cfg/model", config_name="gsam", version_base=None)
def main(cfg: DictConfig) -> None:
    """🚀 Start server with Hydra config."""
    app.state.model = GSAM2Model.from_config(cfg)
    print(f"🌐 Server starting at http://{cfg.server.host}:{cfg.server.port}")
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)


if __name__ == "__main__":
    main()
