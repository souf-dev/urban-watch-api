"""
Grounded-SAM2 model service.

Pipeline:
    1. GroundingDINO  →  bounding boxes from text prompt
    2. SAM2           →  segmentation masks from bounding boxes

If the real model weights are not available, the service falls back to a
deterministic placeholder that returns synthetic detections so the rest
of the API can be developed and tested end-to-end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.config.settings import (
    BOX_THRESHOLD,
    DEVICE,
    GROUNDING_DINO_CHECKPOINT,
    GROUNDING_DINO_CONFIG,
    PROJECT_ROOT,
    SAM2_CHECKPOINT,
    SAM2_CONFIG,
    TEXT_PROMPT,
    TEXT_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────
@dataclass
class Detection:
    """Single detected anomaly."""

    label: str
    confidence: float
    box: list[float]           # [x1, y1, x2, y2]  — pixel coords
    mask: np.ndarray | None    # H×W binary mask (uint8, 0/255)


@dataclass
class InferenceResult:
    """Full result from one inference call."""

    detections: list[Detection] = field(default_factory=list)


# ──────────────────────────────────────────────
# Service
# ──────────────────────────────────────────────
class ModelService:
    """
    Singleton-style service that loads models once and exposes
    `run_inference(image)`.

    Inference is intentionally synchronous — callers are responsible for
    dispatching to a thread-pool executor so the async event loop is never
    blocked (see routes/detect.py).
    """

    def __init__(self) -> None:
        self._device: str = DEVICE if torch.cuda.is_available() else "cpu"
        self._grounding_model: Any | None = None
        self._sam2_predictor: Any | None = None
        self._is_loaded = False

        logger.info("ModelService created — device=%s", self._device)

    # ── Model loading ────────────────────────
    def load_models(self) -> None:
        """
        Load GroundingDINO + SAM2 weights.
        """
        from groundingdino.util.inference import load_model as load_grounding_model
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        grounding_config = self._resolve_path(
            GROUNDING_DINO_CONFIG, package_name="groundingdino"
        )
        grounding_checkpoint = self._resolve_path(GROUNDING_DINO_CHECKPOINT)
        sam2_checkpoint = self._resolve_path(SAM2_CHECKPOINT)

        self._require_file(grounding_config, "GroundingDINO config")
        self._require_file(grounding_checkpoint, "GroundingDINO checkpoint")
        self._require_file(sam2_checkpoint, "SAM2 checkpoint")

        logger.info("Loading GroundingDINO from %s", grounding_checkpoint)
        self._grounding_model = load_grounding_model(
            str(grounding_config),
            str(grounding_checkpoint),
            device=self._device,
        )

        logger.info("Loading SAM2 from %s", sam2_checkpoint)
        sam2_model = build_sam2(
            SAM2_CONFIG,
            str(sam2_checkpoint),
            device=self._device,
        )
        self._sam2_predictor = SAM2ImagePredictor(sam2_model)
        self._is_loaded = True

    # ── Inference ────────────────────────────
    def run_inference(self, image: np.ndarray) -> InferenceResult:
        """
        Run the full Grounded-SAM2 pipeline on an image.

        This method is synchronous and CPU/GPU-bound. Always call it from a
        thread-pool executor in async contexts:

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, model_service.run_inference, img)

        Parameters
        ----------
        image : np.ndarray
            BGR image (H, W, 3) — as returned by cv2.imread / imdecode.

        Returns
        -------
        InferenceResult
            Detections with boxes, masks, labels, and confidence scores.
        """
        if not self._is_loaded:
            self.load_models()

        # TEXT_PROMPT is a module-level constant built once at import time.
        logger.info("Running inference — prompt: %s", TEXT_PROMPT)

        return self._run_real_inference(image)

    # ── Real pipeline (template) ─────────────
    def _run_real_inference(self, image: np.ndarray) -> InferenceResult:
        """
        Run GroundingDINO for boxes, then SAM2 for masks.
        """
        import cv2
        import groundingdino.datasets.transforms as T
        from groundingdino.util.inference import predict
        from PIL import Image
        from torchvision.ops import box_convert

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_tensor, _ = transform(Image.fromarray(image_rgb), None)

        with torch.inference_mode():
            boxes, scores, labels = predict(
                model=self._grounding_model,
                image=image_tensor,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=self._device,
            )

        if len(boxes) == 0:
            return InferenceResult(detections=[])

        boxes_xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        scale = torch.tensor(
            [width, height, width, height],
            dtype=boxes_xyxy.dtype,
            device=boxes_xyxy.device,
        )
        boxes_xyxy = boxes_xyxy * scale
        boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp(0, width - 1)
        boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp(0, height - 1)
        boxes_np = boxes_xyxy.cpu().numpy()

        sam2_predictor = self._sam2_predictor
        if sam2_predictor is None:
            raise RuntimeError("SAM2 predictor was not loaded.")

        sam2_predictor.set_image(image_rgb)
        with torch.inference_mode():
            masks, _, _ = sam2_predictor.predict(
                box=boxes_np,
                multimask_output=False,
            )

        detections: list[Detection] = []
        for idx, box in enumerate(boxes_np):
            mask = masks[idx]
            if mask.ndim == 3:
                mask = mask[0]

            detections.append(
                Detection(
                    label=labels[idx],
                    confidence=float(scores[idx]),
                    box=[float(value) for value in box.tolist()],
                    mask=(mask.astype(np.uint8) * 255),
                )
            )

        return InferenceResult(detections=detections)

    @staticmethod
    def _resolve_path(value: str, package_name: str | None = None) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path

        project_path = PROJECT_ROOT / path
        if project_path.exists():
            return project_path

        if package_name:
            spec = find_spec(package_name)
            if spec and spec.origin:
                package_root = Path(spec.origin).resolve().parent
                package_path = package_root.parent / path
                if package_path.exists():
                    return package_path

                if path.parts and path.parts[0] == package_name:
                    stripped_path = package_root.joinpath(*path.parts[1:])
                    if stripped_path.exists():
                        return stripped_path

        return project_path

    @staticmethod
    def _require_file(path: Path, label: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} not found: {path}. Download the file or set the "
                "matching environment variable."
            )

    # ── Placeholder pipeline ─────────────────
    @staticmethod
    def _run_placeholder_inference(image: np.ndarray) -> InferenceResult:
        """
        Returns 2-3 synthetic detections that look realistic enough to
        test the rest of the API (routing, image overlay, response schema).
        """
        h, w = image.shape[:2]
        rng = np.random.default_rng(seed=42)

        sample_detections = [
            ("crack", 0.87),
            ("rust", 0.74),
            ("scratch", 0.61),
        ]

        detections: list[Detection] = []
        for i, (label, confidence) in enumerate(sample_detections):
            # generate a plausible box in the image
            x1 = rng.integers(0, w // 2)
            y1 = rng.integers(0, h // 2)
            x2 = rng.integers(w // 2, w)
            y2 = rng.integers(h // 2, h)

            # generate a simple elliptical mask inside the box
            mask = np.zeros((h, w), dtype=np.uint8)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            rx, ry = (x2 - x1) // 3, (y2 - y1) // 3
            yy, xx = np.ogrid[:h, :w]
            ellipse = ((xx - cx) ** 2) / max(rx ** 2, 1) + (
                (yy - cy) ** 2
            ) / max(ry ** 2, 1)
            mask[ellipse <= 1] = 255

            detections.append(
                Detection(
                    label=label,
                    confidence=round(confidence, 2),
                    box=[x1, y1, x2, y2],
                    mask=mask,
                )
            )

        return InferenceResult(detections=detections)


# ──────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────
model_service = ModelService()
