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

import numpy as np
import torch

from app.config.settings import (
    BOX_THRESHOLD,
    DEVICE,
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
        self._grounding_model = None
        self._sam2_predictor = None
        self._is_loaded = False

        logger.info("ModelService created — device=%s", self._device)

    # ── Model loading ────────────────────────
    def load_models(self) -> None:
        """
        Load GroundingDINO + SAM2 weights.

        TODO: Replace the placeholder below with real model loading once
        checkpoints are downloaded.  Example (pseudocode):

            from groundingdino.util.inference import load_model as load_gd
            self._grounding_model = load_gd(
                GROUNDING_DINO_CONFIG, GROUNDING_DINO_CHECKPOINT
            )

            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=self._device)
            self._sam2_predictor = SAM2ImagePredictor(sam2_model)

        Performance tip: load both models in torch.float16 on CUDA:
            sam2_model = build_sam2(..., device=self._device)
            sam2_model.half()
        """
        logger.warning(
            "⚠️  Real model weights not loaded — using placeholder inference. "
            "Plug in real checkpoints to enable actual detection."
        )
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

        # ----- Real pipeline (uncomment when weights are available) -----
        # return self._run_real_inference(image)

        # ----- Placeholder pipeline -----
        return self._run_placeholder_inference(image)

    # ── Real pipeline (template) ─────────────
    def _run_real_inference(self, image: np.ndarray) -> InferenceResult:
        """
        Template for the real Grounded-SAM2 pipeline.

        Step 1 — GroundingDINO:
            boxes, scores, labels = predict(
                model=self._grounding_model,
                image=image_transformed,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )

        Step 2 — SAM2 (pass all boxes at once for best throughput):
            self._sam2_predictor.set_image(image_rgb)
            masks, _, _ = self._sam2_predictor.predict(
                box=boxes_xyxy,          # shape (N, 4) — batch all at once
                multimask_output=False,
            )

        Performance tips:
            - Keep both models in float16 on CUDA.
            - Use torch.compile() on the GroundingDINO forward pass (PyTorch 2+).
            - Pass all boxes to SAM2 in one call, never loop.
        """
        raise NotImplementedError(
            "Real model inference is not yet wired — "
            "download checkpoints and uncomment _run_real_inference."
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
            x1 = int(rng.integers(0, w // 2))
            y1 = int(rng.integers(0, h // 2))
            x2 = int(rng.integers(w // 2, w))
            y2 = int(rng.integers(h // 2, h))

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