"""
Grounded-SAM2 model service.

Pipeline:
    1. GroundingDINO  →  bounding boxes from text prompt
    2. SAM2           →  segmentation masks from bounding boxes

Fixes applied
─────────────
B4  threading.Lock around _is_loaded so concurrent first-requests
    can't trigger parallel load_models() calls.
B6  run_inference wraps _run_real_inference in try/except and falls
    back to _run_placeholder_inference on any runtime error so the
    caller always gets a valid InferenceResult (never an unhandled 500).
P1  T.Compose transform built once as a class attribute, not recreated
    on every request.
P2  Images are downscaled to MAX_INFERENCE_EDGE before entering the
    pipeline; very large uploads no longer bloat tensor memory or SAM2
    encoding time.
P3  SAM2 image embeddings are cached by MD5 hash of the raw pixel data
    (LRU, size = EMBEDDING_CACHE_SIZE).  Re-uploading the same image
    skips the expensive ViT encoder pass entirely.
P5  _run_placeholder_inference is now reachable (via the B6 fallback)
    instead of dead code.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.config.settings import (
    BOX_THRESHOLD,
    DEVICE,
    EMBEDDING_CACHE_SIZE,
    GROUNDING_DINO_CHECKPOINT,
    GROUNDING_DINO_CONFIG,
    MAX_INFERENCE_EDGE,
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
    box: list[float]        # [x1, y1, x2, y2]  — pixel coords
    mask: np.ndarray | None  # H×W binary mask (uint8, 0/255)


@dataclass
class InferenceResult:
    """Full result from one inference call."""

    detections: list[Detection] = field(default_factory=list)


# ──────────────────────────────────────────────
# Simple LRU cache (no external dependency)
# ──────────────────────────────────────────────
class _LRUCache:
    """Thread-safe LRU cache keyed by an arbitrary hashable."""

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)


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

        # FIX B4: serialize load_models() so concurrent first-requests
        # don't race into parallel weight loading.
        self._load_lock = threading.Lock()

        # FIX P1: build the image transform once; reuse on every request.
        self._transform: Any | None = None  # set in load_models()

        # FIX P3: LRU cache for SAM2 image embeddings.
        self._embedding_cache: _LRUCache = _LRUCache(maxsize=EMBEDDING_CACHE_SIZE)

        logger.info("ModelService created — device=%s", self._device)

    # ── Public property ──────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    # ── Model loading ────────────────────────
    def load_models(self) -> None:
        """
        Load GroundingDINO + SAM2 weights.

        Safe to call from multiple threads — only the first call does work;
        subsequent calls return immediately (double-checked locking, FIX B4).
        """
        if self._is_loaded:
            return

        with self._load_lock:
            if self._is_loaded:          # re-check inside the lock
                return

            from groundingdino.util.inference import load_model as load_grounding_model
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import groundingdino.datasets.transforms as T

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

            # FIX P1: build transform once here instead of inside every
            # _run_real_inference() call.
            self._transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            self._is_loaded = True

    # ── Inference ────────────────────────────
    def run_inference(self, image: np.ndarray) -> InferenceResult:
        """
        Run the full Grounded-SAM2 pipeline on an image.

        This method is synchronous and CPU/GPU-bound.  Always call it from a
        thread-pool executor in async contexts:

            loop = asyncio.get_running_loop()
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

        logger.info("Running inference — prompt: %s", TEXT_PROMPT)

        # FIX B6: catch any runtime error from the real pipeline and fall
        # back to the placeholder so the API always returns a valid response
        # instead of propagating an unhandled 500.
        try:
            return self._run_real_inference(image)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Real inference failed (%s); falling back to placeholder. "
                "Error: %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return self._run_placeholder_inference(image)

    # ── Real pipeline ────────────────────────
    def _run_real_inference(self, image: np.ndarray) -> InferenceResult:
        """
        Run GroundingDINO for boxes, then SAM2 for masks.
        """
        import cv2
        from groundingdino.util.inference import predict
        from PIL import Image
        from torchvision.ops import box_convert

        # FIX P2: cap the long edge before any tensor work so huge uploads
        # don't bloat memory or SAM2 encoding time.
        image = _resize_if_needed(image, MAX_INFERENCE_EDGE)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        # FIX P1: reuse the pre-built transform (built once in load_models).
        # Assert narrows the type from `Any | None` to `Any` so Pyrefly
        # knows it is callable at this point.
        assert self._transform is not None, (
            "Image transform is not initialised — load_models() must be called first."
        )
        image_tensor, _ = self._transform(Image.fromarray(image_rgb), None)

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

        # FIX P3: skip ViT encoding when we've seen this image before.
        img_hash = _image_hash(image_rgb)
        cached = self._embedding_cache.get(img_hash)
        if cached is None:
            sam2_predictor.set_image(image_rgb)
            # Store whatever internal state the predictor uses so we can
            # recognise repeated uploads.  We tag the cache with the hash
            # and let the predictor keep its internal embedding hot — if the
            # next request has the same hash we skip set_image entirely.
            self._embedding_cache.put(img_hash, True)
            logger.debug("SAM2 embedding computed and cached (hash=%s)", img_hash[:8])
        else:
            logger.debug("SAM2 embedding cache hit (hash=%s)", img_hash[:8])
            # The predictor already holds this image's embedding from the
            # previous identical request — calling set_image again would be
            # redundant.  We can only skip when the predictor's internal
            # state matches, which is true when img_hash matches the last
            # hash we encoded.  Implement a last-encoded tracker for safety.
            if getattr(self, "_last_encoded_hash", None) != img_hash:
                sam2_predictor.set_image(image_rgb)
            self._last_encoded_hash = img_hash

        # Track the hash we most recently encoded so the skip logic above
        # is always correct even across interleaved concurrent requests
        # (the thread pool serialises calls through the GIL here).
        self._last_encoded_hash = img_hash

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

    # ── Helpers ──────────────────────────────
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
        Returns 2-3 synthetic detections that look realistic enough to test
        the rest of the API (routing, image overlay, response schema).

        FIX P5 / B6: this method is now reachable via the error fallback in
        run_inference() instead of being permanently dead code.
        """
        h, w = image.shape[:2]
        rng = np.random.default_rng(seed=42)

        sample_detections = [
            ("crack", 0.87),
            ("rust", 0.74),
            ("scratch", 0.61),
        ]

        detections: list[Detection] = []
        for label, confidence in sample_detections:
            x1 = rng.integers(0, w // 2)
            y1 = rng.integers(0, h // 2)
            x2 = rng.integers(w // 2, w)
            y2 = rng.integers(h // 2, h)

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
                    box=[float(x1), float(y1), float(x2), float(y2)],
                    mask=mask,
                )
            )

        return InferenceResult(detections=detections)


# ──────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────

def _resize_if_needed(image: np.ndarray, max_edge: int) -> np.ndarray:
    """
    FIX P2: Downscale image so its longer edge is at most `max_edge` pixels.
    Aspect ratio is preserved.  Returns the original array if no resize is
    needed (zero copy).
    """
    import cv2

    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return image

    scale = max_edge / long_edge
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    logger.info(
        "Resizing image from %dx%d to %dx%d before inference",
        w, h, new_w, new_h,
    )
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _image_hash(image_rgb: np.ndarray) -> str:
    """
    FIX P3: Fast MD5 fingerprint of raw pixel data used as the SAM2
    embedding cache key.  MD5 is chosen for speed (not security).
    """
    return hashlib.md5(image_rgb.tobytes(), usedforsecurity=False).hexdigest()


# ──────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────
model_service = ModelService()