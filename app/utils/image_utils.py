"""
Image utilities — draw detections on images and persist results.
"""

from __future__ import annotations

import uuid

import cv2
import numpy as np

from app.config.settings import OUTPUTS_DIR
from app.services.model_service import Detection

# Colour palette — one per anomaly class (BGR)
_COLOURS: list[tuple[int, int, int]] = [
    (0, 0, 255),      # red
    (0, 165, 255),    # orange
    (83, 200, 0),     # green (#00C853)
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (255, 0, 255),    # magenta
]


def _colour_for(index: int) -> tuple[int, int, int]:
    return _COLOURS[index % len(_COLOURS)]


def colour_hex_for(index: int) -> str:
    """Return the display hex colour matching the BGR annotation palette."""
    blue, green, red = _colour_for(index)
    return f"#{red:02X}{green:02X}{blue:02X}"


# ──────────────────────────────────────────────
# Draw detections
# ──────────────────────────────────────────────

def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    """
    Overlay bounding boxes, labels, confidence scores, and semi-transparent
    masks on a copy of the input image.

    Uses a single canvas + overlay pair. The overlay is blended once after
    all masks are applied, avoiding redundant full-image copies per detection.

    Returns the annotated image (BGR, uint8).
    """
    canvas = image.copy()
    overlay = image.copy()

    for idx, det in enumerate(detections):
        colour = _colour_for(idx)
        x1, y1, x2, y2 = map(int, det.box)

        # ── Mask (semi-transparent fill, applied to overlay) ──
        if det.mask is not None:
            mask_bool = det.mask.astype(bool)
            overlay[mask_bool] = colour

        # ── Bounding box ──
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)

        # ── Label + confidence ──
        label_text = f"{det.label} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        cv2.rectangle(
            canvas, (x1, y1 - th - 10), (x1 + tw + 6, y1), colour, -1
        )
        cv2.putText(
            canvas,
            label_text,
            (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Blend mask overlay at 40 % opacity — single pass for all detections
    cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
    return canvas


# ──────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────

def save_result_image(image: np.ndarray, prefix: str = "result") -> str:
    """Save an annotated image to the outputs directory. Returns the relative path."""
    # Full UUID (32 hex chars) eliminates collision risk under concurrent load
    filename = f"{prefix}_{uuid.uuid4().hex}.png"
    filepath = OUTPUTS_DIR / filename
    cv2.imwrite(str(filepath), image)
    return f"outputs/{filename}"
