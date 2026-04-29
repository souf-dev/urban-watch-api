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
    (0, 255, 0),      # lime
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
    Overlay bounding boxes, labels, and confidence scores on a copy of the
    input image.

    Returns the annotated image (BGR, uint8).
    """
    canvas = image.copy()

    for idx, det in enumerate(detections):
        colour = _colour_for(idx)
        x1, y1, x2, y2 = map(int, det.box)

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

    return canvas


# ──────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────

def save_result_image(image: np.ndarray, prefix: str = "result") -> str:
    """
    Save an annotated image to the outputs directory.

    FIX B5: the filename now includes a short UUID so concurrent requests
    each write their own file instead of overwriting the shared result.png.

    Returns the relative path served by the /outputs static mount.
    """
    unique_id = uuid.uuid4().hex[:12]
    filename = f"{prefix}_{unique_id}.png"
    filepath = OUTPUTS_DIR / filename
    cv2.imwrite(str(filepath), image)
    return f"outputs/{filename}"