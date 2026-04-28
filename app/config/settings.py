"""
Application settings and default configuration.
"""

import os
from pathlib import Path
from typing import Optional, List


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # urban-watch-api/
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# Default anomaly classes
# ──────────────────────────────────────────────
DEFAULT_ANOMALIES: list[str] = [
    "trash on floor",
    "litter on floor",
    "food waste on floor",
    "liquid spill",
    "oil spill",
    "dirty floor",
    "stain on floor",
    "broken product",
    "damaged package",
    "open package",
    "fallen product",
    "misplaced product",
    "wet floor",
    "dirty shelf",
    "overflowing trash bin",
    "broken shelf",
    "damaged sign",
]


def build_text_prompt(anomalies: Optional[List[str]] = None) -> str:
    """
    Build a Grounding-DINO style text prompt from anomaly labels.

    Each label is separated by '. ' and the string ends with a period.
    Example: "crack. rust. scratch. hole. broken edge. missing screw."
    """
    labels = anomalies or DEFAULT_ANOMALIES
    return ". ".join(labels) + "."


# ──────────────────────────────────────────────
# Cached static prompt (built once at import time)
# ──────────────────────────────────────────────
TEXT_PROMPT: str = build_text_prompt()


# ──────────────────────────────────────────────
# Model settings
# ──────────────────────────────────────────────
GROUNDING_DINO_CONFIG = os.getenv(
    "GROUNDING_DINO_CONFIG",
    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
)
GROUNDING_DINO_CHECKPOINT = os.getenv(
    "GROUNDING_DINO_CHECKPOINT",
    str(PROJECT_ROOT / "checkpoints" / "groundingdino_swint_ogc.pth"),
)
SAM2_CHECKPOINT = os.getenv(
    "SAM2_CHECKPOINT",
    str(PROJECT_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"),
)
SAM2_CONFIG = os.getenv(
    "SAM2_CONFIG",
    "configs/sam2.1/sam2.1_hiera_l.yaml",
)

# Inference thresholds
BOX_THRESHOLD = 0.18
TEXT_THRESHOLD = 0.15

# Upload limits
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

# Device
DEVICE = "cuda"  # falls back to cpu at runtime if unavailable

# Upload limits
MAX_UPLOAD_BYTES: int = 20 * 1024 * 1024  # 20 MB
