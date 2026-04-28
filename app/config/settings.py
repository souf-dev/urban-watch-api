"""
Application settings and default configuration.
"""

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
    "crack",
    "rust",
    "scratch",
    "hole",
    "broken edge",
    "missing screw",
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
GROUNDING_DINO_CONFIG = "GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "groundingdino_swint_ogc.pth"
SAM2_CHECKPOINT = "sam2_hiera_large.pt"
SAM2_CONFIG = "sam2_hiera_l.yaml"

# Inference thresholds
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25

# Device
DEVICE = "cuda"  # falls back to cpu at runtime if unavailable

# Upload limits
MAX_UPLOAD_BYTES: int = 20 * 1024 * 1024  # 20 MB