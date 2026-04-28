"""
Shared pytest fixtures for Urban Watch API tests.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


# ──────────────────────────────────────────────
# App client
# ──────────────────────────────────────────────

@pytest.fixture(scope="session")
def client() -> TestClient:
    """Reusable synchronous test client (models loaded once)."""
    with TestClient(app) as c:
        yield c


# ──────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────

def _encode_image(array: np.ndarray, ext: str = ".jpg") -> bytes:
    """Encode a NumPy BGR image to bytes."""
    ok, buf = cv2.imencode(ext, array)
    assert ok, "cv2.imencode failed in fixture"
    return buf.tobytes()


@pytest.fixture()
def small_image_bytes() -> bytes:
    """200×200 solid-colour JPEG — the happy-path image."""
    img = np.full((200, 200, 3), fill_value=(100, 149, 237), dtype=np.uint8)
    return _encode_image(img, ".jpg")


@pytest.fixture()
def large_image_bytes() -> bytes:
    """Synthetic image whose encoded size exceeds MAX_UPLOAD_BYTES (20 MB).

    We generate a large random array and encode as BMP (no compression) to
    guarantee the byte count is well above the limit.
    """
    from app.config.settings import MAX_UPLOAD_BYTES

    # BMP of a 3000×3000 image ≈ 27 MB uncompressed — safely over 20 MB
    img = np.random.randint(0, 256, (3000, 3000, 3), dtype=np.uint8)
    return _encode_image(img, ".bmp")


@pytest.fixture()
def not_an_image_bytes() -> bytes:
    """Plain-text bytes that cannot be decoded as an image."""
    return b"this is not an image"