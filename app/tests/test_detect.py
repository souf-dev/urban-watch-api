"""
Tests for POST /detect-anomalies
"""

from __future__ import annotations

import io


# ──────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────

def test_detect_returns_success(client, small_image_bytes):
    """Valid image → 200 with detections list and final_image path."""
    response = client.post(
        "/detect-anomalies",
        files={"image": ("test.jpg", io.BytesIO(small_image_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert isinstance(body["detections"], list)
    assert body["final_image"].startswith("outputs/")


def test_detect_detections_have_required_fields(client, small_image_bytes):
    """Each detection must include label, confidence, box, and display color."""
    response = client.post(
        "/detect-anomalies",
        files={"image": ("test.jpg", io.BytesIO(small_image_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    for det in response.json()["detections"]:
        assert "label" in det
        assert "confidence" in det
        assert "box" in det
        assert "color" in det


def test_detect_box_has_four_coords(client, small_image_bytes):
    """Bounding box must be a list of exactly 4 numbers."""
    response = client.post(
        "/detect-anomalies",
        files={"image": ("test.jpg", io.BytesIO(small_image_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    for det in response.json()["detections"]:
        assert len(det["box"]) == 4


def test_detect_confidence_in_range(client, small_image_bytes):
    """Confidence score must be between 0 and 1."""
    response = client.post(
        "/detect-anomalies",
        files={"image": ("test.jpg", io.BytesIO(small_image_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    for det in response.json()["detections"]:
        assert 0.0 <= det["confidence"] <= 1.0


# ──────────────────────────────────────────────
# PNG input
# ──────────────────────────────────────────────

def test_detect_accepts_png(client):
    """API must accept PNG files as well as JPEG."""
    import cv2
    import numpy as np

    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    png_bytes = buf.tobytes()

    response = client.post(
        "/detect-anomalies",
        files={"image": ("test.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["success"] is True


# ──────────────────────────────────────────────
# Error cases
# ──────────────────────────────────────────────

def test_detect_empty_file_returns_400(client):
    """Empty upload → 400."""
    response = client.post(
        "/detect-anomalies",
        files={"image": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_detect_non_image_bytes_returns_400(client, not_an_image_bytes):
    """Undecodable bytes with image/* content-type → 400."""
    response = client.post(
        "/detect-anomalies",
        files={"image": ("bad.jpg", io.BytesIO(not_an_image_bytes), "image/jpeg")},
    )
    assert response.status_code == 400


def test_detect_wrong_content_type_returns_400(client, not_an_image_bytes):
    """Non-image content-type → 400."""
    response = client.post(
        "/detect-anomalies",
        files={"image": ("doc.txt", io.BytesIO(not_an_image_bytes), "text/plain")},
    )
    assert response.status_code == 400
    assert "text/plain" in response.json()["detail"]


def test_detect_oversized_image_returns_413(client, large_image_bytes):
    """Image above MAX_UPLOAD_BYTES → 413."""
    response = client.post(
        "/detect-anomalies",
        files={"image": ("big.bmp", io.BytesIO(large_image_bytes), "image/bmp")},
    )
    assert response.status_code == 413
    assert "large" in response.json()["detail"].lower()


def test_detect_missing_file_returns_422(client):
    """No file in request → 422 Unprocessable Entity (FastAPI validation)."""
    response = client.post("/detect-anomalies")
    assert response.status_code == 422
