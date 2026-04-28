"""
POST /detect-anomalies

Accepts an uploaded image, runs the Grounded-SAM2 pipeline, and returns
detected anomalies with their bounding boxes, masks, and a final annotated
image.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.model_service import model_service
from app.utils.image_utils import draw_detections, save_mask, save_result_image

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/detect-anomalies")
async def detect_anomalies(image: UploadFile = File(...)):
    """
    Detect surface anomalies in the uploaded image.

    **Input**: multipart/form-data with an `image` field.

    **Returns**: JSON with detection results and paths to output images.
    """
    # ── 1. Validate & decode the uploaded image ──────────────────
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got {image.content_type}",
        )

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    np_arr = np.frombuffer(contents, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded file as an image.",
        )

    logger.info(
        "Received image: %s  (%dx%d)",
        image.filename,
        img_bgr.shape[1],
        img_bgr.shape[0],
    )

    # ── 2. Run Grounded-SAM2 inference ───────────────────────────
    result = model_service.run_inference(img_bgr)

    # ── 3. Draw detections & save outputs ────────────────────────
    annotated = draw_detections(img_bgr, result.detections)
    final_image_path = save_result_image(annotated)

    detections_payload: list[dict] = []
    for idx, det in enumerate(result.detections):
        mask_path = save_mask(det.mask, idx) if det.mask is not None else None
        detections_payload.append(
            {
                "label": det.label,
                "confidence": det.confidence,
                "box": det.box,
                "mask_path": mask_path,
            }
        )

    # ── 4. Respond ───────────────────────────────────────────────
    return {
        "success": True,
        "detections": detections_payload,
        "final_image": final_image_path,
    }
