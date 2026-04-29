"""
Urban Watch API — Anomaly Detection Service

FastAPI application entry-point.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config.settings import OUTPUTS_DIR
from app.routes.detect import router as detect_router
from app.services.model_service import model_service

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models on startup; clean up on shutdown.

    FIX B2: load_models() failure no longer crashes the whole process.
    If weights are missing the server starts in degraded mode and every
    inference request returns HTTP 503 with a clear message instead of
    an unhandled 500 traceback.
    """
    logger.info("🚀  Loading Grounded-SAM2 models …")
    try:
        model_service.load_models()
        logger.info("✅  Models ready.")
    except FileNotFoundError as exc:
        logger.warning(
            "⚠️  Model weights not found — running in degraded mode. "
            "Inference requests will return 503 until weights are available. "
            "Details: %s",
            exc,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "❌  Unexpected error while loading models: %s. "
            "Inference requests will return 503.",
            exc,
            exc_info=True,
        )
    yield
    logger.info("🛑  Shutting down.")


# ── App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Urban Watch — Anomaly Detection API",
    description=(
        "Accepts an image and detects surface anomalies (cracks, rust, "
        "scratches, etc.) using the Grounded-SAM2 pipeline."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Serve the /outputs directory so clients can fetch result images directly
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


@app.get("/")
async def serve_index():
    """Serve the frontend dashboard."""
    import os
    return FileResponse(os.path.join("app", "index.html"))


# Register routes
app.include_router(detect_router)


@app.get("/health")
async def health_check():
    """Simple liveness probe."""
    from app.services.model_service import model_service as ms
    return {
        "status": "ok",
        "models_loaded": ms.is_loaded,
    }