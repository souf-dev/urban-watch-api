"""
Urban Watch API — Anomaly Detection Service

FastAPI application entry-point.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
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
    """Load models on startup; clean up on shutdown."""
    logger.info("🚀  Loading Grounded-SAM2 models …")
    model_service.load_models()
    logger.info("✅  Models ready.")
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

# Register routes
app.include_router(detect_router)


@app.get("/health")
async def health_check():
    """Simple liveness probe."""
    return {"status": "ok"}
