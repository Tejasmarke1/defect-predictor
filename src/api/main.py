"""
src/api/main.py
---------------
FastAPI application entry point.
Loads all models at startup via ModelRegistry lifespan handler.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from loguru import logger

from src.api.dependencies import ModelRegistry
from src.api.routers.analyze import router as analyze_router
from src.api.routers.explain import router as explain_router
from src.api.routers.experiments import router as experiments_router
from src.api.routers.health import router as health_router


# ---------------------------------------------------------------------------
# Lifespan — startup + shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Defect Prediction Engine — starting up")
    logger.info("=" * 60)

    registry = ModelRegistry.get_instance()

    if registry.is_healthy:
        logger.success(
            f"All models loaded in {time.time()-t0:.1f}s — API ready"
        )
    else:
        failed = [k for k, v in registry.models_loaded.items() if not v]
        logger.warning(
            f"Startup degraded — failed models: {failed}. "
            "Baseline XGBoost predictions still available."
        )

    yield  # ── API is running ────────────────────────────────────────────

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("Defect Prediction Engine — shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Defect Prediction Engine",
    description=(
        "Predicts which files in a Python codebase are most likely to contain "
        "bugs, using a hybrid XGBoost + GNN architecture with SHAP explainability. "
        "\n\n"
        "**Day 5 model results:** Hybrid AUC 0.8590 | +0.178 over tabular baseline "
        "| 5,463 files across 12 repos | 2,953 GNN embeddings"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(analyze_router)
app.include_router(explain_router)
app.include_router(experiments_router)