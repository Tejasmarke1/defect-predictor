"""
src/api/routers/health.py
-------------------------
GET /health  — detailed model and system health
GET /        — root redirect info
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from loguru import logger

from src.api.dependencies import ModelRegistry, get_registry
from src.api.models import HealthResponse

router = APIRouter(tags=["Health"])

_start_time = time.time()


@router.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "name": "Defect Prediction Engine API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "analyze": "/analyze",
    }


@router.get("/health", response_model=HealthResponse)
async def health_check(
    registry: ModelRegistry = Depends(get_registry),
) -> HealthResponse:
    """
    Returns detailed health status.

    - "healthy"   — all models loaded
    - "degraded"  — XGBoost loaded but GNN or Hybrid missing
                    (API still works with baseline)
    - "unhealthy" — XGBoost itself failed to load
    """
    # Check MLflow connectivity
    mlflow_ok = False
    try:
        import mlflow
        from configs.config import MLFLOW
        mlflow.set_tracking_uri(MLFLOW["tracking_uri"])
        mlflow.search_experiments()
        mlflow_ok = True
    except Exception:
        pass

    xgb_ok    = registry.models_loaded.get("xgboost", False)
    gnn_ok    = registry.models_loaded.get("gnn",     False)
    hybrid_ok = registry.models_loaded.get("hybrid",  False)

    if not xgb_ok:
        status = "unhealthy"
    elif not (gnn_ok and hybrid_ok):
        status = "degraded"
    else:
        status = "healthy"

    logger.debug(f"Health check: {status}")

    return HealthResponse(
        status=status,
        version="1.0.0",
        models_loaded=registry.models_loaded,
        last_analysis=registry.last_analysis,
        total_analyses_run=registry.total_analyses,
        uptime_seconds=registry.uptime_seconds,
        mlflow_connected=mlflow_ok,
    )