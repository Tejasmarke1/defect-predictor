"""
src/api/dependencies.py
-----------------------
Singleton ModelRegistry — loads all models once at startup and holds them
in memory. Never reloads on requests. Also provides an in-memory JobStore.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
import torch
from loguru import logger

# ---------------------------------------------------------------------------
# Paths — resolved relative to project root
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
MODELS_DIR   = ROOT / "models"
DATA_DIR     = ROOT / "data" / "processed"

XGB_PATH        = MODELS_DIR / "xgboost_defect_predictor.json"
GNN_PATH        = MODELS_DIR / "gnn_model.pt"
HYBRID_PATH     = MODELS_DIR / "hybrid_model.pkl"
META_PATH       = MODELS_DIR / "model_meta.json"
EMBEDDINGS_PATH = DATA_DIR   / "gnn_embeddings.pkl"


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Loads all models once at startup and holds them in memory.
    Calling get_instance() 100 times returns the same object — models
    are never reloaded after the first call.
    """

    _instance: Optional["ModelRegistry"] = None
    _startup_time: float = time.time()

    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance._initialized = False
            cls._instance._load_all()
        return cls._instance

    # ------------------------------------------------------------------
    def _load_all(self) -> None:
        """Load every model in try/except — partial failures are tolerated."""
        if getattr(self, "_initialized", False):
            return

        self.models_loaded: dict[str, bool] = {
            "xgboost": False,
            "gnn":     False,
            "hybrid":  False,
            "shap":    False,
        }

        self.xgb_model    = None
        self.gnn_model    = None
        self.gnn_trainer  = None
        self.hybrid_model = None
        self.shap_explainer = None
        self.threshold    = 0.5
        self.feature_names: list[str] = []
        self.gnn_embeddings: dict[str, np.ndarray] = {}
        self.total_analyses: int = 0
        self.last_analysis: Optional[str] = None

        # ---- XGBoost ------------------------------------------------
        try:
            import xgboost as xgb
            from src.models.train import DefectXGBoost
            self.xgb_model = DefectXGBoost()
            self.xgb_model.load(XGB_PATH)
            self.models_loaded["xgboost"] = True
            logger.success(f"XGBoost loaded from {XGB_PATH}")
        except Exception as e:
            logger.error(f"XGBoost load failed: {e}")

        # ---- model_meta.json ----------------------------------------
        try:
            with open(META_PATH) as f:
                meta = json.load(f)
            self.threshold     = float(meta.get("best_threshold", 0.5))
            self.feature_names = meta.get("feature_names", [])
            logger.success(f"model_meta loaded: threshold={self.threshold:.3f}")
        except Exception as e:
            logger.warning(f"model_meta.json not found or unreadable: {e}")

        # ---- GNN + GNNTrainer ---------------------------------------
        try:
            from src.models.gnn_model import CodeGNN, GNNTrainer
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.gnn_model  = CodeGNN()
            self.gnn_trainer = GNNTrainer(model=self.gnn_model, device=device)
            self.gnn_trainer.load(GNN_PATH)
            self.models_loaded["gnn"] = True
            logger.success(f"GNN loaded from {GNN_PATH} (device={device})")
        except Exception as e:
            logger.error(f"GNN load failed: {e}")

        # ---- HybridDefectModel ---------------------------------------
        try:
            from src.models.hybrid_model import HybridDefectModel
            self.hybrid_model = HybridDefectModel(gnn_trainer=self.gnn_trainer)
            self.hybrid_model.load(HYBRID_PATH)
            self.models_loaded["hybrid"] = True
            logger.success(f"HybridDefectModel loaded from {HYBRID_PATH}")
        except Exception as e:
            logger.error(f"HybridDefectModel load failed: {e}")

        # ---- SHAPExplainer -------------------------------------------
        try:
            from src.explainability.shap_explainer import SHAPExplainer
            if self.xgb_model is not None:
                self.shap_explainer = SHAPExplainer(self.xgb_model)
                self.models_loaded["shap"] = True
                logger.success("SHAPExplainer initialised")
        except Exception as e:
            logger.error(f"SHAPExplainer init failed: {e}")

        # ---- GNN embeddings -----------------------------------------
        try:
            with open(EMBEDDINGS_PATH, "rb") as f:
                self.gnn_embeddings = pickle.load(f)
            logger.success(
                f"GNN embeddings loaded: {len(self.gnn_embeddings)} files"
            )
        except Exception as e:
            logger.warning(f"GNN embeddings not loaded: {e}")

        self._initialized = True
        status = "healthy" if self.is_healthy else "degraded"
        logger.info(f"ModelRegistry ready — status={status} | {self.models_loaded}")

    # ------------------------------------------------------------------
    @property
    def is_healthy(self) -> bool:
        """Healthy = at least XGBoost is loaded (baseline always works)."""
        return self.models_loaded.get("xgboost", False)

    @property
    def uptime_seconds(self) -> float:
        return time.time() - ModelRegistry._startup_time

    def bump_analysis_count(self, timestamp: str) -> None:
        self.total_analyses += 1
        self.last_analysis = timestamp


# ---------------------------------------------------------------------------
# FastAPI dependency function
# ---------------------------------------------------------------------------

def get_registry() -> ModelRegistry:
    return ModelRegistry.get_instance()


# ---------------------------------------------------------------------------
# JobStore — in-memory analysis job cache
# ---------------------------------------------------------------------------

class JobStore:
    """
    In-memory store mapping job_id → AnalyzeResponse.
    job_id format: {repo_name}_{uuid4().hex[:8]}
    """

    def __init__(self) -> None:
        self._jobs: dict[str, object] = {}

    @staticmethod
    def make_job_id(repo_name: str) -> str:
        safe = repo_name.replace("/", "_").replace(".", "_")[:30]
        return f"{safe}_{uuid4().hex[:8]}"

    def create(self, job_id: str, placeholder: object) -> None:
        self._jobs[job_id] = placeholder

    def update(self, job_id: str, response: object) -> None:
        self._jobs[job_id] = response

    def get(self, job_id: str) -> Optional[object]:
        return self._jobs.get(job_id)

    def list_recent(self, n: int = 10) -> list:
        items = list(self._jobs.values())
        return items[-n:]

    def __len__(self) -> int:
        return len(self._jobs)


# Module-level singleton
job_store = JobStore()