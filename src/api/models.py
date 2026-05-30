"""
src/api/models.py
-----------------
Pydantic v2 request and response models for the Defect Prediction Engine API.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, HttpUrl, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class SHAPFeature(BaseModel):
    feature_name: str
    shap_value: float
    feature_value: float
    direction: str  # "increases_risk" | "decreases_risk"


class FileRiskScore(BaseModel):
    file_path: str
    risk_score: float           # 0.0 – 1.0
    risk_label: str             # "HIGH" | "MEDIUM" | "LOW"
    rank: int                   # 1 = riskiest
    top_shap_features: list[SHAPFeature]
    lines_of_code: Optional[int] = None
    cyclomatic_complexity: Optional[float] = None
    last_modified_days_ago: Optional[int] = None


class ExperimentSummary(BaseModel):
    run_name: str
    auc: float
    f1: float
    precision_at_20: float
    n_features: int
    timestamp: str


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    repo_url: HttpUrl
    since_days: int = 365
    top_k: int = 20
    use_hybrid: bool = True

    @field_validator("repo_url")
    @classmethod
    def must_be_github(cls, v: HttpUrl) -> HttpUrl:
        host = str(v.host) if v.host else ""
        if "github.com" not in host:
            raise ValueError("Only github.com URLs are supported.")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_range(cls, v: int) -> int:
        if not 1 <= v <= 200:
            raise ValueError("top_k must be between 1 and 200.")
        return v

    @field_validator("since_days")
    @classmethod
    def since_days_range(cls, v: int) -> int:
        if not 1 <= v <= 3650:
            raise ValueError("since_days must be between 1 and 3650.")
        return v


class ExplainRequest(BaseModel):
    file_path: str
    job_id: str


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class AnalyzeResponse(BaseModel):
    job_id: str
    status: str                     # "completed" | "running" | "failed"
    repo_url: str
    repo_name: str
    analysis_time_ms: float
    mining_time_ms: float
    feature_time_ms: float
    prediction_time_ms: float
    model_used: str                 # "hybrid" | "baseline"
    total_files_analyzed: int
    buggy_files_predicted: int
    top_k_results: list[FileRiskScore]
    model_auc: float
    precision_at_k: float
    warnings: list[str] = []
    error: Optional[str] = None


class ExplainResponse(BaseModel):
    file_path: str
    risk_score: float
    risk_label: str
    shap_waterfall: list[SHAPFeature]
    plain_english_summary: str
    similar_files: list[str]
    embedding_neighbors: list[str]


class HealthResponse(BaseModel):
    status: str                     # "healthy" | "degraded" | "unhealthy"
    version: str
    models_loaded: dict[str, bool]
    last_analysis: Optional[str] = None
    total_analyses_run: int
    uptime_seconds: float
    mlflow_connected: bool


class ExperimentsResponse(BaseModel):
    experiment_name: str
    total_runs: int
    best_run: Optional[ExperimentSummary] = None
    all_runs: list[ExperimentSummary]