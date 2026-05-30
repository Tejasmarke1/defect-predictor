"""
src/api/routers/experiments.py
------------------------------
GET /experiments        — all MLflow runs sorted by AUC
GET /experiments/{name} — single run by name
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.api.models import ExperimentSummary, ExperimentsResponse

router = APIRouter(prefix="/experiments", tags=["MLflow"])

_TARGET_EXPERIMENTS = ["defect-prediction", "model-comparison"]


def _parse_run(run) -> ExperimentSummary:
    """Convert an MLflow run object to ExperimentSummary."""
    metrics = run.data.metrics
    params  = run.data.params
    tags    = run.data.tags

    run_name = (
        tags.get("mlflow.runName")
        or run.info.run_name
        or run.info.run_id[:8]
    )

    return ExperimentSummary(
        run_name=run_name,
        auc=round(float(metrics.get("AUC", metrics.get("cv_mean_auc", 0.0))), 4),
        f1=round(float(metrics.get("F1",  metrics.get("cv_mean_f1",  0.0))), 4),
        precision_at_20=round(float(metrics.get("Precision_at_20", 0.0)), 4),
        n_features=int(float(params.get("n_features", 0))),
        timestamp=str(run.info.start_time),
    )


def _load_all_runs() -> list[ExperimentSummary]:
    """Load runs from all target experiments, return sorted by AUC desc."""
    import mlflow
    from configs.config import MLFLOW as MLFLOW_CFG

    mlflow.set_tracking_uri(MLFLOW_CFG["tracking_uri"])

    summaries: list[ExperimentSummary] = []

    for exp_name in _TARGET_EXPERIMENTS:
        try:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is None:
                continue
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["metrics.AUC DESC"],
                max_results=10,
            )
            for _, row in runs.iterrows():
                run_name = row.get("tags.mlflow.runName", row["run_id"][:8])
                summaries.append(ExperimentSummary(
                    run_name=str(run_name),
                    auc=round(float(row.get("metrics.AUC",
                              row.get("metrics.cv_mean_auc", 0.0))), 4),
                    f1=round(float(row.get("metrics.F1",
                             row.get("metrics.cv_mean_f1", 0.0))), 4),
                    precision_at_20=round(
                        float(row.get("metrics.Precision_at_20", 0.0)), 4
                    ),
                    n_features=int(float(row.get("params.n_features", 0) or 0)),
                    timestamp=str(row.get("start_time", "")),
                ))
        except Exception as e:
            logger.warning(f"Could not load experiment '{exp_name}': {e}")

    summaries.sort(key=lambda s: s.auc, reverse=True)
    return summaries


@router.get("/", response_model=ExperimentsResponse)
def get_experiments() -> ExperimentsResponse:
    """All MLflow runs across defect-prediction and model-comparison experiments."""
    try:
        runs = _load_all_runs()
    except Exception as e:
        logger.error(f"MLflow error: {e}")
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {e}")

    return ExperimentsResponse(
        experiment_name=", ".join(_TARGET_EXPERIMENTS),
        total_runs=len(runs),
        best_run=runs[0] if runs else None,
        all_runs=runs,
    )


@router.get("/{run_name}", response_model=ExperimentSummary)
def get_experiment_run(run_name: str) -> ExperimentSummary:
    """Get a specific run by name."""
    try:
        runs = _load_all_runs()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {e}")

    for run in runs:
        if run.run_name == run_name:
            return run

    raise HTTPException(
        status_code=404,
        detail=f"No run named '{run_name}'. Available: {[r.run_name for r in runs]}",
    )