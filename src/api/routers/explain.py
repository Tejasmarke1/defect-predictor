"""
src/api/routers/explain.py
--------------------------
GET /explain/{job_id}/{file_path}  — deep explanation for one file
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from src.api.dependencies import ModelRegistry, get_registry, job_store
from src.api.models import AnalyzeResponse, ExplainResponse, FileRiskScore, SHAPFeature

router = APIRouter(prefix="/explain", tags=["Explainability"])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _build_plain_english(
    file_path: str,
    risk_score: float,
    risk_label: str,
    shap_features: list[SHAPFeature],
) -> str:
    """
    Generate a human-readable risk summary.
    Example:
      "This file has HIGH defect risk (score: 0.84). The main risk factors
       are: high fix_density (0.42, top tier), high code_churn_90d (1823
       lines), and 4 unique authors in last 90 days."
    """
    top = shap_features[:3]
    factors: list[str] = []

    for feat in top:
        direction = "high" if feat.direction == "increases_risk" else "low"
        factors.append(
            f"{direction} {feat.feature_name} "
            f"({feat.feature_value:.2g}, SHAP={feat.shap_value:+.3f})"
        )

    factor_str = "; ".join(factors) if factors else "no dominant factor identified"

    return (
        f"This file has {risk_label} defect risk (score: {risk_score:.2f}). "
        f"The main risk factors are: {factor_str}."
    )


@router.get("/{job_id}/{file_path:path}", response_model=ExplainResponse)
def explain_file(
    job_id: str,
    file_path: str,
    registry: ModelRegistry = Depends(get_registry),
) -> ExplainResponse:
    """
    Deep explanation for a specific file from a completed analysis.

    Returns:
    - Full SHAP waterfall (all features, sorted by |shap_value|)
    - Plain-English risk summary
    - Similar files (closest risk score in same job)
    - GNN embedding neighbors (cosine similarity in embedding space)
    """
    # ── Load job ──────────────────────────────────────────────────────────
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found. Run POST /analyze first.",
        )

    if not isinstance(job, AnalyzeResponse):
        raise HTTPException(
            status_code=400,
            detail=f"Job '{job_id}' has status '{getattr(job, 'status', 'unknown')}' — not yet completed.",
        )

    # ── Find file in results ──────────────────────────────────────────────
    target: FileRiskScore | None = None
    for result in job.top_k_results:
        if result.file_path == file_path:
            target = result
            break

    if target is None:
        available = [r.file_path for r in job.top_k_results[:5]]
        raise HTTPException(
            status_code=404,
            detail=(
                f"File '{file_path}' not found in job '{job_id}' results. "
                f"Available (first 5): {available}"
            ),
        )

    # ── Full SHAP waterfall ───────────────────────────────────────────────
    # top_shap_features already stored from analysis; use those as waterfall
    # sorted by |shap_value| descending
    shap_waterfall = sorted(
        target.top_shap_features,
        key=lambda f: abs(f.shap_value),
        reverse=True,
    )

    # ── Plain-English summary ─────────────────────────────────────────────
    summary = _build_plain_english(
        file_path=file_path,
        risk_score=target.risk_score,
        risk_label=target.risk_label,
        shap_features=shap_waterfall,
    )

    # ── Similar files (closest risk score) ───────────────────────────────
    all_results = job.top_k_results
    similar_files: list[str] = [
        r.file_path
        for r in sorted(
            [r for r in all_results if r.file_path != file_path],
            key=lambda r: abs(r.risk_score - target.risk_score),
        )[:3]
    ]

    # ── GNN embedding neighbors ───────────────────────────────────────────
    embedding_neighbors: list[str] = []

    if registry.gnn_embeddings:
        target_emb = registry.gnn_embeddings.get(file_path)

        if target_emb is not None and np.any(target_emb != 0):
            # Compute cosine similarity to all other files in the job
            job_file_paths = {r.file_path for r in all_results}
            sims: list[tuple[str, float]] = []

            for fp, emb in registry.gnn_embeddings.items():
                if fp == file_path or fp not in job_file_paths:
                    continue
                if np.any(emb != 0):
                    sim = _cosine_similarity(target_emb, emb)
                    sims.append((fp, sim))

            sims.sort(key=lambda x: x[1], reverse=True)
            embedding_neighbors = [fp for fp, _ in sims[:3]]
        else:
            logger.debug(
                f"No non-zero GNN embedding for '{file_path}' — "
                "embedding_neighbors will be empty."
            )

    return ExplainResponse(
        file_path=file_path,
        risk_score=target.risk_score,
        risk_label=target.risk_label,
        shap_waterfall=shap_waterfall,
        plain_english_summary=summary,
        similar_files=similar_files,
        embedding_neighbors=embedding_neighbors,
    )