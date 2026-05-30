"""
src/api/routers/analyze.py
--------------------------
POST /analyze       — full pipeline: mine → features → predict → explain
GET  /analyze/{id}  — retrieve completed analysis
GET  /analyze/      — list recent analyses
"""

from __future__ import annotations

import ast
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from loguru import logger

from configs.config import API, MODEL
from src.api.dependencies import JobStore, ModelRegistry, get_registry, job_store
from src.api.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    FileRiskScore,
    SHAPFeature,
)

router = APIRouter(prefix="/analyze", tags=["Analysis"])

# ---------------------------------------------------------------------------
# Risk label thresholds
# ---------------------------------------------------------------------------
HIGH_THRESHOLD   = 0.70
MEDIUM_THRESHOLD = 0.40


def _risk_label(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "HIGH"
    elif score >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# AST helpers for inline feature computation
# ---------------------------------------------------------------------------

def _ast_cyclomatic(source: str) -> Optional[float]:
    try:
        tree = ast.parse(source)
        branch_nodes = (
            ast.If, ast.For, ast.While, ast.Try,
            ast.ExceptHandler, ast.With, ast.Assert,
        )
        return float(sum(1 for _ in ast.walk(tree) if isinstance(_, branch_nodes)) + 1)
    except Exception:
        return None


def _count_lines(source: str) -> int:
    return len(source.splitlines())


# ---------------------------------------------------------------------------
# Feature extraction from a cloned repo
# ---------------------------------------------------------------------------

def _extract_features_from_repo(
    repo_path: Path,
    since_days: int,
    warnings: list[str],
) -> pd.DataFrame:
    """
    Walk cloned repo, compute process + AST features per Python file.
    Returns a DataFrame with feature_cols + file_path.
    Mirrors the Day 2 pipeline logic without requiring a full git mine.
    """
    import re
    from datetime import timedelta

    ref_date = datetime.now(timezone.utc)
    cutoff   = ref_date - timedelta(days=since_days)

    # ── Collect source files ──────────────────────────────────────────────
    source_files: list[tuple[str, str]] = []  # (rel_path, source_code)

    excluded = ["/test_", "_test.py", "/tests/", "/migrations/", "setup.py"]

    for py_file in repo_path.rglob("*.py"):
        rel = py_file.relative_to(repo_path).as_posix()
        if any(p in rel for p in excluded):
            continue
        try:
            src = py_file.read_text(encoding="utf-8", errors="ignore")
            if src.strip():
                source_files.append((rel, src))
        except Exception:
            pass

    if not source_files:
        raise HTTPException(
            status_code=422,
            detail="No Python source files found in the repository.",
        )

    # ── Git log for process metrics ───────────────────────────────────────
    # git log --format="%H|%ae|%ai|%s" --name-only --diff-filter=AM -- "*.py"
    commit_records: list[dict] = []
    try:
        result = subprocess.run(
            [
                "git", "log",
                f"--after={cutoff.strftime('%Y-%m-%d')}",
                "--format=%H|%ae|%ai|%s",
                "--name-only",
                "--diff-filter=AM",
            ],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        current_meta: dict = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            if "|" in line and line.count("|") >= 2:
                parts = line.split("|", 3)
                if len(parts) >= 3:
                    current_meta = {
                        "hash":   parts[0],
                        "author": parts[1],
                        "date":   pd.to_datetime(parts[2], utc=True, errors="coerce"),
                        "msg":    parts[3] if len(parts) > 3 else "",
                    }
            elif line.endswith(".py") and current_meta:
                commit_records.append({**current_meta, "file_path": line})
    except Exception as e:
        warnings.append(f"Git log failed ({e}); process metrics will be zero.")

    commits_df = pd.DataFrame(commit_records) if commit_records else pd.DataFrame()

    if len(commits_df) < 10:
        warnings.append(
            f"Only {len(commits_df)} commits found in the last {since_days} days. "
            "Results may be unreliable."
        )

    BUG_RE = re.compile(
        r"\b(fix(es|ed|ing)?|bug(fix|s)?|defect|regression|crash)\b",
        re.IGNORECASE,
    )

    # ── Build feature rows ────────────────────────────────────────────────
    rows: list[dict] = []

    for rel_path, source in source_files:
        row: dict = {"file_path": rel_path}

        # Process metrics
        if not commits_df.empty and "file_path" in commits_df.columns:
            fc = commits_df[commits_df["file_path"] == rel_path].copy()
            fc["date"] = pd.to_datetime(fc["date"], utc=True, errors="coerce")

            cutoff_30 = ref_date - pd.Timedelta(days=30)
            cutoff_90 = ref_date - pd.Timedelta(days=90)
            w30 = fc[fc["date"] >= cutoff_30]
            w90 = fc[fc["date"] >= cutoff_90]

            row["commit_count_30d"]  = len(w30)
            row["commit_count_90d"]  = len(w90)
            row["code_churn_30d"]    = len(w30) * 10   # proxy: no diff stats
            row["code_churn_90d"]    = len(w90) * 10
            row["author_count_90d"]  = w90["author"].nunique() if len(w90) else 0
            bug_fixes_90 = int(w90["msg"].apply(lambda m: bool(BUG_RE.search(str(m)))).sum()) if len(w90) else 0
            row["bug_fix_count_90d"] = bug_fixes_90
            row["fix_density"]       = round(bug_fixes_90 / max(len(w90), 1), 4)

            if not fc.empty and fc["date"].notna().any():
                last = fc["date"].max()
                row["days_since_last_change"] = max(0, (ref_date - last).days)
            else:
                row["days_since_last_change"] = since_days

            cc30 = row["commit_count_30d"]
            cc90 = row["commit_count_90d"]
            row["commit_burst_30d"] = round(cc30 / max(cc90 / 3.0, 1), 4)

            if not fc.empty and "author" in fc.columns:
                vc = fc["author"].value_counts()
                row["ownership_score"] = round(float(vc.iloc[0]) / max(len(fc), 1), 4)
            else:
                row["ownership_score"] = 1.0
        else:
            # No git data — fill zeros
            for col in [
                "commit_count_30d", "commit_count_90d", "code_churn_30d",
                "code_churn_90d", "author_count_90d", "bug_fix_count_90d",
                "fix_density", "days_since_last_change", "commit_burst_30d",
                "ownership_score",
            ]:
                row[col] = 0

        # AST features
        try:
            tree = ast.parse(source)
            branch_nodes = (
                ast.If, ast.For, ast.While, ast.Try,
                ast.ExceptHandler, ast.With, ast.Assert,
            )
            cyc = sum(1 for _ in ast.walk(tree) if isinstance(_, branch_nodes)) + 1
            funcs = [
                n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            cog = sum(
                1 for n in ast.walk(tree)
                if isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler))
            )

            def _depth(node: ast.AST, d: int = 0) -> int:
                kids = list(ast.iter_child_nodes(node))
                return max((_depth(k, d + 1) for k in kids), default=d)

            avg_len = float(np.mean([
                (f.end_lineno - f.lineno + 1)
                for f in funcs
                if hasattr(f, "end_lineno") and f.end_lineno
            ])) if funcs else 0.0

            row["ast_cyclomatic_complexity"] = cyc
            row["ast_cognitive_complexity"]  = cog
            row["max_nesting_depth"]         = _depth(tree)
            row["num_functions"]             = len(funcs)
            row["num_classes"]               = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.ClassDef))
            row["avg_function_length"]       = round(avg_len, 2)
            row["max_args_per_function"]     = max((len(f.args.args) for f in funcs), default=0)
            row["num_imports"]               = sum(1 for _ in ast.walk(tree) if isinstance(_, (ast.Import, ast.ImportFrom)))
            row["ast_node_count"]            = sum(1 for _ in ast.walk(tree))
            row["has_try_except"]            = int(any(isinstance(n, ast.Try) for n in ast.walk(tree)))
            # Extra info for response (not model features)
            row["_lines_of_code"]            = _count_lines(source)
        except SyntaxError:
            for col in [
                "ast_cyclomatic_complexity", "ast_cognitive_complexity",
                "max_nesting_depth", "num_functions", "num_classes",
                "avg_function_length", "max_args_per_function",
                "num_imports", "ast_node_count", "has_try_except",
            ]:
                row[col] = 0
            row["_lines_of_code"] = _count_lines(source)

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SHAP explanation helper
# ---------------------------------------------------------------------------

def _compute_shap_for_file(
    registry: ModelRegistry,
    feature_row: pd.Series,
    feature_names: list[str],
) -> list[SHAPFeature]:
    """Compute top-5 SHAP features for one file row."""
    try:
        if registry.shap_explainer is None:
            return []

        X = feature_row[feature_names].values.reshape(1, -1)
        shap_vals = registry.shap_explainer.explain(X)  # shape [1, n_features]

        if shap_vals is None or len(shap_vals) == 0:
            return []

        sv = shap_vals[0]  # shape [n_features]
        indexed = sorted(
            enumerate(sv), key=lambda x: abs(x[1]), reverse=True
        )[:5]

        result: list[SHAPFeature] = []
        for idx, val in indexed:
            if idx >= len(feature_names):
                continue
            result.append(SHAPFeature(
                feature_name=feature_names[idx],
                shap_value=round(float(val), 6),
                feature_value=round(float(feature_row[feature_names[idx]]), 4),
                direction="increases_risk" if val > 0 else "decreases_risk",
            ))
        return result

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def _run_analysis(
    request: AnalyzeRequest,
    job_id: str,
    registry: ModelRegistry,
) -> AnalyzeResponse:
    """
    Full pipeline: clone → features → predict → explain.
    Times each phase independently.
    """
    warnings: list[str] = []
    repo_url  = str(request.repo_url)
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")

    t_start = time.time()

    # ── Phase 1: Clone repo ───────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / repo_name

        max_mb = API.get("max_repo_size_mb", 500)
        logger.info(f"Cloning {repo_url} → {repo_path}")

        clone_result = subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", repo_url, str(repo_path)],
            capture_output=True,
            text=True,
            timeout=API.get("timeout_seconds", 600),
        )

        t_mined = time.time()
        mining_time_ms = (t_mined - t_start) * 1000

        if clone_result.returncode != 0:
            raise HTTPException(
                status_code=422,
                detail=f"Repository clone failed: {clone_result.stderr[:300]}",
            )

        # ── Phase 2: Feature extraction ───────────────────────────────────
        logger.info(f"Extracting features from {repo_path}")
        df_features = _extract_features_from_repo(
            repo_path, request.since_days, warnings
        )

        t_features = time.time()
        feature_time_ms = (t_features - t_mined) * 1000

        # Determine which feature columns the model expects
        drop_meta = {
            "file_path", "is_buggy", "commit_date",
            "repo_name", "repo", "_lines_of_code",
        }
        available_cols = [c for c in df_features.columns if c not in drop_meta]

        # Align to model's expected feature names if available
        model_features = registry.feature_names or available_cols
        feature_cols: list[str] = []
        for col in model_features:
            if col in df_features.columns:
                feature_cols.append(col)
            else:
                df_features[col] = 0.0   # fill missing with zero
                feature_cols.append(col)

        # ── Phase 3: Prediction ───────────────────────────────────────────
        logger.info(f"Predicting on {len(df_features)} files")

        use_hybrid = (
            request.use_hybrid
            and registry.models_loaded.get("hybrid", False)
            and registry.hybrid_model is not None
        )

        try:
            if use_hybrid:
                # Build embeddings for these files from source
                source_codes: dict[str, str] = {}
                for rel_path, source in [
                    (r, (repo_path / r).read_text(encoding="utf-8", errors="ignore"))
                    for r in df_features["file_path"].tolist()
                    if (repo_path / r).exists()
                ]:
                    source_codes[rel_path] = source

                embeddings = registry.gnn_trainer.get_embeddings(source_codes) \
                    if registry.gnn_trainer else {}

                scores = registry.hybrid_model.predict_proba(df_features, embeddings)
                model_used = "hybrid"
                model_auc  = 0.8590   # Day 5 result
            else:
                X = df_features[feature_cols].values.astype(float)
                scores = registry.xgb_model.predict_proba(X)
                model_used = "baseline"
                model_auc  = 0.7926   # Day 3 result
                if request.use_hybrid:
                    warnings.append("Hybrid model unavailable — using XGBoost baseline.")

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)[:200]}",
            )

        t_predicted = time.time()
        prediction_time_ms = (t_predicted - t_features) * 1000
        analysis_time_ms   = (t_predicted - t_start) * 1000

        # ── Build ranked results ──────────────────────────────────────────
        df_features["_score"] = scores
        df_sorted = df_features.sort_values("_score", ascending=False).reset_index(drop=True)

        threshold = registry.threshold
        buggy_count = int((scores >= threshold).sum())

        top_k = min(request.top_k, len(df_sorted))
        top_results: list[FileRiskScore] = []

        for rank, (_, row) in enumerate(df_sorted.head(top_k).iterrows(), start=1):
            shap_feats = _compute_shap_for_file(registry, row, feature_cols)

            top_results.append(FileRiskScore(
                file_path=str(row["file_path"]),
                risk_score=round(float(row["_score"]), 4),
                risk_label=_risk_label(float(row["_score"])),
                rank=rank,
                top_shap_features=shap_feats,
                lines_of_code=int(row.get("_lines_of_code", 0)) or None,
                cyclomatic_complexity=float(row.get("ast_cyclomatic_complexity", 0)) or None,
                last_modified_days_ago=int(row.get("days_since_last_change", 0)) or None,
            ))

        # Precision@K (no ground truth available for new repos)
        precision_at_k = -1.0
        if buggy_count == 0:
            warnings.append("Model predicted no buggy files above threshold.")

        now_iso = datetime.now(timezone.utc).isoformat()
        registry.bump_analysis_count(now_iso)

        response = AnalyzeResponse(
            job_id=job_id,
            status="completed",
            repo_url=repo_url,
            repo_name=repo_name,
            analysis_time_ms=round(analysis_time_ms, 1),
            mining_time_ms=round(mining_time_ms, 1),
            feature_time_ms=round(feature_time_ms, 1),
            prediction_time_ms=round(prediction_time_ms, 1),
            model_used=model_used,
            total_files_analyzed=len(df_features),
            buggy_files_predicted=buggy_count,
            top_k_results=top_results,
            model_auc=model_auc,
            precision_at_k=precision_at_k,
            warnings=warnings,
        )

        job_store.update(job_id, response)
        logger.success(
            f"Analysis complete: job={job_id} files={len(df_features)} "
            f"buggy={buggy_count} model={model_used} "
            f"time={analysis_time_ms:.0f}ms"
        )
        return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/", response_model=AnalyzeResponse)
def analyze_repository(
    request: AnalyzeRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> AnalyzeResponse:
    """
    Full pipeline: clone repo → extract features → predict → explain.

    Times each phase independently:
      mining_time_ms     = git clone time
      feature_time_ms    = AST + process feature extraction
      prediction_time_ms = model inference
      analysis_time_ms   = total (no LLM — pure retrieval + prediction)
    """
    if not registry.is_healthy:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model registry is unhealthy. "
                f"Loaded: {registry.models_loaded}"
            ),
        )

    job_id = JobStore.make_job_id(str(request.repo_url).rstrip("/").split("/")[-1])
    logger.info(f"New analysis job: {job_id}  url={request.repo_url}")

    # Store placeholder so GET can return "running" status
    job_store.create(
        job_id,
        AnalyzeResponse(
            job_id=job_id,
            status="running",
            repo_url=str(request.repo_url),
            repo_name=str(request.repo_url).rstrip("/").split("/")[-1],
            analysis_time_ms=0,
            mining_time_ms=0,
            feature_time_ms=0,
            prediction_time_ms=0,
            model_used="unknown",
            total_files_analyzed=0,
            buggy_files_predicted=0,
            top_k_results=[],
            model_auc=0.0,
            precision_at_k=0.0,
        ),
    )

    try:
        return _run_analysis(request, job_id, registry)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in job {job_id}: {e}")
        error_response = AnalyzeResponse(
            job_id=job_id,
            status="failed",
            repo_url=str(request.repo_url),
            repo_name=str(request.repo_url).rstrip("/").split("/")[-1],
            analysis_time_ms=0,
            mining_time_ms=0,
            feature_time_ms=0,
            prediction_time_ms=0,
            model_used="unknown",
            total_files_analyzed=0,
            buggy_files_predicted=0,
            top_k_results=[],
            model_auc=0.0,
            precision_at_k=0.0,
            error=str(e)[:500],
        )
        job_store.update(job_id, error_response)
        raise HTTPException(status_code=500, detail=str(e)[:300])


@router.get("/", response_model=list[AnalyzeResponse])
async def list_analyses(limit: int = 10) -> list[AnalyzeResponse]:
    """List the most recent analyses."""
    items = job_store.list_recent(limit)
    return [i for i in items if isinstance(i, AnalyzeResponse)]


@router.get("/{job_id}", response_model=AnalyzeResponse)
async def get_analysis(job_id: str) -> AnalyzeResponse:
    """Retrieve a previously completed or running analysis by job_id."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found.",
        )
    return job