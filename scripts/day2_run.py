"""
scripts/day2_run.py
-------------------
Master script for Day 2: Feature Engineering.

Runs the complete pipeline:
  1. Load raw Flask commits  (data/raw/flask_commits.csv)
  2. Load KC1 labeled data   (data/processed/kc1_labeled.csv)
  3. Build and save the feature matrix (data/processed/feature_matrix.csv)
  4. Log key statistics to MLflow under the "feature-engineering" experiment

Usage
-----
    python scripts/day2_run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import numpy as np
import pandas as pd
from loguru import logger

from src.features.feature_pipeline import build_feature_matrix

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_COMMITS_PATH = PROJECT_ROOT / "data" / "raw" / "flask_commits.csv"
LABELED_PATH = PROJECT_ROOT / "data" / "processed" / "labeled_flask.csv"
FEATURE_MATRIX_PATH = PROJECT_ROOT / "data" / "processed" / "feature_matrix.csv"

# ---------------------------------------------------------------------------
# Logging config
# ---------------------------------------------------------------------------

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)
log_file = PROJECT_ROOT / "logs" / "day2_run.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logger.add(log_file, rotation="10 MB", level="DEBUG", enqueue=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path, label: str) -> pd.DataFrame:
    """Load *path* as a CSV, log shape, and abort on failure."""
    if not path.exists():
        logger.error(f"{label} not found at {path}. Did Day 1 complete?")
        sys.exit(1)
    df = pd.read_csv(path)
    logger.info(f"Loaded {label}: {df.shape[0]:,} rows × {df.shape[1]} cols  ({path.name})")
    logger.debug(f"  columns: {list(df.columns)}")
    return df


def _validate_commits(commits_df: pd.DataFrame) -> None:
    """Abort with a clear message if required commit columns are missing."""
    required = {"commit_hash", "commit_date", "file_path", "filename",
                "lines_changed", "author_email"}
    missing = required - set(commits_df.columns)
    if missing:
        logger.error(f"Raw commits DataFrame missing columns: {missing}")
        sys.exit(1)


def _validate_labeled(labeled_df: pd.DataFrame) -> None:
    """Abort if the labeled dataset is missing its minimum required columns."""
    required = {"file_path", "is_buggy"}
    missing = required - set(labeled_df.columns)
    if missing:
        logger.error(f"Labeled DataFrame missing columns: {missing}")
        logger.error(
            "Labeled dataset must have at least 'file_path' and 'is_buggy'. "
            "Check data/processed/labeled_flask.csv."
        )
        sys.exit(1)


def _print_summary(feature_df: pd.DataFrame) -> None:
    """Pretty-print feature matrix summary statistics."""
    label_counts = feature_df["is_buggy"].value_counts().to_dict()
    n_buggy = int(label_counts.get(1, label_counts.get(True, 0)))
    n_clean = int(label_counts.get(0, label_counts.get(False, 0)))
    pct_buggy = 100 * n_buggy / len(feature_df) if len(feature_df) else 0

    numeric_df = feature_df.select_dtypes(include=[np.number]).drop(
        columns=["is_buggy"], errors="ignore"
    )

    logger.info("")
    logger.info("━" * 60)
    logger.info("FEATURE MATRIX SUMMARY")
    logger.info("━" * 60)
    logger.info(f"  Shape          : {feature_df.shape[0]:,} rows × {feature_df.shape[1]} cols")
    logger.info(f"  Buggy modules  : {n_buggy:,}  ({pct_buggy:.1f}%)")
    logger.info(f"  Clean modules  : {n_clean:,}  ({100 - pct_buggy:.1f}%)")
    logger.info(f"  Missing values : {int(numeric_df.isna().sum().sum())}")
    logger.info(f"  Feature columns: {len(numeric_df.columns)}")
    logger.info("")
    logger.info("  Top 5 features by variance:")
    for feat, var in numeric_df.var().sort_values(ascending=False).head(5).items():
        logger.info(f"    {feat:<40s} var={var:.4f}")
    logger.info("━" * 60)


def _log_mlflow(feature_df: pd.DataFrame) -> None:
    """Log feature matrix statistics to MLflow."""
    mlflow.set_experiment("feature-engineering")
    with mlflow.start_run(run_name="day2_feature_matrix"):
        n_rows, n_cols = feature_df.shape
        # 2 non-feature cols: file_path, is_buggy
        mlflow.log_param("n_modules", n_rows)
        mlflow.log_param("n_feature_cols", n_cols - 2)
        mlflow.log_param("short_window_days", 30)
        mlflow.log_param("long_window_days", 90)

        n_buggy = int(feature_df["is_buggy"].sum())
        mlflow.log_metric("n_buggy", n_buggy)
        mlflow.log_metric("n_clean", n_rows - n_buggy)
        mlflow.log_metric("pct_buggy", round(100 * n_buggy / n_rows, 2))

        numeric_df = feature_df.select_dtypes(include=[np.number]).drop(
            columns=["is_buggy"], errors="ignore"
        )
        mlflow.log_metric("missing_values_total", int(numeric_df.isna().sum().sum()))

        for col in ("cyclomatic_complexity", "halstead_volume", "code_churn_90d"):
            if col in numeric_df.columns:
                mlflow.log_metric(f"mean_{col}", round(float(numeric_df[col].mean()), 4))

        mlflow.log_artifact(str(FEATURE_MATRIX_PATH))
        logger.success("MLflow run logged successfully.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full Day 2 feature engineering pipeline."""
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  Adaptive Defect Prediction Engine — Day 2  ║")
    logger.info("║  Feature Engineering Pipeline                ║")
    logger.info("╚══════════════════════════════════════════════╝")

    # 1. Load ---------------------------------------------------------------
    commits_df = _load_csv(RAW_COMMITS_PATH, "raw commits")
    labeled_df = _load_csv(LABELED_PATH, "KC1 labeled data")

    # 2. Validate -----------------------------------------------------------
    _validate_commits(commits_df)
    _validate_labeled(labeled_df)

    # 3. Build feature matrix -----------------------------------------------
    #    window_end is NOT required in labeled_df — the pipeline derives it
    #    automatically from the commit history per module.
    feature_df = build_feature_matrix(commits_df, labeled_df, save=True)

    # 4. Summary ------------------------------------------------------------
    _print_summary(feature_df)

    # 5. MLflow -------------------------------------------------------------
    try:
        _log_mlflow(feature_df)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"MLflow logging failed (non-fatal): {exc}")

    logger.success("Day 2 pipeline complete.")


if __name__ == "__main__":
    main()