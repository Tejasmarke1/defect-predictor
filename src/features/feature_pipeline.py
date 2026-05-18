"""
feature_pipeline.py
-------------------
Orchestrates the full Day 2 feature engineering pipeline.

Two data sources are combined into one feature matrix:

  KC1 (static software metrics dataset)
  ├── Already has: lines_of_code, cyclomatic_complexity, halstead_*, branch_count, …
  └── Provides: is_buggy label + classical Halstead/CK metrics per module

  Flask git commits (mined in Day 1)
  ├── Used for: process metrics (churn, commit count, ownership, etc.)
  └── Used for: AST features extracted from source_code snapshots

Pipeline
--------
1. Clean and validate KC1 columns.
2. Match KC1 modules → commit history to derive window_end per module.
3. Extract process metrics (churn, ownership, burst, …) from commits.
4. Extract AST features from the latest source snapshot per module.
5. Merge KC1 + process + AST into one feature matrix.
6. Impute missing values (zero-fill counts, median for the rest).
7. Apply log1p transform to skewed features.
8. Save to data/processed/feature_matrix.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from configs.config import FEATURES
from src.features.process_features import extract_process_features
from src.features.ast_features import extract_ast_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_PATH = Path("data/processed/feature_matrix.csv")

# KC1 columns that are direct features (already numeric, pass through as-is).
KC1_FEATURE_COLS = [
    "lines_of_code",
    "cyclomatic_complexity",
    "essential_complexity",
    "design_complexity",
    "halstead_length",
    "halstead_volume",
    "halstead_level",
    "halstead_difficulty",
    "halstead_intelligence",
    "halstead_effort",
    "halstead_bugs",
    "halstead_time",
    "lines_of_code_clean",
    "lines_of_comments",
    "lines_blank",
    "lines_code_and_comment",
    "unique_operators",
    "unique_operands",
    "total_operators",
    "total_operands",
    "branch_count",
]

# Right-skewed features that benefit from log1p compression.
LOG1P_FEATURES = [
    # KC1 / Halstead
    "lines_of_code",
    "halstead_length",
    "halstead_volume",
    "halstead_effort",
    "halstead_time",
    "total_operators",
    "total_operands",
    # Process
    "code_churn_30d",
    "code_churn_90d",
    "commit_count_30d",
    "commit_count_90d",
    "author_count_90d",
    "bug_fix_count_90d",
    "days_since_last_change",
    # AST
    "ast_node_count",
    "avg_function_length",
    "num_imports",
]

# Count / binary columns → zero-fill when missing.
ZERO_FILL_FEATURES = [
    "code_churn_30d",
    "code_churn_90d",
    "commit_count_30d",
    "commit_count_90d",
    "author_count_90d",
    "bug_fix_count_90d",
    "commit_burst_30d",
    "has_try_except",
    "num_functions",
    "num_classes",
    "num_imports",
    "ast_node_count",
    "max_nesting_depth",
    "max_args_per_function",
]


# ---------------------------------------------------------------------------
# KC1 preprocessing
# ---------------------------------------------------------------------------

def prepare_kc1(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the KC1 labeled dataset.

    - Casts ``is_buggy`` to int (0/1).
    - Drops non-feature metadata columns (repo_name, dataset_source, window_end).
    - Casts all present KC1_FEATURE_COLS to float.

    Parameters
    ----------
    labeled_df : Raw KC1 DataFrame loaded from CSV.

    Returns
    -------
    Cleaned DataFrame with file_path, is_buggy, and numeric KC1 feature cols.
    """
    df = labeled_df.copy()

    df["is_buggy"] = pd.to_numeric(df["is_buggy"], errors="coerce").fillna(0).astype(int)

    drop_cols = [c for c in ("repo_name", "dataset_source", "window_end") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.debug(f"Dropped metadata columns: {drop_cols}")

    present_kc1 = [c for c in KC1_FEATURE_COLS if c in df.columns]
    for col in present_kc1:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        f"KC1 prepared — {len(df):,} modules | "
        f"{len(present_kc1)} static feature cols | "
        f"buggy={df['is_buggy'].sum()} clean={(df['is_buggy'] == 0).sum()}"
    )
    return df


# ---------------------------------------------------------------------------
# Commit ↔ KC1 module matching
# ---------------------------------------------------------------------------

def _build_labeled_for_process(
    kc1_df: pd.DataFrame,
    commits_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive a window_end timestamp for each KC1 module so that process
    metrics can be computed over a meaningful time window.

    Matching strategy (in order):
    1. Exact match on the full ``file_path`` column.
    2. Bare filename match (e.g. ``KC1.c`` found in commits as ``src/KC1.c``).
    3. Fallback: global latest commit date (module gets zero-valued process features).

    Parameters
    ----------
    kc1_df     : Cleaned KC1 DataFrame (needs file_path, is_buggy).
    commits_df : Raw commits DataFrame (needs file_path, filename, commit_date).

    Returns
    -------
    DataFrame with columns [file_path, window_end, is_buggy].
    """
    last_by_filepath = (
        commits_df.groupby("file_path")["commit_date"]
        .max()
        .reset_index()
        .rename(columns={"commit_date": "window_end"})
    )
    last_by_filename = (
        commits_df.groupby("filename")["commit_date"]
        .max()
        .reset_index()
        .rename(columns={"filename": "file_path", "commit_date": "window_end"})
    )
    global_max = commits_df["commit_date"].max()

    result = kc1_df[["file_path", "is_buggy"]].copy()

    # Pass 1: full path match
    result = result.merge(last_by_filepath, on="file_path", how="left")

    # Pass 2: bare filename match for unresolved rows
    unmatched_mask = result["window_end"].isna()
    if unmatched_mask.any():
        bare_names = result.loc[unmatched_mask, "file_path"].apply(lambda p: Path(p).name)
        tmp = pd.DataFrame({"file_path": result.loc[unmatched_mask, "file_path"].values,
                            "_bare": bare_names.values})
        tmp = tmp.merge(
            last_by_filename.rename(columns={"file_path": "_bare"}),
            on="_bare", how="left"
        )
        result.loc[unmatched_mask, "window_end"] = tmp["window_end"].values

    # Pass 3: global fallback
    still_unmatched = result["window_end"].isna().sum()
    if still_unmatched:
        logger.warning(
            f"{still_unmatched} KC1 module(s) had no matching commits — "
            f"using global fallback date ({global_max.date()}). "
            f"Their process features will be all zeros."
        )
        result["window_end"] = result["window_end"].fillna(global_max)

    matched = len(result) - still_unmatched
    logger.info(
        f"window_end resolved — {matched:,} from git history, "
        f"{still_unmatched} fallback (total {len(result):,})"
    )
    return result


# ---------------------------------------------------------------------------
# Source code resolution
# ---------------------------------------------------------------------------

def _get_source_for_module(
    file_path: str,
    window_end: pd.Timestamp,
    commits_df: pd.DataFrame,
) -> Optional[str]:
    """
    Return the most recent source_code snapshot for *file_path* up to *window_end*.

    Matches on both full ``file_path`` and bare ``filename`` to maximise hit rate
    between KC1 module names and git paths.

    Parameters
    ----------
    file_path  : KC1 module identifier (may be a bare name like ``KC1.c``).
    window_end : Latest allowable commit date.
    commits_df : Raw commits DataFrame (must have source_code column).

    Returns
    -------
    Source code string, or None if unavailable / not Python.
    """
    if "source_code" not in commits_df.columns:
        return None

    bare = Path(file_path).name
    mask = (
        (commits_df["file_path"].eq(file_path) | commits_df["filename"].eq(bare))
        & (commits_df["commit_date"] <= window_end)
    )
    candidates = commits_df[mask].dropna(subset=["source_code"])

    if candidates.empty:
        return None

    src = candidates.sort_values("commit_date").iloc[-1]["source_code"]
    return src if isinstance(src, str) and src.strip() else None


# ---------------------------------------------------------------------------
# Imputation & transformation
# ---------------------------------------------------------------------------

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing numeric values in the feature matrix.

    - ZERO_FILL_FEATURES → 0.
    - All other numeric columns → column median.
    - ``is_buggy`` is never modified.

    Parameters
    ----------
    df : Feature matrix that may contain NaNs.

    Returns
    -------
    DataFrame with no missing values in numeric feature columns.
    """
    logger.info("Imputing missing values …")
    df = df.copy()

    for col in ZERO_FILL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    for col in df.select_dtypes(include=[np.number]).columns:
        if col == "is_buggy":
            continue
        if df[col].isna().all():
            df[col] = df[col].fillna(0)
            logger.debug(f"  zero-fill '{col}' → all values missing")
            continue
        if df[col].isna().any():
            med = df[col].median()
            if pd.isna(med):
                df[col] = df[col].fillna(0)
                logger.debug(f"  zero-fill '{col}' → median unavailable")
            else:
                df[col] = df[col].fillna(med)
                logger.debug(f"  median-fill '{col}' → {med:.4f}")

    return df


def apply_log1p_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p to heavily right-skewed features to compress their range.

    Clips to 0 first to guard against any negative edge cases.

    Parameters
    ----------
    df : Feature matrix (post-imputation).

    Returns
    -------
    DataFrame with transformed columns (names unchanged).
    """
    logger.info("Applying log1p transforms …")
    df = df.copy()
    applied = [c for c in LOG1P_FEATURES if c in df.columns]
    for col in applied:
        df[col] = np.log1p(df[col].clip(lower=0))
    logger.debug(f"  log1p applied to: {applied}")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_feature_matrix(
    commits_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build the complete feature matrix by fusing KC1, process, and AST features.

    Parameters
    ----------
    commits_df : Raw Flask commits DataFrame from Day 1 mining.
    labeled_df : KC1 labeled dataset (data/processed/kc1_labeled.csv).
    save       : If True, write the result to OUTPUT_PATH.

    Returns
    -------
    DataFrame — one row per KC1 module, all feature groups merged,
    ``is_buggy`` and ``file_path`` guaranteed present.
    """
    logger.info("=" * 60)
    logger.info("Building feature matrix …")
    logger.info(f"  commits rows  : {len(commits_df):,}")
    logger.info(f"  KC1 rows      : {len(labeled_df):,}")

    # ── 0. Normalise commit dates ─────────────────────────────────────────
    commits_df = commits_df.copy()
    commits_df["commit_date"] = (
        pd.to_datetime(commits_df["commit_date"], utc=True, errors="coerce")
        .dt.tz_localize(None)
    )

    # ── 1. KC1 static features ────────────────────────────────────────────
    logger.info("Step 1 — Preparing KC1 static features …")
    kc1_df = prepare_kc1(labeled_df)

    # ── 2. Derive window_end per module ───────────────────────────────────
    logger.info("Step 2 — Matching KC1 modules to commit history …")
    labeled_for_process = _build_labeled_for_process(kc1_df, commits_df)

    # ── 3. Process features ───────────────────────────────────────────────
    logger.info("Step 3 — Extracting process features …")
    process_df = extract_process_features(commits_df, labeled_for_process)
    # Drop columns that belong to KC1 (is_buggy) or are internal bookkeeping
    process_df = process_df.drop(columns=["is_buggy", "window_end"], errors="ignore")
    logger.info(f"  Process matrix: {process_df.shape}")

    # ── 4. AST features ───────────────────────────────────────────────────
    logger.info("Step 4 — Extracting AST features …")
    ast_records = []
    for _, row in labeled_for_process.iterrows():
        source = _get_source_for_module(row["file_path"], row["window_end"], commits_df)
        feats = extract_ast_features(source)
        feats["file_path"] = row["file_path"]
        ast_records.append(feats)
    ast_df = pd.DataFrame(ast_records)
    logger.info(f"  AST matrix: {ast_df.shape}")

    # ── 5. Merge ──────────────────────────────────────────────────────────
    logger.info("Step 5 — Merging KC1 + process + AST …")
    feature_df = kc1_df.merge(process_df, on="file_path", how="left")
    feature_df = feature_df.merge(ast_df, on="file_path", how="left")
    logger.info(f"  Merged shape: {feature_df.shape}")

    # ── 6. Imputation ─────────────────────────────────────────────────────
    feature_df = impute_missing_values(feature_df)

    # ── 7. Log1p ──────────────────────────────────────────────────────────
    feature_df = apply_log1p_transforms(feature_df)

    # ── 8. Column ordering ────────────────────────────────────────────────
    priority = ["file_path", "is_buggy"]
    rest = [c for c in feature_df.columns if c not in priority]
    feature_df = feature_df[priority + rest]

    # ── 9. Validation ─────────────────────────────────────────────────────
    assert "is_buggy" in feature_df.columns, "is_buggy label column missing!"
    assert "file_path" in feature_df.columns, "file_path identifier column missing!"
    remaining_nulls = feature_df.select_dtypes(include=[np.number]).isna().sum().sum()
    if remaining_nulls > 0:
        logger.warning(f"  {remaining_nulls} NaNs remain after imputation — check data.")

    n_buggy = int(feature_df["is_buggy"].sum())
    n_clean = int((feature_df["is_buggy"] == 0).sum())
    logger.info(f"  Label split   : buggy={n_buggy}, clean={n_clean}")
    logger.info(f"  Feature count : {len(rest)}")

    # ── 10. Save ──────────────────────────────────────────────────────────
    if save:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(OUTPUT_PATH, index=False)
        logger.success(f"Feature matrix saved → {OUTPUT_PATH}  {feature_df.shape}")

    return feature_df


def load_feature_matrix(path: str | Path = OUTPUT_PATH) -> pd.DataFrame:
    """
    Load a previously saved feature matrix from CSV.

    Parameters
    ----------
    path : Path to the feature matrix CSV.

    Returns
    -------
    DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {path}. Run the pipeline first."
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded feature matrix from {path}  shape={df.shape}")
    return df