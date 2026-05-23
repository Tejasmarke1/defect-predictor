"""
feature_pipeline.py
-------------------
Orchestrates the full Day 2 feature engineering pipeline.

Data sources
------------
KC1 (NASA static software metrics — C codebase)
  Already contains: lines_of_code, cyclomatic_complexity, halstead_*, branch_count, …
  Provides: is_buggy labels + all classical static metrics.

Flask git commits (mined in Day 1 — Python codebase)
  Used for: process metrics (churn, ownership, burst, …) when a KC1 module
  filename happens to match a file in the commit history.
  Used for: AST structural features extracted from source_code snapshots.

IMPORTANT — cross-dataset reality check
----------------------------------------
KC1 is a NASA C project; the Flask commits are Python. Their filenames will
almost never overlap, so process and AST features will be NaN/missing for
most KC1 modules after the merge. This is expected and correct behaviour.
The pipeline does NOT zero-fill these columns — it leaves them as NaN so
that the model knows the signal is absent, not zero.

If you want process features to matter, mine commits from the same C repo
that produced KC1 (NASA IV&V facility) and pass those as commits_df.

Column collision resolution
----------------------------
Both KC1 and the AST extractor produce a 'cyclomatic_complexity' column.
KC1's version (McCabe, from the PROMISE dataset) is authoritative and is
kept as-is. The AST version is renamed to 'ast_cyclomatic_complexity' to
avoid pandas _x/_y suffix noise.

Pipeline steps
--------------
1.  Clean and validate KC1 columns.
2.  Attempt to match KC1 filenames → commit history for window_end.
3.  Extract process metrics (will be NaN for unmatched modules).
4.  Extract AST features from source snapshots (will be NaN for unmatched).
5.  Resolve column collisions before merging.
6.  Merge KC1 + process + AST.
7.  Impute: NaN for process/AST columns → NaN kept (not zero-filled) so the
    model can use a missing-indicator; only genuine count features are 0-filled.
8.  Apply log1p to skewed KC1 Halstead features.
9.  Save to data/processed/feature_matrix.csv.
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

KC1_FEATURE_COLS = [
    "lines_of_code",
    "cyclomatic_complexity",   # KC1 / McCabe — authoritative
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

# AST extractor outputs this name too — rename it before merging.
AST_CYCLOMATIC_RENAME = {
    "cyclomatic_complexity": "ast_cyclomatic_complexity",
    "cognitive_complexity":  "ast_cognitive_complexity",
}

# Log1p: only KC1 Halstead features (already present and numeric).
# Process/AST columns intentionally excluded — they're mostly NaN for KC1.
LOG1P_FEATURES = [
    "lines_of_code",
    "halstead_length",
    "halstead_volume",
    "halstead_effort",
    "halstead_time",
    "total_operators",
    "total_operands",
]

# Process count columns: fill with 0 only when a git match WAS found
# (i.e. window_end came from commit history, not the fallback).
# Handled explicitly in impute_missing_values via the has_git_match flag.
PROCESS_COUNT_COLS = [
    "code_churn_30d",
    "code_churn_90d",
    "commit_count_30d",
    "commit_count_90d",
    "author_count_90d",
    "bug_fix_count_90d",
    "commit_burst_30d",
]


# ---------------------------------------------------------------------------
# KC1 preprocessing
# ---------------------------------------------------------------------------

def prepare_kc1(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the KC1 labeled dataset.

    - Cast is_buggy to int (0/1).
    - Drop non-feature metadata columns.
    - Cast KC1_FEATURE_COLS to float (coerce non-numeric to NaN).
    - Rename lines_code_and_comment if needed (column name varies across
      PROMISE downloads: 'lines_code_and_comment' vs 'loccodeandcomment').

    Parameters
    ----------
    labeled_df : Raw KC1 DataFrame from CSV.

    Returns
    -------
    Cleaned DataFrame with file_path, is_buggy, and numeric KC1 feature cols.
    """
    df = labeled_df.copy()

    # Normalise common column name variant
    if "loccodeandcomment" in df.columns and "lines_code_and_comment" not in df.columns:
        df = df.rename(columns={"loccodeandcomment": "lines_code_and_comment"})

    df["is_buggy"] = pd.to_numeric(df["is_buggy"], errors="coerce").fillna(0).astype(int)

    drop_cols = [c for c in ("repo_name", "dataset_source", "window_end") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.debug(f"Dropped metadata columns: {drop_cols}")

    present = [c for c in KC1_FEATURE_COLS if c in df.columns]
    missing = [c for c in KC1_FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning(f"KC1 feature cols not found (will be absent from matrix): {missing}")
    for col in present:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        f"KC1 prepared — {len(df):,} modules | "
        f"{len(present)} static cols | "
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
    Attempt to match each KC1 module to a commit history entry to derive window_end.

    Returns a DataFrame with [file_path, window_end, is_buggy, has_git_match].
    ``has_git_match`` is True only when a real commit was found — the pipeline
    uses this flag to decide whether process features are meaningful or absent.

    Matching order:
      1. Exact file_path match.
      2. Bare filename match (e.g. KC1.c vs src/KC1.c).
      3. Fallback: global latest commit date + has_git_match=False.

    Parameters
    ----------
    kc1_df     : Cleaned KC1 DataFrame.
    commits_df : Raw commits DataFrame.

    Returns
    -------
    DataFrame with columns [file_path, window_end, is_buggy, has_git_match].
    """
    last_by_filepath = (
        commits_df.groupby("file_path")["commit_date"].max()
        .reset_index().rename(columns={"commit_date": "window_end"})
    )
    last_by_filename = (
        commits_df.groupby("filename")["commit_date"].max()
        .reset_index().rename(columns={"filename": "file_path", "commit_date": "window_end"})
    )
    global_max = commits_df["commit_date"].max()

    result = kc1_df[["file_path", "is_buggy"]].copy()
    result["has_git_match"] = False

    # Pass 1: exact file_path
    result = result.merge(last_by_filepath, on="file_path", how="left")
    matched_1 = result["window_end"].notna()
    result.loc[matched_1, "has_git_match"] = True

    # Pass 2: bare filename for still-unmatched rows
    unmatched = ~matched_1
    if unmatched.any():
        bare = result.loc[unmatched, "file_path"].apply(lambda p: Path(p).name)
        tmp = pd.DataFrame({
            "file_path": result.loc[unmatched, "file_path"].values,
            "_bare": bare.values,
        }).merge(
            last_by_filename.rename(columns={"file_path": "_bare"}),
            on="_bare", how="left"
        )
        new_window = tmp["window_end"].values
        result.loc[unmatched, "window_end"] = new_window
        newly_matched = unmatched & result["window_end"].notna()
        result.loc[newly_matched, "has_git_match"] = True

    # Pass 3: global fallback for remaining unmatched
    still_unmatched = result["window_end"].isna().sum()
    result["window_end"] = result["window_end"].fillna(global_max)

    matched_total = result["has_git_match"].sum()
    logger.info(
        f"Module→commit matching: {matched_total} matched from git history, "
        f"{still_unmatched} fallback (total {len(result):,}). "
        f"Process/AST features will be NaN for fallback modules."
    )
    return result


# ---------------------------------------------------------------------------
# Source code resolution
# ---------------------------------------------------------------------------

def _get_source_for_module(
    file_path: str,
    window_end: pd.Timestamp,
    has_git_match: bool,
    commits_df: pd.DataFrame,
) -> Optional[str]:
    """
    Retrieve the most recent source_code snapshot up to window_end.

    Returns None immediately when has_git_match is False (no point searching).

    Parameters
    ----------
    file_path     : KC1 module identifier.
    window_end    : Latest allowable commit date.
    has_git_match : Whether this module has a real commit history entry.
    commits_df    : Raw commits DataFrame.

    Returns
    -------
    Source code string, or None.
    """
    if not has_git_match or "source_code" not in commits_df.columns:
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
    Selectively fill missing values.

    Strategy
    --------
    - KC1 static features: median-fill (they should never be NaN after prepare_kc1,
      but guard anyway).
    - Process count columns: leave as NaN — missing means no git history, not zero.
      The XGBoost model handles NaN natively; a missing-indicator is more honest
      than zero-filling and falsely implying zero churn.
    - is_buggy: never touched.

    Parameters
    ----------
    df : Feature matrix (may contain NaNs in process/AST columns).

    Returns
    -------
    DataFrame with KC1 NaNs filled; process/AST NaNs preserved.
    """
    logger.info("Imputing missing values …")
    df = df.copy()

    kc1_numeric = [c for c in KC1_FEATURE_COLS if c in df.columns]
    for col in kc1_numeric:
        if df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
            logger.debug(f"  KC1 median-fill '{col}' → {med:.4f}")

    remaining = df.select_dtypes(include=[np.number]).isna().sum()
    remaining = remaining[remaining > 0]
    if not remaining.empty:
        logger.info(
            f"  NaNs preserved in {len(remaining)} process/AST cols "
            f"(no git match for those modules) — XGBoost handles NaN natively."
        )
    return df


def apply_log1p_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p to KC1 Halstead / LOC features (right-skewed, always present).

    Process and AST features are intentionally excluded — they are mostly NaN
    for KC1, and transforming NaN produces NaN anyway.

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
    Build the complete feature matrix.

    When KC1 modules cannot be matched to the commit history (which is
    expected when using a C dataset with Python commits), the process and
    AST feature columns will contain NaN.  This is intentional — the static
    KC1 features alone are valid and complete for model training.

    Parameters
    ----------
    commits_df : Raw commits DataFrame from Day 1.
    labeled_df : KC1 labeled dataset.
    save       : Write result to OUTPUT_PATH when True.

    Returns
    -------
    DataFrame — one row per KC1 module, is_buggy and file_path guaranteed.
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

    # ── 2. Match modules → commit history ────────────────────────────────
    logger.info("Step 2 — Matching KC1 modules to commit history …")
    labeled_for_process = _build_labeled_for_process(kc1_df, commits_df)

    # ── 3. Process features ───────────────────────────────────────────────
    logger.info("Step 3 — Extracting process features …")
    # Only run on matched modules to avoid polluting unmatched ones with zeros
    matched_df = labeled_for_process[labeled_for_process["has_git_match"]].copy()
    if matched_df.empty:
        logger.warning(
            "  No KC1 modules matched the commit history. "
            "Process features will be all NaN. This is expected when "
            "commits_df is from a different codebase than KC1."
        )
        process_df = pd.DataFrame(columns=["file_path"])
    else:
        process_df = extract_process_features(commits_df, matched_df)
        process_df = process_df.drop(columns=["is_buggy", "window_end"], errors="ignore")
        logger.info(f"  Process matrix: {process_df.shape} (matched modules only)")

    # ── 4. AST features ───────────────────────────────────────────────────
    logger.info("Step 4 — Extracting AST features …")
    ast_records = []
    for _, row in labeled_for_process.iterrows():
        source = _get_source_for_module(
            row["file_path"], row["window_end"],
            bool(row["has_git_match"]), commits_df
        )
        feats = extract_ast_features(source)
        # Rename colliding column names BEFORE merge
        feats = {AST_CYCLOMATIC_RENAME.get(k, k): v for k, v in feats.items()}
        feats["file_path"] = row["file_path"]
        # Mark as all-NaN for unmatched modules (source was None)
        if source is None:
            feats = {k: (np.nan if k != "file_path" else v) for k, v in feats.items()}
        ast_records.append(feats)
    ast_df = pd.DataFrame(ast_records)
    logger.info(f"  AST matrix: {ast_df.shape}")

    # ── 5. Merge ──────────────────────────────────────────────────────────
    logger.info("Step 5 — Merging KC1 + process + AST …")
    feature_df = kc1_df.merge(process_df, on="file_path", how="left")
    feature_df = feature_df.merge(ast_df,     on="file_path", how="left")

    # Sanity check: no _x/_y suffix columns should exist
    collision_cols = [c for c in feature_df.columns if c.endswith("_x") or c.endswith("_y")]
    if collision_cols:
        logger.warning(f"  Column merge collisions detected: {collision_cols}")
        # Drop the _y duplicates (KC1 _x is authoritative) and strip suffix
        drop = [c for c in collision_cols if c.endswith("_y")]
        feature_df = feature_df.drop(columns=drop)
        feature_df.columns = [c.removesuffix("_x") for c in feature_df.columns]
        logger.info(f"  Collision resolved — dropped {drop}")

    logger.info(f"  Merged shape: {feature_df.shape}")

    # ── 6. Imputation ─────────────────────────────────────────────────────
    feature_df = impute_missing_values(feature_df)

    # ── 7. Log1p (KC1 Halstead/LOC only) ─────────────────────────────────
    feature_df = apply_log1p_transforms(feature_df)

    # ── 8. Drop internal bookkeeping column ──────────────────────────────
    feature_df = feature_df.drop(columns=["has_git_match"], errors="ignore")

    # ── 9. Column ordering ────────────────────────────────────────────────
    priority = ["file_path", "is_buggy"]
    rest = [c for c in feature_df.columns if c not in priority]
    feature_df = feature_df[priority + rest]

    # ── 10. Validation ────────────────────────────────────────────────────
    assert "is_buggy"   in feature_df.columns
    assert "file_path"  in feature_df.columns

    n_buggy = int(feature_df["is_buggy"].sum())
    n_clean = int((feature_df["is_buggy"] == 0).sum())
    kc1_nan  = feature_df[[c for c in KC1_FEATURE_COLS if c in feature_df.columns]].isna().sum().sum()
    proc_nan = feature_df[[c for c in feature_df.columns if "churn" in c or "commit" in c]].isna().sum().sum()

    logger.info(f"  Label split       : buggy={n_buggy}, clean={n_clean}")
    logger.info(f"  Feature count     : {len(rest)}")
    logger.info(f"  KC1 NaNs remaining: {kc1_nan}  (should be 0)")
    logger.info(f"  Process NaNs      : {proc_nan}  (expected when no git match)")

    # ── 11. Save ──────────────────────────────────────────────────────────
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