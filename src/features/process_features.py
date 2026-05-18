"""
process_features.py
-------------------
Extract process (git-history) metrics for each (file_path, window_end) pair
from the raw commits DataFrame produced by the Day 1 mining pipeline.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import timedelta
from loguru import logger

from configs.config import FEATURES

SHORT_WINDOW = FEATURES["short_window_days"]
LONG_WINDOW = FEATURES["long_window_days"]
MIN_COMMITS = FEATURES["min_commits"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_datetime(series: pd.Series) -> pd.Series:
    """Coerce a Series to tz-naive datetime64."""
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    return dt.dt.tz_localize(None) if dt.dt.tz is not None else dt


def _window(df: pd.DataFrame, ref: pd.Timestamp, days: int) -> pd.DataFrame:
    """Return rows of *df* whose commit_date falls within *days* before *ref*."""
    cutoff = ref - timedelta(days=days)
    return df[(df["commit_date"] >= cutoff) & (df["commit_date"] <= ref)]


# ---------------------------------------------------------------------------
# per-metric functions
# ---------------------------------------------------------------------------

def compute_code_churn(file_df: pd.DataFrame, ref: pd.Timestamp) -> dict:
    """
    Compute total lines added+deleted (code churn) in 30-day and 90-day windows.

    Parameters
    ----------
    file_df : DataFrame filtered to a single file, all time.
    ref     : The reference date (window_end).

    Returns
    -------
    dict with keys code_churn_30d, code_churn_90d.
    """
    w30 = _window(file_df, ref, SHORT_WINDOW)
    w90 = _window(file_df, ref, LONG_WINDOW)
    return {
        "code_churn_30d": int(w30["lines_changed"].sum()),
        "code_churn_90d": int(w90["lines_changed"].sum()),
    }


def compute_commit_counts(file_df: pd.DataFrame, ref: pd.Timestamp) -> dict:
    """
    Compute number of distinct commits touching this file in 30- and 90-day windows.

    Parameters
    ----------
    file_df : DataFrame filtered to a single file.
    ref     : Reference date.

    Returns
    -------
    dict with keys commit_count_30d, commit_count_90d.
    """
    w30 = _window(file_df, ref, SHORT_WINDOW)
    w90 = _window(file_df, ref, LONG_WINDOW)
    return {
        "commit_count_30d": int(w30["commit_hash"].nunique()),
        "commit_count_90d": int(w90["commit_hash"].nunique()),
    }


def compute_author_count(file_df: pd.DataFrame, ref: pd.Timestamp) -> dict:
    """
    Count unique authors who touched this file in the 90-day window.

    Parameters
    ----------
    file_df : DataFrame filtered to a single file.
    ref     : Reference date.

    Returns
    -------
    dict with key author_count_90d.
    """
    w90 = _window(file_df, ref, LONG_WINDOW)
    return {"author_count_90d": int(w90["author_email"].nunique())}


def compute_bug_fix_metrics(file_df: pd.DataFrame, ref: pd.Timestamp) -> dict:
    """
    Compute bug-fix commit count (90-day) and fix density (lifetime).

    Parameters
    ----------
    file_df : DataFrame filtered to a single file.
    ref     : Reference date.

    Returns
    -------
    dict with keys bug_fix_count_90d, fix_density.
    """
    past = file_df[file_df["commit_date"] <= ref]
    w90 = _window(file_df, ref, LONG_WINDOW)

    bug_fix_90 = int(w90["is_bug_fix"].sum()) if "is_bug_fix" in w90.columns else 0
    total_lifetime = len(past)
    bug_lifetime = int(past["is_bug_fix"].sum()) if "is_bug_fix" in past.columns else 0

    fix_density = bug_lifetime / total_lifetime if total_lifetime > 0 else 0.0
    return {
        "bug_fix_count_90d": bug_fix_90,
        "fix_density": round(fix_density, 6),
    }


def compute_days_since_last_change(file_df: pd.DataFrame, ref: pd.Timestamp) -> dict:
    """
    Compute how many days elapsed since the most recent commit before *ref*.

    Parameters
    ----------
    file_df : DataFrame filtered to a single file.
    ref     : Reference date.

    Returns
    -------
    dict with key days_since_last_change (float; NaN if no prior commits).
    """
    past = file_df[file_df["commit_date"] <= ref]
    if past.empty:
        return {"days_since_last_change": np.nan}
    last = past["commit_date"].max()
    return {"days_since_last_change": float((ref - last).days)}


def compute_commit_burst(file_df: pd.DataFrame, ref: pd.Timestamp) -> dict:
    """
    Detect whether the 30-day commit count is unusually high vs the 90-day baseline.

    A burst is defined as the 30-day count being ≥ 2× the expected 30-day count
    derived from the 90-day window (i.e. commit_count_30d / (commit_count_90d/3) >= 2).

    Parameters
    ----------
    file_df : DataFrame filtered to a single file.
    ref     : Reference date.

    Returns
    -------
    dict with key commit_burst_30d (int 0/1).
    """
    counts = compute_commit_counts(file_df, ref)
    c30 = counts["commit_count_30d"]
    c90 = counts["commit_count_90d"]
    baseline = c90 / 3.0 if c90 > 0 else 0.0
    burst = int(c30 >= 2 * baseline and c30 > 0)
    return {"commit_burst_30d": burst}


def compute_ownership_score(file_df: pd.DataFrame, ref: pd.Timestamp) -> dict:
    """
    Compute the ownership score — fraction of commits from the top contributor (lifetime up to ref).

    A score close to 1.0 means one person owns the file (low diffusion).
    A score close to 0 means many contributors (high diffusion, higher risk).

    Parameters
    ----------
    file_df : DataFrame filtered to a single file.
    ref     : Reference date.

    Returns
    -------
    dict with key ownership_score (float in [0, 1]).
    """
    past = file_df[file_df["commit_date"] <= ref]
    if past.empty:
        return {"ownership_score": 0.0}
    counts = past["author_email"].value_counts()
    top_fraction = counts.iloc[0] / len(past) if len(past) > 0 else 0.0
    return {"ownership_score": round(float(top_fraction), 6)}


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def extract_process_features(
    commits_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For every row in *labeled_df* (identified by file_path + window_end),
    compute all process metrics from *commits_df*.

    Parameters
    ----------
    commits_df : Raw commits DataFrame from Day 1 (all repos/files).
    labeled_df : DataFrame with at minimum columns [file_path, window_end, is_buggy].

    Returns
    -------
    DataFrame with one row per (file_path, window_end) and all process feature columns.
    """
    logger.info("Starting process feature extraction …")

    commits_df = commits_df.copy()
    commits_df["commit_date"] = _to_datetime(commits_df["commit_date"])

    # Normalise is_bug_fix to bool
    if "is_bug_fix" in commits_df.columns:
        commits_df["is_bug_fix"] = commits_df["is_bug_fix"].fillna(False).astype(bool)

    labeled_df = labeled_df.copy()
    labeled_df["window_end"] = _to_datetime(labeled_df["window_end"])

    records = []
    skipped = 0
    empty_schema = {
        "file_path": pd.Series(dtype="object"),
        "window_end": pd.Series(dtype="datetime64[ns]"),
        "is_buggy": pd.Series(dtype="int64"),
        "code_churn_30d": pd.Series(dtype="float64"),
        "code_churn_90d": pd.Series(dtype="float64"),
        "commit_count_30d": pd.Series(dtype="float64"),
        "commit_count_90d": pd.Series(dtype="float64"),
        "author_count_90d": pd.Series(dtype="float64"),
        "bug_fix_count_90d": pd.Series(dtype="float64"),
        "fix_density": pd.Series(dtype="float64"),
        "days_since_last_change": pd.Series(dtype="float64"),
        "commit_burst_30d": pd.Series(dtype="float64"),
        "ownership_score": pd.Series(dtype="float64"),
    }

    for _, row in labeled_df.iterrows():
        fp = row["file_path"]
        ref = row["window_end"]

        file_df = commits_df[commits_df["file_path"] == fp].copy()

        if len(file_df) < MIN_COMMITS:
            skipped += 1
            logger.debug(f"Skipping {fp} — fewer than {MIN_COMMITS} commits.")
            continue

        rec: dict = {"file_path": fp, "window_end": ref, "is_buggy": row.get("is_buggy", np.nan)}

        rec.update(compute_code_churn(file_df, ref))
        rec.update(compute_commit_counts(file_df, ref))
        rec.update(compute_author_count(file_df, ref))
        rec.update(compute_bug_fix_metrics(file_df, ref))
        rec.update(compute_days_since_last_change(file_df, ref))
        rec.update(compute_commit_burst(file_df, ref))
        rec.update(compute_ownership_score(file_df, ref))

        records.append(rec)

    logger.info(f"Process features extracted for {len(records)} files; skipped {skipped} (< {MIN_COMMITS} commits).")
    if not records:
        return pd.DataFrame(empty_schema)

    df = pd.DataFrame(records)
    return df.reindex(columns=empty_schema.keys())