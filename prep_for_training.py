"""
prep_for_training.py
--------------------
Cleans and prepares the raw feature matrix into a model-ready dataset.

Steps
-----
1.  Load labeled_combined.csv
2.  Drop leaky and non-feature columns
3.  Preserve window_end as commit_date for temporal splitting
4.  Apply per-repo buggy rate sanity cap (max 60%)
5.  Per-repo median imputation with global median fallback
6.  Validate zero NaNs
7.  Print feature inventory and per-repo breakdown
8.  Save outputs:
        feature_matrix_clean.csv      — model input (no repo_name, no commit_date)
        feature_matrix_with_repo.csv  — model input + repo_name
        feature_matrix_final.csv      — model input + repo_name + commit_date (for temporal split)
        feature_inventory.txt         — human-readable feature list

Leaky columns removed
---------------------
    total_commits       — lifetime count, directly encodes label history
    bug_fix_commits     — literally counts bug-fix commits = label leakage
    bug_fix_count_90d   — redundant with fix_density, borderline leaky
    first_seen          — date string, not a numeric feature
    last_seen           — date string, not a numeric feature
    window_end          — preserved as commit_date then dropped from features
    commits_in_window   — windowed labeler artifact
    bug_fix_commits_in_window — windowed labeler artifact
    dataset_source      — metadata

Per-repo buggy rate cap
-----------------------
    Repos where SZZ over-labels (e.g. django "Fixed #XXXXX" on feature commits,
    sqlalchemy path mismatch) get their least-suspicious clean files restored.
    Cap is 50% — any repo above this threshold has excess buggy files flipped
    to clean, prioritising files with the fewest bug_touch_count.
    sqlalchemy is handled separately: dropped entirely if 0 files were mined
    with the new strict SZZ (indicates src_path mismatch) and replaced with
    the correctly-labeled subset from labeled_combined.csv.

Run
---
    python prep_for_training.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parent
LABELED_COMBINED = ROOT / "data" / "processed" / "labeled_combined.csv"
OUT_CLEAN        = ROOT / "data" / "processed" / "feature_matrix_clean.csv"
OUT_WITH_REPO    = ROOT / "data" / "processed" / "feature_matrix_with_repo.csv"
OUT_FINAL        = ROOT / "data" / "processed" / "feature_matrix_final.csv"
OUT_INVENTORY    = ROOT / "data" / "processed" / "feature_inventory.txt"

# ── columns that must never enter the model ───────────────────────────────────
LEAKY_OR_META = {
    "total_commits",
    "bug_fix_commits",
    "bug_fix_count_90d",
    "first_seen",
    "last_seen",
    "dataset_source",
    "commits_in_window",
    "bug_fix_commits_in_window",
    "window_end",
    # window_end is NOT in this set — we rename it to commit_date first
}

# Always kept, never treated as features
KEEP_ALWAYS = {"file_path", "is_buggy", "repo_name", "commit_date"}

# Per-repo maximum acceptable buggy rate
# Repos above this get excess buggy labels flipped to clean
MAX_BUGGY_RATE = 0.50

# Repos known to have unreliable SZZ due to commit message conventions
# These get a stricter cap
STRICT_CAP_REPOS = {
    "django":     0.35,   # "Fixed #XXXXX" used for features too
    "sqlalchemy": 0.40,   # src_path mismatch inflates labels
    "aiohttp":    0.45,
    "falcon":     0.45,
    "tornado":    0.45,
    "redis-py":   0.45,
    "werkzeug":   0.45,
    "click":      0.45,
    "celery":     0.45,
}

# Keywords used to classify features in the summary
PROCESS_KEYWORDS = [
    "churn", "commit_count", "author_count", "fix_density",
    "days_since", "burst", "ownership",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    if not LABELED_COMBINED.exists():
        logger.error(f"Required file not found: {LABELED_COMBINED}")
        logger.error("Run scripts/day5_mine_repos.py first.")
        sys.exit(1)

    df = pd.read_csv(LABELED_COMBINED)
    logger.info(f"Loaded: {df.shape}  columns: {list(df.columns)}")

    if "repo_name" not in df.columns:
        logger.error("labeled_combined.csv missing 'repo_name' column.")
        sys.exit(1)

    repos = sorted(df["repo_name"].unique())
    logger.info(f"Repos: {repos}")
    return df


def preserve_commit_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use last_seen (real last commit date per file) as commit_date.
    Prefer last_seen over window_end — window_end is synthetic (set to
    today at mining time) while last_seen is the actual last commit date
    extracted from git history.
    """
    if "last_seen" in df.columns:
        df = df.rename(columns={"last_seen": "commit_date"})
        logger.info(
            f"Using last_seen as commit_date "
            f"(range: {df['commit_date'].min()} to {df['commit_date'].max()})"
        )
        # Drop window_end since last_seen is better
        if "window_end" in df.columns:
            df = df.drop(columns=["window_end"])
    elif "window_end" in df.columns:
        df = df.rename(columns={"window_end": "commit_date"})
        logger.warning("last_seen not found — using window_end as commit_date (synthetic dates).")
    else:
        logger.warning("No date column found — temporal split will use stratified random.")
    return df


def drop_leaky(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in LEAKY_OR_META if c in df.columns]
    absent  = [c for c in LEAKY_OR_META if c not in df.columns]
    if present:
        df = df.drop(columns=present)
        logger.info(f"Dropped leaky/meta columns ({len(present)}): {present}")
    if absent:
        logger.debug(f"Already absent: {absent}")
    return df


def apply_buggy_rate_cap(df: pd.DataFrame) -> pd.DataFrame:
    """
    For repos where SZZ over-labels, flip excess buggy→clean.

    Strategy: within each over-labeled repo, sort files by
    fix_density ascending (least bug-fix activity = least likely
    to be truly buggy) and flip the excess ones to clean.
    """
    df = df.copy()
    rows_flipped_total = 0

    for repo, cap in {**{r: MAX_BUGGY_RATE for r in df["repo_name"].unique()},
                      **STRICT_CAP_REPOS}.items():
        mask_repo = df["repo_name"] == repo
        repo_df   = df[mask_repo]

        if len(repo_df) == 0:
            continue

        current_rate = repo_df["is_buggy"].mean()
        if current_rate <= cap:
            continue

        # How many buggy files to flip
        n_buggy    = int(repo_df["is_buggy"].sum())
        n_target   = int(len(repo_df) * cap)
        n_to_flip  = n_buggy - n_target

        if n_to_flip <= 0:
            continue

        # Sort buggy files by fix_density ascending — least suspicious first
        sort_col = "fix_density" if "fix_density" in df.columns else "is_buggy"
        buggy_idx = (
            repo_df[repo_df["is_buggy"] == 1]
            .sort_values(sort_col, ascending=True)
            .index[:n_to_flip]
        )
        df.loc[buggy_idx, "is_buggy"] = 0
        rows_flipped_total += len(buggy_idx)

        new_rate = df[mask_repo]["is_buggy"].mean()
        logger.info(
            f"  {repo}: {current_rate*100:.1f}% → {new_rate*100:.1f}% buggy "
            f"(flipped {len(buggy_idx)} files, cap={cap*100:.0f}%)"
        )

    logger.info(f"Buggy rate capping complete — {rows_flipped_total} files flipped total.")
    return df


def audit_nans(df: pd.DataFrame, feature_cols: list[str], label: str) -> None:
    nan_counts = df[feature_cols].isna().sum()
    nan_cols   = nan_counts[nan_counts > 0].sort_values(ascending=False)
    logger.info(f"--- NaN audit {label} ---")
    if nan_cols.empty:
        logger.info("  No NaNs found ✓")
        return
    for col, n in nan_cols.items():
        logger.info(f"  {col:<40} {n:>5}  ({n/len(df)*100:.1f}%)")


def impute(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df     = df.copy()
    to_fill = [c for c in feature_cols if df[c].isna().any()]
    if not to_fill:
        logger.info("Nothing to impute.")
        return df

    logger.info(f"Imputing {len(to_fill)} columns via per-repo median …")
    for col in to_fill:
        df[col] = df[col].fillna(
            df.groupby("repo_name")[col].transform("median")
        )
        still_nan = df[col].isna().sum()
        if still_nan:
            gm = df[col].median()
            df[col] = df[col].fillna(gm)
            logger.debug(f"  {col}: {still_nan} → global median {gm:.4f}")
    return df


def validate(df: pd.DataFrame, feature_cols: list[str]) -> None:
    total_nan = df[feature_cols].isna().sum().sum()
    if total_nan:
        logger.error(f"Imputation incomplete — {total_nan} NaNs remain!")
        sys.exit(1)
    assert "is_buggy"  in df.columns
    assert "file_path" in df.columns
    logger.success(
        f"Validation passed — zero NaNs across {len(feature_cols)} feature columns ✓"
    )


def classify_features(cols: list[str]) -> tuple[list[str], list[str]]:
    process = [c for c in cols if any(k in c for k in PROCESS_KEYWORDS)]
    ast     = [c for c in cols if c not in process]
    return process, ast


def print_summary(df: pd.DataFrame, feature_cols: list[str]) -> None:
    process_cols, ast_cols = classify_features(feature_cols)

    lines = []
    lines.append("=" * 62)
    lines.append("FINAL FEATURE MATRIX — READY FOR TRAINING")
    lines.append("=" * 62)
    lines.append(f"  Rows              : {len(df):,}")
    lines.append(f"  Feature columns   : {len(feature_cols)}")
    lines.append(f"  Buggy             : {int(df['is_buggy'].sum()):,}  "
                 f"({df['is_buggy'].mean():.1%})")
    lines.append(f"  Clean             : {int((df['is_buggy']==0).sum()):,}  "
                 f"({1-df['is_buggy'].mean():.1%})")
    lines.append(f"  Repos             : {df['repo_name'].nunique()}")

    if "commit_date" in df.columns:
        valid_dates = pd.to_datetime(df['commit_date'], errors='coerce').dropna()
        if not valid_dates.empty:
            lines.append(f"  Date range        : {valid_dates.min().strftime('%Y-%m-%d')} → "
                         f"{valid_dates.max().strftime('%Y-%m-%d')}")
        else:
            lines.append("  Date range        : Unknown (all NaT)")

    lines.append("")
    lines.append(f"  PROCESS features ({len(process_cols)}):")
    for c in process_cols:
        lines.append(f"    {c}")
    lines.append("")
    lines.append(f"  AST features ({len(ast_cols)}):")
    for c in ast_cols:
        lines.append(f"    {c}")
    lines.append("")
    lines.append("Per-repo breakdown:")

    per_repo = (
        df.groupby("repo_name")
        .agg(files=("file_path", "count"), bug_ratio=("is_buggy", "mean"))
        .assign(bug_ratio=lambda x: x["bug_ratio"].map("{:.1%}".format))
        .sort_values("files", ascending=False)
    )
    lines.append(per_repo.to_string())
    lines.append("=" * 62)

    for line in lines:
        logger.info(line)

    OUT_INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    OUT_INVENTORY.write_text("\n".join(lines),encoding="utf-8")
    logger.info(f"Inventory → {OUT_INVENTORY}")


def main() -> None:
    # 1. Load
    df = load()

    # 2. Preserve commit_date BEFORE dropping leaky columns
    df = preserve_commit_date(df)

    # 3. Drop leaky columns (window_end already renamed, won't be dropped)
    df = drop_leaky(df)

    # 4. Identify feature columns
    feature_cols = [c for c in df.columns if c not in KEEP_ALWAYS]
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # 5. Apply buggy rate cap BEFORE imputation
    logger.info("Applying per-repo buggy rate caps ...")
    df = apply_buggy_rate_cap(df)

    # 6. Audit NaNs before imputation
    audit_nans(df, feature_cols, "BEFORE imputation")

    # 7. Impute
    df = impute(df, feature_cols)

    # 8. Audit NaNs after imputation
    audit_nans(df, feature_cols, "AFTER imputation")

    # 9. Validate
    validate(df, feature_cols)

    # 10. Summary
    print_summary(df, feature_cols)

    # 11. Save three outputs
    OUT_CLEAN.parent.mkdir(parents=True, exist_ok=True)

    # Clean: features + labels only, no repo_name, no commit_date
    clean_cols = ["file_path", "is_buggy"] + feature_cols
    df[clean_cols].to_csv(OUT_CLEAN, index=False)
    logger.success(f"Clean matrix        → {OUT_CLEAN}  {df[clean_cols].shape}")

    # With repo: adds repo_name for stratified splits
    with_repo_cols = ["repo_name", "file_path", "is_buggy"] + feature_cols
    df[with_repo_cols].to_csv(OUT_WITH_REPO, index=False)
    logger.success(f"With repo           → {OUT_WITH_REPO}  {df[with_repo_cols].shape}")

    # Final: adds commit_date for temporal splitting in day4_run.py
    final_cols = ["repo_name", "file_path", "is_buggy", "commit_date"] + feature_cols
    df[final_cols].to_csv(OUT_FINAL, index=False)
    logger.success(f"Final (with dates)  → {OUT_FINAL}  {df[final_cols].shape}")

    logger.success("prep_for_training.py complete.")
    print()
    print("Next: python scripts/day4_run.py")


if __name__ == "__main__":
    main()