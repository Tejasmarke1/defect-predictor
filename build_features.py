"""
build_features_fast.py
----------------------
Replaces build_features_combined.py for the windowed dataset.

Key optimisations vs the original pipeline
-------------------------------------------
1. AST deduplication  — parse each unique (file_path, window_end) ONCE,
                        then join back to all (file, window) rows.
                        Reduces AST calls from ~46k to ~8-10k.

2. Vectorised process — group commits by file_path upfront, compute all
                        window metrics with pandas vectorised ops instead
                        of a Python loop over labeled_df rows.

3. Progress bars      — tqdm on both steps so you can see real progress.

Expected runtime: 3-6 minutes (vs 30+ min original).

Run
---
    python build_features_fast.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.ast_features import extract_ast_features  # noqa: E402

# ── paths ─────────────────────────────────────────────────────────────────────
COMMITS_PATH  = Path("data/raw/combined_commits.csv")
LABELED_PATH  = Path("data/processed/labeled_windowed.csv")
OUT_MATRIX    = Path("data/processed/feature_matrix.csv")

# ── process feature windows ───────────────────────────────────────────────────
SHORT_DAYS = 30
LONG_DAYS  = 90


# ─────────────────────────────────────────────────────────────────────────────
# 1. Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    for p in (COMMITS_PATH, LABELED_PATH):
        if not p.exists():
            logger.error(f"Required file not found: {p}")
            sys.exit(1)

    commits = pd.read_csv(COMMITS_PATH)
    commits["commit_date"] = (
        pd.to_datetime(commits["commit_date"], utc=True, errors="coerce")
        .dt.tz_localize(None)
    )
    commits = commits.dropna(subset=["commit_date"])
    commits["is_bug_fix"] = commits["is_bug_fix"].fillna(False).astype(bool)
    commits["lines_changed"] = commits["lines_changed"].fillna(0)
    commits["file_path"] = commits["file_path"].fillna("unknown").str.replace("\\", "/", regex=False)

    labeled = pd.read_csv(LABELED_PATH)
    labeled["window_end"] = pd.to_datetime(labeled["window_end"], errors="coerce")
    labeled["is_buggy"]   = labeled["is_buggy"].astype(int)
    labeled["file_path"]  = labeled["file_path"].str.replace("\\", "/", regex=False)

    logger.info(f"Commits  : {len(commits):,} rows | {commits['repo_name'].nunique()} repos")
    logger.info(f"Labeled  : {len(labeled):,} (file, window) pairs | bug ratio: {labeled['is_buggy'].mean():.1%}")
    return commits, labeled


# ─────────────────────────────────────────────────────────────────────────────
# 2. Vectorised process features
# ─────────────────────────────────────────────────────────────────────────────

def _compute_process_for_row(
    file_commits: pd.DataFrame,
    window_end: pd.Timestamp,
) -> dict:
    """Compute all process metrics for one (file, window_end) pair."""
    past   = file_commits[file_commits["commit_date"] <= window_end]
    w30    = past[past["commit_date"] >= window_end - pd.Timedelta(days=SHORT_DAYS)]
    w90    = past[past["commit_date"] >= window_end - pd.Timedelta(days=LONG_DAYS)]

    c30 = w30["commit_hash"].nunique()
    c90 = w90["commit_hash"].nunique()

    total = len(past)
    bug_lifetime = int(past["is_bug_fix"].sum()) if total > 0 else 0
    fix_density  = bug_lifetime / total if total > 0 else 0.0

    last_date = past["commit_date"].max() if not past.empty else None
    days_since = float((window_end - last_date).days) if last_date is not None else np.nan

    ownership = 0.0
    if not past.empty and "author_email" in past.columns:
        counts = past["author_email"].value_counts()
        ownership = float(counts.iloc[0] / len(past))

    baseline = c90 / 3.0 if c90 > 0 else 0.0
    burst = int(c30 >= 2 * baseline and c30 > 0)

    return {
        "code_churn_30d":        int(w30["lines_changed"].sum()),
        "code_churn_90d":        int(w90["lines_changed"].sum()),
        "commit_count_30d":      c30,
        "commit_count_90d":      c90,
        "author_count_90d":      int(w90["author_email"].nunique()) if "author_email" in w90.columns else 0,
        "fix_density":           round(fix_density, 6),
        "days_since_last_change": days_since,
        "commit_burst_30d":      burst,
        "ownership_score":       round(ownership, 6),
    }


def extract_process_features_fast(
    commits: pd.DataFrame,
    labeled: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute process features for all (file_path, window_end) pairs.

    Groups commits by file_path first — each file's history is loaded
    once, then all its windows are processed in a tight inner loop.
    Much faster than the original which re-filtered the full commits
    DataFrame for every single row.
    """
    logger.info("Extracting process features (grouped by file) ...")

    # Pre-group commits by file_path — O(n) once instead of O(n*k)
    commit_groups = {
        fp: grp.reset_index(drop=True)
        for fp, grp in commits.groupby("file_path")
    }

    # All unique (file_path, window_end) pairs to compute
    pairs = (
        labeled[["file_path", "window_end"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    logger.info(f"  Unique (file, window_end) pairs: {len(pairs):,}")

    records = []
    skipped = 0

    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Process features"):
        fp  = row["file_path"]
        ref = row["window_end"]

        file_commits = commit_groups.get(fp)
        if file_commits is None or len(file_commits) < 1:
            skipped += 1
            continue

        rec = {"file_path": fp, "window_end": ref}
        rec.update(_compute_process_for_row(file_commits, ref))
        records.append(rec)

    logger.info(f"  Process features: {len(records):,} pairs computed, {skipped} skipped (no commits)")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Deduplicated AST features
# ─────────────────────────────────────────────────────────────────────────────

def _get_source(
    file_path: str,
    window_end: pd.Timestamp,
    commit_groups: dict[str, pd.DataFrame],
) -> str | None:
    """Get most recent source snapshot for file_path up to window_end."""
    file_commits = commit_groups.get(file_path)
    if file_commits is None or "source_code" not in file_commits.columns:
        return None

    candidates = file_commits[
        file_commits["commit_date"] <= window_end
    ].dropna(subset=["source_code"])

    if candidates.empty:
        return None

    src = candidates.sort_values("commit_date").iloc[-1]["source_code"]
    return src if isinstance(src, str) and src.strip() else None


def extract_ast_features_fast(
    commits: pd.DataFrame,
    labeled: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract AST features for all (file_path, window_end) pairs,
    but only parse each unique combination ONCE.

    The same file at the same window_end appears many times in the
    labeled DataFrame (once per overlapping window slice) — deduplicating
    before parsing reduces AST calls by ~5-8x.
    """
    logger.info("Extracting AST features (deduplicated) ...")

    commit_groups = {
        fp: grp.reset_index(drop=True)
        for fp, grp in commits.groupby("file_path")
    }

    # Unique (file_path, window_end) only
    pairs = (
        labeled[["file_path", "window_end"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    logger.info(f"  Unique (file, window_end) pairs: {len(pairs):,}  (vs {len(labeled):,} total rows)")

    records = []
    parse_ok = 0
    parse_fail = 0

    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="AST features"):
        fp  = row["file_path"]
        ref = row["window_end"]

        source = _get_source(fp, ref, commit_groups)
        feats  = extract_ast_features(source)

        if source is not None:
            parse_ok += 1
        else:
            parse_fail += 1
            # All-zero defaults already returned by extract_ast_features(None)

        # Rename colliding columns before merge
        rename = {
            "cyclomatic_complexity": "ast_cyclomatic_complexity",
            "cognitive_complexity":  "ast_cognitive_complexity",
        }
        feats = {rename.get(k, k): v for k, v in feats.items()}
        feats["file_path"]  = fp
        feats["window_end"] = ref
        records.append(feats)

    logger.info(f"  AST parsed: {parse_ok:,} ok | {parse_fail:,} no source (zeros used)")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Imputation
# ─────────────────────────────────────────────────────────────────────────────

def impute(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Per-repo median imputation with global median fallback."""
    df = df.copy()
    to_fill = [c for c in feature_cols if df[c].isna().any()]
    if not to_fill:
        return df

    logger.info(f"Imputing {len(to_fill)} columns ...")
    for col in to_fill:
        df[col] = df[col].fillna(
            df.groupby("repo_name")[col].transform("median")
        )
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("Fast feature pipeline — windowed dataset")
    logger.info("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    commits, labeled = load_data()

    # ── Process features ──────────────────────────────────────────────────
    process_df = extract_process_features_fast(commits, labeled)

    # ── AST features ──────────────────────────────────────────────────────
    ast_df = extract_ast_features_fast(commits, labeled)

    # ── Merge everything ──────────────────────────────────────────────────
    logger.info("Merging labeled + process + AST ...")

    feature_df = labeled.merge(
        process_df, on=["file_path", "window_end"], how="left"
    )
    feature_df = feature_df.merge(
        ast_df, on=["file_path", "window_end"], how="left"
    )

    logger.info(f"Merged shape: {feature_df.shape}")

    # ── Impute ────────────────────────────────────────────────────────────
    meta_cols    = {"file_path", "is_buggy", "window_start", "window_end",
                    "repo_name", "commits_in_window"}
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]
    feature_df   = impute(feature_df, feature_cols)

    # ── Validate ──────────────────────────────────────────────────────────
    nan_total = feature_df[feature_cols].isna().sum().sum()
    if nan_total:
        logger.warning(f"{nan_total} NaNs remain after imputation (check for empty repos)")
    else:
        logger.success("Zero NaNs in feature columns")

    # ── Final summary ─────────────────────────────────────────────────────
    n_buggy = int(feature_df["is_buggy"].sum())
    n_clean = int((feature_df["is_buggy"] == 0).sum())

    logger.info("=" * 60)
    logger.info(f"  Final shape   : {feature_df.shape}")
    logger.info(f"  Buggy         : {n_buggy:,}  ({n_buggy/len(feature_df):.1%})")
    logger.info(f"  Clean         : {n_clean:,}  ({n_clean/len(feature_df):.1%})")
    logger.info(f"  Features      : {len(feature_cols)}")
    logger.info(f"  Repos         : {feature_df['repo_name'].nunique()}")

    per_repo = (
        feature_df.groupby("repo_name")
        .agg(windows=("file_path","count"), bug_ratio=("is_buggy","mean"))
        .assign(bug_ratio=lambda x: x["bug_ratio"].map("{:.1%}".format))
        .sort_values("windows", ascending=False)
    )
    logger.info(f"\n{per_repo.to_string()}")
    logger.info("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────────
    OUT_MATRIX.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(OUT_MATRIX, index=False)
    logger.success(f"Feature matrix → {OUT_MATRIX}  {feature_df.shape}")

    logger.info("Next: run prep_for_training.py to drop leaky columns and impute")


if __name__ == "__main__":
    main()
    
    
    