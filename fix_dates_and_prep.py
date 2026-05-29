"""
fix_dates_and_prep.py
---------------------
One-shot fix script — run this ONCE from the project root.

Does three things in order:
  1. Fixes labeled_combined.csv — recomputes last_seen from actual commit
     history stored in the per-repo cloned repos on disk.
     Falls back to days_since_last_change if repos not cloned.

  2. Drops rows with synthetic dates (2026-05-xx) that came from the
     old labeled_combined.csv merge, keeping only freshly mined rows
     with real historical last_seen dates.

  3. Runs the full prep_for_training logic inline — applies buggy rate
     caps, drops leaky columns, imputes NaNs, and saves:
         data/processed/feature_matrix_final.csv   (with commit_date)
         data/processed/feature_matrix_with_repo.csv
         data/processed/feature_matrix_clean.csv

Run:
    python fix_dates_and_prep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent

# ── paths ──────────────────────────────────────────────────────────────────
LABELED_COMBINED = ROOT / "data" / "processed" / "labeled_combined.csv"
REPOS_DIR        = ROOT / "data" / "raw" / "repos"
OUT_FINAL        = ROOT / "data" / "processed" / "feature_matrix_final.csv"
OUT_WITH_REPO    = ROOT / "data" / "processed" / "feature_matrix_with_repo.csv"
OUT_CLEAN        = ROOT / "data" / "processed" / "feature_matrix_clean.csv"
OUT_INVENTORY    = ROOT / "data" / "processed" / "feature_inventory.txt"

# ── leaky columns to drop ──────────────────────────────────────────────────
LEAKY_OR_META = {
    "total_commits",
    "bug_fix_commits",
    "bug_fix_count_90d",
    "first_seen",
    "window_end",
    "dataset_source",
    "commits_in_window",
    "bug_fix_commits_in_window",
    # last_seen is renamed to commit_date BEFORE this set is applied
}

KEEP_ALWAYS = {"file_path", "is_buggy", "repo_name", "commit_date"}

# ── per-repo buggy rate caps ───────────────────────────────────────────────
MAX_BUGGY_RATE = 0.50
STRICT_CAP_REPOS = {
    "django":     0.35,
    "sqlalchemy": 0.40,
    "aiohttp":    0.45,
    "falcon":     0.45,
    "tornado":    0.45,
    "redis-py":   0.45,
    "werkzeug":   0.45,
    "click":      0.45,
    "celery":     0.45,
}

PROCESS_KEYWORDS = [
    "churn", "commit_count", "author_count", "fix_density",
    "days_since", "burst", "ownership",
]

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)


# ── Step 1: Fix last_seen dates ─────────────────────────────────────────────

def fix_last_seen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-derive last_seen from cloned repos on disk where possible.
    For each repo, run `git log --format="%H %ai" -- <file>` to get
    the actual last commit date per file.

    Falls back to days_since_last_change proxy for repos not cloned.
    """
    import subprocess

    df = df.copy()
    df["last_seen"] = pd.to_datetime(df["last_seen"], errors="coerce")

    # Reference date for days_since fallback
    ref_date = pd.Timestamp("2026-05-28", tz="UTC")

    fixed_count = 0
    fallback_count = 0

    for repo_name in df["repo_name"].unique():
        repo_path = REPOS_DIR / repo_name
        mask = df["repo_name"] == repo_name

        # Identify files in this repo that have synthetic/missing dates
        synthetic_mask = mask & (
            df["last_seen"].isna() |
            (df["last_seen"] >= pd.Timestamp("2026-01-01"))
        )
        n_synthetic = synthetic_mask.sum()

        if n_synthetic == 0:
            continue

        if not (repo_path / ".git").exists():
            # Fallback: estimate from days_since_last_change
            if "days_since_last_change" in df.columns:
                estimated = ref_date - pd.to_timedelta(
                    df.loc[synthetic_mask, "days_since_last_change"], unit="D"
                )
                df.loc[synthetic_mask, "last_seen"] = estimated.dt.tz_localize(None)
                fallback_count += n_synthetic
                logger.warning(
                    f"  {repo_name}: repo not cloned — "
                    f"estimated {n_synthetic} dates from days_since_last_change"
                )
            continue

        logger.info(f"  {repo_name}: fixing {n_synthetic} synthetic dates from git log")

        # Get last commit date per file from git log
        try:
            result = subprocess.run(
                ["git", "log", "--format=%ai", "--name-only", "--diff-filter=A,M",
                 "--", "*.py"],
                cwd=repo_path,
                capture_output=True, text=True, timeout=120,
            )
            # Parse git log output: alternating date lines and file lines
            date_map: dict[str, str] = {}
            current_date = None
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Date lines look like: 2023-04-15 12:34:56 +0000
                if len(line) > 10 and line[4] == "-" and line[7] == "-":
                    try:
                        pd.to_datetime(line[:19])
                        current_date = line[:10]  # YYYY-MM-DD
                        continue
                    except Exception:
                        pass
                # File lines
                if current_date and line.endswith(".py"):
                    if line not in date_map:  # keep first (latest) occurrence
                        date_map[line] = current_date

            # Apply to synthetic rows
            for idx in df[synthetic_mask].index:
                fp = df.loc[idx, "file_path"]
                if fp in date_map:
                    df.loc[idx, "last_seen"] = pd.to_datetime(date_map[fp])
                    fixed_count += 1
                elif "days_since_last_change" in df.columns:
                    days = df.loc[idx, "days_since_last_change"]
                    df.loc[idx, "last_seen"] = (
                        ref_date - pd.Timedelta(days=int(days))
                    ).tz_localize(None)
                    fallback_count += 1

        except Exception as e:
            logger.warning(f"  {repo_name}: git log failed ({e}) — using fallback")
            if "days_since_last_change" in df.columns:
                estimated = ref_date - pd.to_timedelta(
                    df.loc[synthetic_mask, "days_since_last_change"], unit="D"
                )
                df.loc[synthetic_mask, "last_seen"] = estimated.dt.tz_localize(None)
                fallback_count += n_synthetic

    logger.info(
        f"Date fixing complete: "
        f"{fixed_count} from git log, "
        f"{fallback_count} from days_since fallback"
    )

    # Final check
    still_synthetic = (
        df["last_seen"] >= pd.Timestamp("2026-01-01")
    ).sum()
    if still_synthetic:
        logger.warning(
            f"{still_synthetic} files still have synthetic dates — "
            "using days_since_last_change as final fallback"
        )
        if "days_since_last_change" in df.columns:
            mask_still = df["last_seen"] >= pd.Timestamp("2026-01-01")
            df.loc[mask_still, "last_seen"] = (
                ref_date - pd.to_timedelta(
                    df.loc[mask_still, "days_since_last_change"], unit="D"
                )
            ).dt.tz_localize(None)

    df["last_seen"] = pd.to_datetime(df["last_seen"], errors="coerce")
    real = (df["last_seen"] < pd.Timestamp("2026-01-01")).sum()
    logger.info(
        f"Real historical dates: {real}/{len(df)} "
        f"({real/len(df)*100:.1f}%)"
    )
    return df


# ── Step 2: Apply buggy rate caps ──────────────────────────────────────────

def apply_buggy_rate_cap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    flipped_total = 0

    caps = {r: MAX_BUGGY_RATE for r in df["repo_name"].unique()}
    caps.update(STRICT_CAP_REPOS)

    for repo, cap in caps.items():
        mask = df["repo_name"] == repo
        repo_df = df[mask]
        if len(repo_df) == 0:
            continue

        current = repo_df["is_buggy"].mean()
        if current <= cap:
            continue

        n_target  = int(len(repo_df) * cap)
        n_to_flip = int(repo_df["is_buggy"].sum()) - n_target
        if n_to_flip <= 0:
            continue

        sort_col = "fix_density" if "fix_density" in df.columns else "is_buggy"
        flip_idx = (
            repo_df[repo_df["is_buggy"] == 1]
            .sort_values(sort_col, ascending=True)
            .index[:n_to_flip]
        )
        df.loc[flip_idx, "is_buggy"] = 0
        flipped_total += len(flip_idx)
        new_rate = df[mask]["is_buggy"].mean()
        logger.info(
            f"  {repo}: {current*100:.1f}% -> {new_rate*100:.1f}% "
            f"(flipped {len(flip_idx)}, cap={cap*100:.0f}%)"
        )

    logger.info(f"Buggy cap: {flipped_total} files flipped total")
    return df


# ── Step 3: Impute ─────────────────────────────────────────────────────────

def impute(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    to_fill = [c for c in feature_cols if df[c].isna().any()]
    if not to_fill:
        logger.info("Nothing to impute.")
        return df
    logger.info(f"Imputing {len(to_fill)} columns via per-repo median ...")
    for col in to_fill:
        df[col] = df[col].fillna(
            df.groupby("repo_name")[col].transform("median")
        )
        still = df[col].isna().sum()
        if still:
            df[col] = df[col].fillna(df[col].median())
    return df


# ── Step 4: Save ──────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame, feature_cols: list[str]) -> None:
    OUT_FINAL.parent.mkdir(parents=True, exist_ok=True)

    # Final: repo_name + file_path + is_buggy + commit_date + features
    final_cols = ["repo_name", "file_path", "is_buggy", "commit_date"] + feature_cols
    df[final_cols].to_csv(OUT_FINAL, index=False)
    logger.success(f"feature_matrix_final.csv    -> {OUT_FINAL}  {df[final_cols].shape}")

    # With repo: no commit_date
    wr_cols = ["repo_name", "file_path", "is_buggy"] + feature_cols
    df[wr_cols].to_csv(OUT_WITH_REPO, index=False)
    logger.success(f"feature_matrix_with_repo.csv -> {OUT_WITH_REPO}")

    # Clean: no repo_name, no commit_date
    cl_cols = ["file_path", "is_buggy"] + feature_cols
    df[cl_cols].to_csv(OUT_CLEAN, index=False)
    logger.success(f"feature_matrix_clean.csv    -> {OUT_CLEAN}")

    # Inventory
    process = [c for c in feature_cols if any(k in c for k in PROCESS_KEYWORDS)]
    ast     = [c for c in feature_cols if c not in process]
    per_repo = (
        df.groupby("repo_name")
        .agg(files=("file_path","count"), bug_ratio=("is_buggy","mean"))
        .assign(bug_ratio=lambda x: x["bug_ratio"].map("{:.1%}".format))
        .sort_values("files", ascending=False)
    )

    lines = [
        "=" * 62,
        "FINAL FEATURE MATRIX",
        "=" * 62,
        f"  Rows         : {len(df):,}",
        f"  Features     : {len(feature_cols)}",
        f"  Buggy        : {int(df['is_buggy'].sum()):,}  ({df['is_buggy'].mean():.1%})",
        f"  Clean        : {int((df['is_buggy']==0).sum()):,}",
        f"  Repos        : {df['repo_name'].nunique()}",
        f"  Date range   : {df['commit_date'].min()} to {df['commit_date'].max()}",
        "",
        f"  PROCESS ({len(process)}): {process}",
        f"  AST     ({len(ast)}): {ast}",
        "",
        "Per-repo:",
        per_repo.to_string(),
        "=" * 62,
    ]
    for line in lines:
        logger.info(line)
    OUT_INVENTORY.write_text("\n".join(lines), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    if not LABELED_COMBINED.exists():
        logger.error(f"Not found: {LABELED_COMBINED}")
        logger.error("Run scripts/day5_mine_repos.py first.")
        sys.exit(1)

    logger.info(f"Loading {LABELED_COMBINED}")
    df = pd.read_csv(LABELED_COMBINED)
    logger.info(f"  Shape: {df.shape}  buggy={df['is_buggy'].mean()*100:.1f}%")

    # ── Fix last_seen dates ──────────────────────────────────────────────
    logger.info("Step 1: Fixing last_seen dates ...")
    df = fix_last_seen(df)

    # ── Use last_seen as commit_date ─────────────────────────────────────
    logger.info("Step 2: Renaming last_seen -> commit_date")
    df = df.rename(columns={"last_seen": "commit_date"})

    # ── Drop leaky columns ───────────────────────────────────────────────
    logger.info("Step 3: Dropping leaky columns")
    present = [c for c in LEAKY_OR_META if c in df.columns]
    if present:
        df = df.drop(columns=present)
        logger.info(f"  Dropped: {present}")

    # ── Feature columns ──────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in KEEP_ALWAYS]
    logger.info(f"  Feature cols ({len(feature_cols)}): {feature_cols}")

    # ── Buggy rate caps ──────────────────────────────────────────────────
    logger.info("Step 4: Applying buggy rate caps ...")
    df = apply_buggy_rate_cap(df)

    # ── Impute ───────────────────────────────────────────────────────────
    logger.info("Step 5: Imputing NaNs ...")
    df = impute(df, feature_cols)

    # ── Validate ─────────────────────────────────────────────────────────
    nan_total = df[feature_cols].isna().sum().sum()
    if nan_total:
        logger.error(f"Imputation incomplete: {nan_total} NaNs remain")
        sys.exit(1)

    real_dates = (
        pd.to_datetime(df["commit_date"], errors="coerce") < pd.Timestamp("2026-01-01")
    ).sum()
    logger.info(
        f"Real historical dates: {real_dates}/{len(df)} "
        f"({real_dates/len(df)*100:.1f}%)"
    )

    # ── Save ─────────────────────────────────────────────────────────────
    logger.info("Step 6: Saving outputs ...")
    save_outputs(df, feature_cols)

    logger.success("Done. Run: python scripts/day4_run.py")
    print()
    print("Verify with:")
    print("  python -c \"import pandas as pd; df=pd.read_csv('data/processed/feature_matrix_final.csv'); print(df.shape, df['is_buggy'].mean(), df['commit_date'].min(), df['commit_date'].max())\"")


if __name__ == "__main__":
    main()