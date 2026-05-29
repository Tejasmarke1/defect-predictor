"""
Day 5 — Multi-Repository Mining Pipeline
=========================================
Mines Python OSS repositories using PyDriller, applies strict SZZ bug
labeling, extracts process + AST features, and produces:

    data/processed/combined_source_codes.pkl   ← PRIMARY: {file_path: source}
    data/processed/labeled_combined.csv        ← labels + features per file
    data/processed/repo_stats.csv              ← per-repo mining summary
    data/raw/repos/                            ← cloned repositories

The combined_source_codes.pkl is the key output — it gives the GNN real
embeddings for all files across all 12 repos instead of zeros.

labeled_combined.csv feeds directly into prep_for_training.py which
already exists and handles imputation, leakage removal, and validation.

SZZ improvements over v1
------------------------
  - Stricter keyword list (removed: handle, resolve, correct, wrong,
    patch, workaround, typo, mistake, fail, problem)
  - MAX_FILES_PER_BUGFIX_COMMIT = 15  (skip large refactor commits)
  - MIN_BUG_TOUCHES = 2               (file needs 2+ targeted fixes)

Usage
-----
    python scripts/day5_mine_repos.py                        # all repos
    python scripts/day5_mine_repos.py --repos django requests
    python scripts/day5_mine_repos.py --source-only          # pkl only, skip features
    python scripts/day5_mine_repos.py --dry-run
"""

from __future__ import annotations

import argparse
import ast as ast_module
import os
import pickle
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from pydriller import Repository
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Repository registry — matches your existing labeled_combined.csv repos
# ---------------------------------------------------------------------------

REPOS: list[dict] = [
    {
        "name": "flask",
        "url": "https://github.com/pallets/flask",
        "src_paths": ["src/flask", "flask"],
    },
    {
        "name": "django",
        "url": "https://github.com/django/django",
        "src_paths": ["django"],
    },
    {
        "name": "requests",
        "url": "https://github.com/psf/requests",
        "src_paths": ["src/requests", "requests"],
    },
    {
        "name": "werkzeug",
        "url": "https://github.com/pallets/werkzeug",
        "src_paths": ["src/werkzeug", "werkzeug"],
    },
    {
        "name": "click",
        "url": "https://github.com/pallets/click",
        "src_paths": ["src/click", "click"],
    },
    {
        "name": "tornado",
        "url": "https://github.com/tornadoweb/tornado",
        "src_paths": ["tornado"],
    },
    {
        "name": "aiohttp",
        "url": "https://github.com/aio-libs/aiohttp",
        "src_paths": ["aiohttp"],
    },
    {
        "name": "celery",
        "url": "https://github.com/celery/celery",
        "src_paths": ["celery"],
    },
    {
        "name": "sqlalchemy",
        "url": "https://github.com/sqlalchemy/sqlalchemy",
        "src_paths": ["lib/sqlalchemy"],
    },
    {
        "name": "pymongo",
        "url": "https://github.com/mongodb/mongo-python-driver",
        "src_paths": ["pymongo", "bson", "gridfs"],
    },
    {
        "name": "redis-py",
        "url": "https://github.com/redis/redis-py",
        "src_paths": ["redis"],
    },
    {
        "name": "falcon",
        "url": "https://github.com/falconry/falcon",
        "src_paths": ["falcon"],
    },
]

# ---------------------------------------------------------------------------
# SZZ — strict keyword pattern
# Removed from v1: handle, resolve, correct, wrong, patch,
#                  workaround, typo, mistake, fail, problem
# ---------------------------------------------------------------------------

BUG_FIX_PATTERN = re.compile(
    r"\b("
    r"fix(es|ed|ing)?|"
    r"bug(fix|s)?|"
    r"defect(s)?|"
    r"regression(s)?|"
    r"crash(es|ed|ing)?|"
    r"segfault|"
    r"deadlock|"
    r"memory[\s_-]?leak|"
    r"stack[\s_-]?overflow|"
    r"infinite[\s_-]?loop|"
    r"null[\s_-]?pointer|"
    r"index[\s_-]?error|"
    r"key[\s_-]?error|"
    r"attribute[\s_-]?error|"
    r"type[\s_-]?error|"
    r"value[\s_-]?error|"
    r"runtime[\s_-]?error"
    r")\b",
    re.IGNORECASE,
)

# SZZ quality gates
MAX_FILES_PER_BUGFIX_COMMIT = 15   # commits touching >15 files = refactor, skip
MIN_BUG_TOUCHES = 2                # file needs ≥2 targeted bug-fix commits

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPOS_DIR            = ROOT / "data" / "raw" / "repos"
PROCESSED_DIR        = ROOT / "data" / "processed"
COMBINED_SOURCE_PATH = PROCESSED_DIR / "combined_source_codes.pkl"
LABELED_COMBINED     = PROCESSED_DIR / "labeled_combined.csv"
REPO_STATS_PATH      = PROCESSED_DIR / "repo_stats.csv"


# ---------------------------------------------------------------------------
# Step 1 — Clone / update
# ---------------------------------------------------------------------------

def clone_or_update_repo(repo: dict, dry_run: bool = False) -> Path:
    local_path = REPOS_DIR / repo["name"]

    if dry_run:
        logger.info(f"[DRY RUN] {repo['url']} → {local_path}")
        return local_path

    REPOS_DIR.mkdir(parents=True, exist_ok=True)

    if (local_path / ".git").exists():
        logger.info(f"Pulling latest: {repo['name']}")
        result = subprocess.run(
            ["git", "pull", "--quiet"],
            cwd=local_path, capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning(f"git pull failed for {repo['name']}: {result.stderr[:150]}")
    else:
        logger.info(f"Cloning {repo['url']} → {local_path}")
        result = subprocess.run(
            ["git", "clone", "--quiet", repo["url"], str(local_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed: {result.stderr}")

    return local_path


# ---------------------------------------------------------------------------
# Step 2 — File filter
# ---------------------------------------------------------------------------

def is_python_source(file_path: str, src_paths: list[str]) -> bool:
    """True if file is non-test Python source within the repo's src_paths."""
    if not file_path.endswith(".py"):
        return False

    excluded_patterns = [
        "/test_", "_test.py", "/tests/", "/test/",
        "/migrations/", "/conftest", "setup.py",
        "__pycache__", ".pyc",
    ]
    if any(p in file_path for p in excluded_patterns):
        return False

    if src_paths == ["."]:
        return True

    return any(
        file_path.startswith(sp + "/") or
        file_path.startswith(sp + "\\") or
        f"/{sp}/" in f"/{file_path}"
        for sp in src_paths
    )


# ---------------------------------------------------------------------------
# Step 3 — Mine commits with strict SZZ
# ---------------------------------------------------------------------------

def mine_repo(
    repo_path: Path,
    repo_cfg: dict,
    since: datetime,
) -> tuple[dict[str, int], dict[str, list[dict]], dict[str, str]]:
    """
    Returns:
        bug_touch_count : {file_path: n_targeted_bugfix_commits}
        file_history    : {file_path: [commit_record, ...]}
        latest_source   : {file_path: source_code_string}
    """
    bug_touch_count: dict[str, int] = defaultdict(int)
    file_history:    dict[str, list[dict]] = defaultdict(list)
    latest_source:   dict[str, str] = {}

    logger.info(f"  Mining {repo_path.name} since {since.date()} ...")

    try:
        commits = list(Repository(
            str(repo_path),
            since=since,
            only_no_merge=True,
        ).traverse_commits())
    except Exception as e:
        logger.error(f"PyDriller error on {repo_path.name}: {e}")
        return {}, {}, {}

    for commit in tqdm(commits, desc=f"  {repo_path.name}", leave=False):
        is_bugfix = bool(BUG_FIX_PATTERN.search(commit.msg))
        commit_dt = commit.committer_date

        # Collect Python files modified in this commit
        py_files = [
            m for m in commit.modified_files
            if m.new_path and is_python_source(m.new_path, repo_cfg["src_paths"])
        ]

        if not py_files:
            continue

        # SZZ gate: skip large commits (refactors disguised as bug fixes)
        if is_bugfix and len(py_files) > MAX_FILES_PER_BUGFIX_COMMIT:
            is_bugfix = False

        for mod in py_files:
            fp = mod.new_path

            # Count targeted bug-fix touches per file
            if is_bugfix:
                bug_touch_count[fp] += 1

            # Build commit history for process metrics
            file_history[fp].append({
                "commit_hash":  commit.hash,
                "commit_date":  commit_dt,
                "author":       commit.author.email,
                "is_bugfix":    int(is_bugfix),
                "lines_added":  mod.added_lines,
                "lines_deleted": mod.deleted_lines,
                "churn":        mod.added_lines + mod.deleted_lines,
            })

            # Keep latest non-empty source code
            if mod.source_code and mod.source_code.strip():
                latest_source[fp] = mod.source_code

    # Apply MIN_BUG_TOUCHES threshold
    buggy_files = {
        fp: 1
        for fp, count in bug_touch_count.items()
        if count >= MIN_BUG_TOUCHES
    }

    n_commits = len(commits)
    n_bugfix  = sum(1 for c in commits if BUG_FIX_PATTERN.search(c.msg))
    logger.info(
        f"  {repo_path.name}: {n_commits} commits "
        f"({n_bugfix} bugfix={n_bugfix/max(n_commits,1)*100:.0f}%), "
        f"{len(file_history)} py files, "
        f"{len(buggy_files)} buggy (≥{MIN_BUG_TOUCHES} touches)"
    )

    return dict(buggy_files), dict(file_history), latest_source


# ---------------------------------------------------------------------------
# Step 4 — Process features
# ---------------------------------------------------------------------------

def process_features(
    file_path: str,
    history: list[dict],
    ref_date: datetime,
) -> dict:
    if not history:
        return {}

    df = pd.DataFrame(history)
    df["commit_date"] = pd.to_datetime(df["commit_date"], utc=True)
    df = df.sort_values("commit_date")

    cutoff_30 = ref_date - pd.Timedelta(days=30)
    cutoff_90 = ref_date - pd.Timedelta(days=90)
    r30 = df[df["commit_date"] >= cutoff_30]
    r90 = df[df["commit_date"] >= cutoff_90]

    total = len(df)
    top_author = df["author"].value_counts().iloc[0] if total else 0

    last_change = df["commit_date"].max()
    days_since = max(0, (ref_date - last_change).days)

    cc30 = len(r30)
    cc90 = len(r90)

    return {
        "total_commits":       total,
        "bug_fix_commits":     int(df["is_bugfix"].sum()),
        "code_churn_30d":      int(r30["churn"].sum()),
        "code_churn_90d":      int(r90["churn"].sum()),
        "commit_count_30d":    cc30,
        "commit_count_90d":    cc90,
        "author_count_90d":    r90["author"].nunique(),
        "bug_fix_count_90d":   int(r90["is_bugfix"].sum()),
        "fix_density":         round(r90["is_bugfix"].sum() / max(cc90, 1), 4),
        "days_since_last_change": days_since,
        "commit_burst_30d":    round(cc30 / max(cc90, 1), 4),
        "ownership_score":     round(top_author / max(total, 1), 4),
    }


# ---------------------------------------------------------------------------
# Step 5 — AST features
# ---------------------------------------------------------------------------

def ast_features(source: str) -> dict:
    if not source or not source.strip():
        return {}
    try:
        tree = ast_module.parse(source)
    except SyntaxError:
        return {}

    branch_nodes = (
        ast_module.If, ast_module.For, ast_module.While,
        ast_module.Try, ast_module.ExceptHandler,
        ast_module.With, ast_module.Assert,
    )
    cyclomatic = sum(1 for _ in ast_module.walk(tree) if isinstance(_, branch_nodes)) + 1

    def _depth(node: ast_module.AST, d: int = 0) -> int:
        kids = list(ast_module.iter_child_nodes(node))
        return max((_depth(k, d+1) for k in kids), default=d)

    funcs = [
        n for n in ast_module.walk(tree)
        if isinstance(n, (ast_module.FunctionDef, ast_module.AsyncFunctionDef))
    ]
    avg_len = float(np.mean([
        (f.end_lineno - f.lineno + 1)
        for f in funcs
        if hasattr(f, "end_lineno") and f.end_lineno
    ])) if funcs else 0.0

    cognitive = sum(
        1 for n in ast_module.walk(tree)
        if isinstance(n, (ast_module.If, ast_module.For, ast_module.While,
                          ast_module.ExceptHandler))
    )

    return {
        "ast_cyclomatic_complexity": cyclomatic,
        "ast_cognitive_complexity":  cognitive,
        "max_nesting_depth":         _depth(tree),
        "num_functions":             len(funcs),
        "num_classes":               sum(1 for _ in ast_module.walk(tree)
                                        if isinstance(_, ast_module.ClassDef)),
        "avg_function_length":       round(avg_len, 2),
        "max_args_per_function":     max((len(f.args.args) for f in funcs), default=0),
        "num_imports":               sum(1 for _ in ast_module.walk(tree)
                                        if isinstance(_, (ast_module.Import,
                                                          ast_module.ImportFrom))),
        "ast_node_count":            sum(1 for _ in ast_module.walk(tree)),
        "has_try_except":            int(any(isinstance(n, ast_module.Try)
                                            for n in ast_module.walk(tree))),
    }


# ---------------------------------------------------------------------------
# Step 6 — Build per-repo feature DataFrame
# ---------------------------------------------------------------------------

MIN_COMMITS_PER_FILE = 3   # skip files with very sparse history


def build_features(
    repo_name: str,
    buggy_files: dict[str, int],
    file_history: dict[str, list[dict]],
    latest_source: dict[str, str],
) -> pd.DataFrame:
    ref_date = datetime.now(timezone.utc)
    rows = []

    all_fps = set(file_history) | set(latest_source)

    for fp in tqdm(all_fps, desc=f"  Features [{repo_name}]", leave=False):
        history = file_history.get(fp, [])
        source  = latest_source.get(fp, "")

        if len(history) < MIN_COMMITS_PER_FILE:
            continue

        pf = process_features(fp, history, ref_date)
        af = ast_features(source)

        if not pf:
            continue

        row = {
            "repo_name":   repo_name,
            "file_path":   fp,
            "is_buggy":    int(buggy_files.get(fp, 0)),
            "window_end":  ref_date.strftime("%Y-%m-%d %H:%M:%S"),
            "first_seen":  pd.to_datetime(history[0]["commit_date"]).strftime("%Y-%m-%d")
                           if history else "",
            "last_seen":   pd.to_datetime(history[-1]["commit_date"]).strftime("%Y-%m-%d")
                           if history else "",
        }
        row.update(pf)
        row.update(af)
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        buggy_n = df["is_buggy"].sum()
        logger.info(
            f"  {repo_name}: {len(df)} files, "
            f"{buggy_n} buggy ({buggy_n/len(df)*100:.1f}%)"
        )
    else:
        logger.warning(f"  {repo_name}: 0 files extracted")
    return df


# ---------------------------------------------------------------------------
# Step 7 — Extract source from already-cloned repos (fast path)
# ---------------------------------------------------------------------------

def extract_source_from_cloned(
    repo_path: Path,
    repo_cfg: dict,
) -> dict[str, str]:
    """
    Walk the cloned repo on disk and read current source files.
    Much faster than re-mining all commits when we only need source code.
    """
    sources: dict[str, str] = {}

    for py_file in repo_path.rglob("*.py"):
        rel = py_file.relative_to(repo_path).as_posix()
        if not is_python_source(rel, repo_cfg["src_paths"]):
            continue
        try:
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                sources[rel] = text
        except Exception:
            pass

    logger.info(f"  {repo_cfg['name']}: {len(sources)} source files read from disk")
    return sources


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    repo_filter: Optional[list[str]] = None,
    dry_run: bool = False,
    source_only: bool = False,
    since_year: int = 2015,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    since = datetime(since_year, 1, 1, tzinfo=timezone.utc)

    repos_to_mine = [
        r for r in REPOS
        if repo_filter is None or r["name"] in repo_filter
    ]
    if not repos_to_mine:
        logger.error(f"No repos matched filter: {repo_filter}")
        sys.exit(1)

    logger.info(f"Repos to process: {[r['name'] for r in repos_to_mine]}")

    if dry_run:
        for r in repos_to_mine:
            logger.info(f"  {r['name']:20s}  {r['url']}")
        return

    # -----------------------------------------------------------------------
    # Load existing labeled_combined.csv if present — we'll merge into it
    # -----------------------------------------------------------------------
    existing_df: Optional[pd.DataFrame] = None
    if LABELED_COMBINED.exists():
        existing_df = pd.read_csv(LABELED_COMBINED)
        existing_repos = set(existing_df["repo_name"].unique()) \
                         if "repo_name" in existing_df.columns else set()
        logger.info(
            f"Existing labeled_combined.csv: {len(existing_df)} rows, "
            f"repos: {sorted(existing_repos)}"
        )
    else:
        existing_repos = set()

    # -----------------------------------------------------------------------
    # Load existing combined source codes if present
    # -----------------------------------------------------------------------
    all_sources: dict[str, str] = {}
    if COMBINED_SOURCE_PATH.exists():
        with open(COMBINED_SOURCE_PATH, "rb") as f:
            all_sources = pickle.load(f)
        logger.info(f"Loaded existing combined_source_codes.pkl: {len(all_sources)} files")

    # -----------------------------------------------------------------------
    # Process each repo
    # -----------------------------------------------------------------------
    new_dfs: list[pd.DataFrame] = []

    for repo_cfg in repos_to_mine:
        repo_name = repo_cfg["name"]
        logger.info(f"\n{'='*60}\nProcessing: {repo_name}\n{'='*60}")
        t0 = time.time()

        # Clone / update
        try:
            repo_path = clone_or_update_repo(repo_cfg)
        except RuntimeError as e:
            logger.error(f"Skipping {repo_name}: {e}")
            continue

        # ------------------------------------------------------------------
        # Source-only mode: just read files from disk, skip feature mining
        # ------------------------------------------------------------------
        if source_only:
            src = extract_source_from_cloned(repo_path, repo_cfg)
            all_sources.update(src)
            logger.info(f"  Source-only: added {len(src)} files")
            continue

        # ------------------------------------------------------------------
        # Full mining mode
        # ------------------------------------------------------------------
        buggy_files, file_history, latest_source = mine_repo(
            repo_path, repo_cfg, since
        )

        if not file_history:
            logger.warning(f"No files mined from {repo_name} — skipping features")
            # Still capture source codes
            src = extract_source_from_cloned(repo_path, repo_cfg)
            all_sources.update(src)
            continue

        # Build features
        repo_df = build_features(repo_name, buggy_files, file_history, latest_source)

        if len(repo_df) > 0:
            new_dfs.append(repo_df)

        # Update sources — prefer PyDriller (latest commit) over disk read
        all_sources.update(latest_source)
        # Fill gaps with disk read for files not touched since `since`
        disk_src = extract_source_from_cloned(repo_path, repo_cfg)
        for fp, src in disk_src.items():
            if fp not in all_sources:
                all_sources[fp] = src

        logger.info(f"  Done in {time.time()-t0:.0f}s")

    # -----------------------------------------------------------------------
    # Save combined_source_codes.pkl — always, even source-only mode
    # -----------------------------------------------------------------------
    with open(COMBINED_SOURCE_PATH, "wb") as f:
        pickle.dump(all_sources, f)
    logger.info(f"Saved combined_source_codes.pkl: {len(all_sources)} files → {COMBINED_SOURCE_PATH}")

    if source_only:
        print(f"\nSource-only mode complete. {len(all_sources)} files in combined_source_codes.pkl")
        print("Run day4_run.py to retrain with full GNN coverage.")
        return

    # -----------------------------------------------------------------------
    # Merge new feature data with existing labeled_combined.csv
    # -----------------------------------------------------------------------
    if not new_dfs and existing_df is None:
        logger.error("No feature data to save.")
        sys.exit(1)

    parts = []
    if existing_df is not None:
        parts.append(existing_df)
    if new_dfs:
        parts.append(pd.concat(new_dfs, ignore_index=True))

    combined = pd.concat(parts, ignore_index=True)

    # Deduplicate: keep latest row per (repo_name, file_path)
    combined = (
        combined.sort_values("window_end")
        .drop_duplicates(subset=["repo_name", "file_path"], keep="last")
        .reset_index(drop=True)
    )

    combined.to_csv(LABELED_COMBINED, index=False)
    logger.info(f"Saved labeled_combined.csv: {len(combined)} files → {LABELED_COMBINED}")

    # -----------------------------------------------------------------------
    # Repo stats
    # -----------------------------------------------------------------------
    stats = (
        combined.groupby("repo_name")
        .agg(
            files    =("file_path", "count"),
            buggy    =("is_buggy",  "sum"),
            bug_rate =("is_buggy",  "mean"),
        )
        .assign(bug_rate=lambda x: (x["bug_rate"]*100).round(1).astype(str)+"%")
        .sort_values("files", ascending=False)
        .reset_index()
    )
    stats.to_csv(REPO_STATS_PATH, index=False)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "="*68)
    print("DAY 5 MINING COMPLETE")
    print("="*68)
    print(stats.to_string(index=False))
    print("-"*68)
    print(f"Total files     : {len(combined)}")
    print(f"Total buggy     : {int(combined['is_buggy'].sum())}  "
          f"({combined['is_buggy'].mean()*100:.1f}%)")
    print(f"Source codes    : {len(all_sources)}")
    print(f"GNN coverage    : {sum(1 for fp in combined['file_path'] if fp in all_sources)}"
          f" / {len(combined)} files have source")
    print("="*68)
    print()
    print("Next steps:")
    print("  1. python prep_for_training.py          # clean + impute features")
    print("  2. python scripts/day4_run.py           # retrain GNN + hybrid")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Day 5 — Multi-repo defect mining pipeline"
    )
    parser.add_argument(
        "--repos", nargs="+", metavar="REPO",
        help="Repos to mine. Default: all. e.g. --repos django requests",
    )
    parser.add_argument(
        "--source-only", action="store_true",
        help="Only extract source codes from disk (fast, no commit mining). "
             "Use when repos are already cloned and labeled_combined.csv exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without cloning or mining.",
    )
    parser.add_argument(
        "--since", type=int, default=2015, metavar="YEAR",
        help="Mine commits since this year (default: 2015).",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "day5_mining.log", rotation="50 MB", level="DEBUG")

    main(
        repo_filter=args.repos,
        dry_run=args.dry_run,
        source_only=args.source_only,
        since_year=args.since,
    )