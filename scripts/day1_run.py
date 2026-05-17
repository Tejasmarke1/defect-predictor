"""
Day 1 Master Script
====================
Run this to execute the complete Day 1 pipeline:
    1. Download PROMISE dataset (instant)
    2. Start Flask repo mining (runs in background)
    3. Run SZZ labeling on PROMISE data
    4. Print summary + next steps

Usage:
    python scripts/day1_run.py
    python scripts/day1_run.py --skip-mining   # if you want PROMISE only
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.mining.promise_loader import PromiseLoader
from src.mining.szz_labeler import SZZLabeler


def run_promise_pipeline():
    """Step 1: Load and label PROMISE data — works instantly."""
    logger.info("━" * 60)
    logger.info("STEP 1: Loading PROMISE Dataset")
    logger.info("━" * 60)

    loader = PromiseLoader()

    # Load KC1 as primary
    df_kc1 = loader.load("KC1")
    stats = {
        "samples": len(df_kc1),
        "bug_ratio": df_kc1["is_buggy"].mean(),
        "features": len([c for c in df_kc1.columns if c not in ["file_path", "repo_name", "dataset_source", "is_buggy"]]),
    }

    logger.success(f"KC1 loaded: {stats['samples']} samples, {stats['bug_ratio']:.1%} buggy, {stats['features']} features")

    # Save processed version
    output_path = Path("data/processed/kc1_labeled.csv")
    df_kc1.to_csv(output_path, index=False)
    logger.success(f"Saved to {output_path}")

    return df_kc1


def run_mining_pipeline(repo_url: str = "https://github.com/pallets/flask"):
    """Step 2: Mine a real Git repo — takes 5-15 minutes."""
    logger.info("━" * 60)
    logger.info(f"STEP 2: Mining Repository — {repo_url}")
    logger.info("━" * 60)
    logger.info("This takes 5-15 minutes depending on repo size.")
    logger.info("Let it run while you review the PROMISE data above.")

    from src.mining.git_miner import GitMiner

    miner = GitMiner(
        repo_url=repo_url,
        since=datetime(2019, 1, 1),  # 5 years of history
    )

    df_commits = miner.mine()
    summary = miner.get_summary(df_commits)

    # Label the mined data
    logger.info("Running SZZ labeling on mined commits...")
    labeler = SZZLabeler(df_commits, window_days=90)

    df_labeled = labeler.create_file_labels()
    stats = labeler.get_label_statistics(df_labeled)

    # If too few windowed samples, use simple labels
    if stats["total_samples"] < 50:
        logger.warning("Few windowed samples — using simple file-level labels")
        df_labeled = labeler.create_simple_file_labels()

    return df_commits, df_labeled


def print_day1_summary(df_promise, df_mined=None):
    """Print end-of-day summary."""
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("DAY 1 COMPLETE — SUMMARY")
    logger.info("=" * 60)

    logger.success(f"✓ PROMISE KC1: {len(df_promise)} labeled samples ready")
    logger.success(f"  Bug ratio: {df_promise['is_buggy'].mean():.1%}")
    logger.success(f"  Features: {len(df_promise.columns) - 3}")

    if df_mined is not None:
        logger.success(f"✓ Flask mined commits saved to data/raw/flask_commits.csv")

    logger.info("\nFiles created today:")
    files = [
        "data/processed/kc1_labeled.csv",
        "data/raw/flask_commits.csv" if df_mined is not None else None,
    ]
    for f in files:
        if f and Path(f).exists():
            size = Path(f).stat().st_size / 1024
            logger.info(f"  {f} ({size:.1f} KB)")

    logger.info("\n" + "=" * 60)
    logger.info("TOMORROW — Day 2: Feature Engineering")
    logger.info("=" * 60)
    logger.info("You will build:")
    logger.info("  • Process metrics (churn, author count, fix density)")
    logger.info("  • AST structural features (complexity, nesting depth)")
    logger.info("  • Final feature matrix ready for XGBoost")
    logger.info("\nGit commit before sleeping:")
    logger.info('  git add -A && git commit -m "Day 1: data pipeline + PROMISE loading"')


def main():
    parser = argparse.ArgumentParser(description="Day 1 Pipeline")
    parser.add_argument("--skip-mining", action="store_true", help="Skip git mining (PROMISE only)")
    parser.add_argument("--repo", default="https://github.com/pallets/flask", help="Repo URL to mine")
    args = parser.parse_args()

    logger.info("🚀 DEFECT PREDICTOR — DAY 1")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Always run PROMISE (fast)
    df_promise = run_promise_pipeline()

    # Mining is optional but recommended
    df_mined = None
    if not args.skip_mining:
        try:
            _, df_mined = run_mining_pipeline(args.repo)
        except Exception as e:
            logger.error(f"Mining failed (non-fatal): {e}")
            logger.info("Continuing with PROMISE data only — you can mine later")

    print_day1_summary(df_promise, df_mined)


if __name__ == "__main__":
    main()