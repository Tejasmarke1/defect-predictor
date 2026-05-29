"""
Day 3 — Master Orchestration Script
=====================================
Runs the full Day 3 pipeline in sequence:
  1. train.py   → Train XGBoost, save model + OOF predictions
  2. evaluate.py → Generate evaluation plots
  3. shap_explainer.py → Global SHAP analysis

Prints a final summary: AUC, F1 at Mean P@20, top 3 SHAP features.

Usage:
    python scripts/day3_run.py
    python scripts/day3_run.py --drop-halstead   # drop redundant Halstead feats
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

# Add project root to path so configs and src are importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.train import run_training
from src.models.evaluate import run_evaluation
from src.explainability.shap_explainer import run_shap_analysis


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the master script."""
    parser = argparse.ArgumentParser(description="Day 3 — Defect Predictor Pipeline")
    parser.add_argument(
        "--drop-halstead",
        action="store_true",
        help="Drop highly correlated Halstead features before training.",
    )
    return parser.parse_args()


def print_summary(train_summary: dict, eval_results: dict, top_shap) -> None:
    """
    Print a concise end-of-day summary to stdout.

    Args:
        train_summary: Dict from run_training() with mean_auc, mean_f1, mean_ap.
        eval_results: Dict from run_evaluation() with roc_auc, ap, classification_report.
        top_shap: pd.Series of global feature importances (Mean |SHAP|).
    """
    top3 = top_shap.head(3)
    separator = "=" * 60

    logger.info(separator)
    logger.info("DAY 3 PIPELINE COMPLETE — SUMMARY")
    logger.info(separator)
    logger.info(f"  CV Mean AUC-ROC   : {train_summary.get('mean_auc', 0):.4f}")
    logger.info(f"  CV Mean AP        : {train_summary.get('mean_ap', 0):.4f}")
    logger.info(f"  CV Mean F1        : {train_summary.get('mean_f1', 0):.4f}")
    logger.info(f"  OOF AUC-ROC       : {eval_results['roc_auc']:.4f}")
    logger.info(f"  OOF Avg Precision : {eval_results['ap']:.4f}")
    logger.info("  Top 3 SHAP Features:")
    for feat, val in top3.items():
        logger.info(f"    {feat:<35} {val:.5f}")
    logger.info(separator)

    # Expected KC1 AUC range sanity check
    auc = eval_results["roc_auc"]
    if auc > 0.78:
        logger.success(
            f"✓ AUC={auc:.4f} exceeds the expected KC1 range [0.70, 0.78] — excellent result!"
        )
    elif auc >= 0.70:
        logger.success(f"✓ AUC={auc:.4f} is within the expected KC1 range [0.70, 0.78]")
    elif auc >= 0.65:
        logger.warning(
            f"⚠ AUC={auc:.4f} is below 0.70. KC1 static-only models typically reach 0.70–0.78. "
            "Try --drop-halstead to reduce multicollinearity."
        )
    else:
        logger.error(
            f"✗ AUC={auc:.4f} is below 0.65. Check feature_matrix.csv and label integrity."
        )


def main() -> None:
    """Entry point: parse args, run pipeline, print summary."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("STEP 1/3 — Model Training")
    logger.info("=" * 60)
    train_summary = run_training(drop_halstead_redundant=args.drop_halstead)

    logger.info("=" * 60)
    logger.info("STEP 2/3 — Model Evaluation")
    logger.info("=" * 60)
    eval_results = run_evaluation()

    logger.info("=" * 60)
    logger.info("STEP 3/3 — SHAP Explainability")
    logger.info("=" * 60)
    importance, shap_vals, explainer, X, file_paths = run_shap_analysis()

    print_summary(train_summary, eval_results, importance)


if __name__ == "__main__":
    main()
