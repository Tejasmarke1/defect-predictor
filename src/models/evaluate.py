"""
Day 3 — Model Evaluation
=========================
Loads saved XGBoost model and OOF predictions, produces a full suite of
diagnostic plots saved to reports/figures/.
"""

from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
OOF_PATH = ROOT / "data" / "processed" / "oof_predictions.csv"
MODEL_PATH = ROOT / "models" / "xgboost_defect_predictor.json"
META_PATH = ROOT / "models" / "model_meta.json"
FIGURES_DIR = ROOT / "reports" / "figures"


def load_artifacts() -> tuple:
    """
    Load the saved XGBoost model, model metadata, and OOF predictions.

    Returns:
        model: Loaded XGBClassifier.
        meta: Dict from model_meta.json (feature_names, threshold, etc.).
        oof_df: DataFrame with columns [file_path, is_buggy, oof_prob, oof_pred].
    """
    logger.info("Loading model artifacts...")
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    with open(META_PATH) as f:
        meta = json.load(f)

    oof_df = pd.read_csv(OOF_PATH)
    logger.info(f"OOF shape: {oof_df.shape}")
    return model, meta, oof_df


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    """
    Plot and save a labelled confusion matrix heatmap.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels at best threshold.
        save_path: Output PNG path.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    labels = ["Clean", "Buggy"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (OOF @ best threshold)")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {save_path}")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path) -> float:
    """
    Plot ROC curve with AUC annotation and save to disk.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        save_path: Output PNG path.

    Returns:
        roc_auc: Computed AUC-ROC value.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — XGBoost KC1 Defect Predictor (OOF)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC curve saved → {save_path}  (AUC={roc_auc:.4f})")
    return roc_auc


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path) -> float:
    """
    Plot Precision-Recall curve with Average Precision annotation.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        save_path: Output PNG path.

    Returns:
        ap: Average Precision score.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="darkorange", lw=2,
            label=f"PR curve (AP = {ap:.4f})")
    ax.axhline(y=baseline, color="gray", linestyle="--", lw=1,
               label=f"Baseline (prevalence={baseline:.2%})")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — XGBoost KC1 (OOF)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PR curve saved → {save_path}  (AP={ap:.4f})")
    return ap


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path) -> None:
    """
    Plot calibration (reliability) curve to assess probability trustworthiness.

    A perfectly calibrated model produces a diagonal line. Deviation indicates
    over/under-confidence in predicted probabilities.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        save_path: Output PNG path.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="uniform"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(mean_predicted_value, fraction_of_positives,
            "s-", color="tomato", lw=2, label="XGBoost (OOF)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve — XGBoost KC1 Defect Predictor")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Calibration curve saved → {save_path}")

def precision_recall_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_values: list = [10, 20, 50, 100],
) -> pd.DataFrame:
    """
    Compute Precision@K and Recall@K — how precise is the model
    when you inspect only the top-K highest-risk modules?

    This is the operationally relevant metric for code review:
    teams inspect a fixed budget of files per sprint, not all flagged files.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        k_values: List of K values to evaluate.

    Returns:
        DataFrame with columns [K, precision_at_k, recall_at_k, bugs_found].
    """
    sorted_idx = np.argsort(y_prob)[::-1]
    total_bugs = y_true.sum()
    rows = []

    for k in k_values:
        if k > len(y_true):
            continue
        top_k = y_true[sorted_idx[:k]]
        bugs_found = top_k.sum()
        rows.append({
            "K": k,
            "precision_at_k": bugs_found / k,
            "recall_at_k": bugs_found / total_bugs,
            "bugs_found": int(bugs_found),
            "total_bugs": int(total_bugs),
        })
        

    return pd.DataFrame(rows)

def run_evaluation() -> dict:
    """
    Full evaluation pipeline: load artifacts → compute all metrics → save plots.

    Returns:
        results: Dict with roc_auc, ap, classification_report_str,
                 best_threshold, confusion_matrix.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model, meta, oof_df = load_artifacts()

    y_true = oof_df["is_buggy"].values
    y_prob = oof_df["oof_prob"].values
    y_pred = oof_df["oof_pred"].values
    best_threshold = meta["best_threshold"]

    logger.info(f"Using best threshold: {best_threshold:.3f}")

    # --- Classification report ---
    report = classification_report(y_true, y_pred, target_names=["Clean", "Buggy"])
    logger.info(f"\nClassification Report:\n{report}")

    # --- Plots ---
    plot_confusion_matrix(
        y_true, y_pred,
        FIGURES_DIR / "confusion_matrix.png",
    )
    roc_auc = plot_roc_curve(
        y_true, y_prob,
        FIGURES_DIR / "roc_curve.png",
    )
    ap = plot_pr_curve(
        y_true, y_prob,
        FIGURES_DIR / "pr_curve.png",
    )
    plot_calibration_curve(
        y_true, y_prob,
        FIGURES_DIR / "calibration_curve.png",
    )

    cm = confusion_matrix(y_true, y_pred)
    results = {
        "roc_auc": roc_auc,
        "ap": ap,
        "classification_report": report,
        "best_threshold": best_threshold,
        "confusion_matrix": cm.tolist(),
    }
    precision_recall_at_k_df = precision_recall_at_k(y_true, y_prob)
    results["precision_recall_at_k"] = precision_recall_at_k_df.to_dict(orient="records")
    logger.info(f"\nPrecision@K:\n{precision_recall_at_k_df}")
    
    logger.info("Evaluation complete. All plots saved.")
    return results



if __name__ == "__main__":
    run_evaluation()