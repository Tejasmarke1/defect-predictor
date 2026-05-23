"""
Day 3 — XGBoost Defect Prediction Training Pipeline
=====================================================
Trains an XGBoost classifier on KC1 static features with StratifiedKFold CV,
threshold tuning, OOF predictions, and MLflow experiment tracking.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score,
)
from loguru import logger

from configs.config import MODEL

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
FEATURE_MATRIX_PATH = ROOT / "data" / "processed" / "feature_matrix.csv"
OOF_PREDICTIONS_PATH = ROOT / "data" / "processed" / "oof_predictions.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "xgboost_defect_predictor.json"

# ---------------------------------------------------------------------------
# Halstead multicollinearity note
# ---------------------------------------------------------------------------
# The following Halstead features form a highly correlated cluster
# (pairwise |r| >= 0.90 per EDA): halstead_effort, halstead_time,
# halstead_volume, halstead_length, total_operators, total_operands.
# XGBoost redistributes importance across correlated features, which can mask
# the "true" driver. If model performance is poor (AUC < 0.70), consider
# dropping these and retaining only: halstead_difficulty, halstead_bugs,
# halstead_intelligence as the most semantically distinct representatives.
HALSTEAD_REDUNDANT = [
    "halstead_effort",
    "halstead_time",
    "halstead_volume",
    "halstead_length",
    "total_operators",
    "total_operands",
]


def load_and_prepare_data(path: Path, drop_halstead_redundant: bool = False):
    """
    Load feature matrix, drop NaN-dominant columns, and add future-proof flag.

    Args:
        path: Path to feature_matrix.csv.
        drop_halstead_redundant: If True, drops highly correlated Halstead
            features to reduce multicollinearity. Recommended if AUC < 0.70.

    Returns:
        X (pd.DataFrame): Feature matrix ready for training.
        y (pd.Series): Binary target (0/1).
        feature_names (list[str]): Column names used as features.
    """
    logger.info(f"Loading feature matrix from {path}")
    df = pd.read_csv(path)
    logger.info(f"Raw shape: {df.shape}")

    # Drop columns with >50% NaN (process + AST features are 100% NaN for KC1
    # because KC1 is a C codebase with no matching Python git history)
    nan_frac = df.isnull().mean()
    drop_cols = nan_frac[nan_frac > 0.50].index.tolist()
    logger.info(f"Dropping {len(drop_cols)} NaN-dominant columns: {drop_cols}")
    df = df.drop(columns=drop_cols)

    # Drop non-feature identifier columns
    for col in ["file_path", "Unnamed: 0"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Target
    y = df["is_buggy"].astype(int)
    df = df.drop(columns=["is_buggy"])

    # Future-proof binary flag: will be 1 when a real matched git repo is
    # available. Currently 0 for all KC1 rows.
    df["has_process_data"] = 0
    logger.info("Added feature 'has_process_data' = 0 (future-proof placeholder)")

    if drop_halstead_redundant:
        cols_to_drop = [c for c in HALSTEAD_REDUNDANT if c in df.columns]
        logger.warning(
            f"Dropping {len(cols_to_drop)} redundant Halstead features "
            f"to reduce multicollinearity: {cols_to_drop}"
        )
        df = df.drop(columns=cols_to_drop)

    feature_names = df.columns.tolist()
    logger.info(f"Final feature count: {len(feature_names)}")
    logger.info(f"Class distribution — buggy: {y.mean():.2%}")
    return df, y, feature_names


# In src/models/train.py — replace find_best_threshold()
def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.40,
) -> float:
    """
    Find threshold that maximises precision subject to recall >= min_recall.
    Falls back to max-F1 threshold if no threshold satisfies the recall floor.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        min_recall: Minimum acceptable recall (default 0.40).

    Returns:
        best_threshold: Float that maximises precision at recall >= min_recall.
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_precision, best_thr = 0.0, 0.5
    fallback_f1, fallback_thr = 0.0, 0.5

    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        # Track best F1 as fallback
        if f1 > fallback_f1:
            fallback_f1 = f1
            fallback_thr = float(thr)

        # Main objective: maximise precision with recall floor
        if r >= min_recall and p > best_precision:
            best_precision = p
            best_thr = float(thr)

    if best_precision == 0.0:
        logger.warning(
            f"No threshold achieved recall >= {min_recall:.0%}. "
            f"Falling back to max-F1 threshold={fallback_thr:.2f}"
        )
        return fallback_thr

    return best_thr


def train_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    fold_idx: int,
) -> dict:
    """
    Train XGBoost on one CV fold, evaluate on validation set, tune threshold.

    Args:
        X_train: Training features for this fold.
        y_train: Training labels for this fold.
        X_val: Validation features for this fold.
        y_val: Validation labels for this fold.
        fold_idx: 1-based fold number for logging.

    Returns:
        metrics: Dict containing AUC, F1, Precision, Recall, AP,
                 best_threshold, val_probs, best_estimators.
    """
    params = {
        "n_estimators": MODEL.get("n_estimators", 400),
        "max_depth": MODEL.get("max_depth", 6),
        "learning_rate": MODEL.get("learning_rate", 0.05),
        "subsample": MODEL.get("subsample", 0.8),
        "colsample_bytree": MODEL.get("colsample_bytree", 0.8),
        "scale_pos_weight": MODEL.get("scale_pos_weight", 5),
        "random_state": MODEL.get("random_state", 42),
        "eval_metric": "aucpr",
        "verbosity": 0,
    }

    model = xgb.XGBClassifier(
        **params,
        early_stopping_rounds=MODEL.get("early_stopping_rounds", 30),
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_probs = model.predict_proba(X_val)[:, 1]
    best_thr = find_best_threshold(y_val.values, val_probs)
    val_preds = (val_probs >= best_thr).astype(int)

    metrics = {
        "auc": roc_auc_score(y_val, val_probs),
        "f1": f1_score(y_val, val_preds, zero_division=0),
        "precision": precision_score(y_val, val_preds, zero_division=0),
        "recall": recall_score(y_val, val_preds, zero_division=0),
        "ap": average_precision_score(y_val, val_probs),
        "best_threshold": best_thr,
        "val_probs": val_probs,
        "best_estimators": model.best_iteration + 1,
    }
    logger.info(
        f"Fold {fold_idx} | AUC={metrics['auc']:.4f} | "
        f"F1={metrics['f1']:.4f} | P={metrics['precision']:.4f} | "
        f"R={metrics['recall']:.4f} | AP={metrics['ap']:.4f} | "
        f"thr={metrics['best_threshold']:.2f} | "
        f"trees={metrics['best_estimators']}"
    )
    return metrics


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int,
) -> xgb.XGBClassifier:
    """
    Retrain XGBoost on the full dataset with a fixed tree count.

    Uses the average best_iteration from early-stopped CV folds so the
    final model benefits from all data without overfitting.

    Args:
        X: Full feature matrix (all rows).
        y: Full binary target vector.
        n_estimators: Tree count derived from avg CV early-stopping iteration.

    Returns:
        Fitted XGBClassifier trained on the complete dataset.
    """
    params = {
        "n_estimators": n_estimators,
        "max_depth": MODEL.get("max_depth", 6),
        "learning_rate": MODEL.get("learning_rate", 0.05),
        "subsample": MODEL.get("subsample", 0.8),
        "colsample_bytree": MODEL.get("colsample_bytree", 0.8),
        "scale_pos_weight": MODEL.get("scale_pos_weight", 3),
        "min_child_weight": MODEL.get("min_child_weight", 5),
        "gamma": MODEL.get("gamma", 1.0),
        "random_state": MODEL.get("random_state", 42),
        "verbosity": 0,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    return model


def run_training(drop_halstead_redundant: bool = False) -> dict:
    """
    Full end-to-end training pipeline.

    Steps:
        1. Load and prepare feature matrix.
        2. StratifiedKFold CV with per-fold threshold tuning.
        3. Aggregate metrics; warn if AUC < 0.65.
        4. Save OOF predictions.
        5. Retrain final model on full dataset.
        6. Save model and metadata; log everything to MLflow.

    Args:
        drop_halstead_redundant: Whether to drop correlated Halstead features.

    Returns:
        summary: Dict with mean_auc, mean_f1, mean_ap, best_threshold,
                 model_path, feature_names.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    # Load raw CSV once to preserve file_path for OOF output
    raw_df = pd.read_csv(FEATURE_MATRIX_PATH)
    file_paths = raw_df["file_path"] if "file_path" in raw_df.columns else pd.Series(range(len(raw_df)))

    X, y, feature_names = load_and_prepare_data(
        FEATURE_MATRIX_PATH,
        drop_halstead_redundant=drop_halstead_redundant,
    )

    mlflow.set_experiment("model-training")
    with mlflow.start_run(run_name="xgboost_kc1_cv"):

        mlflow.log_params({k: v for k, v in MODEL.items()})
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("pos_rate", round(float(y.mean()), 4))
        mlflow.log_param("drop_halstead_redundant", drop_halstead_redundant)

        skf = StratifiedKFold(
            n_splits=MODEL.get("n_splits", 5),
            shuffle=True,
            random_state=MODEL.get("random_state", 42),
        )

        fold_metrics = []
        oof_probs = np.zeros(len(X))

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            metrics = train_fold(X_train, y_train, X_val, y_val, fold_idx)
            oof_probs[val_idx] = metrics["val_probs"]
            fold_metrics.append(metrics)

            mlflow.log_metrics({
                f"fold{fold_idx}_auc": metrics["auc"],
                f"fold{fold_idx}_f1": metrics["f1"],
                f"fold{fold_idx}_precision": metrics["precision"],
                f"fold{fold_idx}_recall": metrics["recall"],
                f"fold{fold_idx}_ap": metrics["ap"],
                f"fold{fold_idx}_threshold": metrics["best_threshold"],
            })

        # ---- Aggregate ----
        mean_auc = float(np.mean([m["auc"] for m in fold_metrics]))
        std_auc = float(np.std([m["auc"] for m in fold_metrics]))
        mean_f1 = float(np.mean([m["f1"] for m in fold_metrics]))
        mean_ap = float(np.mean([m["ap"] for m in fold_metrics]))
        best_threshold = float(np.mean([m["best_threshold"] for m in fold_metrics]))
        avg_estimators = int(np.mean([m["best_estimators"] for m in fold_metrics]))

        logger.info("=" * 60)
        logger.info(
            f"CV Summary | AUC={mean_auc:.4f}±{std_auc:.4f} | "
            f"F1={mean_f1:.4f} | AP={mean_ap:.4f}"
        )
        logger.info(f"Best threshold (avg): {best_threshold:.3f}")
        logger.info(f"Avg estimators (early-stopped): {avg_estimators}")

        # ---- Sanity checks ----
        if mean_auc < 0.65:
            logger.warning(
                f"⚠ AUC={mean_auc:.4f} is BELOW the expected floor of 0.65 for KC1. "
                "Try drop_halstead_redundant=True or verify feature_matrix.csv integrity."
            )
        assert mean_auc >= 0.50, (
            f"AUC={mean_auc:.4f} — model performs worse than random! "
            "Check label column and feature matrix."
        )
        # Expected KC1 range: 0.70–0.78 (warn, don't assert)
        if mean_auc < 0.70:
            logger.warning(
                f"AUC={mean_auc:.4f} is below the expected KC1 range of 0.70–0.78. "
                "This is acceptable if features are KC1-static only."
            )

        mlflow.log_metrics({
            "cv_mean_auc": mean_auc,
            "cv_std_auc": std_auc,
            "cv_mean_f1": mean_f1,
            "cv_mean_ap": mean_ap,
            "best_threshold": best_threshold,
            "avg_estimators": avg_estimators,
        })

        # ---- OOF predictions ----
        oof_df = pd.DataFrame({
            "file_path": file_paths.values,
            "is_buggy": y.values,
            "oof_prob": oof_probs,
            "oof_pred": (oof_probs >= best_threshold).astype(int),
        })
        oof_df.to_csv(OOF_PREDICTIONS_PATH, index=False)
        logger.info(f"OOF predictions saved → {OOF_PREDICTIONS_PATH}")
        mlflow.log_artifact(str(OOF_PREDICTIONS_PATH))

        # ---- Final model ----
        logger.info(
            f"Retraining final model on full dataset "
            f"with {avg_estimators} trees..."
        )
        final_model = train_final_model(X, y, n_estimators=avg_estimators)
        final_model.save_model(str(MODEL_PATH))
        logger.info(f"Final model saved → {MODEL_PATH}")
        mlflow.xgboost.log_model(final_model, artifact_path="xgboost_model")
        mlflow.log_artifact(str(MODEL_PATH))

        # ---- Metadata sidecar ----
        meta = {
            "feature_names": feature_names,
            "best_threshold": best_threshold,
            "cv_mean_auc": mean_auc,
            "cv_mean_f1": mean_f1,
            "avg_estimators": avg_estimators,
        }
        meta_path = MODEL_DIR / "model_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        mlflow.log_artifact(str(meta_path))

        summary = {
            "mean_auc": mean_auc,
            "mean_f1": mean_f1,
            "mean_ap": mean_ap,
            "best_threshold": best_threshold,
            "model_path": str(MODEL_PATH),
            "feature_names": feature_names,
        }
        logger.info("Training pipeline complete.")
        return summary


if __name__ == "__main__":
    run_training()