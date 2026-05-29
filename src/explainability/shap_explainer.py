"""
Day 3 — SHAP Explainability Module
====================================
Computes global and local SHAP explanations for the XGBoost defect predictor.

Global outputs:
  - reports/figures/shap_summary.png   (beeswarm plot)
  - reports/figures/shap_importance.png (mean |SHAP| bar plot)
  - data/processed/shap_values.csv

Local output (per module):
  - reports/figures/shap_waterfall_<module_slug>.png
  - Returns dict {feature: shap_value} for programmatic use
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "xgboost_defect_predictor.json"
META_PATH = ROOT / "models" / "model_meta.json"
FEATURE_MATRIX_PATH = ROOT / "data" / "processed" / "feature_matrix_final.csv"
SHAP_VALUES_PATH = ROOT / "data" / "processed" / "shap_values.csv"
FIGURES_DIR = ROOT / "reports" / "figures"


def _slugify(text: str) -> str:
    """Convert a file path to a safe filename slug."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)[:60]


def load_model_and_data() -> tuple:
    """
    Load the saved XGBoost model, metadata, and prepared feature matrix.

    The feature matrix is preprocessed identically to training: drop NaN-dominant
    columns, drop identifiers, add has_process_data flag.

    Returns:
        model: Loaded XGBClassifier.
        meta: Dict from model_meta.json.
        X: Feature DataFrame aligned to training feature_names.
        file_paths: Series of file path identifiers.
    """
    logger.info("Loading model for SHAP analysis...")
    
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    with open(META_PATH) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]

    df = pd.read_csv(FEATURE_MATRIX_PATH)
    file_paths = df["file_path"] if "file_path" in df.columns else pd.Series(range(len(df)))

    # Mirror preprocessing from train.py
    nan_frac = df.isnull().mean()
    drop_cols = nan_frac[nan_frac > 0.50].index.tolist()
    df = df.drop(columns=drop_cols, errors="ignore")
    for col in ["file_path", "Unnamed: 0", "is_buggy"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if "has_process_data" not in df.columns:
        df["has_process_data"] = 0

    # Align columns to training feature order
    X = df[feature_names]
    logger.info(f"Feature matrix aligned: {X.shape}")
    return model, meta, X, file_paths


def compute_shap_values(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values for all samples using TreeExplainer.

    TreeExplainer is exact and efficient for tree-based models. Returns
    SHAP values for the positive class (index 1) of the binary classifier.

    Args:
        model: Fitted XGBClassifier.
        X: Feature matrix (n_samples × n_features).

    Returns:
        shap_values: np.ndarray of shape (n_samples, n_features).
    """
    logger.info("Computing SHAP values via TreeExplainer (this may take ~30s)...")
    explainer = shap.TreeExplainer(getattr(model, "calibrated_classifiers_", [model])[0].estimator if hasattr(model, "calibrated_classifiers_") else model)
    shap_output = explainer.shap_values(X)

    # For XGBoost binary classifier, shap_values is already (n, f)
    # Some SHAP versions return a list [neg_class, pos_class]
    if isinstance(shap_output, list):
        shap_vals = shap_output[1]
    else:
        shap_vals = shap_output

    logger.info(f"SHAP values computed: {shap_vals.shape}")
    return shap_vals, explainer


def plot_shap_summary(
    shap_vals: np.ndarray,
    X: pd.DataFrame,
    save_path: Path,
) -> None:
    """
    Generate and save a SHAP beeswarm (summary) plot.

    Each dot represents one sample; colour encodes feature value (red=high,
    blue=low); x-axis position encodes SHAP impact on model output.

    Args:
        shap_vals: SHAP values array (n_samples, n_features).
        X: Feature matrix with column names.
        save_path: Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_vals, X,
        plot_type="dot",
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary (beeswarm) saved → {save_path}")


def plot_shap_importance(
    shap_vals: np.ndarray,
    X: pd.DataFrame,
    save_path: Path,
) -> pd.Series:
    """
    Generate and save a SHAP bar plot of mean absolute SHAP values.

    Mean |SHAP| is a model-consistent global feature importance measure,
    more reliable than XGBoost's native gain/split counts for correlated features.

    Args:
        shap_vals: SHAP values array (n_samples, n_features).
        X: Feature matrix with column names.
        save_path: Output PNG path.

    Returns:
        importance: pd.Series sorted by mean |SHAP| descending.
    """
    mean_abs_shap = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=X.columns,
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    mean_abs_shap.head(20).sort_values().plot(
        kind="barh", ax=ax, color="steelblue"
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (Mean |SHAP|) — Top 20 Features")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"SHAP importance bar plot saved → {save_path}")
    logger.info(f"Top 5 SHAP features:\n{mean_abs_shap.head(5).to_string()}")
    return mean_abs_shap


def save_shap_values(
    shap_vals: np.ndarray,
    X: pd.DataFrame,
    file_paths: pd.Series,
    save_path: Path,
) -> None:
    """
    Persist the full SHAP values matrix to CSV for downstream analysis.

    Args:
        shap_vals: SHAP values array (n_samples, n_features).
        X: Feature matrix (provides column names).
        file_paths: Module identifiers for the index column.
        save_path: Output CSV path.
    """
    shap_df = pd.DataFrame(shap_vals, columns=X.columns)
    shap_df.insert(0, "file_path", file_paths.values)
    shap_df.to_csv(save_path, index=False)
    logger.info(f"SHAP values matrix saved → {save_path}  ({shap_df.shape})")


def explain_module(
    file_path: str,
    X: pd.DataFrame,
    shap_vals: np.ndarray,
    explainer,
    file_paths: pd.Series,
    save_figure: bool = True,
) -> dict:
    """
    Generate a local SHAP explanation for a single module.

    Finds the row whose file_path matches the argument, computes a waterfall
    plot (showing how each feature pushes the prediction from the base value),
    and returns a feature→SHAP-value mapping sorted by absolute impact.

    Args:
        file_path: Identifier of the module to explain (must match feature_matrix).
        X: Full feature matrix.
        shap_vals: SHAP values array (n_samples, n_features).
        explainer: Fitted shap.TreeExplainer instance.
        file_paths: Series of identifiers aligned to X.
        save_figure: If True, saves a waterfall PNG to reports/figures/.

    Returns:
        explanation: Dict {feature: shap_value} sorted by |shap_value| desc.
                     Also contains "__base_value__" and "__prediction_shap_sum__".

    Raises:
        ValueError: If file_path is not found in the dataset.
    """
    matches = file_paths[file_paths == file_path]
    if len(matches) == 0:
        raise ValueError(
            f"file_path '{file_path}' not found in dataset. "
            f"Available sample: {file_paths.iloc[0]}"
        )

    idx = matches.index[0]
    row_shap = shap_vals[idx]
    row_X = X.iloc[idx]

    if save_figure:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        slug = _slugify(file_path)
        fig_path = FIGURES_DIR / f"shap_waterfall_{slug}.png"

        # Build a shap.Explanation object for the waterfall plot
        shap_explanation = shap.Explanation(
            values=row_shap,
            base_values=explainer.expected_value
            if not isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value[1],
            data=row_X.values,
            feature_names=X.columns.tolist(),
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(shap_explanation, show=False, max_display=15)
        plt.title(f"SHAP Waterfall — {Path(file_path).name}")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Waterfall plot saved → {fig_path}")

    # Build explanation dict sorted by |SHAP| descending
    base_val = (
        explainer.expected_value
        if not isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value[1]
    )
    explanation = {
        "__base_value__": float(base_val),
        "__prediction_shap_sum__": float(row_shap.sum()),
    }
    feature_impact = {
        feat: float(sv)
        for feat, sv in zip(X.columns, row_shap)
    }
    sorted_impact = dict(
        sorted(feature_impact.items(), key=lambda kv: abs(kv[1]), reverse=True)
    )
    explanation.update(sorted_impact)
    return explanation


def run_shap_analysis() -> pd.Series:
    """
    Full SHAP pipeline: load data → compute SHAP → save global plots + CSV.

    Returns:
        mean_abs_shap: pd.Series of global feature importances (Mean |SHAP|).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    model, meta, X, file_paths = load_model_and_data()
    shap_vals, explainer = compute_shap_values(model, X)

    plot_shap_summary(shap_vals, X, FIGURES_DIR / "shap_summary.png")
    importance = plot_shap_importance(shap_vals, X, FIGURES_DIR / "shap_importance.png")
    save_shap_values(shap_vals, X, file_paths, SHAP_VALUES_PATH)

    logger.info("SHAP analysis complete.")
    return importance, shap_vals, explainer, X, file_paths


if __name__ == "__main__":
    run_shap_analysis()
