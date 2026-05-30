"""
Day 4 — Hybrid Model: GNN structural embeddings (32 dims) + tabular features (22 dims)
→ 54-dim XGBoost input.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from loguru import logger

from configs.config import MLFLOW, MODEL
from src.models.evaluate import TemporalEvaluator
from src.models.gnn_model import GNNTrainer
from src.models.train import DefectXGBoost

TEMPORAL_TEST_MONTHS: int = MODEL["test_months"]
XGB_DEFAULTS: dict = MODEL["xgb_defaults"]
EMBEDDING_DIM: int = MODEL["gnn"]["embedding_dim"]  # 32


# ---------------------------------------------------------------------------
# HybridDefectModel
# ---------------------------------------------------------------------------

class HybridDefectModel:
    """
    Combines GNN structural embeddings (32 dims) with tabular
    process + static features (22 dims) → 54-dim XGBoost input.

    XGBoost handles the tabular features well; GNN handles structural
    patterns. Neither alone matches the combination.
    """

    def __init__(self, gnn_trainer: GNNTrainer, xgb_params: dict | None = None) -> None:
        self.gnn_trainer = gnn_trainer
        self.xgb_params = xgb_params or XGB_DEFAULTS
        self.xgb = DefectXGBoost(**{k: v for k, v in self.xgb_params.items() if k != "eval_metric"})
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------
    def _get_tabular_cols(self, df: pd.DataFrame) -> list[str]:
        """Return feature columns (drop metadata cols)."""
        drop_cols = {"is_buggy", "file_path", "commit_date", "label", "window_start", "window_end", "repo_name","repo"}
        return [c for c in df.columns if c not in drop_cols]

    # ------------------------------------------------------------------
    def build_hybrid_features(
        self,
        df_features: pd.DataFrame,
        embeddings: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        For each row in df_features:
          1. Look up GNN embedding by file_path
          2. Concatenate with tabular feature row
          3. If embedding missing → use np.zeros(32)
        Returns array of shape [N, 54].
        """
        if not self._feature_cols:
            self._feature_cols = self._get_tabular_cols(df_features)
            
        tab_cols = self._feature_cols
        
        missing = [c for c in tab_cols if c not in df_features.columns]
        for c in missing:
            df_features[c] = 0.0

        tab_matrix = df_features[tab_cols].values.astype(np.float32)  # [N, 22]

        emb_rows: list[np.ndarray] = []
        for _, row in df_features.iterrows():
            fp = row.get("file_path", "")
            emb = embeddings.get(str(fp), np.zeros(EMBEDDING_DIM, dtype=np.float32))
            emb_rows.append(emb.astype(np.float32))

        emb_matrix = np.vstack(emb_rows)  # [N, 32]
        hybrid = np.hstack([tab_matrix, emb_matrix])  # [N, 54]
        return hybrid

    # ------------------------------------------------------------------
    def train(
        self,
        df_train: pd.DataFrame,
        embeddings: dict[str, np.ndarray],
    ) -> dict:
        """Train XGBoost on 54-dim features. Return evaluation metrics."""
        X_train = self.build_hybrid_features(df_train, embeddings)
        y_train = df_train["is_buggy"].values

        logger.info(f"Training HybridDefectModel on {X_train.shape} feature matrix.")
        self.xgb.train(X_train, y_train)

        scores = self.xgb.predict_proba(X_train)
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
        auc = roc_auc_score(y_train, scores)
        ap = average_precision_score(y_train, scores)
        preds = (scores >= 0.5).astype(int)
        f1 = f1_score(y_train, preds, zero_division=0)

        metrics = {"train_auc": auc, "train_ap": ap, "train_f1": f1}
        logger.info(f"Hybrid train metrics: {metrics}")
        return metrics

    # ------------------------------------------------------------------
    def predict_proba(
        self,
        df_features: pd.DataFrame,
        embeddings: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Return defect probability scores, shape [N]."""
        X = self.build_hybrid_features(df_features, embeddings)
        return self.xgb.predict_proba(X)

    # ------------------------------------------------------------------
    def get_feature_names(self) -> list[str]:
        """54 feature names: tabular names + gnn_emb_0 … gnn_emb_31."""
        tab = list(self._feature_cols)
        gnn = [f"gnn_emb_{i}" for i in range(EMBEDDING_DIM)]
        return tab + gnn

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "xgb": self.xgb,
            "feature_cols": self._feature_cols,
            "xgb_params": self.xgb_params,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"HybridDefectModel saved to {path}")

    def load(self, path: Path) -> None:
        path = Path(path)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.xgb = payload["xgb"]
        self._feature_cols = payload["feature_cols"]
        self.xgb_params = payload["xgb_params"]
        logger.info(f"HybridDefectModel loaded from {path}")


# ---------------------------------------------------------------------------
# Three-way comparison
# ---------------------------------------------------------------------------

def run_three_way_comparison(
    df_features: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    gnn_trainer: GNNTrainer,
) -> pd.DataFrame:
    """
    Runs and logs three MLflow runs under experiment 'model-comparison':

    Run 1 — 'baseline_random_split':
        DefectXGBoost on 22 tabular features, random 80/20 split
        Expected: AUC ~0.83, F1 ~0.52 (inflated — data leakage)

    Run 2 — 'baseline_temporal_split':
        DefectXGBoost on 22 tabular features, temporal split (test_months=3)
        Expected: AUC ~0.7926, F1 ~0.44 (honest — matches Day 3)

    Run 3 — 'hybrid_temporal_split':
        HybridDefectModel on 54 features, temporal split
        Target: AUC > 0.82, Precision@20 > 0.78

    Returns DataFrame with columns:
        run_name | AUC | F1 | Precision | Recall | Precision@20 | n_features

    Raises warning (not error) if hybrid does not beat baseline.
    """
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split

    mlflow.set_tracking_uri(MLFLOW["tracking_uri"])
    mlflow.set_experiment("model-comparison")

    evaluator = TemporalEvaluator()

    # Identify tabular feature columns
    # We alias window_end for temporal split
    if "window_end" in df_features.columns:
        df_features["commit_date"] = pd.to_datetime(df_features["window_end"])
        
    drop_cols = {"is_buggy", "file_path", "commit_date", "label", "window_start", "window_end", "repo_name","repo"}
    tab_cols = [c for c in df_features.columns if c not in drop_cols]
    n_tab = len(tab_cols)

    results: list[dict] = []

    # ------------------------------------------------------------------
    # Helper to compute all metrics
    # ------------------------------------------------------------------
    def _metrics(y_true: np.ndarray, y_scores: np.ndarray, run_name: str, n_features: int) -> dict:
        threshold = 0.5
        y_pred = (y_scores >= threshold).astype(int)
        auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        p_at_20 = _precision_at_k(y_true, y_scores, 20)
        return {
            "run_name": run_name,
            "AUC": round(auc, 4),
            "F1": round(f1, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "Precision@20": round(p_at_20, 4),
            "n_features": n_features,
        }

    def _precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
        if k <= 0 or len(y_true) == 0:
            return 0.0
        top_k_idx = np.argsort(y_scores)[::-1][:k]
        return float(y_true[top_k_idx].sum()) / k

    # ------------------------------------------------------------------
    # Run 1 — baseline_random_split
    # ------------------------------------------------------------------
    logger.info("Running: baseline_random_split")
    with mlflow.start_run(run_name="baseline_random_split"):
        X_all = df_features[tab_cols].values
        y_all = df_features["is_buggy"].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
        xgb1 = DefectXGBoost()
        xgb1.train(X_tr, y_tr)
        scores1 = xgb1.predict_proba(X_te)

        m1 = _metrics(y_te, scores1, "baseline_random_split", n_tab)
        mlflow.log_param("n_features", n_tab)
        mlflow.log_param("split", "random_80_20")
        for k, v in m1.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k.replace("@", "_at_"), v)
        results.append(m1)
        logger.info(f"  AUC={m1['AUC']:.4f}  F1={m1['F1']:.4f}  P@20={m1['Precision@20']:.4f}")

    # ------------------------------------------------------------------
    # Run 2 — baseline_temporal_split
    # ------------------------------------------------------------------
    logger.info("Running: baseline_temporal_split")
    with mlflow.start_run(run_name="baseline_temporal_split"):
        if "commit_date" in df_features.columns:
            df_train2, df_test2 = evaluator.temporal_train_test_split(
                df_features, test_months=TEMPORAL_TEST_MONTHS
            )
        else:
            # No commit_date — stratified 80/20 to preserve class ratio in test set
            df_train2, df_test2 = train_test_split(
                df_features, test_size=0.2, random_state=42,
                stratify=df_features["is_buggy"] if "is_buggy" in df_features.columns else None,
            )
            logger.warning("No commit_date column — using stratified 80/20 split for temporal baseline.")

        X_tr2 = df_train2[tab_cols].values
        y_tr2 = df_train2["is_buggy"].values
        X_te2 = df_test2[tab_cols].values
        y_te2 = df_test2["is_buggy"].values

        xgb2 = DefectXGBoost()
        xgb2.train(X_tr2, y_tr2)
        scores2 = xgb2.predict_proba(X_te2)

        m2 = _metrics(y_te2, scores2, "baseline_temporal_split", n_tab)
        mlflow.log_param("n_features", n_tab)
        mlflow.log_param("split", f"temporal_{TEMPORAL_TEST_MONTHS}mo")
        for k, v in m2.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k.replace("@", "_at_"), v)
        results.append(m2)
        logger.info(f"  AUC={m2['AUC']:.4f}  F1={m2['F1']:.4f}  P@20={m2['Precision@20']:.4f}")

    # ------------------------------------------------------------------
    # Run 3 — hybrid_temporal_split
    # ------------------------------------------------------------------
    logger.info("Running: hybrid_temporal_split")
    with mlflow.start_run(run_name="hybrid_temporal_split"):
        if "commit_date" in df_features.columns:
            df_train3, df_test3 = evaluator.temporal_train_test_split(
                df_features, test_months=TEMPORAL_TEST_MONTHS
            )
        else:
            df_train3, df_test3 = train_test_split(
                df_features, test_size=0.2, random_state=42,
                stratify=df_features["is_buggy"] if "is_buggy" in df_features.columns else None,
            )

        hybrid = HybridDefectModel(gnn_trainer=gnn_trainer)
        hybrid.train(df_train3, embeddings)
        scores3 = hybrid.predict_proba(df_test3, embeddings)
        y_te3 = df_test3["is_buggy"].values

        n_hybrid = n_tab + MODEL["gnn"]["embedding_dim"]  # 54
        m3 = _metrics(y_te3, scores3, "hybrid_temporal_split", n_hybrid)
        mlflow.log_param("n_features", n_hybrid)
        mlflow.log_param("split", f"temporal_{TEMPORAL_TEST_MONTHS}mo")
        mlflow.log_param("embedding_dim", MODEL["gnn"]["embedding_dim"])
        for k, v in m3.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k.replace("@", "_at_"), v)
        results.append(m3)
        logger.info(f"  AUC={m3['AUC']:.4f}  F1={m3['F1']:.4f}  P@20={m3['Precision@20']:.4f}")

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    comparison_df = pd.DataFrame(results)

    print("\n" + "=" * 72)
    print("MODEL COMPARISON — Day 4 Results")
    print("=" * 72)
    print(comparison_df.to_string(index=False))
    print("=" * 72)
    print(f"Day 3 baseline (temporal): AUC=0.7926  P@20=0.75")
    print("=" * 72 + "\n")

    # Warn if hybrid does not beat baseline
    baseline_auc = m2["AUC"]
    hybrid_auc = m3["AUC"]
    if hybrid_auc <= baseline_auc:
        warnings.warn(
            f"Hybrid AUC ({hybrid_auc:.4f}) did not beat temporal baseline ({baseline_auc:.4f}). "
            "Consider longer GNN training or richer source coverage.",
            UserWarning,
            stacklevel=2,
        )

    return comparison_df