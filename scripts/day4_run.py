"""
Day 4 run script — GNN training + hybrid model comparison.

Steps:
  1.  Load feature_matrix_final.csv
  2.  Load combined_source_codes.pkl (all 12 repos) or flask fallback
  3.  Build labels dict from is_buggy column
  4.  Initialize CodeGNN + GNNTrainer
  5.  Build PyG dataset
  6.  Train GNN
  7.  Save GNN to models/gnn_model.pt
  8.  Get embeddings for all files
  9.  Save embeddings to data/processed/gnn_embeddings.pkl
  10. Run three-way comparison
  11. Save hybrid model to models/hybrid_model.pkl
  12. Print final metrics + git commit command
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from loguru import logger

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import MLFLOW, MODEL
from src.models.gnn_model import CodeGNN, GNNTrainer
from src.models.hybrid_model import HybridDefectModel, run_three_way_comparison

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR      = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"

FEATURE_MATRIX_PATH  = PROCESSED_DIR / "feature_matrix_final.csv"
COMBINED_SOURCE_PATH = PROCESSED_DIR / "combined_source_codes.pkl"
FLASK_COMMITS_PATH   = DATA_DIR / "raw" / "flask_commits.csv"

GNN_MODEL_PATH    = MODELS_DIR / "gnn_model.pt"
EMBEDDINGS_PATH   = PROCESSED_DIR / "gnn_embeddings.pkl"
HYBRID_MODEL_PATH = MODELS_DIR / "hybrid_model.pkl"

# ---------------------------------------------------------------------------
# Configure loguru
# ---------------------------------------------------------------------------
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
logger.add(log_dir / "day4_run.log", rotation="10 MB", level="DEBUG")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_source_codes() -> dict[str, str]:
    """
    Load source codes for GNN embedding.
    Prefers combined_source_codes.pkl (all 12 repos, 3602 files).
    Falls back to flask_commits.csv if pkl not found.
    """
    if COMBINED_SOURCE_PATH.exists():
        with open(COMBINED_SOURCE_PATH, "rb") as f:
            source_codes = pickle.load(f)
        logger.info(f"  Loaded combined_source_codes.pkl: {len(source_codes)} files")
        return source_codes

    logger.warning(
        "combined_source_codes.pkl not found — falling back to flask_commits.csv. "
        "Run scripts/day5_mine_repos.py --source-only for full GNN coverage."
    )

    if not FLASK_COMMITS_PATH.exists():
        logger.error(f"flask_commits.csv not found at {FLASK_COMMITS_PATH}")
        return {}

    flask_df = pd.read_csv(FLASK_COMMITS_PATH)
    source_col = next(
        (c for c in ["source_code", "source", "content", "code"]
         if c in flask_df.columns),
        None,
    )
    if source_col is None:
        logger.error("No source code column found in flask_commits.csv.")
        return {}

    if "commit_date" in flask_df.columns:
        flask_df["commit_date"] = pd.to_datetime(flask_df["commit_date"], errors="coerce")
        latest = (
            flask_df.sort_values("commit_date")
            .drop_duplicates(subset=["file_path"], keep="last")
        )
    else:
        latest = flask_df.drop_duplicates(subset=["file_path"], keep="last")

    source_codes = dict(
        zip(
            latest["file_path"].astype(str),
            latest[source_col].fillna("").astype(str),
        )
    )
    logger.info(f"  Loaded flask_commits.csv: {len(source_codes)} files")
    return source_codes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Step 1: Load feature matrix
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading feature_matrix_final.csv")

    matrix_path = FEATURE_MATRIX_PATH
    if not matrix_path.exists():
        fallback = PROCESSED_DIR / "feature_matrix_collapsed.csv"
        if fallback.exists():
            matrix_path = fallback
            logger.warning(f"feature_matrix_final.csv not found — using {fallback.name}")
        else:
            logger.error("No feature matrix found. Run prep_for_training.py first.")
            sys.exit(1)

    df_features = pd.read_csv(matrix_path)

    drop_meta = {
        "is_buggy", "file_path", "commit_date",
        "label", "window_start", "window_end",
        "repo_name", "repo",
    }
    feature_cols = [c for c in df_features.columns if c not in drop_meta]

    logger.info(f"  Shape: {df_features.shape}")
    logger.info(f"  Feature columns ({len(feature_cols)}): {feature_cols}")
    assert "is_buggy"  in df_features.columns, "Missing 'is_buggy' column"
    assert "file_path" in df_features.columns, "Missing 'file_path' column"
    logger.info(f"  Confirmed {len(feature_cols)} features + is_buggy + file_path ✓")

    # ------------------------------------------------------------------
    # Step 2: Load source codes
    # ------------------------------------------------------------------
    logger.info("Step 2: Loading source codes")
    source_codes = load_source_codes()

    if not source_codes:
        logger.error("No source codes loaded. Cannot build GNN dataset.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Build labels dict
    # ------------------------------------------------------------------
    logger.info("Step 3: Building labels dict from feature_matrix")

    # groupby().max() — file is buggy if ANY window was buggy
    labels: dict[str, int] = (
        df_features.groupby("file_path")["is_buggy"]
        .max()
        .astype(int)
        .to_dict()
    )
    n_buggy = sum(labels.values())
    logger.info(f"  Total labeled files: {len(labels)}")
    logger.info(f"  Buggy: {n_buggy} ({100 * n_buggy / len(labels):.1f}%)")

    # ------------------------------------------------------------------
    # Step 4: Initialize CodeGNN + GNNTrainer
    # ------------------------------------------------------------------
    logger.info("Step 4: Initialising CodeGNN + GNNTrainer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")

    model   = CodeGNN()
    trainer = GNNTrainer(model=model, device=device)
    logger.info(f"  CodeGNN parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Step 5: Build PyG dataset
    # ------------------------------------------------------------------
    logger.info("Step 5: Building PyG dataset")
    dataset     = trainer.build_dataset(source_codes=source_codes, labels=labels)
    total_files = len(source_codes)
    parseable   = len(dataset)
    skipped     = total_files - parseable

    print(f"\n  Total files:     {total_files}")
    print(f"  Parseable files: {parseable}")
    print(f"  Skipped files:   {skipped}\n")

    if parseable == 0:
        logger.error("No parseable files — GNN will produce zero embeddings.")
        logger.info("Hybrid model will rely entirely on tabular features.")

    # ------------------------------------------------------------------
    # Step 6: Train GNN
    # ------------------------------------------------------------------
    logger.info("Step 6: Training GNN")
    epochs = MODEL["gnn"]["epochs"]

    if parseable > 0:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW["tracking_uri"])
        mlflow.set_experiment(MLFLOW["experiment_name"])
        with mlflow.start_run(run_name="gnn_training"):
            mlflow.log_param("epochs",          epochs)
            mlflow.log_param("hidden_dim",      MODEL["gnn"]["hidden_dim"])
            mlflow.log_param("embedding_dim",   MODEL["gnn"]["embedding_dim"])
            mlflow.log_param("num_layers",      MODEL["gnn"]["num_layers"])
            mlflow.log_param("dropout",         MODEL["gnn"]["dropout"])
            mlflow.log_param("lr",              MODEL["gnn"]["lr"])
            mlflow.log_param("parseable_files", parseable)
            epoch_losses = trainer.train(dataset, epochs=epochs)
            mlflow.log_metric("final_train_loss", epoch_losses[-1])
        logger.info(f"  GNN training complete — final loss: {epoch_losses[-1]:.4f}")
    else:
        logger.warning("Skipping GNN training — no parseable files.")
        epoch_losses = [0.0]

    # ------------------------------------------------------------------
    # Step 7: Save GNN
    # ------------------------------------------------------------------
    logger.info("Step 7: Saving GNN model")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save(GNN_MODEL_PATH)

    # ------------------------------------------------------------------
    # Step 8: Get embeddings
    # ------------------------------------------------------------------
    logger.info("Step 8: Extracting embeddings for all files")
    embeddings = trainer.get_embeddings(source_codes)

    # Coverage report
    n_real = sum(
        1 for fp in df_features["file_path"].astype(str)
        if fp in embeddings and any(embeddings[fp] != 0)
    )
    logger.info(
        f"  Embeddings: {len(embeddings)} files total | "
        f"{n_real}/{len(df_features)} feature-matrix files have real embeddings "
        f"({n_real/len(df_features)*100:.1f}% coverage)"
    )

    # ------------------------------------------------------------------
    # Step 9: Save embeddings
    # ------------------------------------------------------------------
    logger.info("Step 9: Saving embeddings")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    logger.info(f"  Saved to {EMBEDDINGS_PATH}")

    # ------------------------------------------------------------------
    # Step 10: Three-way comparison
    # ------------------------------------------------------------------
    logger.info("Step 10: Running three-way model comparison")
    comparison_df = run_three_way_comparison(
        df_features=df_features,
        embeddings=embeddings,
        gnn_trainer=trainer,
    )

    # ------------------------------------------------------------------
    # Step 11: Save hybrid model
    # ------------------------------------------------------------------
    logger.info("Step 11: Saving hybrid model")
    from src.models.evaluate import TemporalEvaluator
    evaluator = TemporalEvaluator()

    if "commit_date" in df_features.columns:
        df_train_full, _ = evaluator.temporal_train_test_split(
            df_features, test_months=MODEL["test_months"]
        )
    else:
        from sklearn.model_selection import train_test_split as _tts
        df_train_full, _ = _tts(
            df_features, test_size=0.2, random_state=42,
            stratify=df_features["is_buggy"],
        )

    hybrid_model = HybridDefectModel(gnn_trainer=trainer)
    hybrid_model.train(df_train_full, embeddings)
    hybrid_model.save(HYBRID_MODEL_PATH)

    # ------------------------------------------------------------------
    # Step 12: Final summary
    # ------------------------------------------------------------------
    hybrid_row   = comparison_df[comparison_df["run_name"] == "hybrid_temporal_split"]
    baseline_row = comparison_df[comparison_df["run_name"] == "baseline_temporal_split"]

    hybrid_auc   = float(hybrid_row["AUC"].values[0])   if len(hybrid_row)   else 0.0
    hybrid_p20   = float(hybrid_row["Precision@20"].values[0]) if len(hybrid_row) else 0.0
    baseline_auc = float(baseline_row["AUC"].values[0]) if len(baseline_row) else 0.0
    baseline_p20 = float(baseline_row["Precision@20"].values[0]) if len(baseline_row) else 0.0

    day3_auc = 0.7926
    day3_p20 = 0.75

    print("\n" + "=" * 72)
    print("DAY 4 FINAL SUMMARY")
    print("=" * 72)
    print(f"GNN Architecture:   3-layer GCN  |  hidden=64  |  embed=32")
    print(f"Dataset:            {parseable} parseable files  |  {epochs} epochs")
    print(f"GNN coverage:       {n_real}/{len(df_features)} feature-matrix files "
          f"({n_real/len(df_features)*100:.1f}%)")
    print()
    print(f"{'Model':<28} {'AUC':>7}  {'P@20':>7}")
    print("-" * 46)
    print(f"{'Day 3 baseline':<28} {day3_auc:>7.4f}  {day3_p20:>7.2f}")
    print(f"{'Temporal baseline':<28} {baseline_auc:>7.4f}  {baseline_p20:>7.4f}")
    print(f"{'Hybrid (Day 4)':<28} {hybrid_auc:>7.4f}  {hybrid_p20:>7.4f}")
    print()
    print(f"Delta vs Day 3:     AUC {hybrid_auc - day3_auc:+.4f}   "
          f"P@20 {hybrid_p20 - day3_p20:+.4f}")
    print()
    print("Artifacts saved:")
    print(f"  {GNN_MODEL_PATH}")
    print(f"  {EMBEDDINGS_PATH}")
    print(f"  {HYBRID_MODEL_PATH}")
    print("=" * 72)
    print("\nGit commit command:")
    print(
        "git add src/models/gnn_model.py src/models/hybrid_model.py "
        "scripts/day4_run.py scripts/day5_mine_repos.py "
        "prep_for_training.py configs/config.py "
        "models/gnn_model.pt models/hybrid_model.pkl "
        "data/processed/gnn_embeddings.pkl "
        "data/processed/feature_matrix_final.csv && "
        f'git commit -m "Day 5: Multi-repo GNN+Hybrid — '
        f'AUC {hybrid_auc:.4f} P@20 {hybrid_p20:.4f} | '
        f'{len(df_features)} files 12 repos {n_real} GNN embeddings"'
    )
    print()


if __name__ == "__main__":
    main()