"""
Central configuration management.
All magic numbers and settings live here.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Create dirs if they don't exist
for d in [RAW_DIR, PROCESSED_DIR, DATASETS_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Mining Config ──────────────────────────────────────────────────────────────
MINING = {
    # Keywords to identify bug-fix commits (SZZ algorithm)
    "bug_keywords": [
        "fix", "bug", "defect", "error", "fault", "issue",
        "patch", "correct", "resolve", "close", "closes",
        "closed", "hotfix", "regression", "revert"
    ],
    # Only analyze Python files
    "file_extensions": [".py"],
    # Ignore test files (they have different bug patterns)
    "ignore_paths": ["test_", "_test", "tests/", "test/", "conftest"],
    # Max file size to analyze (bytes)
    "max_file_size": 100_000,
}

# ── Feature Engineering Config ─────────────────────────────────────────────────
FEATURES = {
    # Rolling window for process metrics (days)
    "short_window_days": 30,
    "long_window_days": 90,
    # Minimum commits for a file to be included
    "min_commits": 2,
    # GNN node feature dimension
    "ast_node_types": [
        "Module", "FunctionDef", "AsyncFunctionDef", "ClassDef",
        "Return", "Delete", "Assign", "AugAssign", "AnnAssign",
        "For", "AsyncFor", "While", "If", "With", "AsyncWith",
        "Raise", "Try", "Import", "ImportFrom", "Call", "Compare",
        "BoolOp", "BinOp", "UnaryOp", "Lambda", "IfExp",
        "Attribute", "Subscript", "Name", "Constant", "List",
        "Tuple", "Dict", "Set", "comprehension"
    ],
}

# ── Model Config ───────────────────────────────────────────────────────────────
MODEL = {
    # Temporal split - test on last N months
    "test_months": 12,
    # XGBoost defaults (will be tuned by Optuna)
    "xgb_defaults": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_splits": 5,
        "min_child_weight": 5,       # NEW — forces confident splits only
        "gamma": 1.0,
        "early_stopping_rounds": 30,
    },
    # GNN config
    "gnn": {
        "hidden_dim": 64,
        "embedding_dim": 32,
        "num_layers": 3,
        "dropout": 0.3,
        "epochs": 200,
        "lr": 0.001,
        "batch_size": 32,
    },
    # Optuna tuning
    "optuna_trials": 50,
    "optuna_timeout": 300,  # seconds
}

# ── MLflow Config ──────────────────────────────────────────────────────────────
MLFLOW = {
    "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", str(ROOT_DIR / "mlruns")),
    "experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME", "defect-prediction"),
}

# ── API Config ─────────────────────────────────────────────────────────────────
API = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", 8000)),
    "max_repo_size_mb": int(os.getenv("MAX_REPO_SIZE_MB", 500)),
    "timeout_seconds": int(os.getenv("ANALYSIS_TIMEOUT_SECONDS", 300)),
}

# ── GitHub Config ──────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", None)