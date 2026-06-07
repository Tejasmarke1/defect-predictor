# 🔍 Adaptive Defect Prediction Engine

> Predicts which files in a Python codebase are most likely to contain bugs using a hybrid architecture combining Graph Neural Networks on AST structure with XGBoost on process metrics — with full SHAP explainability and a production FastAPI backend.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-orange)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![PyTorch Geometric](https://img.shields.io/badge/GNN-PyTorch_Geometric-EE4C2C)](https://pytorch-geometric.readthedocs.io/)
[![Docker](https://img.shields.io/badge/deploy-Docker-2496ED)](https://www.docker.com/)

---

## 📊 Results

| Model | AUC-ROC | F1 | Precision@20 | Split |
|---|---|---|---|---|
| XGBoost baseline (random) | 0.7902 | 0.699 | 0.90 | Random 80/20 — inflated |
| XGBoost baseline (temporal) | 0.6812 | 0.194 | 0.35 | Temporal — honest |
| **GNN + XGBoost Hybrid** | **0.8590** | **0.323** | **0.45** | **Temporal — honest** |

**Dataset:** 5,463 files across 12 Python OSS repositories (Django, Flask, SQLAlchemy, Celery, aiohttp, Werkzeug, Requests, Tornado, Celery, Redis-py, PyMongo, Falcon)

**Key finding:** The hybrid GNN architecture delivers **+0.178 AUC lift** over the tabular baseline on a proper temporal split. The random-split baseline (0.79) is inflated by ~15% due to future data leakage — a critical mistake in most published defect prediction work that this project explicitly avoids.

---

## 🎯 What It Does

Given a GitHub repository URL, the system:

1. **Mines** git history to extract file-level change patterns using PyDriller + SZZ algorithm
2. **Labels** files as defect-prone using strict SZZ quality gates (keyword filter + commit-size cap + minimum bug-touch threshold)
3. **Extracts** 19 features across two families: process metrics (churn, author count, fix density, commit burst) and AST structural metrics (cyclomatic complexity, nesting depth, function count)
4. **Embeds** Python source code as AST graphs via a 3-layer GCNConv network producing 32-dimensional structural embeddings per file
5. **Predicts** using a 51-dimensional hybrid XGBoost (19 tabular + 32 GNN) with temporal validation
6. **Explains** every prediction with SHAP — top-5 feature contributions per file with direction labels
7. **Serves** results via a FastAPI REST API — POST a GitHub URL, get a risk-ranked file list with plain-English summaries

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: GitHub Repo URL                    │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │      Git Mining Pipeline     │
              │  PyDriller + Strict SZZ     │
              │  • Keyword filter           │
              │  • Max 15 files/commit      │
              │  • Min 2 bug touches        │
              └──────────────┬──────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                                       │
┌────────▼────────┐                   ┌──────────▼──────────┐
│ Process Metrics  │                   │   AST Graph Builder  │
│ 9 features:      │                   │  Python src → AST   │
│ • code_churn_90d │                   │  → PyG Data object  │
│ • fix_density    │                   │  Node features:     │
│ • author_count   │                   │  • 38-dim type OHE  │
│ • commit_burst   │                   │  • depth, children  │
│ • ownership_score│                   │  (40 dims total)    │
└────────┬────────┘                   └──────────┬──────────┘
         │                                       │
         │                             ┌──────────▼──────────┐
         │                             │    CodeGNN           │
         │                             │  GCNConv(40→64)     │
         │                             │  GCNConv(64→64)     │
         │                             │  GCNConv(64→32)     │
         │                             │  global_mean_pool   │
         │                             │  → 32-dim embedding │
         │                             └──────────┬──────────┘
         │                                        │
         └──────────────────┬─────────────────────┘
                 19 tabular + 32 GNN = 51 features
                            │
              ┌─────────────▼─────────────┐
              │   HybridDefectModel       │
              │   XGBoost on 51 dims      │
              │   Temporal train/test     │
              │   AUC 0.859               │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   SHAP Explainability     │
              │   Per-file top-5 SHAP    │
              │   Plain-English summary  │
              │   Embedding neighbors    │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   FastAPI REST API        │
              │   POST /analyze           │
              │   GET  /explain/{file}    │
              │   GET  /health            │
              │   GET  /experiments       │
              └───────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git (required by PyDriller for repo cloning)
- CUDA GPU optional — CPU works fine for inference

### Setup

```bash
git clone https://github.com/yourusername/defect-predictor
cd defect-predictor

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run the full training pipeline

```bash
# Day 1-2: Mine repos + extract features (uses pre-mined data if present)
python scripts/day5_mine_repos.py --repos flask django requests

# Prep feature matrix (imputation, leakage removal, date fixing)
python fix_dates_and_prep.py

# Day 3: XGBoost baseline
python scripts/day3_run.py

# Day 4-5: GNN + Hybrid model (takes ~40 min on CPU for 200 epochs)
python scripts/day4_run.py
```

### Start the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at `http://localhost:8000/docs`

### Quick prediction

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/pallets/flask",
    "since_days": 365,
    "top_k": 10,
    "use_hybrid": true
  }'
```

### Docker

```bash
# Build and run
docker compose up --build

# With MLflow UI
docker compose --profile mlflow up
# MLflow UI → http://localhost:5000
# API        → http://localhost:8000
```

### Smoke tests

```bash
# Starts server, runs all endpoint tests, prints timing summary
python scripts/day5_run.py
```

---

## 📁 Project Structure

```
defect-predictor/
├── configs/
│   └── config.py                     # All hyperparams and paths
├── data/
│   ├── raw/
│   │   ├── flask_commits.csv         # Mined commit history
│   │   └── repos/                    # Cloned repositories
│   └── processed/
│       ├── feature_matrix_final.csv  # 5,463 files, 19 features + dates
│       ├── labeled_combined.csv      # Raw labels from SZZ mining
│       ├── combined_source_codes.pkl # {file_path: source} for GNN
│       ├── gnn_embeddings.pkl        # {file_path: array(32)}
│       ├── oof_predictions.csv       # Out-of-fold XGBoost scores
│       └── shap_values.csv           # SHAP values per file
├── models/
│   ├── xgboost_defect_predictor.json # Day 3 baseline
│   ├── model_meta.json               # threshold, feature names
│   ├── gnn_model.pt                  # CodeGNN weights (200 epochs)
│   └── hybrid_model.pkl              # HybridDefectModel
├── src/
│   ├── mining/
│   │   ├── git_miner.py              # PyDriller pipeline
│   │   ├── szz_labeler.py            # Strict SZZ implementation
│   │   └── promise_loader.py         # PROMISE KC1 benchmark
│   ├── features/
│   │   ├── process_features.py       # Churn, authors, fix density
│   │   ├── ast_features.py           # Cyclomatic, nesting, imports
│   │   └── feature_pipeline.py       # Combines all features
│   ├── models/
│   │   ├── train.py                  # DefectXGBoost + CV pipeline
│   │   ├── evaluate.py               # TemporalEvaluator + metrics
│   │   ├── gnn_model.py              # CodeGNN + GNNTrainer
│   │   └── hybrid_model.py           # HybridDefectModel + comparison
│   ├── explainability/
│   │   └── shap_explainer.py         # SHAP analysis
│   └── api/
│       ├── main.py                   # FastAPI app + lifespan
│       ├── models.py                 # Pydantic v2 schemas
│       ├── dependencies.py           # ModelRegistry singleton + JobStore
│       └── routers/
│           ├── analyze.py            # POST /analyze
│           ├── explain.py            # GET  /explain/{job_id}/{file}
│           ├── health.py             # GET  /health
│           └── experiments.py        # GET  /experiments
├── notebooks/
│   ├── day1_eda.ipynb                # Git history EDA
│   ├── day3_models.ipynb             # XGBoost baseline analysis
│   ├── day4_gnn.ipynb                # GNN training + embeddings
│   └── day5_api_test.ipynb           # Live API end-to-end tests
├── scripts/
│   ├── day3_run.py                   # XGBoost training pipeline
│   ├── day4_run.py                   # GNN + Hybrid training
│   ├── day5_mine_repos.py            # Multi-repo mining
│   └── day5_run.py                   # API smoke tests
├── fix_dates_and_prep.py             # One-shot data prep + date fix
├── prep_for_training.py              # Feature cleaning + imputation
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🔌 API Reference

All endpoints return JSON. Server runs on `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

### `POST /analyze`

Clone a repository and predict defect risk for all Python files.

**Request:**
```json
{
  "repo_url": "https://github.com/pallets/flask",
  "since_days": 365,
  "top_k": 20,
  "use_hybrid": true
}
```

**Response (truncated):**
```json
{
  "job_id": "flask_a3f92c1b",
  "status": "completed",
  "model_used": "hybrid",
  "total_files_analyzed": 77,
  "buggy_files_predicted": 31,
  "analysis_time_ms": 45320,
  "mining_time_ms": 38100,
  "feature_time_ms": 4200,
  "prediction_time_ms": 1020,
  "model_auc": 0.859,
  "top_k_results": [
    {
      "file_path": "src/flask/app.py",
      "risk_score": 0.847,
      "risk_label": "HIGH",
      "rank": 1,
      "top_shap_features": [
        {
          "feature_name": "fix_density",
          "shap_value": 0.312,
          "feature_value": 0.42,
          "direction": "increases_risk"
        }
      ],
      "lines_of_code": 1843,
      "cyclomatic_complexity": 47.0,
      "last_modified_days_ago": 12
    }
  ]
}
```

### `GET /explain/{job_id}/{file_path}`

Deep SHAP explanation for a specific file from a completed analysis.

**Response:**
```json
{
  "file_path": "src/flask/app.py",
  "risk_score": 0.847,
  "risk_label": "HIGH",
  "plain_english_summary": "This file has HIGH defect risk (score: 0.85). The main risk factors are: high fix_density (0.42, SHAP=+0.312); high code_churn_90d (1823.00, SHAP=+0.198); high ast_cyclomatic_complexity (47.00, SHAP=+0.143).",
  "similar_files": ["src/flask/blueprints.py", "src/flask/wrappers.py"],
  "embedding_neighbors": ["src/flask/testing.py"],
  "shap_waterfall": [...]
}
```

### `GET /health`

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": {"xgboost": true, "gnn": true, "hybrid": true, "shap": true},
  "total_analyses_run": 14,
  "uptime_seconds": 3847.2,
  "mlflow_connected": true
}
```

### `GET /experiments`

Returns all MLflow runs from `defect-prediction` and `model-comparison` experiments, sorted by AUC.

---

## 🧠 Key Technical Decisions

### Why strict SZZ quality gates?
Naive SZZ labeled 78% of files as buggy — nearly impossible to learn from. Three fixes reduced this to 41%: (1) removing overly broad keywords like "handle", "resolve", "correct"; (2) skipping commits that touch >15 files (refactors, not targeted fixes); (3) requiring a file to be touched by ≥2 bug-fix commits before being labeled buggy. Without these gates the model learned to predict "buggy" for almost everything.

### Why temporal validation instead of random split?
Random splits leak future information into training. A file modified in 2024 might appear in both the training set (in one time window) and the test set (in another), letting the model see patterns it shouldn't have access to at prediction time. Temporal splits respect the deployment reality: you train on history and predict on future changes. The gap between random AUC (0.79) and temporal AUC (0.68 baseline) quantifies exactly how much this matters.

### Why GCNConv and not GAT or GraphSAGE?
ASTs are deterministic tree-structured graphs without meaningful edge weights. GAT's attention mechanism adds no value when all edges are syntactic parent-child relationships — there's no ambiguity about which neighbors matter. GCNConv's spectral aggregation (normalised sum of neighbour features) is exactly right for propagating structural information up and down a syntax tree.

### Why XGBoost for the final classifier instead of an MLP?
Process metrics are tabular, low-dimensional, and mixed-scale. Tree methods handle this natively without feature normalisation or architecture choices. The GNN handles structural features; XGBoost handles everything else. The hybrid beats either component alone (+0.178 AUC over tabular baseline).

### Why SHAP over built-in feature importance?
XGBoost feature importance is global and unsigned — it tells you which features matter on average but not *why this specific file* got a high score. SHAP is local, signed, and consistent: it tells you exactly which features pushed *this file's* score up or down, and by how much. This is the difference between a tool developers ignore and one they actually trust.

---

## 📈 MLflow Experiment Tracking

All training runs are tracked with MLflow. View the UI:

```bash
mlflow ui --backend-store-uri ./mlruns
# Open http://localhost:5000
```

Tracked per run: AUC, F1, Precision@K, n_features, threshold, train/test split type, GNN epochs and loss.

---

## 🔬 Dataset Details

| Repo | Files | Buggy rate | Notes |
|---|---|---|---|
| django | 1,778 | 35.0% | Capped from 71.6% — "Fixed #XXXXX" on features |
| sqlalchemy | 630 | 39.9% | Capped from 96.8% — src_path mismatch |
| celery | 254 | 44.9% | Complex async codebase |
| pymongo | 252 | 48.8% | |
| werkzeug | 222 | 44.8% | |
| redis-py | 192 | 44.7% | |
| falcon | 170 | 44.9% | |
| aiohttp | 115 | 44.8% | |
| tornado | 91 | 44.4% | |
| flask | 77 | 42.9% | Primary dev/test repo |
| click | 71 | 44.7% | |
| requests | 42 | 31.0% | Smallest, cleanest |

**Temporal split:** Train on commits before May 2025, test on May 2025+.
Train: 4,763 files | Test: 700 files

---

## 🙏 References

- SZZ Algorithm: Śliwerski, Zimmermann & Zeller (2005). *When Do Changes Induce Fixes?*
- PROMISE Repository: http://promise.site.uottawa.ca/SERepository/
- PyDriller: Spadini et al. (2018). *PyDriller: Python Framework for Mining Software Repositories*
- PyTorch Geometric: Fey & Lenssen (2019). *Fast Graph Representation Learning with PyTorch Geometric*
- XGBoost: Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System*
- SHAP: Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions*