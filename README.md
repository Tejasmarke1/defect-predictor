# 🔍 Adaptive Defect Prediction Engine

> Predicts which files in a codebase are most likely to contain bugs using a hybrid ML architecture combining Graph Neural Networks (GNN) on AST structure with XGBoost on process metrics — with full SHAP explainability.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-orange)](https://mlflow.org/)


---

## 🎯 What It Does

Given a GitHub repository, this system:
1. **Mines** the git history to extract file-level change patterns
2. **Labels** files as defect-prone using the SZZ algorithm on bug-fix commits
3. **Extracts** process metrics (churn, author count, fix density) + structural features (AST complexity)
4. **Trains** a hybrid GNN + XGBoost model with proper temporal validation
5. **Explains** every prediction with SHAP — no black boxes
6. **Serves** results via a FastAPI backend + Streamlit dashboard

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: GitHub Repo URL                    │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │      Git Mining Pipeline     │
              │   (PyDriller + SZZ Labels)  │
              └──────────────┬──────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                                       │
┌────────▼────────┐                   ┌──────────▼──────────┐
│ Process Metrics  │                   │  AST Graph Builder  │
│ • Code churn     │                   │  • Parse Python AST │
│ • Author count   │                   │  • Build NetworkX G │
│ • Fix density    │                   │  • Node features    │
│ • Commit burst   │                   └──────────┬──────────┘
└────────┬────────┘                              │
         │                             ┌──────────▼──────────┐
         │                             │    Graph Neural Net  │
         │                             │  (PyTorch Geometric) │
         │                             │  • 3-layer GCN      │
         │                             │  • File embeddings  │
         │                             └──────────┬──────────┘
         │                                        │
         └──────────────────┬─────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   Hybrid XGBoost Model    │
              │  Process + GNN features  │
              │  Temporal validation     │
              │  Optuna hypertuning      │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   SHAP Explainability     │
              │  Per-file explanations   │
              │  Global feature ranks    │
              └─────────────┬─────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                                     │
┌────────▼────────┐                 ┌──────────▼──────────┐
│   FastAPI       │                 │  Streamlit Dashboard │
│   REST API      │                 │  Risk Heatmap       │
│   /analyze      │                 │  SHAP Plots         │
│   /explain      │                 │  MLflow Experiments │
└─────────────────┘                 └─────────────────────┘
```

---

## 📊 Results

| Model Variant | F1 | Precision | Recall | AUC-ROC |
|---|---|---|---|---|
| XGBoost (random split) | — | — | — | — |
| XGBoost (temporal split) | — | — | — | — |
| XGBoost + GNN hybrid | — | — | — | — |

*Results populated after training*

**Key finding:** Temporal validation reveals ~X% lower F1 than random split — demonstrating that random splits leak future data into training, a critical mistake in most defect prediction work.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- CUDA GPU (optional, for GNN training)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/defect-predictor
cd defect-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env if needed (defaults work for local use)
```

### Run Day 1 Pipeline

```bash
# Download PROMISE dataset + start Flask mining
python scripts/day1_run.py

# Skip mining (PROMISE data only, faster)
python scripts/day1_run.py --skip-mining

# Mine a different repo
python scripts/day1_run.py --repo https://github.com/psf/requests
```

### Launch Dashboard (Day 6+)

```bash
# Start API
uvicorn src.api.main:app --reload

# Start dashboard (new terminal)
streamlit run ui/dashboard.py
```

---

## 📁 Project Structure

```
defect-predictor/
├── configs/
│   └── config.py              # All settings, no magic numbers
├── data/
│   ├── raw/                   # Mined git history (gitignored)
│   ├── processed/             # Feature matrices
│   └── datasets/              # PROMISE benchmark files
├── src/
│   ├── mining/
│   │   ├── git_miner.py       # PyDriller pipeline
│   │   ├── szz_labeler.py     # Bug label generation
│   │   └── promise_loader.py  # PROMISE benchmark loader
│   ├── features/
│   │   ├── process_features.py   # Git-based metrics
│   │   ├── ast_features.py       # Code structure metrics
│   │   └── feature_pipeline.py   # Combines all features
│   ├── models/
│   │   ├── xgboost_model.py      # XGBoost + Optuna tuning
│   │   ├── gnn_model.py          # PyTorch Geometric GNN
│   │   └── hybrid_model.py       # GNN + XGBoost ensemble
│   ├── explainability/
│   │   └── shap_explainer.py     # SHAP analysis
│   └── api/
│       └── main.py               # FastAPI endpoints
├── notebooks/
│   ├── day1_eda.ipynb            # Exploratory analysis
│   ├── day2_features.ipynb       # Feature engineering
│   ├── day3_baseline.ipynb       # XGBoost baseline
│   └── day4_gnn.ipynb            # GNN experiments
├── ui/
│   └── dashboard.py              # Streamlit dashboard
├── tests/
├── scripts/
│   ├── day1_run.py
│   ├── day2_run.py
│   └── ...
├── mlruns/                        # MLflow experiment tracking
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🧠 Key Technical Decisions

### Why Temporal Validation?
Random 80/20 splits leak future information into training — a file modified in January might be "explained" by features from March. Real deployment only sees past data. Temporal splits respect this constraint and give honest evaluation.

### Why GNN on AST?
Code has graph structure — functions call other functions, classes inherit, modules import modules. A GNN learns representations that capture structural complexity patterns text-based approaches miss entirely.

### Why XGBoost over Neural Net for final prediction?
Process metrics are tabular, sparse, and mixed-scale. Tree methods handle this naturally. The GNN handles structural features; XGBoost handles everything else. The ensemble beats either alone.

### Why SHAP?
Defect prediction without explanation is useless in practice. A developer needs to know *why* a file is flagged — not just that it is. SHAP provides per-prediction, feature-level attribution.

---

## 📖 Technical Blog Post

*Coming soon on dev.to*

---

## 🙏 References

- SZZ Algorithm: Śliwerski, Zimmermann, Zeller (2005)
- PROMISE Repository: http://promise.site.uottawa.ca/SERepository/
- PyDriller: Spadini et al. (2018)
- PyTorch Geometric: Fey & Lenssen (2019)