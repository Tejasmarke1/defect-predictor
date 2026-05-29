import pickle
import sys
import uuid
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

import numpy as np
import pandas as pd
import uvicorn
import mlflow
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Ensure project root is in the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import MODELS_DIR, PROCESSED_DIR, MLFLOW
from src.models.hybrid_model import HybridDefectModel

class AnalyzeRequest(BaseModel):
    repo_url: str

class AnalyzeResponse(BaseModel):
    job_id: str
    message: str

class PredictResponse(BaseModel):
    file_path: str
    is_buggy_probability: float
    is_buggy_prediction: bool

class ReportResponse(BaseModel):
    job_id: str
    status: str
    predictions: Optional[List[PredictResponse]] = None

class ExplainResponse(BaseModel):
    file_path: str
    explanation: Dict[str, float]

class HybridAPIState:
    model: HybridDefectModel | None = None
    embeddings: Dict[str, np.ndarray] = {}
    shap_values: pd.DataFrame | None = None
    jobs: Dict[str, Dict[str, Any]] = {}

state = HybridAPIState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup - load models, embeddings, and SHAP values
    model_path = MODELS_DIR / "hybrid_model.pkl"
    embeddings_path = PROCESSED_DIR / "gnn_embeddings.pkl"
    shap_path = PROCESSED_DIR / "shap_values.csv"
    
    try:
        if model_path.exists():
            # Initialize with dummy args, load will overwrite internal state
            model = HybridDefectModel(gnn_trainer=None) # type: ignore
            model.load(model_path)
            state.model = model
            print(f"Loaded HybridDefectModel from {model_path}")
            
        if embeddings_path.exists():
            with open(embeddings_path, "rb") as f:
                state.embeddings = pickle.load(f)
            print(f"Loaded {len(state.embeddings)} embeddings from {embeddings_path}")
            
        if shap_path.exists():
            state.shap_values = pd.read_csv(shap_path)
            print(f"Loaded SHAP values from {shap_path}")
    except Exception as e:
        print(f"Warning: Failed to load model, embeddings or SHAP values: {e}")
        
    yield
    # Cleanup
    state.model = None
    state.embeddings.clear()
    state.jobs.clear()


app = FastAPI(
    title="Hybrid Defect Predictor API",
    description="API for predicting software defects and reviewing analysis using the Hybrid model and XGBoost.",
    version="1.0.0",
    lifespan=lifespan,
)

async def _run_analysis_pipeline(job_id: str, repo_url: str):
    """Mock analysis pipeline running in background."""
    state.jobs[job_id]["status"] = "processing"
    
    # Simulate processing time
    await asyncio.sleep(2)
    
    # In a real app we'd clone the repo, extract features, and run prediction using state.model.
    # Here we'll mock some predictions to demonstrate the endpoint structure.
    
    predictions = [
        PredictResponse(
            file_path="src/main.py",
            is_buggy_probability=0.85,
            is_buggy_prediction=True
        ),
        PredictResponse(
            file_path="src/utils.py",
            is_buggy_probability=0.12,
            is_buggy_prediction=False
        )
    ]
    
    state.jobs[job_id]["status"] = "completed"
    state.jobs[job_id]["predictions"] = predictions


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_repo(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Takes repo URL, returns predictions."""
    job_id = str(uuid.uuid4())
    state.jobs[job_id] = {"status": "pending", "repo_url": req.repo_url}
    
    background_tasks.add_task(_run_analysis_pipeline, job_id, req.repo_url)
    
    return AnalyzeResponse(job_id=job_id, message="Analysis job started.")

@app.get("/report/{job_id}", response_model=ReportResponse)
def get_report(job_id: str):
    """Get analysis results for a given job ID."""
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job ID not found.")
        
    job = state.jobs[job_id]
    return ReportResponse(
        job_id=job_id,
        status=job["status"],
        predictions=job.get("predictions")
    )

@app.get("/explain/{file:path}", response_model=ExplainResponse)
def explain_file(file: str):
    """SHAP explanation for specific file."""
    # Handle URL encoding if any
    file_path = unquote(file)
    
    if state.shap_values is None:
        raise HTTPException(status_code=503, detail="SHAP values are not loaded on the server.")
        
    # Find row by file_path
    if "file_path" in state.shap_values.columns:
        row = state.shap_values[state.shap_values["file_path"] == file_path]
    elif file_path in state.shap_values.index:
        row = state.shap_values.loc[[file_path]]
    else:
        row = pd.DataFrame()

    if row.empty:
        raise HTTPException(status_code=404, detail=f"No SHAP explanation found for file: {file_path}")
        
    row_dict = row.drop(columns=[c for c in ["file_path", "is_buggy", "label"] if c in row.columns]).iloc[0].to_dict()
    sorted_impact = dict(sorted(row_dict.items(), key=lambda kv: abs(kv[1]), reverse=True))
        
    return ExplainResponse(
        file_path=file_path,
        explanation=sorted_impact
    )

@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Service status."""
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "embeddings_loaded": len(state.embeddings) > 0,
        "shap_loaded": state.shap_values is not None,
    }


@app.get("/experiments")
def list_experiments() -> Dict[str, Any]:
    """MLflow experiment list."""
    try:
        if MLFLOW.get("tracking_uri"):
            mlflow.set_tracking_uri(MLFLOW["tracking_uri"])
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        
        return {
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage,
                }
                for exp in experiments
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch MLflow experiments: {e}")

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
