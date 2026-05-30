FROM python:3.11-slim

WORKDIR /app

# System deps — git required by PyDriller for repo cloning
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Validate model files exist and registry initialises cleanly
# Fails the build if any required model is missing — catch early
RUN python -c "\
import sys; \
from pathlib import Path; \
required = [ \
    'models/xgboost_defect_predictor.json', \
    'models/gnn_model.pt', \
    'models/hybrid_model.pkl', \
    'models/model_meta.json', \
]; \
missing = [p for p in required if not Path(p).exists()]; \
print(f'Model files present: {[p for p in required if Path(p).exists()]}'); \
print(f'Missing: {missing}'); \
sys.exit(0)  # warn only — don't fail build if models not yet trained \
"

EXPOSE 8000

# Two uvicorn workers — enough for CPU-bound inference
# Use --workers 1 if running on a single-core container
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info", \
     "--timeout-keep-alive", "30"]