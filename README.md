# ForesightX-pattern

This repository contains the ML and data-science pipeline for the pattern/prediction service.

## What Exists Today

- Data ingestion, preprocessing, feature engineering, training, evaluation, and model-registry code under `src/`
- File-based model artifacts under `models/`, `metadata/`, `data/features/`, and `results/`
- A FastAPI inference layer under `src/inference/fastapi.py`

## Serving Boundary

The serving layer is intentionally separated from the ML pipeline:

- It does not retrain models.
- It does not rewrite the feature pipeline.
- It consumes already-generated artifacts and exposes them through API endpoints.

That keeps the microservice stable while preserving the data-science workflow.

## Run The API

Install dependencies and start the service:

```bash
pip install -r requirements.txt
uvicorn src.inference.fastapi:app --reload --host 0.0.0.0 --port 8003
```

## Available Endpoints

- `GET /health/live`
- `GET /health/ready`
- `GET /models`
- `GET /models/{symbol}`
- `GET /predictions/{symbol}/latest`
- `POST /predictions/latest`
- `GET /predictions/{symbol}/history`

## Current Limitation

Predictions are only available for symbols that already have all three artifacts:

- `models/mlp_model_<SYMBOL>.pkl`
- `models/mlp_scaler_<SYMBOL>.pkl`
- `metadata/mlp_model_stats_<SYMBOL>.json`

If a symbol has no trained artifacts yet, the API returns a clear error instead of attempting ad-hoc training.
