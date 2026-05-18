# ForesightX Pattern

ForesightX Pattern is a production-oriented ML microservice split into two layers:

1. Offline ML system for ingestion, preprocessing, feature engineering, sequence building, training, evaluation, MLflow tracking, and DVC orchestration.
2. Online inference service for FastAPI prediction, cached model loading, latest-bar feature building, and Monte Carlo dropout confidence estimation.

## Architecture

```text
foresightx_pattern/
├── foresightx_pattern/
│   ├── app/                  # FastAPI inference layer
│   └── ml/                   # Offline ML layer
├── configs/                 # Runtime and training config
├── artifacts/               # Data, reports, trained model bundle
├── tests/                   # Unit and API tests
├── dvc.yaml
├── mlflow_config.py
├── Dockerfile
└── docker-compose.yml
```

## Model Design

- Global foundation model trained across multiple tickers.
- Sequence encoder: linear projection + LSTM/GRU.
- Stock adaptation: offline-trained `nn.Embedding(num_stocks, embedding_dim)`.
- Fusion: encoder latent + stock embedding.
- Head: dense layers -> next 3 hourly close predictions.
- Runtime confidence: ONNX prediction plus metric-calibrated intervals from the training evaluation report.

## Feature Contract

The offline and online systems share the same feature contract:

- OHLCV
- returns
- RSI
- EMA fast/slow
- MACD, signal, histogram
- rolling mean/std
- time features: hour, day-of-week, sinusoidal encodings

Only past bars are used. Targets are the next 3 trading-hour closes. Trading hours are filtered explicitly; the service does not assume a 24h market.

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

For a training-only environment without API/database extras:

```bash
pip install -r requirements.train.txt
```

Run the DVC pipeline:

```bash
dvc repro
```

Or run training directly:

```bash
python3 -m foresightx_pattern.ml.training.train
```

To convert an existing trained `model.pt` bundle without retraining:

```bash
python3 -m foresightx_pattern.ml.training.export_onnx --model-dir artifacts/model
```

Artifacts written locally:

- `artifacts/data/raw_market.parquet`
- `artifacts/data/processed_market.parquet`
- `artifacts/data/feature_store.parquet`
- `artifacts/model/model.pt`
- `artifacts/model/model.onnx`
- `artifacts/model/scaler.pkl`
- `artifacts/model/metadata.json`
- `artifacts/reports/evaluation.json`

MLflow:

- `mlflow_config.py` configures tracking from `configs/default.yaml`
- training logs params, metrics, and artifacts
- best model is registered and transitioned to `Production` when registry support is available

## Inference

Start the API:

```bash
dvc pull artifacts/model
uvicorn foresightx_pattern.app.main:app --reload --host 0.0.0.0 --port 8003
```

Endpoint:

```http
POST /predict
Content-Type: application/json

{
  "ticker": "RELIANCE.NS",
  "timestamp": "2026-04-19T10:00:00+05:30"
}
```

Response:

```json
{
  "ticker": "TATAMOTORS.NS",
  "predictions": [624.1, 626.4, 628.0],
  "confidence": 0.91,
  "intervals": [[620.0, 628.0], [622.2, 630.6], [623.5, 632.5]],
  "model_version": "7"
}
```

Inference flow:

1. Fetch latest hourly bars for the requested ticker.
2. Rebuild the shared feature set using only past data.
3. Load the cached model bundle once from `artifacts/model` or `FORESIGHTX_ARTIFACTS_DIR`.
4. Map `ticker -> stock_id`.
5. Run prediction.
6. Build confidence intervals from stored evaluation metrics.
7. Return predictions, confidence, intervals, and model version.

## Deployment Scope

This repository has two roles:

- offline data-science work: data preparation, training, evaluation, notebooks, MLflow/DVC artifacts, and reports
- online inference: the FastAPI service that exposes health and prediction endpoints

The production Docker image is intentionally inference-only. It installs `requirements.inference.txt` and copies only:

- `foresightx_pattern/app`
- the runtime ML modules required by feature building and model execution
- `configs/default.yaml`
- the latest model bundle from `artifacts/model` (`model.onnx`, `scaler.pkl`, `metadata.json`)

Training code, PyTorch model definitions, notebooks, papers, raw/processed data, MLflow runs, and reports are excluded from the runtime image to keep EC2 resource usage low.

## Tests

```bash
pytest tests/test_model.py tests/test_features.py tests/test_api.py
```

Test coverage includes:

- feature pipeline output
- model forward pass shape
- `/predict` API contract

## Deployment

Build the container:

```bash
docker build -f ForesightX-Pattern/Dockerfile -t foresightx-pattern ..
```

Run with Redis sidecar:

```bash
docker compose -f docker-compose.yml up --build
```

Redis is included for future cache extension; current in-process caching is implemented in `FeatureService`.
