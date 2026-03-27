# ForesightX - About

**"The Future Has Signals. We Decode Them."**

## Project Goal

Build a production-ready system that ingests market & alternative data, trains and backtests a suite of time-series and ML models, serves predictions (point + interval) via REST API, automatically retrains when needed, monitors data/model health, and provides interactive explanations & dashboards — while clearly labeling it as **not financial advice**.

---

## High-Level Architecture

```
Frontend (nothingamazing.com UI)
    ↓
API Gateway (FastAPI) → Serves predictions + dashboards
    ↓
Prediction Microservice → Model server with saved artifacts
    ↓
Data Ingestion & Feature Service → Stores raw data & engineered features
    ↓
Training Orchestration → Airflow / Kubeflow / Prefect
    ↓
Experiment Tracking → MLflow
    ↓
Data Versioning → DVC + Artifact Store (S3)
    ↓
Model Registry → MLflow / KServe
    ↓
Deployment → Kubernetes + Helm
    ↓
Monitoring & Alerting → Prometheus + Grafana + Evidently
    ↓
Logging → ELK Stack / Loki

Optional: Streaming layer with Kafka for live market ticks
```

---

## Components & Tech Stack

### Core Technology
- **Language:** Python 3.10+

### Data Sources
- **Market Data:** Yahoo Finance (yfinance), Alpha Vantage, IEX Cloud, Tiingo, Quandl
- **Alternative Data:** Twitter/Reddit sentiment, news headlines (newsapi), Google Trends, macro indicators (FRED)
- **Advanced:** Optional paid tick feeds for high-frequency data

### Data Infrastructure
- **Storage:** AWS S3 / MinIO (local), PostgreSQL for metadata, Parquet for time-series
- **Feature Store:** Feast or Redis cache for online features
- **Versioning:** DVC or LakeFS

### ML Pipeline
- **Orchestration:** Apache Airflow, Kubeflow Pipelines, or Prefect
- **Experiment Tracking:** MLflow
- **Model Registry:** MLflow Model Registry

### Serving & Deployment
- **API Server:** FastAPI + Uvicorn (containerized)
- **Model Serving:** KServe / TorchServe (optional)
- **Container Orchestration:** Kubernetes with Helm charts
- **CI/CD:** GitHub Actions or GitLab CI

### Monitoring & Observability
- **Metrics:** Prometheus + Grafana
- **Drift Detection:** Evidently (data & model drift)
- **Error Tracking:** Sentry
- **Explainability:** SHAP, Captum (PyTorch), LIME

### Backtesting
- **Frameworks:** backtrader, vectorbt, or custom walk-forward engine

### Optional Components
- **Streaming:** Apache Kafka for real-time data
- **Caching:** Redis for features and signals
- **Graph DB:** RedisGraph for signal storage

---

## Data Sources & Ingestion

### Data Layers

1. **Historical OHLCV Data**
   - Daily/minute bars: Yahoo Finance (yfinance), Alpha Vantage (rate-limited)
   - Premium sources: IEX Cloud (paid), Tiingo (paid)

2. **Fundamental Data**
   - Financial statements via Tiingo, Alpha Vantage

3. **Alternative Data**
   - Social sentiment: Twitter/Reddit via API
   - News: newsapi for headlines
   - Trends: Google Trends
   - Macro indicators: FRED API
   - Options: Implied volatility (if available)

4. **Market Microstructure** *(Optional)*
   - High-frequency data from paid feeds

### Ingestion Pattern

```python
import yfinance as yf

def fetch_market_data(symbol, start, end, interval='1d'):
    """
    Fetch market data and store as Parquet
    """
    data = yf.download(symbol, start=start, end=end, interval=interval)
    data.to_parquet(f's3://bucket/raw/{symbol}/{interval}/{start}_{end}.parquet')
    
    # Log metadata: source, query params, timestamp, rate limits
    return data
```

**Best Practices:**
- Use incremental ingestion with date partitioning
- Store provenance metadata (source, timestamp, query params)
- Log rate-limit responses and errors
- Write to Parquet format for efficient storage

---

## Feature Engineering

### Offline + Online Feature Pipeline

### Offline + Online Feature Pipeline

#### Core Features

1. **Lag Features**
   - `price_{t-1..t-n}`, `return_{t-1..t-n}`

2. **Technical Indicators**
   - Moving Averages: SMA, EMA
   - Momentum: RSI, MACD
   - Volatility: Bollinger Bands, ATR

3. **Volatility Features**
   - Rolling standard deviation
   - Realized volatility

4. **Volume Features**
   - Volume change
   - VWAP (Volume-Weighted Average Price)

5. **Calendar Features**
   - Day of week, month
   - Holiday flags

6. **Fundamental Ratios**
   - P/E, P/B, EPS growth

7. **Sentiment Scores**
   - Headline sentiment
   - Social media sentiment aggregates

8. **Macro Indicators**
   - Lagged values from economic data

### Feature Computation & Storage

- **Batch Processing:** Use Spark/Polars for large-scale feature computation
- **Storage:** Feature store (Feast) or Parquet files
- **Online Inference:** Lightweight feature service computes recent features in real-time
- **Scaling:** Store preprocessed features with versioning for reproducibility

---

## Model Suite

### Ensemble Approach

Build multiple models and ensemble them. Save all checkpoints to registry.

#### 1. Baseline Classical Models

#### 1. Baseline Classical Models

- **ARIMA / SARIMAX:** Statistical baseline for time-series
- **Prophet:** Facebook's model for seasonality baseline

#### 2. Machine Learning Models

- **Gradient Boosting:** XGBoost/LightGBM on engineered features (strong baseline)
- **MLP:** Multi-layer Perceptron on windowed features

#### 3. Deep Time-Series Models

- **LSTM / GRU:** Sequence-to-sequence forecasting
- **Temporal Fusion Transformer (TFT):** Excellent for interpretability & multivariate forecasting
- **Transformers:** Informer/Autoformer for longer horizons

#### 4. Probabilistic / Interval Predictions

- **Quantile Regression:** LightGBM with quantile loss
- **Bayesian Models:** MC-dropout for uncertainty intervals

#### 5. Ensemble Strategy

- **Weighted Blend:** Based on past validation performance
- **Stacking:** Meta-learner combines predictions from base models

---

## Backtesting & Evaluation

### Methodology

**Time-Series Walk-Forward Validation:**
- Train on `[t0..tN]`, validate on next window, roll forward
- Prevents data leakage with strict chronological splits
- Keep separate test set for final evaluation (most recent unseen period)

### Evaluation Metrics

#### Forecast Accuracy
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

#### Directional Accuracy
- Sign accuracy (up/down predictions)
- Precision/Recall for directional signals

#### Economic Metrics (Paper Trading)
- **Sharpe Ratio:** Risk-adjusted returns
- **Max Drawdown:** Largest peak-to-trough decline
- **Cumulative Returns:** Account for transaction costs & slippage

#### Probabilistic Forecast Calibration
- CRPS (Continuous Ranked Probability Score)
- Coverage analysis for predicted intervals

**Important:** Always include transaction costs, slippage simulation, and realistic constraints in backtests.

---

## Experiment Tracking & Model Governance

### MLflow Integration

- **Track:** Parameters, metrics, artifacts (scalers, feature preprocessing)
- **Registry:** Store optimal models with stages (Staging, Production)
- **Versioning:** Record dataset version (DVC commit or dataset hash) for traceability
- **Model Card:** Maintain documentation for each release
  - Intended use & limitations
  - Data used & evaluation metrics
  - Model architecture & performance

### Best Practices

- Log model signatures for input/output validation
- Track experiment lineage and reproducibility
- Implement model approval workflows
- Maintain audit logs for production deployments

---

## Training & CI/CD Pipeline

### Pipeline Stages

1. **Data Ingestion Job** (Daily scheduled)
2. **Feature Computation Job** (Batch processing)
3. **Training Job** (Scheduled or triggered)
4. **Backtesting & Evaluation Job**
5. **Model Registration** (If passing thresholds → staging)
6. **Canary Deployment** (Small traffic % → monitor → promote/rollback)

### GitHub Actions Example

```yaml
name: Train Model
on:
  workflow_dispatch:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run training pipeline
        env:
          S3_BUCKET: ${{ secrets.S3_BUCKET }}
        run: |
          python src/model/train_model.py --config configs/train.yaml
      
      - name: Upload model artifacts
      - name: Upload model artifacts
        run: aws s3 cp models/latest s3://$S3_BUCKET/models/ --recursive
```

**Quality Gates:**
- Run linting and unit tests before training
- Implement gating conditions for model promotion
- Require passing performance thresholds

---

## Model Serving (FastAPI)

### Production API Server

```python
from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

app = FastAPI()
model = mlflow.pyfunc.load_model("models:/my_stock_model/Production")

@app.post("/predict")
def predict(payload: dict):
    """
    Make prediction for given features
    payload: {symbol: 'AAPL', features: {...}}
    """
    df = pd.DataFrame([payload['features']])
    preds = model.predict(df)
    
    return {
        "prediction": preds.tolist(),
        "symbol": payload['symbol']
    }
```

### Deployment Strategy

- **Containerization:** Docker with optimized Dockerfile
- **Load Balancing:** Deploy behind Kubernetes Service + Ingress (NGINX)
- **Auto-scaling:** Horizontal Pod Autoscaler (HPA)
- **Model Management:** Use KServe for simplified model serving

### API Response Format

Return comprehensive prediction data:
- Point forecast
- Quantile predictions (uncertainty intervals)
- Confidence scores
- Recommended signal (buy/hold/sell)
- SHAP explanation snippets

---

## Kubernetes & Rollout Strategy

### Infrastructure

- **Service & Ingress:** NGINX Ingress Controller
- **Auto-scaling:** HPA based on CPU, memory, or custom metrics (latency)
- **Model Serving:** KServe for simplified autoscaling and model management

### Canary Deployment

1. Route 10% of traffic to new model deployment
2. Monitor prediction quality and error rates
3. Promote to 100% if stable, or rollback if issues detected

### Horizontal Pod Autoscaler Example

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: predictor-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: predictor
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
```

---

## Monitoring & Drift Detection

### Data & Model Monitoring

**Evidently AI:**
- Compute drift metrics (population drift, feature drift)
- Generate metrics dashboards
- Track prediction distribution changes

**Prometheus + Grafana:**
- System metrics (latency, error rates)
- Custom metrics:
  - Prediction distribution
  - Latency percentiles
  - Missing feature percentage
  - Model confidence scores

### Alerting Strategy

Trigger alerts when:
- Drift metric exceeds threshold
- Prediction distribution shifts significantly
- Model performance degrades on holdout set
- Error rates spike

### Retraining Triggers

**Rule-Based:**
- Retrain when drift metric > X **AND** performance on holdout < Y

**ML-Based (Advanced):**
- Secondary model predicts when to retrain based on monitoring signals

---

## Explainability & Transparency

### Model Interpretability

- **SHAP:** Per-prediction explanations and aggregate feature importance
- **Captum:** For PyTorch models
- **LIME:** For simpler models

### User-Facing Explanations

**Natural Language Rationale:**
> "Model increased probability due to rising momentum and positive sentiment"

**Features:**
- Short explanation snippets in UI
- Feature importance visualizations
- Anomaly log for extreme predictions with SHAP snapshots

**Transparency:**
- Clear display of which features influenced the prediction
- Confidence intervals and uncertainty metrics
- Historical model performance metrics

---

## Backtesting & Paper Trading

### Implementation

**Frameworks:** backtrader, vectorbt, or custom engine

**Simulation Requirements:**
- Realistic order fills
- Transaction fees & costs
- Market latencies
- Slippage modeling

### Safety & Compliance

**Production Signals:**
- Display signals only (no automatic execution)
- Require explicit human approval for live trades
- Gate all production signals with manual review

**Legal Disclaimer:**
> ⚠️ **"Not Financial Advice"** — Prominently displayed
> - Do not provide trading advice
> - Do not accept money or handle live-trading instructions
> - Ensure compliance with regulations before any money-handling features

---

## Testing & Quality Gates

### Test Suite

**Unit Tests:**
- Preprocessing modules
- Feature calculation functions
- Model I/O operations

**Integration Tests:**
- End-to-end data pipeline on sample datasets
- API endpoint testing

**Regression Tests:**
- Track model performance on canonical test sets
- Fail CI if model drifts beyond acceptable range

**Data Quality Checks:**
- Missing value detection
- Duplicate record identification
- Schema drift validation
- Data range and distribution checks

---

## Security & Compliance

### Security Best Practices

**Secrets Management:**
- Use Kubernetes Secrets or HashiCorp Vault
- **Never** commit secrets to repository
- Rotate credentials regularly

**API Security:**
- Rate limiting on all endpoints
- Authentication: OAuth 2.0 / API keys
- Input validation and sanitization

**Data Privacy:**
- Avoid logging PII (Personally Identifiable Information)
- Comply with GDPR/CCPA if handling user data
- Implement data retention policies

### Legal Compliance

- Display clear terms of service and disclaimers
- Seek legal counsel before handling money or live trading
- Ensure regulatory compliance (SEC, FINRA if applicable)
- Document data sources and usage rights

---

## Repository Structure

```
ForesightX/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── make_dataset.py      # Data ingestion scripts
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py    # Feature engineering
│   ├── model/
│   │   ├── __init__.py
│   │   ├── train_model.py       # Training pipelines
│   │   └── predict_model.py     # Prediction & serving
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py         # Plotting & dashboards
├── data/
│   ├── external/                # External data sources
│   ├── interim/                 # Intermediate processed data
│   ├── processed/               # Final feature sets
│   └── raw/                     # Raw data from sources
├── models/                      # Trained model artifacts
├── notebooks/                   # Jupyter notebooks for exploration
│   └── exp1.ipynb
├── references/                  # Data dictionaries, manuals
├── reports/                     # Analysis reports
│   └── figures/                 # Generated graphics
├── docs/                        # Documentation
│   ├── commands.rst
│   ├── conf.py
│   ├── getting-started.rst
│   ├── index.rst
│   ├── make.bat
│   └── Makefile
├── .env                         # Environment variables (not in git)
├── .gitignore
├── About.md                     # This file
├── LICENSE
├── Makefile                     # Build automation
├── README.md                    # Project overview
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── test_environment.py          # Environment validation
└── tox.ini                      # Testing automation
```

### Future Additions

As the project grows, consider adding:
- `infra/` - Kubernetes manifests, Helm charts, Terraform
- `configs/` - Configuration files for training/serving
- `tests/` - Comprehensive test suite
- `api/` - FastAPI application code
- `monitoring/` - Drift detection and monitoring scripts

---

## Code Examples

### Technical Indicators (pandas)

```python
def sma(series, window):
    """Simple Moving Average"""
    return series.rolling(window).mean()

def rsi(series, window=14):
    """Relative Strength Index"""
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = -delta.clip(upper=0).rolling(window).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))
```

### Simple Backtesting Strategy

```python
import numpy as np

def simple_strategy(preds, price_series, threshold=0.01):
    """
    Vectorized backtesting strategy
    """
    positions = (preds > threshold).astype(int)  # Buy signal
    returns = price_series.pct_change().shift(-1) * positions
    cum_returns = (1 + returns.fillna(0)).cumprod()
    return cum_returns
```

---


---

## Dashboard / UX Ideas for nothingamazing.com

### Main Features

1. **Home Dashboard**
   - Interactive chart (Plotly) with predicted bands vs actuals
   - Model toggle to compare different predictions
   - Real-time confidence intervals

2. **Signal Feed**
   - Live trading signals with confidence scores
   - SHAP-based explanations (one-liners)
   - Historical signal performance

3. **Experiment Tracker**
   - Model lineage visualization
   - Dataset hashes and versions
   - Metric time series and comparisons

4. **Model Health Monitor**
   - Public-facing health badge (🟢 Green / 🟡 Yellow / 🔴 Red)
   - Based on drift metrics and evaluation scores
   - Recent performance statistics

5. **Operations Panel** *(Admin)*
   - Manual retrain/promote buttons
   - Audit log of deployments
   - System metrics and alerts

---

## Risks & Mitigations

### Technical Risks

| Risk | Mitigation |
|------|------------|
| **Data Leakage** | Enforce strict chronological splits; unit tests to detect leakage |
| **Overfitting** | Robust cross-validation, early stopping, conservative ensembles |
| **Model Degradation** | Comprehensive monitoring + automated retraining & rollback |
| **Cost Overruns** | Use spot instances, efficient infrastructure, cache features |

### Business Risks

| Risk | Mitigation |
|------|------------|
| **Regulatory Issues** | Clear disclaimers, avoid live-money advice, seek legal counsel |
| **Liability** | Terms of service, "Not Financial Advice" warnings |
| **Data Quality** | Robust validation, multiple data sources, quality checks |

---

## Minimal Viable Product (MVP)

### Phase 1: Foundation *(Weeks 1-2)*

✅ **Infrastructure Setup**
- Repository skeleton with proper structure
- Local docker-compose setup
- Kubernetes manifests placeholder

✅ **Data Pipeline**
- Ingestion pipeline for 1-3 tickers using yfinance
- Persist data to Parquet format
- Basic data validation

✅ **Feature Engineering**
- Module with core technical indicators
- Feature storage system

### Phase 2: Modeling *(Weeks 3-4)*

✅ **Model Training**
- Train XGBoost model + baseline ARIMA
- Log experiments to MLflow
- Implement walk-forward validation

✅ **Backtesting**
- Simple backtesting module
- Transaction cost simulation
- Performance metrics calculation

### Phase 3: Serving *(Weeks 5-6)*

✅ **API Development**
- FastAPI server for predictions
- Load registered model from MLflow
- Return prediction + SHAP explanations

✅ **Basic UI**
- Simple web page calling `/predict` endpoint
- Chart visualization with predictions
- Display SHAP explanations

### Phase 4: Operations *(Weeks 7-8)*

✅ **Monitoring**
- Prometheus metrics for latency & errors
- Evidently report for feature drift
- Basic alerting rules

✅ **CI/CD Pipeline**
- GitHub Actions workflow
- Automated testing and deployment

---

## CI/CD Checklist

### Pre-Deployment

- [ ] Linting + unit tests run on every PR
- [ ] Unit tests for feature engineering modules
- [ ] Unit tests for model I/O operations
- [ ] Integration tests for data pipeline

### Deployment Pipeline

- [ ] On merge to `main`: Build Docker image
- [ ] Push image to container registry
- [ ] Deploy to staging namespace in Kubernetes
- [ ] Run integration tests against staging
- [ ] Manual or automated promotion to production (based on thresholds)

### Post-Deployment

- [ ] Monitor model performance metrics
- [ ] Track drift detection alerts
- [ ] Review audit logs
- [ ] Update documentation

---

## Next Steps

1. **Complete MVP deliverables** as outlined above
2. **Expand data sources** to include alternative data
3. **Implement advanced models** (LSTM, TFT)
4. **Build comprehensive monitoring** dashboards
5. **Deploy to production** with full observability
6. **Iterate based on feedback** and performance metrics

---

## Resources & References

- **MLflow:** https://mlflow.org/
- **Evidently AI:** https://www.evidentlyai.com/
- **Feast:** https://feast.dev/
- **KServe:** https://kserve.github.io/
- **SHAP:** https://shap.readthedocs.io/

---

**Last Updated:** December 13, 2025  
**Version:** 2.0  
**Author:** Aditya Pratap Singh Tomar