from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from src.db.session import check_database_connection, close_database, get_session_factory
from src.inference.config import InferenceSettings
from src.inference.on_demand_training import OnDemandModelTrainer, TrainingRequest, TrainingResponse
from src.inference.schemas import (
    HealthResponse,
    HistoricalPredictionResponse,
    ModelDetailResponse,
    ModelListResponse,
    PredictionRequest,
    PredictionResponse,
    SignalPredictionRequest,
    SignalPredictionResponse,
)
from src.inference.service import InferenceService, InferenceServiceError


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


settings = InferenceSettings()
configure_logging(settings.log_level)
logger = logging.getLogger(settings.service_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    session_factory = get_session_factory(settings.database_url)
    await check_database_connection(settings.database_url)
    service = InferenceService(settings=settings, logger=logger, session_factory=session_factory)
    available_symbols = service.warmup()
    await service.sync_registry()
    
    # Initialize on-demand trainer
    on_demand_trainer = OnDemandModelTrainer(models_dir="models", metadata_dir="metadata")
    
    app.state.settings = settings
    app.state.inference_service = service
    app.state.on_demand_trainer = on_demand_trainer
    app.state.session_factory = session_factory
    logger.info("Inference service startup complete. Available models: %s", available_symbols or "none")
    logger.info("On-demand training service initialized")
    try:
        yield
    finally:
        await close_database()


app = FastAPI(
    title="ForesightX Pattern Inference Service",
    version="1.0.0",
    description="API layer for serving stock-return predictions from trained ForesightX model artifacts.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_inference_service() -> InferenceService:
    return app.state.inference_service


def get_on_demand_trainer() -> OnDemandModelTrainer:
    return app.state.on_demand_trainer


def translate_error(exc: InferenceServiceError) -> HTTPException:
    detail = str(exc)
    status_code = status.HTTP_404_NOT_FOUND if "not found" in detail.lower() or "missing" in detail.lower() else status.HTTP_422_UNPROCESSABLE_ENTITY
    return HTTPException(status_code=status_code, detail=detail)


@app.get("/", tags=["meta"])
async def root(service: InferenceService = Depends(get_inference_service)) -> dict[str, object]:
    return {
        "service": settings.service_name,
        "status": "ok",
        "available_models": service.list_available_symbols(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health/live", response_model=HealthResponse, tags=["health"])
async def live(service: InferenceService = Depends(get_inference_service)) -> HealthResponse:
    return HealthResponse(
        service=settings.service_name,
        status="ok",
        available_models=service.list_available_symbols(),
        timestamp=datetime.now(timezone.utc),
    )


@app.get("/health/ready", response_model=HealthResponse, tags=["health"])
async def ready(service: InferenceService = Depends(get_inference_service)) -> HealthResponse:
    symbols = service.list_available_symbols()
    return HealthResponse(
        service=settings.service_name,
        status="ready" if symbols else "degraded",
        available_models=symbols,
        timestamp=datetime.now(timezone.utc),
    )


@app.get("/models", response_model=ModelListResponse, tags=["models"])
async def list_models(service: InferenceService = Depends(get_inference_service)) -> ModelListResponse:
    models = [service.get_model_summary(symbol) for symbol in service.list_available_symbols()]
    return ModelListResponse(models=models)


@app.get("/models/{symbol}", response_model=ModelDetailResponse, tags=["models"])
async def get_model(symbol: str, service: InferenceService = Depends(get_inference_service)) -> ModelDetailResponse:
    try:
        return service.get_model_detail(symbol)
    except InferenceServiceError as exc:
        raise translate_error(exc) from exc


@app.get("/predictions/{symbol}/latest", response_model=PredictionResponse, tags=["predictions"])
async def predict_latest(
    symbol: str,
    as_of_date: datetime | None = Query(default=None),
    service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    try:
        return await service.predict_latest(symbol, as_of_date=as_of_date)
    except InferenceServiceError as exc:
        raise translate_error(exc) from exc


@app.post("/predictions/latest", response_model=PredictionResponse, tags=["predictions"])
async def predict_latest_post(
    payload: PredictionRequest,
    service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    try:
        return await service.predict_latest(payload.symbol, as_of_date=payload.as_of_date)
    except InferenceServiceError as exc:
        raise translate_error(exc) from exc


@app.get("/predictions/{symbol}/history", response_model=HistoricalPredictionResponse, tags=["predictions"])
async def prediction_history(
    symbol: str,
    limit: int = Query(default=30, ge=1, le=365),
    service: InferenceService = Depends(get_inference_service),
) -> HistoricalPredictionResponse:
    try:
        return service.get_prediction_history(symbol, limit=limit)
    except InferenceServiceError as exc:
        raise translate_error(exc) from exc


@app.post("/predict", response_model=SignalPredictionResponse, tags=["predictions"])
async def predict_signal(
    payload: SignalPredictionRequest,
    service: InferenceService = Depends(get_inference_service),
) -> SignalPredictionResponse:
    try:
        prediction = await service.predict_latest(payload.ticker, as_of_date=payload.as_of_date)
        signal_label, signal_confidence = service._signal_from_prediction(prediction.predicted_return)
        return SignalPredictionResponse(
            prediction_id=prediction.prediction_id,
            symbol=prediction.symbol,
            prediction=signal_label,
            confidence=signal_confidence,
            predicted_return=prediction.predicted_return,
            latest_close=prediction.latest_close,
            predicted_next_close=prediction.predicted_next_close,
        )
    except InferenceServiceError as exc:
        raise translate_error(exc) from exc


# =====================================================================
# ON-DEMAND TRAINING ENDPOINTS
# =====================================================================


@app.post("/train", response_model=TrainingResponse, tags=["training"])
async def train_on_demand(
    request: TrainingRequest,
    trainer: OnDemandModelTrainer = Depends(get_on_demand_trainer),
) -> TrainingResponse:
    """
    Train a new MLP model for any stock ticker on-demand.
    
    This endpoint:
    1. Fetches last N days (default 15) of OHLCV data
    2. Engineers technical indicator features
    3. Trains an MLP neural network
    4. Saves model artifacts
    5. Returns prediction for next trading day
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., "AAPL", "NVDA", "TATAMOTORS.BO")
    days : int
        Number of trading days of historical data (default 15, range 5-252)
    retrain : bool
        Force retrain even if model exists (default False)
    
    Returns:
    --------
    TrainingResponse
        Training status, metrics, and next-day prediction
    
    Example:
    --------
    POST /train
    {
        "ticker": "AAPL",
        "days": 15,
        "retrain": false
    }
    """
    return await trainer.train_async(request)


@app.get("/train/models", tags=["training"])
async def list_trained_models() -> dict:
    """List all on-demand trained models."""
    from pathlib import Path
    
    models_dir = Path("models")
    if not models_dir.exists():
        return {"models": [], "count": 0}
    
    model_files = list(models_dir.glob("mlp_model_*.pkl"))
    model_names = [f.stem.replace("mlp_model_", "") for f in model_files]
    
    return {
        "models": sorted(model_names),
        "count": len(model_names),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/train/models/{ticker}", tags=["training"])
async def get_trained_model_info(ticker: str) -> dict:
    """Get metadata for a trained model."""
    from pathlib import Path
    import json
    
    metadata_file = Path("metadata") / f"mlp_model_stats_{ticker}.json"
    
    if not metadata_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No trained model metadata found for {ticker}",
        )
    
    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model metadata: {str(e)}",
        )
