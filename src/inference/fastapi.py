from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from src.inference.config import InferenceSettings
from src.inference.schemas import (
    HealthResponse,
    HistoricalPredictionResponse,
    ModelDetailResponse,
    ModelListResponse,
    PredictionRequest,
    PredictionResponse,
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
    service = InferenceService(settings=settings, logger=logger)
    available_symbols = service.warmup()
    app.state.settings = settings
    app.state.inference_service = service
    logger.info("Inference service startup complete. Available models: %s", available_symbols or "none")
    yield


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
        return service.predict_latest(symbol, as_of_date=as_of_date)
    except InferenceServiceError as exc:
        raise translate_error(exc) from exc


@app.post("/predictions/latest", response_model=PredictionResponse, tags=["predictions"])
async def predict_latest_post(
    payload: PredictionRequest,
    service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    try:
        return service.predict_latest(payload.symbol, as_of_date=payload.as_of_date)
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
