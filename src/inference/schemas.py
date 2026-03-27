from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    service: str
    status: str
    available_models: list[str]
    timestamp: datetime


class PredictionRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=32)
    as_of_date: datetime | None = None


class PredictionResponse(BaseModel):
    symbol: str
    as_of_date: datetime
    predicted_return: float
    predicted_direction: str
    latest_close: float
    predicted_next_close: float
    model_type: str
    model_timestamp: datetime | None = None
    features_used: int


class ModelSummary(BaseModel):
    symbol: str
    model_type: str
    features_count: int
    model_timestamp: datetime | None = None
    metrics: dict[str, float | int | str]


class ModelListResponse(BaseModel):
    models: list[ModelSummary]


class ModelDetailResponse(ModelSummary):
    feature_names: list[str]
    model_file: str
    scaler_file: str
    metadata_file: str


class HistoricalPredictionPoint(BaseModel):
    as_of_date: datetime
    actual_return: float
    predicted_return: float
    absolute_error: float
    direction_correct: bool


class HistoricalPredictionResponse(BaseModel):
    symbol: str
    points: list[HistoricalPredictionPoint]
