from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    timestamp: datetime | None = None


class PredictionResponse(BaseModel):
    ticker: str
    predictions: list[float]
    confidence: float
    intervals: list[list[float]]
    model_version: str
