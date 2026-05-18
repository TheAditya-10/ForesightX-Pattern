from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from foresightx_pattern.app.schemas.prediction import PredictionResponse
from foresightx_pattern.app.services.confidence_service import ConfidenceService
from foresightx_pattern.app.services.feature_service import FeatureService
from foresightx_pattern.app.services.model_loader import ModelLoader
from foresightx_pattern.ml.utils.config import AppSettings


@dataclass(slots=True)
class InferenceService:
    settings: AppSettings
    model_loader: ModelLoader
    feature_service: FeatureService
    confidence_service: ConfidenceService

    def predict(self, ticker: str, timestamp: datetime | None = None) -> PredictionResponse:
        bundle = self.model_loader.load()
        stock_to_id = bundle.metadata["stock_to_id"]
        normalized = ticker.upper()
        if normalized not in stock_to_id:
            raise ValueError(f"Ticker {normalized} is not part of the trained stock embedding map")
        _, sequence = self.feature_service.latest_sequence(normalized, bundle.scaler, timestamp)
        mean, confidence, intervals = self.confidence_service.predict_with_confidence(
            bundle.model,
            sequence,
            stock_to_id[normalized],
            bundle.metadata.get("metrics", {}),
        )
        return PredictionResponse(
            ticker=normalized,
            predictions=mean,
            confidence=confidence,
            intervals=intervals,
            model_version=str(bundle.metadata.get("model_version", "local")),
        )
