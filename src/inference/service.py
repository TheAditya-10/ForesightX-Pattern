from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.inference.config import InferenceSettings
from src.inference.schemas import (
    HistoricalPredictionPoint,
    HistoricalPredictionResponse,
    ModelDetailResponse,
    ModelSummary,
    PredictionResponse,
)


class InferenceServiceError(RuntimeError):
    """Raised when model or feature artifacts are unavailable or invalid."""


@dataclass(slots=True)
class LoadedArtifacts:
    symbol: str
    metadata: dict[str, Any]
    model: Any
    scaler: Any


class InferenceService:
    def __init__(self, settings: InferenceSettings, logger: logging.Logger | None = None) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger(settings.service_name)
        self._artifact_cache: dict[str, LoadedArtifacts] = {}

    def list_available_symbols(self) -> list[str]:
        symbols: list[str] = []
        for metadata_path in sorted(self.settings.metadata_dir.glob("mlp_model_stats_*.json")):
            symbol = metadata_path.stem.removeprefix("mlp_model_stats_")
            if self._artifact_paths(symbol)["model"].exists() and self._artifact_paths(symbol)["scaler"].exists():
                symbols.append(symbol)
        return symbols

    def get_model_summary(self, symbol: str) -> ModelSummary:
        artifacts = self._load_artifacts(symbol)
        metadata = artifacts.metadata
        return ModelSummary(
            symbol=artifacts.symbol,
            model_type=str(metadata.get("model_type", "MLP")),
            features_count=int(metadata.get("features_count", 0)),
            model_timestamp=self._parse_datetime(metadata.get("timestamp")),
            metrics=self._coerce_metrics(metadata.get("metrics", {})),
        )

    def get_model_detail(self, symbol: str) -> ModelDetailResponse:
        artifacts = self._load_artifacts(symbol)
        metadata = artifacts.metadata
        paths = self._artifact_paths(symbol)
        return ModelDetailResponse(
            symbol=artifacts.symbol,
            model_type=str(metadata.get("model_type", "MLP")),
            features_count=int(metadata.get("features_count", 0)),
            model_timestamp=self._parse_datetime(metadata.get("timestamp")),
            metrics=self._coerce_metrics(metadata.get("metrics", {})),
            feature_names=[str(name) for name in metadata.get("feature_names", [])],
            model_file=str(paths["model"]),
            scaler_file=str(paths["scaler"]),
            metadata_file=str(paths["metadata"]),
        )

    def predict_latest(self, symbol: str, as_of_date: datetime | None = None) -> PredictionResponse:
        artifacts = self._load_artifacts(symbol)
        metadata = artifacts.metadata
        feature_names = metadata.get("feature_names", [])
        if not feature_names:
            raise InferenceServiceError(f"Feature metadata missing for symbol '{symbol}'")

        features_frame = self._load_features(symbol)
        missing_features = [name for name in feature_names if name not in features_frame.columns]
        if missing_features:
            raise InferenceServiceError(
                f"Feature file for '{symbol}' is missing required columns: {', '.join(missing_features[:5])}"
            )

        row = self._select_feature_row(features_frame, feature_names, as_of_date=as_of_date)
        values = row[feature_names].astype(float).to_numpy().reshape(1, -1)
        scaled_values = artifacts.scaler.transform(values)
        predicted_return = float(artifacts.model.predict(scaled_values)[0])
        latest_close = float(row["Close"])
        predicted_next_close = latest_close * (1.0 + predicted_return)

        return PredictionResponse(
            symbol=artifacts.symbol,
            as_of_date=pd.to_datetime(row["Date"]).to_pydatetime(),
            predicted_return=predicted_return,
            predicted_direction="up" if predicted_return >= 0 else "down",
            latest_close=latest_close,
            predicted_next_close=predicted_next_close,
            model_type=str(metadata.get("model_type", "MLP")),
            model_timestamp=self._parse_datetime(metadata.get("timestamp")),
            features_used=len(feature_names),
        )

    def get_prediction_history(self, symbol: str, limit: int = 30) -> HistoricalPredictionResponse:
        normalized = symbol.strip().upper()
        history_path = self.settings.results_dir / f"predictions_{normalized}.csv"
        if not history_path.exists():
            raise InferenceServiceError(f"Historical predictions file not found for symbol '{normalized}'")

        frame = pd.read_csv(history_path)
        required_columns = {"Date", "Actual", "Predicted", "Abs_Error", "Direction_Correct"}
        if not required_columns.issubset(frame.columns):
            raise InferenceServiceError(f"Historical predictions file for '{normalized}' has an unexpected schema")

        limit = max(1, min(limit, 365))
        tail = frame.tail(limit)
        points = [
            HistoricalPredictionPoint(
                as_of_date=pd.to_datetime(row["Date"]).to_pydatetime(),
                actual_return=float(row["Actual"]),
                predicted_return=float(row["Predicted"]),
                absolute_error=float(row["Abs_Error"]),
                direction_correct=bool(row["Direction_Correct"]),
            )
            for _, row in tail.iterrows()
        ]
        return HistoricalPredictionResponse(symbol=normalized, points=points)

    def warmup(self) -> list[str]:
        symbols = self.list_available_symbols()
        for symbol in symbols:
            self._load_artifacts(symbol)
        return symbols

    def _artifact_paths(self, symbol: str) -> dict[str, Path]:
        normalized = symbol.strip().upper()
        return {
            "model": self.settings.models_dir / f"mlp_model_{normalized}.pkl",
            "scaler": self.settings.models_dir / f"mlp_scaler_{normalized}.pkl",
            "metadata": self.settings.metadata_dir / f"mlp_model_stats_{normalized}.json",
        }

    def _load_artifacts(self, symbol: str) -> LoadedArtifacts:
        normalized = symbol.strip().upper()
        if normalized in self._artifact_cache:
            return self._artifact_cache[normalized]

        paths = self._artifact_paths(normalized)
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            raise InferenceServiceError(
                f"Model artifacts not found for symbol '{normalized}'. Missing: {', '.join(missing)}"
            )

        try:
            with paths["metadata"].open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            with paths["model"].open("rb") as handle:
                model = pickle.load(handle)
            with paths["scaler"].open("rb") as handle:
                scaler = pickle.load(handle)
        except Exception as exc:
            raise InferenceServiceError(f"Failed to load model artifacts for symbol '{normalized}': {exc}") from exc

        artifacts = LoadedArtifacts(symbol=normalized, metadata=metadata, model=model, scaler=scaler)
        self._artifact_cache[normalized] = artifacts
        return artifacts

    def _load_features(self, symbol: str) -> pd.DataFrame:
        normalized = symbol.strip().upper()
        features_path = self.settings.features_dir / f"features_{normalized}.csv"
        if not features_path.exists():
            raise InferenceServiceError(f"Feature file not found for symbol '{normalized}'")

        try:
            frame = pd.read_csv(features_path)
        except Exception as exc:
            raise InferenceServiceError(f"Failed to load features for symbol '{normalized}': {exc}") from exc
        if "Date" not in frame.columns or "Close" not in frame.columns:
            raise InferenceServiceError(f"Feature file for '{normalized}' is missing required Date/Close columns")
        return frame

    def _select_feature_row(
        self,
        frame: pd.DataFrame,
        feature_names: list[str],
        *,
        as_of_date: datetime | None,
    ) -> pd.Series:
        candidate_frame = frame.dropna(subset=feature_names).copy()
        if candidate_frame.empty:
            raise InferenceServiceError("No inference-ready rows were found in the feature file")

        candidate_frame["Date"] = pd.to_datetime(candidate_frame["Date"])
        if as_of_date is None:
            return candidate_frame.iloc[-1]

        filtered = candidate_frame[candidate_frame["Date"] <= pd.Timestamp(as_of_date)]
        if filtered.empty:
            raise InferenceServiceError(f"No inference-ready rows found on or before {as_of_date.isoformat()}")
        return filtered.iloc[-1]

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None

    def _coerce_metrics(self, metrics: dict[str, Any]) -> dict[str, float | int | str]:
        coerced: dict[str, float | int | str] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, str)):
                coerced[str(key)] = value
        return coerced
