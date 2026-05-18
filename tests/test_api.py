from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

from foresightx_pattern.app.main import create_app
from foresightx_pattern.ml.features.engineering import feature_columns
from foresightx_pattern.ml.models.full_model import ForesightXPatternModel
from foresightx_pattern.ml.utils.config import load_settings


def test_predict_endpoint_returns_predictions(tmp_path: Path):
    settings = load_settings()
    settings.model_dir = tmp_path / "model"
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    feature_names = feature_columns()
    scaler = StandardScaler()
    scaler.fit(pd.DataFrame(np.random.randn(200, len(feature_names)), columns=feature_names))
    with (settings.model_dir / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)
    metadata = {
        "model_version": "test",
        "sequence_length": 48,
        "feature_names": feature_names,
        "stock_to_id": {"TATAMOTORS.NS": 0},
        "input_dim": len(feature_names),
        "num_stocks": 1,
        "projection_dim": 64,
        "hidden_dim": 128,
        "embedding_dim": 16,
        "num_layers": 2,
        "dropout": 0.2,
        "encoder_type": "lstm",
    }
    (settings.model_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    model = ForesightXPatternModel(input_dim=len(feature_names), num_stocks=1)
    model.eval()
    torch.onnx.export(
        model,
        (
            torch.zeros(1, metadata["sequence_length"], len(feature_names), dtype=torch.float32),
            torch.zeros(1, dtype=torch.long),
        ),
        settings.model_dir / "model.onnx",
        input_names=["sequence", "stock_id"],
        output_names=["predictions"],
        dynamic_axes={
            "sequence": {0: "batch"},
            "stock_id": {0: "batch"},
            "predictions": {0: "batch"},
        },
        opset_version=17,
    )

    def fake_provider(ticker, _settings, timestamp):
        timestamps = []
        for day in pd.bdate_range("2024-01-01", periods=20):
            for hour in range(9, 16):
                timestamps.append(pd.Timestamp(day).tz_localize("Asia/Kolkata") + pd.Timedelta(hours=hour))
        return pd.DataFrame(
            {
                "Timestamp": timestamps,
                "Ticker": [ticker] * len(timestamps),
                "Open": np.linspace(100, 180, len(timestamps)),
                "High": np.linspace(101, 181, len(timestamps)),
                "Low": np.linspace(99, 179, len(timestamps)),
                "Close": np.linspace(100, 180, len(timestamps)),
                "Volume": np.linspace(1_000, 2_000, len(timestamps)),
            }
        )

    client = TestClient(create_app(settings=settings, data_provider=fake_provider))
    response = client.post("/predict", json={"ticker": "TATAMOTORS.NS"})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["predictions"]) == 3
    assert len(payload["intervals"]) == 3
    assert 0 <= payload["confidence"] <= 1
