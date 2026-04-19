from __future__ import annotations

import numpy as np
import pandas as pd

from foresightx_pattern.ml.data.preprocessing import clean_market_data
from foresightx_pattern.ml.features.engineering import build_feature_frame, feature_columns
from foresightx_pattern.ml.utils.config import load_settings


def test_feature_pipeline_generates_required_columns():
    settings = load_settings()
    timestamps = []
    for day in pd.bdate_range("2024-01-01", periods=20):
        for hour in range(9, 16):
            timestamps.append(pd.Timestamp(day).tz_localize("Asia/Kolkata") + pd.Timedelta(hours=hour))
    frame = pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Ticker": ["TATAMOTORS.NS"] * len(timestamps),
            "Open": np.linspace(100, 180, len(timestamps)),
            "High": np.linspace(101, 181, len(timestamps)),
            "Low": np.linspace(99, 179, len(timestamps)),
            "Close": np.linspace(100, 180, len(timestamps)),
            "Volume": np.linspace(1_000, 2_000, len(timestamps)),
        }
    )
    clean = clean_market_data(frame, settings)
    features = build_feature_frame(clean, settings)
    required = set(feature_columns()) | {"target_t1", "target_t2", "target_t3"}
    assert required.issubset(features.columns)
    assert features["Timestamp"].is_monotonic_increasing
    assert not features.empty
