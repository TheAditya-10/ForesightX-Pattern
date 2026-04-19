from __future__ import annotations

import numpy as np
import pandas as pd

from foresightx_pattern.ml.utils.config import AppSettings


def _compute_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100.0)


def build_feature_frame(frame: pd.DataFrame, settings: AppSettings) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    feature_frames: list[pd.DataFrame] = []
    rsi_window = settings.features.get("rsi_window", 14)
    ema_fast = settings.features.get("ema_fast", 12)
    ema_slow = settings.features.get("ema_slow", 26)
    macd_signal = settings.features.get("macd_signal", 9)
    rolling_windows = settings.features.get("rolling_windows", [5, 10, 20])
    horizon = settings.data.get("horizon", 3)

    for ticker, ticker_frame in frame.groupby("Ticker", sort=True):
        data = ticker_frame.sort_values("Timestamp").copy()
        close = data["Close"]
        data["return_1"] = close.pct_change()
        data["ema_fast"] = close.ewm(span=ema_fast, adjust=False).mean()
        data["ema_slow"] = close.ewm(span=ema_slow, adjust=False).mean()
        data["macd"] = data["ema_fast"] - data["ema_slow"]
        data["macd_signal"] = data["macd"].ewm(span=macd_signal, adjust=False).mean()
        data["macd_hist"] = data["macd"] - data["macd_signal"]
        data["rsi"] = _compute_rsi(close, rsi_window)
        for window in rolling_windows:
            data[f"rolling_mean_{window}"] = close.rolling(window).mean()
            data[f"rolling_std_{window}"] = close.rolling(window).std()
        data["hour"] = data["Timestamp"].dt.hour
        data["day_of_week"] = data["Timestamp"].dt.dayofweek
        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
        data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
        data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
        for step in range(1, horizon + 1):
            data[f"target_t{step}"] = data["Close"].shift(-step)
        feature_frames.append(data)

    features = pd.concat(feature_frames, ignore_index=True)
    features = features.dropna().reset_index(drop=True)
    return features


def feature_columns() -> list[str]:
    return [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "return_1",
        "rsi",
        "ema_fast",
        "ema_slow",
        "macd",
        "macd_signal",
        "macd_hist",
        "rolling_mean_5",
        "rolling_std_5",
        "rolling_mean_10",
        "rolling_std_10",
        "rolling_mean_20",
        "rolling_std_20",
        "hour",
        "day_of_week",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    ]
