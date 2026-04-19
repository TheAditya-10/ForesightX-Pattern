from __future__ import annotations

import pandas as pd

from foresightx_pattern.ml.utils.config import AppSettings


def filter_trading_hours(frame: pd.DataFrame, settings: AppSettings) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    data = frame.copy()
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], utc=True)
    timezone_name = settings.data.get("market_timezone", "Asia/Kolkata")
    localized = data["Timestamp"].dt.tz_convert(timezone_name)
    data["Timestamp"] = localized
    open_hour = settings.data.get("market_open_hour", 9)
    close_hour = settings.data.get("market_close_hour", 15)
    close_minute = settings.data.get("market_close_minute", 30)
    minutes = localized.dt.hour * 60 + localized.dt.minute
    start_minutes = open_hour * 60
    end_minutes = close_hour * 60 + close_minute
    mask = localized.dt.dayofweek < 5
    mask &= minutes >= start_minutes
    mask &= minutes <= end_minutes
    return data.loc[mask].sort_values(["Ticker", "Timestamp"]).drop_duplicates(["Ticker", "Timestamp"])


def clean_market_data(frame: pd.DataFrame, settings: AppSettings) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    data = filter_trading_hours(frame, settings)
    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    data[numeric_columns] = data.groupby("Ticker")[numeric_columns].ffill().bfill()
    data = data.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return data.reset_index(drop=True)
