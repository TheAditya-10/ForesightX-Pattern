from __future__ import annotations

import pandas as pd

from foresightx_pattern.ml.utils.markets import market_profile_for_ticker
from foresightx_pattern.ml.utils.config import AppSettings


def filter_trading_hours(frame: pd.DataFrame, settings: AppSettings) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    data = frame.copy()
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], utc=True)
    filtered_frames: list[pd.DataFrame] = []

    for ticker, ticker_frame in data.groupby("Ticker", sort=True):
        local = ticker_frame.copy()
        profile = market_profile_for_ticker(ticker)
        localized = local["Timestamp"].dt.tz_convert(profile.timezone)

        # For daily bars, keep weekdays only and preserve UTC timestamps.
        if len(localized) >= 2:
            median_step = localized.sort_values().diff().median()
            if pd.notna(median_step) and median_step >= pd.Timedelta(hours=20):
                mask = localized.dt.dayofweek < 5
                filtered_frames.append(
                    local.loc[mask].sort_values(["Ticker", "Timestamp"]).drop_duplicates(["Ticker", "Timestamp"])
                )
                continue

        minutes = localized.dt.hour * 60 + localized.dt.minute
        start_minutes = profile.open_hour * 60 + profile.open_minute
        end_minutes = profile.close_hour * 60 + profile.close_minute
        mask = localized.dt.dayofweek < 5
        mask &= minutes >= start_minutes
        mask &= minutes <= end_minutes
        filtered_frames.append(
            local.loc[mask].sort_values(["Ticker", "Timestamp"]).drop_duplicates(["Ticker", "Timestamp"])
        )

    if not filtered_frames:
        return data.iloc[0:0].copy()
    return pd.concat(filtered_frames, ignore_index=True)


def clean_market_data(frame: pd.DataFrame, settings: AppSettings) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    data = filter_trading_hours(frame, settings)
    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    data[numeric_columns] = data.groupby("Ticker")[numeric_columns].ffill().bfill()
    data = data.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return data.reset_index(drop=True)
