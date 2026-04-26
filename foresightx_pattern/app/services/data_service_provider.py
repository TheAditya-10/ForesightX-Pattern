from __future__ import annotations

from typing import Any

import httpx
import pandas as pd

from foresightx_pattern.ml.utils.config import AppSettings


def load_bars_from_data_service(ticker: str, settings: AppSettings, timestamp: pd.Timestamp | None) -> pd.DataFrame:
    params: dict[str, Any] = {
        "interval": settings.data.get("interval", "1h"),
        "limit": max(int(settings.data.get("sequence_length", 48)) * 6, 240),
    }
    response = httpx.get(
        f"{settings.data_service_url.rstrip('/')}/bars/{ticker.upper()}",
        params=params,
        timeout=20.0,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("points", [])
    if not rows:
        raise ValueError(f"No bar data returned by data service for {ticker}")

    frame = pd.DataFrame(rows).rename(
        columns={
            "timestamp": "Timestamp",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    frame["Ticker"] = ticker.upper()
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], utc=True)
    if timestamp is not None:
        cutoff = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        frame = frame.loc[frame["Timestamp"] <= cutoff]
    return frame[["Timestamp", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
