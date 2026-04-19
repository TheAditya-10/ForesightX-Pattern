from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable

import pandas as pd
import yfinance as yf

from foresightx_pattern.ml.utils.config import AppSettings


LOGGER = logging.getLogger(__name__)


def _normalize_history(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized = frame.reset_index().rename(columns=str.title)
    if "Datetime" in normalized.columns:
        normalized = normalized.rename(columns={"Datetime": "Timestamp"})
    elif "Date" in normalized.columns:
        normalized = normalized.rename(columns={"Date": "Timestamp"})
    normalized["Ticker"] = ticker.upper()
    return normalized[["Timestamp", "Ticker", "Open", "High", "Low", "Close", "Volume"]]


def fetch_market_data(
    tickers: Iterable[str],
    settings: AppSettings,
    end_timestamp: datetime | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    period = settings.data.get("period", "730d")
    interval = settings.data.get("interval", "1h")
    for ticker in tickers:
        LOGGER.info("Fetching market data for %s", ticker)
        frame = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            prepost=False,
            threads=False,
        )
        if end_timestamp is not None and not frame.empty:
            frame = frame.loc[frame.index <= pd.Timestamp(end_timestamp)]
        frame = _normalize_history(frame, ticker)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["Timestamp", "Ticker", "Open", "High", "Low", "Close", "Volume"])
    return pd.concat(frames, ignore_index=True)
