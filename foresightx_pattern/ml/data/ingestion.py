from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Iterable

import pandas as pd
import yfinance as yf

from foresightx_pattern.ml.utils.config import AppSettings


LOGGER = logging.getLogger(__name__)


def _normalize_history(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    normalized = frame.reset_index().rename(columns=str.title)
    if "Datetime" in normalized.columns:
        normalized = normalized.rename(columns={"Datetime": "Timestamp"})
    elif "Date" in normalized.columns:
        normalized = normalized.rename(columns={"Date": "Timestamp"})
    normalized["Ticker"] = ticker.upper()
    return normalized[["Timestamp", "Ticker", "Open", "High", "Low", "Close", "Volume"]]


def _download_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    attempts = 3
    for attempt in range(1, attempts + 1):
        frame = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            prepost=False,
            threads=False,
            multi_level_index=True,
        )
        if not frame.empty:
            return frame
        if attempt < attempts:
            time.sleep(attempt)
    return pd.DataFrame()


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
        frame = _download_history(ticker, period=period, interval=interval)
        if end_timestamp is not None and not frame.empty:
            frame = frame.loc[frame.index <= pd.Timestamp(end_timestamp)]
        frame = _normalize_history(frame, ticker)
        if not frame.empty:
            frames.append(frame)
        else:
            LOGGER.warning("No market data returned for %s; skipping ticker from training run", ticker)
    if not frames:
        return pd.DataFrame(columns=["Timestamp", "Ticker", "Open", "High", "Low", "Close", "Volume"])
    return pd.concat(frames, ignore_index=True)
