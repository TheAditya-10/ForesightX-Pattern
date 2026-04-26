from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd
import yfinance as yf

from foresightx_pattern.ml.data.preprocessing import clean_market_data
from foresightx_pattern.ml.features.engineering import build_feature_frame
from foresightx_pattern.ml.features.sequences import build_latest_sequence
from foresightx_pattern.ml.utils.config import AppSettings


@dataclass(slots=True)
class CachedFrame:
    expires_at: datetime
    frame: pd.DataFrame


class FeatureService:
    def __init__(self, settings: AppSettings, data_provider) -> None:
        self.settings = settings
        self.data_provider = data_provider
        self._cache: dict[str, CachedFrame] = {}

    def latest_sequence(self, ticker: str, scaler, timestamp: datetime | None) -> tuple[pd.DataFrame, object]:
        frame = self._load_frame(ticker, timestamp)
        clean = clean_market_data(frame, self.settings)
        features = build_feature_frame(clean, self.settings)
        seq_len = self.settings.data.get("sequence_length", 48)
        if len(features) < seq_len:
            raise ValueError(f"Not enough bars to build a {seq_len}-step sequence for {ticker}")
        return features, build_latest_sequence(features, scaler, seq_len)

    def _load_frame(self, ticker: str, timestamp: datetime | None) -> pd.DataFrame:
        cache_key = f"{ticker}:{timestamp.isoformat() if timestamp else 'latest'}"
        now = datetime.now(UTC)
        cached = self._cache.get(cache_key)
        if cached and cached.expires_at > now:
            return cached.frame.copy()
        frame = self.data_provider(ticker, self.settings, pd.Timestamp(timestamp) if timestamp else None)
        ttl_seconds = int(self.settings.cache.get("feature_ttl_seconds", 300))
        self._cache[cache_key] = CachedFrame(expires_at=now + timedelta(seconds=ttl_seconds), frame=frame.copy())
        return frame


def download_latest_market_data(ticker: str, settings: AppSettings, timestamp: pd.Timestamp | None) -> pd.DataFrame:
    attempts = [
        (ticker, settings.data.get("serving_period", "90d"), settings.data.get("interval", "1h")),
        (ticker, "6mo", "1d"),
    ]
    frame = pd.DataFrame()
    for candidate_ticker, period, interval in attempts:
        frame = yf.download(
            candidate_ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            prepost=False,
            threads=False,
            multi_level_index=True,
        )
        if not frame.empty:
            break
    if frame.empty:
        raise ValueError(f"No market data returned for {ticker}")

    # yfinance may return MultiIndex columns for single tickers in some versions.
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.reset_index().rename(columns={"Datetime": "Timestamp", "Date": "Timestamp"})
    if timestamp is not None:
        cutoff = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        frame = frame.loc[pd.to_datetime(frame["Timestamp"], utc=True) <= cutoff]
    frame["Ticker"] = ticker.upper()
    return frame[["Timestamp", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
