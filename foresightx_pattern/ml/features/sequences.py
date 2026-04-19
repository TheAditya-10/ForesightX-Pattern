from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from foresightx_pattern.ml.features.engineering import feature_columns
from foresightx_pattern.ml.utils.config import AppSettings


@dataclass(slots=True)
class SequenceDatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    stock_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    stock_val: np.ndarray
    scaler: StandardScaler
    stock_to_id: dict[str, int]
    feature_names: list[str]


def build_sequence_dataset(features: pd.DataFrame, settings: AppSettings) -> SequenceDatasetBundle:
    seq_len = settings.data.get("sequence_length", 48)
    val_split = settings.training.get("validation_split", 0.2)
    features_list = feature_columns()
    stock_to_id = {ticker: idx for idx, ticker in enumerate(sorted(features["Ticker"].unique()))}
    train_rows: list[pd.DataFrame] = []
    val_rows: list[pd.DataFrame] = []
    for _, ticker_frame in features.groupby("Ticker", sort=True):
        cutoff = int(len(ticker_frame) * (1 - val_split))
        train_rows.append(ticker_frame.iloc[:cutoff].copy())
        val_rows.append(ticker_frame.iloc[max(seq_len, cutoff - seq_len):].copy())
    scaler = StandardScaler()
    scaler.fit(pd.concat(train_rows, ignore_index=True)[features_list])
    scaled = features.copy()
    scaled[features_list] = scaler.transform(scaled[features_list])
    train_scaled = pd.concat(
        [scaled[scaled["Ticker"] == frame["Ticker"].iloc[0]].iloc[: len(frame)] for frame in train_rows],
        ignore_index=True,
    )
    val_frames: list[pd.DataFrame] = []
    for frame in val_rows:
        ticker = frame["Ticker"].iloc[0]
        source = scaled[scaled["Ticker"] == ticker]
        start = max(0, len(source) - len(frame))
        val_frames.append(source.iloc[start:].copy())
    val_scaled = pd.concat(val_frames, ignore_index=True)
    X_train, y_train, stock_train = _frame_to_sequences(train_scaled, seq_len, stock_to_id)
    X_val, y_val, stock_val = _frame_to_sequences(val_scaled, seq_len, stock_to_id)
    return SequenceDatasetBundle(
        X_train=X_train,
        y_train=y_train,
        stock_train=stock_train,
        X_val=X_val,
        y_val=y_val,
        stock_val=stock_val,
        scaler=scaler,
        stock_to_id=stock_to_id,
        feature_names=features_list,
    )


def build_latest_sequence(
    features: pd.DataFrame,
    scaler: StandardScaler,
    seq_len: int,
) -> np.ndarray:
    columns = feature_columns()
    latest = features.sort_values("Timestamp").tail(seq_len).copy()
    latest[columns] = scaler.transform(latest[columns])
    return latest[columns].to_numpy(dtype=np.float32)


def _frame_to_sequences(
    frame: pd.DataFrame,
    seq_len: int,
    stock_to_id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    stock_ids: list[int] = []
    columns = feature_columns()
    target_cols = ["target_t1", "target_t2", "target_t3"]
    for ticker, ticker_frame in frame.groupby("Ticker", sort=True):
        ticker_frame = ticker_frame.sort_values("Timestamp")
        values = ticker_frame[columns].to_numpy(dtype=np.float32)
        targets = ticker_frame[target_cols].to_numpy(dtype=np.float32)
        for end in range(seq_len, len(ticker_frame) + 1):
            X_list.append(values[end - seq_len : end])
            y_list.append(targets[end - 1])
            stock_ids.append(stock_to_id[ticker])
    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
        np.asarray(stock_ids, dtype=np.int64),
    )
