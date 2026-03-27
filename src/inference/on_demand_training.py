"""
On-Demand Model Training for ForesightX Pattern Service
========================================================

This module provides API endpoints to train models on-demand for any stock ticker
using the existing model architecture (MLP with feature engineering pipeline).

Features:
- Fetch last 15 days (or custom period) of market data
- Engineer features using the production feature pipeline
- Train MLP model on the ticker
- Save trained model artifacts
- Return prediction for the next trading day

Author: GitHub Copilot with Aditya Pratap Singh Tomar
Date: March 2026
"""

import asyncio
import json
import logging
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


class OnDemandTrainingError(RuntimeError):
    """Raised when on-demand training fails."""
    pass


class TrainingRequest(BaseModel):
    """Request schema for on-demand model training."""
    ticker: str = Field(..., min_length=1, max_length=20)
    days: int = Field(default=15, ge=5, le=252)  # 5 to 252 trading days
    retrain: bool = Field(default=False)  # Force retrain even if model exists


class TrainingResponse(BaseModel):
    """Response schema for training completion."""
    ticker: str
    success: bool
    message: str
    model_file: str | None = None
    scaler_file: str | None = None
    training_time_seconds: float | None = None
    metrics: dict[str, float] | None = None
    prediction: dict[str, Any] | None = None


class OnDemandModelTrainer:
    """
    On-demand MLP model trainer for any stock ticker.
    
    Workflow:
    1. Fetch historical price data (OHLCV)
    2. Engineer features (technical indicators, patterns)
    3. Scale features
    4. Train MLPRegressor
    5. Save model artifacts
    6. Generate prediction for next day
    """

    def __init__(self, models_dir: str = "models", metadata_dir: str = "metadata"):
        """Initialize trainer with artifact directories."""
        self.models_dir = Path(models_dir)
        self.metadata_dir = Path(metadata_dir)
        self.logger = logging.getLogger("OnDemandTrainer")

        # Create directories if needed
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

        # Model configuration (matches production MLPRegressor)
        self.model_config = {
            "hidden_layer_sizes": (128, 64, 32),
            "activation": "relu",
            "solver": "adam",
            "learning_rate": "adaptive",
            "learning_rate_init": 0.001,
            "max_iter": 200,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 15,
            "random_state": 42,
        }

    async def train_async(self, request: TrainingRequest) -> TrainingResponse:
        """
        Train model asynchronously (executes CPU tasks in thread pool).
        
        Parameters:
        -----------
        request : TrainingRequest
            Training parameters (ticker, days, retrain flag)
            
        Returns:
        --------
        TrainingResponse
            Training results including model path and prediction
        """
        try:
            # Check if model exists and retrain not requested
            model_file = self.models_dir / f"mlp_model_{request.ticker}.pkl"
            if model_file.exists() and not request.retrain:
                self.logger.info(f"Model already exists for {request.ticker}, loading cached model")
                return await self._load_and_predict(request.ticker)

            return await asyncio.to_thread(self.train_sync, request)

        except Exception as e:
            self.logger.error(f"Async training failed for {request.ticker}: {e}")
            return TrainingResponse(
                ticker=request.ticker,
                success=False,
                message=f"Training failed: {str(e)}",
            )

    def train_sync(self, request: TrainingRequest) -> TrainingResponse:
        """
        Synchronous training pipeline (runs in thread pool).
        
        Parameters:
        -----------
        request : TrainingRequest
            Training parameters
            
        Returns:
        --------
        TrainingResponse
            Training results
        """
        start_time = datetime.now()
        ticker = request.ticker

        try:
            self.logger.info(f"Starting on-demand training for {ticker} ({request.days} days)")

            # Step 1: Fetch data
            self.logger.info(f"[1/5] Fetching {request.days} days of price data...")
            df_prices = self._fetch_price_data(ticker, request.days)

            # Step 2: Engineer features
            self.logger.info("[2/5] Engineering features...")
            df_features = self._engineer_features(df_prices)

            # Step 3: Prepare training data
            self.logger.info("[3/5] Preparing training data...")
            X, y, feature_names = self._prepare_training_data(df_features)

            if len(X) < 10:
                raise OnDemandTrainingError(f"Insufficient data after feature engineering ({len(X)} rows)")

            # Step 4: Train model
            self.logger.info("[4/5] Training MLP model...")
            model, scaler, metrics = self._train_model(X, y)

            # Step 5: Save artifacts
            self.logger.info("[5/5] Saving model artifacts...")
            model_file, scaler_file, metadata_file = self._save_artifacts(
                ticker, model, scaler, feature_names, metrics
            )

            # Step 6: Generate prediction
            self.logger.info("Generating next-day prediction...")
            prediction = self._predict_next(model, scaler, df_features.iloc[-1], ticker)

            duration = (datetime.now() - start_time).total_seconds()

            self.logger.info(
                f"✓ Training complete in {duration:.2f}s | "
                f"RMSE: {metrics['rmse']:.6f} | "
                f"MAE: {metrics['mae']:.6f}"
            )

            return TrainingResponse(
                ticker=ticker,
                success=True,
                message=f"Model trained successfully in {duration:.1f}s",
                model_file=str(model_file),
                scaler_file=str(scaler_file),
                training_time_seconds=duration,
                metrics=metrics,
                prediction=prediction,
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Training failed for {ticker} after {duration:.2f}s: {e}")
            return TrainingResponse(
                ticker=ticker,
                success=False,
                message=f"Training failed: {str(e)}",
                training_time_seconds=duration,
            )

    async def _load_and_predict(self, ticker: str) -> TrainingResponse:
        """Load existing model and generate prediction."""
        try:
            model_file = self.models_dir / f"mlp_model_{ticker}.pkl"
            scaler_file = self.models_dir / f"mlp_scaler_{ticker}.pkl"

            with open(model_file, "rb") as f:
                model = pickle.load(f)
            with open(scaler_file, "rb") as f:
                scaler = pickle.load(f)

            # Get latest data for prediction
            df_prices = self._fetch_price_data(ticker, 30)
            df_features = self._engineer_features(df_prices)
            prediction = self._predict_next(model, scaler, df_features.iloc[-1], ticker)

            return TrainingResponse(
                ticker=ticker,
                success=True,
                message="Using cached model",
                prediction=prediction,
            )
        except Exception as e:
            self.logger.error(f"Failed to load cached model for {ticker}: {e}")
            return TrainingResponse(
                ticker=ticker,
                success=False,
                message=f"Failed to load model: {str(e)}",
            )

    def _fetch_price_data(self, ticker: str, days: int) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days + 10)  # Extra buffer for indicators

            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if df.empty:
                raise OnDemandTrainingError(f"No data available for {ticker}")

            # Reset index and rename columns
            df.reset_index(inplace=True)
            df.columns = [col.lower() for col in df.columns]

            self.logger.info(f"Fetched {len(df)} rows for {ticker}")
            return df

        except Exception as e:
            raise OnDemandTrainingError(f"Failed to fetch data for {ticker}: {str(e)}")

    def _engineer_features(self, df_prices: pd.DataFrame) -> pd.DataFrame:
        """Engineer technical indicators and derived features."""
        df = df_prices.copy()

        try:
            # Direction and returns
            df["returns"] = df["close"].pct_change()
            df["direction"] = (df["returns"] > 0).astype(int)

            # Volatility (20-day rolling)
            df["volatility"] = df["returns"].rolling(20).std()

            # SMA features
            for window in [5, 20, 50]:
                df[f"sma_{window}"] = df["close"].rolling(window).mean()
                df[f"sma_ratio_{window}"] = df["close"] / df[f"sma_{window}"]

            # RSI (14-period)
            df["rsi"] = self._calc_rsi(df["close"], 14)

            # MACD
            exp1 = df["close"].ewm(span=12, adjust=False).mean()
            exp2 = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = exp1 - exp2
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_histogram"] = df["macd"] - df["macd_signal"]

            # Bollinger Bands
            df["bb_middle"] = df["close"].rolling(20).mean()
            bb_std = df["close"].rolling(20).std()
            df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
            df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
            df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

            # Volume features
            df["volume_sma"] = df["volume"].rolling(20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_sma"]

            # ATR (14-period)
            df["atr"] = self._calc_atr(df, 14)

            # Lag features
            for lag in [1, 2, 3, 5]:
                df[f"return_lag_{lag}"] = df["returns"].shift(lag)
                df[f"close_lag_{lag}"] = df["close"].shift(lag)

            # Target (next day return)
            df["target_return"] = df["returns"].shift(-1)

            # Drop NaN rows
            df.dropna(inplace=True)

            self.logger.info(f"Engineered {len(df.columns)} features for {len(df)} rows")
            return df

        except Exception as e:
            raise OnDemandTrainingError(f"Feature engineering failed: {str(e)}")

    def _prepare_training_data(self, df_features: pd.DataFrame) -> tuple:
        """Prepare data for model training."""
        try:
            # Select features (exclude price, datetime, target)
            exclude_cols = ["date", "open", "high", "low", "close", "volume", "returns", "direction", "target_return"]
            feature_cols = [col for col in df_features.columns if col not in exclude_cols]

            X = df_features[feature_cols].values
            y = df_features["target_return"].values

            self.logger.info(f"Training data: X{X.shape}, y{y.shape}, {len(feature_cols)} features")
            return X, y, feature_cols

        except Exception as e:
            raise OnDemandTrainingError(f"Data preparation failed: {str(e)}")

    def _train_model(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Train MLP model with train/val split."""
        try:
            # Split data (80/20 chronological)
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Train model
            model = MLPRegressor(**self.model_config)
            model.fit(X_train_scaled, y_train)

            # Evaluate
            y_val_pred = model.predict(X_val_scaled)
            rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            mae = mean_absolute_error(y_val, y_val_pred)

            metrics = {
                "rmse": float(rmse),
                "mae": float(mae),
                "train_size": len(X_train),
                "val_size": len(X_val),
                "iterations": int(model.n_iter_),
            }

            self.logger.info(f"Model trained: RMSE={rmse:.6f}, MAE={mae:.6f}")
            return model, scaler, metrics

        except Exception as e:
            raise OnDemandTrainingError(f"Model training failed: {str(e)}")

    def _save_artifacts(
        self, ticker: str, model: MLPRegressor, scaler: StandardScaler, feature_names: list, metrics: dict
    ) -> tuple:
        """Save trained model and artifacts to disk."""
        try:
            timestamp = datetime.now().isoformat()

            # Save model
            model_file = self.models_dir / f"mlp_model_{ticker}.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(model, f)

            # Save scaler
            scaler_file = self.models_dir / f"mlp_scaler_{ticker}.pkl"
            with open(scaler_file, "wb") as f:
                pickle.dump(scaler, f)

            # Save metadata
            metadata = {
                "ticker": ticker,
                "timestamp": timestamp,
                "model_type": "MLP",
                "architecture": self.model_config["hidden_layer_sizes"],
                "features_count": len(feature_names),
                "feature_names": feature_names,
                "metrics": metrics,
                "config": {k: v for k, v in self.model_config.items() if k not in ["random_state"]},
            }

            metadata_file = self.metadata_dir / f"mlp_model_stats_{ticker}.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Artifacts saved: {model_file}, {scaler_file}, {metadata_file}")
            return model_file, scaler_file, metadata_file

        except Exception as e:
            raise OnDemandTrainingError(f"Failed to save artifacts: {str(e)}")

    def _predict_next(self, model: MLPRegressor, scaler: StandardScaler, last_row: pd.Series, ticker: str) -> dict:
        """Generate prediction for next trading day."""
        try:
            # Prepare features from last row
            exclude_cols = ["date", "open", "high", "low", "close", "volume", "returns", "direction", "target_return"]
            feature_cols = [col for col in last_row.index if col not in exclude_cols]
            X_last = last_row[feature_cols].values.reshape(1, -1)

            # Scale and predict
            X_last_scaled = scaler.transform(X_last)
            predicted_return = float(model.predict(X_last_scaled)[0])

            # Determine direction
            direction = "bullish" if predicted_return > 0 else "bearish" if predicted_return < 0 else "neutral"

            # Estimate confidence
            confidence = min(0.55 + abs(predicted_return) * 25, 0.95)

            return {
                "ticker": ticker,
                "predicted_return": round(predicted_return, 6),
                "predicted_return_pct": round(predicted_return * 100, 2),
                "direction": direction,
                "confidence": round(confidence, 4),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise OnDemandTrainingError(f"Prediction failed: {str(e)}")

    @staticmethod
    def _calc_rsi(close, period=14):
        """Calculate RSI indicator."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _calc_atr(df, period=14):
        """Calculate ATR indicator."""
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - df["close"].shift())
        tr3 = abs(df["low"] - df["close"].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
