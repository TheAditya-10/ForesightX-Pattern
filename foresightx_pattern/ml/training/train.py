from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from foresightx_pattern.ml.data.ingestion import fetch_market_data
from foresightx_pattern.ml.data.preprocessing import clean_market_data
from foresightx_pattern.ml.evaluation.evaluate import evaluate_regression_model
from foresightx_pattern.ml.features.engineering import build_feature_frame
from foresightx_pattern.ml.features.sequences import build_sequence_dataset
from foresightx_pattern.ml.models.full_model import ForesightXPatternModel
from foresightx_pattern.ml.utils.config import load_settings
from foresightx_pattern.ml.utils.logging import configure_logging
from foresightx_pattern.ml.utils.mlflow_utils import log_run_artifacts, mark_model_production


LOGGER = logging.getLogger(__name__)


def train_pipeline() -> dict[str, float | str]:
    settings = load_settings()
    configure_logging(settings.service.get("log_level", "INFO"))
    raw = fetch_market_data(settings.tickers, settings)
    clean = clean_market_data(raw, settings)
    features = build_feature_frame(clean, settings)
    raw.to_parquet(settings.raw_data_path, index=False)
    clean.to_parquet(settings.processed_data_path, index=False)
    features.to_parquet(settings.features_path, index=False)
    bundle = build_sequence_dataset(features, settings)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(bundle.X_train),
            torch.from_numpy(bundle.stock_train),
            torch.from_numpy(bundle.y_train),
        ),
        batch_size=settings.training.get("batch_size", 64),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(bundle.X_val),
            torch.from_numpy(bundle.stock_val),
            torch.from_numpy(bundle.y_val),
        ),
        batch_size=settings.training.get("batch_size", 64),
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForesightXPatternModel(
        input_dim=bundle.X_train.shape[-1],
        num_stocks=len(bundle.stock_to_id),
        projection_dim=settings.model.get("projection_dim", 64),
        hidden_dim=settings.model.get("hidden_dim", 128),
        embedding_dim=settings.model.get("embedding_dim", 16),
        num_layers=settings.model.get("num_layers", 2),
        dropout=settings.model.get("dropout", 0.2),
        encoder_type=settings.model.get("encoder_type", "lstm"),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.training.get("learning_rate", 1e-3))
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    patience = settings.training.get("early_stopping_patience", 5)
    wait = 0

    for epoch in range(settings.training.get("epochs", 10)):
        model.train()
        train_losses: list[float] = []
        for x_batch, stock_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            stock_batch = stock_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(x_batch, stock_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for x_batch, stock_batch, y_batch in val_loader:
                predictions = model(x_batch.to(device), stock_batch.to(device))
                loss = loss_fn(predictions, y_batch.to(device))
                val_losses.append(loss.item())
        mean_val = float(np.mean(val_losses))
        LOGGER.info("Epoch %s train_loss=%.5f val_loss=%.5f", epoch + 1, np.mean(train_losses), mean_val)
        if mean_val < best_val:
            best_val = mean_val
            wait = 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                LOGGER.info("Early stopping triggered at epoch %s", epoch + 1)
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid checkpoint")
    model.load_state_dict(best_state)

    metrics = evaluate_regression_model(model, val_loader, device)
    _persist_artifacts(model, bundle, metrics, settings)
    return metrics


def _persist_artifacts(model, bundle, metrics: dict[str, float], settings) -> None:
    model_dir = settings.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pt"
    onnx_path = model_dir / "model.onnx"
    scaler_path = model_dir / "scaler.pkl"
    metadata_path = model_dir / "metadata.json"
    report_path = settings.reports_dir / "evaluation.json"
    mlflow_model_dir = model_dir / "model_artifact"
    mlflow_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    with scaler_path.open("wb") as handle:
        pickle.dump(bundle.scaler, handle)
    metadata = {
        "model_name": "foresightx_pattern_foundation",
        "model_version": "local",
        "sequence_length": settings.data.get("sequence_length", 48),
        "horizon": settings.data.get("horizon", 3),
        "feature_names": bundle.feature_names,
        "stock_to_id": bundle.stock_to_id,
        "input_dim": len(bundle.feature_names),
        "num_stocks": len(bundle.stock_to_id),
        "projection_dim": settings.model.get("projection_dim", 64),
        "hidden_dim": settings.model.get("hidden_dim", 128),
        "embedding_dim": settings.model.get("embedding_dim", 16),
        "num_layers": settings.model.get("num_layers", 2),
        "dropout": settings.model.get("dropout", 0.2),
        "encoder_type": settings.model.get("encoder_type", "lstm"),
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), mlflow_model_dir / "model.pt")
    _export_onnx_model(model, onnx_path, metadata)
    _export_onnx_model(model, mlflow_model_dir / "model.onnx", metadata)
    (mlflow_model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    run_id = log_run_artifacts(
        settings=settings,
        run_name="foundation-training",
        params={
            "tickers": ",".join(settings.tickers),
            "sequence_length": settings.data.get("sequence_length", 48),
            "embedding_dim": settings.model.get("embedding_dim", 16),
            "epochs": settings.training.get("epochs", 10),
        },
        metrics=metrics,
        artifact_paths=[
            settings.raw_data_path,
            settings.processed_data_path,
            settings.features_path,
            report_path,
            scaler_path,
            metadata_path,
            onnx_path,
            mlflow_model_dir,
        ],
    )
    version = mark_model_production(settings, metadata["model_name"], run_id, model_dir)
    metadata["model_version"] = str(version)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _export_onnx_model(model, output_path: Path, metadata: dict) -> None:
    model.eval()
    sequence_length = int(metadata["sequence_length"])
    input_dim = int(metadata["input_dim"])
    dummy_sequence = torch.zeros(1, sequence_length, input_dim, dtype=torch.float32)
    dummy_stock_id = torch.zeros(1, dtype=torch.long)
    torch.onnx.export(
        model.cpu(),
        (dummy_sequence, dummy_stock_id),
        output_path,
        input_names=["sequence", "stock_id"],
        output_names=["predictions"],
        dynamic_axes={
            "sequence": {0: "batch"},
            "stock_id": {0: "batch"},
            "predictions": {0: "batch"},
        },
        opset_version=17,
    )


if __name__ == "__main__":
    train_pipeline()
