from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_regression_model(model, data_loader, device) -> dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, stock_batch, y_batch in data_loader:
            outputs = model(x_batch.to(device), stock_batch.to(device)).cpu().numpy()
            preds.append(outputs)
            targets.append(y_batch.numpy())
    y_true = np.concatenate(targets, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)))),
        "mae": float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))),
    }
