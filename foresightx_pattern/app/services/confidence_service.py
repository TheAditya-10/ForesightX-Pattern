from __future__ import annotations

import numpy as np
import torch


class ConfidenceService:
    def __init__(self, samples: int = 20) -> None:
        self.samples = samples

    def predict_with_confidence(self, model, sequence: np.ndarray, stock_id: int) -> tuple[list[float], float, list[list[float]]]:
        tensor = torch.tensor(sequence[None, :, :], dtype=torch.float32)
        stock_tensor = torch.tensor([stock_id], dtype=torch.long)
        outputs: list[np.ndarray] = []
        model.train()
        with torch.no_grad():
            for _ in range(self.samples):
                outputs.append(model(tensor, stock_tensor).cpu().numpy()[0])
        model.eval()
        stacked = np.stack(outputs, axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        intervals = np.stack([mean - 1.96 * std, mean + 1.96 * std], axis=1)
        confidence = float(np.exp(-np.mean(std / np.maximum(np.abs(mean), 1.0))))
        return mean.tolist(), confidence, intervals.tolist()
