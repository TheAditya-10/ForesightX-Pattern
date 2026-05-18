from __future__ import annotations

import numpy as np


class ConfidenceService:
    def __init__(self, interval_multiplier: float = 1.96) -> None:
        self.interval_multiplier = interval_multiplier

    def predict_with_confidence(
        self,
        model,
        sequence: np.ndarray,
        stock_id: int,
        metrics: dict | None = None,
    ) -> tuple[list[float], float, list[list[float]]]:
        mean = np.asarray(model.predict(sequence, stock_id), dtype=np.float32)
        std_value = float((metrics or {}).get("rmse") or (metrics or {}).get("mae") or 0.0)
        std = np.full_like(mean, std_value, dtype=np.float32)
        intervals = np.stack(
            [mean - self.interval_multiplier * std, mean + self.interval_multiplier * std],
            axis=1,
        )
        confidence = float(np.exp(-np.mean(std / np.maximum(np.abs(mean), 1.0))))
        return mean.tolist(), confidence, intervals.tolist()
