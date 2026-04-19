from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import torch

from foresightx_pattern.ml.models.full_model import ForesightXPatternModel
from foresightx_pattern.ml.utils.config import AppSettings


@dataclass(slots=True)
class LoadedModelBundle:
    model: ForesightXPatternModel
    scaler: object
    metadata: dict


class ModelLoader:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._cached: LoadedModelBundle | None = None

    def load(self) -> LoadedModelBundle:
        if self._cached is not None:
            return self._cached
        model_dir = self.settings.model_dir
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        with (model_dir / "scaler.pkl").open("rb") as handle:
            scaler = pickle.load(handle)
        model = ForesightXPatternModel(
            input_dim=metadata["input_dim"],
            num_stocks=metadata["num_stocks"],
            projection_dim=metadata["projection_dim"],
            hidden_dim=metadata["hidden_dim"],
            embedding_dim=metadata["embedding_dim"],
            num_layers=metadata["num_layers"],
            dropout=metadata["dropout"],
            encoder_type=metadata["encoder_type"],
        )
        state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        self._cached = LoadedModelBundle(model=model, scaler=scaler, metadata=metadata)
        return self._cached
