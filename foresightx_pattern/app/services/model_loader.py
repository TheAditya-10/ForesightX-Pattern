from __future__ import annotations

import json
import pickle
from dataclasses import dataclass

import onnxruntime as ort

from foresightx_pattern.ml.utils.config import AppSettings


@dataclass(slots=True)
class LoadedModelBundle:
    model: "OnnxModelSession"
    scaler: object
    metadata: dict


class OnnxModelSession:
    def __init__(self, model_path: str) -> None:
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.sequence_input = self.session.get_inputs()[0].name
        self.stock_input = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, sequence, stock_id: int):
        import numpy as np

        inputs = {
            self.sequence_input: sequence[None, :, :].astype(np.float32, copy=False),
            self.stock_input: np.array([stock_id], dtype=np.int64),
        }
        return self.session.run([self.output_name], inputs)[0][0]


class ModelLoader:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._cached: LoadedModelBundle | None = None

    def load(self) -> LoadedModelBundle:
        if self._cached is not None:
            return self._cached
        model_dir = self.settings.model_dir
        required_files = [
            model_dir / "metadata.json",
            model_dir / "scaler.pkl",
            model_dir / "model.onnx",
        ]
        missing = [path.name for path in required_files if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Pattern model artifacts are missing from "
                f"{model_dir}. Missing: {', '.join(missing)}. "
                "Provide the trained bundle via DVC/MLflow deployment artifacts or set "
                "FORESIGHTX_ARTIFACTS_DIR to a directory containing model.onnx, scaler.pkl, and metadata.json."
            )
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        with (model_dir / "scaler.pkl").open("rb") as handle:
            scaler = pickle.load(handle)
        model = OnnxModelSession(str(model_dir / "model.onnx"))
        self._cached = LoadedModelBundle(model=model, scaler=scaler, metadata=metadata)
        return self._cached
