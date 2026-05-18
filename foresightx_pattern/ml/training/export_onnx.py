from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from foresightx_pattern.ml.models.full_model import ForesightXPatternModel
from foresightx_pattern.ml.training.train import _export_onnx_model
from foresightx_pattern.ml.utils.config import load_settings


def export_existing_model(model_dir: Path | None = None) -> Path:
    settings = load_settings()
    target_dir = model_dir or settings.model_dir
    metadata = json.loads((target_dir / "metadata.json").read_text(encoding="utf-8"))
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
    state = torch.load(target_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    output_path = target_dir / "model.onnx"
    _export_onnx_model(model, output_path, metadata)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an existing ForesightX Pattern model.pt to model.onnx.")
    parser.add_argument("--model-dir", type=Path, default=None, help="Directory containing model.pt and metadata.json.")
    args = parser.parse_args()
    output_path = export_existing_model(args.model_dir)
    print(output_path)


if __name__ == "__main__":
    main()
