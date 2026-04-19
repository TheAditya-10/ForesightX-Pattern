from __future__ import annotations

import torch

from foresightx_pattern.ml.models.full_model import ForesightXPatternModel


def test_model_forward_returns_three_step_forecast():
    model = ForesightXPatternModel(input_dim=24, num_stocks=4, embedding_dim=16)
    x = torch.randn(8, 48, 24)
    stock_ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    output = model(x, stock_ids)
    assert output.shape == (8, 3)
