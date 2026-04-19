from __future__ import annotations

import torch
from torch import nn


class SequenceBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        encoder_type: str = "lstm",
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, projection_dim)
        encoder_cls = nn.LSTM if encoder_type.lower() == "lstm" else nn.GRU
        encoder_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = encoder_cls(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=encoder_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        outputs, hidden = self.encoder(projected)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return hidden[-1]
