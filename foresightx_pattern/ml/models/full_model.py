from __future__ import annotations

import torch
from torch import nn

from foresightx_pattern.ml.models.backbone import SequenceBackbone
from foresightx_pattern.ml.models.embedding import StockEmbedding


class ForesightXPatternModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_stocks: int,
        projection_dim: int = 64,
        hidden_dim: int = 128,
        embedding_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.2,
        encoder_type: str = "lstm",
    ) -> None:
        super().__init__()
        self.backbone = SequenceBackbone(
            input_dim=input_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            encoder_type=encoder_type,
        )
        self.embedding = StockEmbedding(num_stocks=num_stocks, embedding_dim=embedding_dim)
        fused_dim = hidden_dim + embedding_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, x: torch.Tensor, stock_ids: torch.Tensor) -> torch.Tensor:
        latent = self.backbone(x)
        stock_embedding = self.embedding(stock_ids)
        fused = torch.cat([latent, stock_embedding], dim=-1)
        return self.head(fused)
