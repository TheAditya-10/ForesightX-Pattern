from __future__ import annotations

from torch import nn


class StockEmbedding(nn.Module):
    def __init__(self, num_stocks: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_stocks, embedding_dim=embedding_dim)

    def forward(self, stock_ids):
        return self.embedding(stock_ids)
