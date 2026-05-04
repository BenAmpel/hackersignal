"""Heads attached to the CTE backbone.

MLMHead        — per-position classifier over the hash-bucket vocabulary,
                 reduced to a small LM-head dimension by a tied linear
                 layer; for memory reasons the output vocabulary is the
                 full bucket count but the projection is lightweight.
ProjectionHead — MLP projection for contrastive pretraining (SimCSE-style).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MLMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, h: Tensor) -> Tensor:
        h = torch.nn.functional.gelu(self.fc1(h))
        h = self.ln(h)
        return self.decoder(h)


class ProjectionHead(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Linear(d_in, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
