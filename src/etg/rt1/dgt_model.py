"""Full Diachronic Graph Transformer (DGT).

Inputs per spell:  ETGSnapshot
Output per spell:  per-node embedding [V, d_model]

The model keeps a running history of the last W spells' embeddings and,
at each spell, projects the current encoding through TemporalAttention
to produce the final embedding that gets compared across time.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import Config
from .dgt_layers import GTMHMSALayer, TemporalAttention
from .etg_builder import ETGSnapshot
from .laplacian_pe import random_sign_flip


def _build_adjacency_mask(
    edge_index: Tensor, V: int, k_hop: int, device: torch.device
) -> Tensor:
    """[V, V] additive mask: 0 on k-hop neighbors, -inf elsewhere.

    Self-loops are always unmasked.
    """
    A = torch.zeros((V, V), dtype=torch.bool, device=device)
    A[edge_index[0], edge_index[1]] = True
    A[edge_index[1], edge_index[0]] = True  # symmetric reachability
    A.fill_diagonal_(True)
    if k_hop > 1:
        reach = A.clone()
        power = A.clone()
        for _ in range(k_hop - 1):
            power = power.float() @ A.float()
            power = power > 0
            reach = reach | power
        A = reach
    mask = torch.zeros((V, V), device=device, dtype=torch.float32)
    mask.masked_fill_(~A, float("-inf"))
    return mask


class DGT(nn.Module):
    def __init__(self, config: Config, feature_dim: int):
        super().__init__()
        self.config = config
        d = config.dgt_hidden_dim
        # Input projection: concat(node features, LapPE) → d_model
        self.input_proj = nn.Linear(feature_dim + config.lap_pe_k, d)
        self.layers = nn.ModuleList(
            [
                GTMHMSALayer(d, config.dgt_num_heads, config.dgt_dropout)
                for _ in range(config.dgt_num_layers)
            ]
        )
        self.temporal = TemporalAttention(d, config.dgt_dropout)
        self.W = config.dgt_temporal_window
        # Classification head used only at training time (edge-reconstruction).
        self.edge_score = nn.Bilinear(d, d, 1)

    def encode_spell(
        self, snapshot: ETGSnapshot, train: bool, device: torch.device
    ) -> Tensor:
        x = snapshot["x"].to(device)
        pe = snapshot["lap_pe"].to(device)
        if train:
            pe = random_sign_flip(pe)
        h = self.input_proj(torch.cat([x, pe], dim=-1))
        V = x.shape[0]
        mask = _build_adjacency_mask(
            snapshot["edge_index"].to(device), V, self.config.dgt_k_hop, device
        )
        for layer in self.layers:
            h = layer(h, mask)
        return h  # [V, d]

    def forward(
        self,
        snapshots: list[ETGSnapshot],
        device: torch.device,
        train: bool = True,
    ) -> list[Tensor]:
        """Run the encoder across all spells with temporal attention.

        Returns one [V, d] embedding per spell.
        """
        raw_list: list[Tensor] = []
        out_list: list[Tensor] = []
        for snap in snapshots:
            h = self.encode_spell(snap, train=train, device=device)
            raw_list.append(h)
            history = raw_list[-self.W :]
            out_list.append(self.temporal(history))
        return out_list

    # ---- edge reconstruction helpers (used by train_dgt) ----

    def score_edges(self, h: Tensor, edge_index: Tensor) -> Tensor:
        src = h[edge_index[0]]
        dst = h[edge_index[1]]
        return self.edge_score(src, dst).squeeze(-1)
