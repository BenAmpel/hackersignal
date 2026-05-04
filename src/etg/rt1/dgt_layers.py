"""Core DGT layers:
- GTMHMSALayer  : Graph Transformer with Multi-Head Masked Self-Attention.
- TemporalAttention : per-spell embedding projection with attention across
                      previous W spells ("Embedding Projection with Temporal
                      Attention" in the proposal).

All layers are pure PyTorch so they run on CUDA / MPS / CPU identically.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class GTMHMSALayer(nn.Module):
    """Multi-head masked self-attention over graph nodes.

    Attention logits: A_r = (QK^T)/sqrt(d_k) + additive_mask
    additive_mask is 0 on edges in the k-hop neighborhood and -inf elsewhere,
    so each node only attends to (masked) neighbors.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # Two-layer FFN per the proposal: W1,b1 then W2,b2 with GELU in between.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, additive_mask: Tensor) -> Tensor:
        """
        x            : [V, d_model]
        additive_mask: [V, V] with 0 on attended edges and -inf elsewhere
        """
        V = x.shape[0]
        Q = self.W_q(x).view(V, self.num_heads, self.d_k).transpose(0, 1)  # [H, V, d_k]
        K = self.W_k(x).view(V, self.num_heads, self.d_k).transpose(0, 1)
        Vv = self.W_v(x).view(V, self.num_heads, self.d_k).transpose(0, 1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [H, V, V]
        scores = scores + additive_mask.unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, Vv)  # [H, V, d_k]
        ctx = ctx.transpose(0, 1).contiguous().view(V, self.d_model)  # [V, d_model]
        out = self.W_o(ctx)
        h = self.ln1(x + self.dropout(out))
        h = self.ln2(h + self.dropout(self.ffn(h)))
        return h


class TemporalAttention(nn.Module):
    """Attention across the last W time-spells to refine the current embedding.

    Given [H_{t-W+1}, ..., H_t], each of shape [V, d], compute a temporal
    attention for each node across spells and produce a projected embedding
    Emb_{v,t} = ReLU(W_p · sum_k α_{t,k} · H_k).
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, history: list[Tensor]) -> Tensor:
        """
        history: list of [V, d] tensors, newest last (length K<=W).
        returns: [V, d] projected embedding at the newest spell.
        """
        stack = torch.stack(history, dim=1)  # [V, K, d]
        V, K, d = stack.shape
        q = self.query_proj(stack[:, -1:, :])  # [V, 1, d] — current spell is query
        k = self.key_proj(stack)
        v = self.value_proj(stack)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)  # [V, 1, K]
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, v).squeeze(1)  # [V, d]
        out = self.out_proj(ctx)
        return self.act(out + stack[:, -1, :])  # residual on current spell
