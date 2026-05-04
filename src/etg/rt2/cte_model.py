"""Context Transformer Encoder (CTE).

Stages per the proposal (Figure 4):
  embedding → BiLSTM → 1D-CNN bank (kernels 3,4,5) → Local-guided Global
  Context Attention → sentence representation.

Also used by:
  - Masked Prediction pretext head (per-token cross-entropy over buckets).
  - Projection head for contrastive pretraining.
  - Sentence representation for supervised triplet fine-tuning.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import Config
from .cte_heads import MLMHead, ProjectionHead
from .trigram_hasher import PAD_ID


class LocalGlobalContextAttention(nn.Module):
    """Per-position attention guided by a global pooled representation.

    A(h^L_p, h^g) = softmax((h^L_p W h^g^T) / sqrt(d))
    context     c = sum_p A_p · h^L_p
    refined   r_p = h^L_p + c
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_model, d_model) * (1.0 / math.sqrt(d_model)))

    def forward(self, h_local: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        h_local: [B, L, d]
        mask   : [B, L]  bool (True where valid, False for padding)
        returns (refined [B, L, d], pooled [B, d])
        """
        # Build a global query from masked mean.
        mask_f = mask.float().unsqueeze(-1)
        pooled = (h_local * mask_f).sum(dim=1) / (mask_f.sum(dim=1).clamp(min=1.0))  # [B, d]
        # Compute scores: (h^L W h^g)
        Wh_g = pooled @ self.W  # [B, d]
        scores = torch.einsum("bld,bd->bl", h_local, Wh_g) / math.sqrt(h_local.shape[-1])
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)  # [B, L]
        context = torch.einsum("bl,bld->bd", attn, h_local)  # [B, d]
        refined = h_local + context.unsqueeze(1)
        # Attention-pooled sentence representation.
        sent = context
        return refined, sent


class CTE(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.trigram_buckets, config.cte_emb_dim, padding_idx=PAD_ID
        )
        self.lstm = nn.LSTM(
            input_size=config.cte_emb_dim,
            hidden_size=config.cte_lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        lstm_out = 2 * config.cte_lstm_hidden
        # Parallel 1D convolution bank (same padding so L is preserved).
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=lstm_out,
                    out_channels=config.cte_cnn_channels,
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in config.cte_cnn_kernels
            ]
        )
        d_local = config.cte_cnn_channels * len(config.cte_cnn_kernels)
        self.local_proj = nn.Linear(d_local, config.cte_attn_dim)
        self.lg_attn = LocalGlobalContextAttention(config.cte_attn_dim)
        self.dropout = nn.Dropout(config.cte_dropout)
        # Heads
        self.mlm_head = MLMHead(config.cte_attn_dim, config.trigram_buckets)
        self.proj_head = ProjectionHead(config.cte_attn_dim, config.cte_attn_dim)

    def encode_tokens(self, input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        input_ids: [B, L]
        returns: (refined [B, L, d_attn], sentence [B, d_attn], mask [B, L])
        """
        mask = input_ids != PAD_ID
        x = self.embedding(input_ids)  # [B, L, e]
        lstm_out, _ = self.lstm(x)       # [B, L, 2H]
        # 1D-CNN needs [B, C, L]
        conv_in = lstm_out.transpose(1, 2)  # [B, 2H, L]
        conv_outs = []
        for conv in self.convs:
            out = conv(conv_in)  # [B, C_k, L']  — may drop 1 position for even kernels
            if out.shape[-1] > conv_in.shape[-1]:
                out = out[..., : conv_in.shape[-1]]
            elif out.shape[-1] < conv_in.shape[-1]:
                pad = torch.zeros(
                    out.shape[0],
                    out.shape[1],
                    conv_in.shape[-1] - out.shape[-1],
                    device=out.device,
                    dtype=out.dtype,
                )
                out = torch.cat([out, pad], dim=-1)
            conv_outs.append(torch.relu(out))
        local = torch.cat(conv_outs, dim=1)  # [B, d_local, L]
        local = local.transpose(1, 2)         # [B, L, d_local]
        local = self.local_proj(local)
        local = self.dropout(local)
        refined, sent = self.lg_attn(local, mask)
        return refined, sent, mask

    def forward_mlm(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        refined, sent, mask = self.encode_tokens(input_ids)
        logits = self.mlm_head(refined)
        return logits, mask

    def forward_contrastive(self, input_ids: Tensor) -> Tensor:
        _, sent, _ = self.encode_tokens(input_ids)
        return self.proj_head(sent)

    def encode_sentence(self, input_ids: Tensor) -> Tensor:
        _, sent, _ = self.encode_tokens(input_ids)
        return sent
