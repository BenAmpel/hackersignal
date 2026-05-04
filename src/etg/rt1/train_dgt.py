"""Train the DGT.

Loss = edge-reconstruction BPR (positive edges vs sampled negatives at the
current spell) + λ · temporal-consistency loss on embeddings across
adjacent spells, weighted by per-word frequency.

Only words whose frequency is > 0 in the current spell contribute to the
reconstruction loss (the `node_mask`).
"""

from __future__ import annotations

import random

import torch
from torch import Tensor, nn
from torch.optim import Adam

from ..config import Config
from ..logging_utils import get_logger
from .dgt_model import DGT
from .etg_builder import ETGSnapshot


def _bpr_loss(pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
    """-log σ(pos - neg), averaged."""
    return -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()


def _temporal_loss(h_prev: Tensor, h_curr: Tensor, freq_prev: Tensor, freq_curr: Tensor) -> Tensor:
    """L2 drift weighted by min(freq_prev, freq_curr)."""
    w = torch.minimum(freq_prev.float(), freq_curr.float())
    if w.sum().item() == 0:
        return torch.tensor(0.0, device=h_prev.device)
    w = w / (w.max() + 1e-12)
    diff = (h_prev - h_curr).pow(2).sum(dim=-1)  # [V]
    return (w * diff).mean()


def _sample_negative_edges(V: int, pos_edges: Tensor, n_samples: int, rng: random.Random) -> Tensor:
    """Sample `n_samples` negative edges not in pos_edges (best-effort)."""
    pos_set = set()
    # Take a bounded snapshot to keep negative sampling O(E) worst case.
    for src, dst in zip(pos_edges[0].tolist()[:4096], pos_edges[1].tolist()[:4096]):
        pos_set.add((src, dst))
    src_list = []
    dst_list = []
    attempts = 0
    while len(src_list) < n_samples and attempts < n_samples * 20:
        s = rng.randrange(V)
        d = rng.randrange(V)
        if s != d and (s, d) not in pos_set:
            src_list.append(s)
            dst_list.append(d)
        attempts += 1
    if not src_list:
        # Fallback: random even if overlapping (tiny graph case).
        src_list = [rng.randrange(V) for _ in range(n_samples)]
        dst_list = [rng.randrange(V) for _ in range(n_samples)]
    return torch.tensor([src_list, dst_list], dtype=torch.long)


def train_dgt(
    model: DGT,
    snapshots: list[ETGSnapshot],
    config: Config,
    device: torch.device,
) -> dict[str, list[float]]:
    logger = get_logger("etg.train_dgt")
    rng = random.Random(config.seed)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=config.dgt_lr)
    history = {"loss": [], "recon": [], "temporal": []}
    V = snapshots[0]["x"].shape[0]

    for epoch in range(config.dgt_epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model(snapshots, device=device, train=True)
        recon_total = torch.tensor(0.0, device=device)
        temp_total = torch.tensor(0.0, device=device)
        for t, (snap, h) in enumerate(zip(snapshots, embeddings)):
            pos_ei = snap["edge_index"].to(device)
            # Subsample positive edges if the spell has many of them.
            if pos_ei.shape[1] > config.dgt_batch_size_edges:
                idx = torch.randperm(pos_ei.shape[1])[: config.dgt_batch_size_edges]
                pos_ei_batch = pos_ei[:, idx]
            else:
                pos_ei_batch = pos_ei
            neg_ei = _sample_negative_edges(
                V,
                pos_ei_batch,
                pos_ei_batch.shape[1] * config.dgt_recon_neg_samples,
                rng,
            ).to(device)
            pos_scores = model.score_edges(h, pos_ei_batch)
            neg_scores = model.score_edges(h, neg_ei)
            # Expand pos_scores to match neg_scores (negatives per positive).
            K = config.dgt_recon_neg_samples
            pos_expanded = pos_scores.unsqueeze(1).expand(-1, K).reshape(-1)
            recon_total = recon_total + _bpr_loss(pos_expanded, neg_scores)
            if t > 0:
                temp_total = temp_total + _temporal_loss(
                    embeddings[t - 1],
                    h,
                    snap_freq_prev := snapshots[t - 1]["freq"].to(device),
                    snap["freq"].to(device),
                )
                _ = snap_freq_prev  # silence linter
        recon_total = recon_total / len(snapshots)
        temp_total = temp_total / max(1, len(snapshots) - 1)
        loss = recon_total + config.dgt_temporal_lambda * temp_total
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        history["loss"].append(float(loss.item()))
        history["recon"].append(float(recon_total.item()))
        history["temporal"].append(float(temp_total.item()))
        logger.info(
            f"DGT epoch {epoch + 1}/{config.dgt_epochs} "
            f"loss={loss.item():.4f} recon={recon_total.item():.4f} temp={temp_total.item():.4f}"
        )
    return history


@torch.no_grad()
def extract_embeddings(
    model: DGT, snapshots: list[ETGSnapshot], device: torch.device
) -> list[Tensor]:
    """Return embeddings per spell in eval mode (no sign flip)."""
    model.eval()
    return [h.detach().cpu() for h in model(snapshots, device=device, train=False)]
