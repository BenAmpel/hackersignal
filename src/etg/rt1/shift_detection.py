"""Cosine-distance-based semantic shift detection.

For each word present in spells t-1 and t, compute
    d_v^t = 1 - cos(z_v^{t-1}, z_v^t)

Shifted words are those with d > per-pair quantile threshold. Optional
permutation test assigns an approximate p-value under a null of no
meaningful drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class ShiftResult:
    t_prev: int
    t_curr: int
    word_ids: np.ndarray    # node ids present in both spells
    scores: np.ndarray      # cosine distance per word_id
    threshold: float
    shifted_ids: np.ndarray # subset of word_ids above threshold


def _cosine_distance(a: Tensor, b: Tensor) -> Tensor:
    a_n = torch.nn.functional.normalize(a, dim=-1)
    b_n = torch.nn.functional.normalize(b, dim=-1)
    return 1.0 - (a_n * b_n).sum(dim=-1)


def detect_shifts_pairwise(
    embeddings: list[Tensor],
    snapshots_node_masks: list[Tensor],
    top_quantile: float = 0.05,
) -> list[ShiftResult]:
    """Return one ShiftResult per adjacent (t-1, t) pair."""
    results: list[ShiftResult] = []
    for t in range(1, len(embeddings)):
        m_prev = snapshots_node_masks[t - 1]
        m_curr = snapshots_node_masks[t]
        present = (m_prev & m_curr).cpu().numpy()
        ids = np.where(present)[0]
        if len(ids) == 0:
            results.append(
                ShiftResult(
                    t_prev=t - 1,
                    t_curr=t,
                    word_ids=ids,
                    scores=np.array([]),
                    threshold=0.0,
                    shifted_ids=np.array([], dtype=np.int64),
                )
            )
            continue
        z_prev = embeddings[t - 1][ids]
        z_curr = embeddings[t][ids]
        d = _cosine_distance(z_prev, z_curr).cpu().numpy()
        threshold = float(np.quantile(d, 1.0 - top_quantile))
        shifted = ids[d >= threshold]
        results.append(
            ShiftResult(
                t_prev=t - 1,
                t_curr=t,
                word_ids=ids,
                scores=d,
                threshold=threshold,
                shifted_ids=shifted,
            )
        )
    return results


def permutation_pvalues(
    embeddings: list[Tensor],
    snapshots_node_masks: list[Tensor],
    t_prev: int,
    t_curr: int,
    n_trials: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Monte Carlo permutation p-value per present word.

    The null: the spell label (prev vs curr) is uninformative for that word
    — we resample which of (z_{t-1}, z_t) is "current" and recompute the
    cosine distance between shuffled pairs drawn from the marginal. A
    p-value is the fraction of trials with drift >= observed drift.
    """
    rng = np.random.default_rng(seed)
    m_prev = snapshots_node_masks[t_prev].cpu().numpy()
    m_curr = snapshots_node_masks[t_curr].cpu().numpy()
    ids = np.where(m_prev & m_curr)[0]
    if len(ids) == 0:
        return np.array([])
    z_prev = embeddings[t_prev][ids].cpu().numpy()
    z_curr = embeddings[t_curr][ids].cpu().numpy()
    # Observed drift
    obs = 1 - (z_prev * z_curr).sum(axis=-1) / (
        np.linalg.norm(z_prev, axis=-1) * np.linalg.norm(z_curr, axis=-1) + 1e-12
    )
    counts = np.zeros(len(ids), dtype=np.int64)
    for _ in range(n_trials):
        shuffled_idx = rng.permutation(len(ids))
        z_shuf = z_curr[shuffled_idx]
        null = 1 - (z_prev * z_shuf).sum(axis=-1) / (
            np.linalg.norm(z_prev, axis=-1) * np.linalg.norm(z_shuf, axis=-1) + 1e-12
        )
        counts += null >= obs
    return (counts + 1) / (n_trials + 1)
