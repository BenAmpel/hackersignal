"""Pure-numpy information-retrieval metrics. Validated against sklearn."""

from __future__ import annotations

import numpy as np


def hit_rate_at_k(ranks: np.ndarray, k: int) -> float:
    """Given 1-indexed ranks of the ground-truth item per query, return HR@k."""
    return float(np.mean(ranks <= k))


def mrr_at_k(ranks: np.ndarray, k: int) -> float:
    """Mean reciprocal rank, truncated at k. Ranks > k contribute 0."""
    rr = np.where(ranks <= k, 1.0 / ranks, 0.0)
    return float(np.mean(rr))


def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    """nDCG@k with binary relevance (single-positive-per-query setting).

    DCG_q = 1 / log2(rank + 1) if rank <= k else 0
    IDCG  = 1 / log2(2)        (the positive is always achievable at rank 1)
    """
    dcg = np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)
    idcg = 1.0 / np.log2(np.array(2.0))
    return float(np.mean(dcg) / idcg)


def mean_average_precision(ranks: np.ndarray) -> float:
    """With a single positive per query, MAP = mean(1 / rank)."""
    return float(np.mean(1.0 / np.asarray(ranks, dtype=np.float64)))


def ranks_from_scores(
    scores: np.ndarray, positive_indices: np.ndarray
) -> np.ndarray:
    """
    scores:            [Q, C] — similarity score per candidate per query.
    positive_indices:  [Q]     — index into the candidate dimension of the
                                 positive for each query.
    Returns 1-indexed ranks (int) per query.
    """
    order = np.argsort(-scores, axis=1)  # descending
    ranks = np.empty(scores.shape[0], dtype=np.int64)
    for q in range(scores.shape[0]):
        pos = int(positive_indices[q])
        # np.where gives 0-indexed position; add 1.
        ranks[q] = int(np.where(order[q] == pos)[0][0]) + 1
    return ranks


def all_ranking_metrics(ranks: np.ndarray, k: int = 10) -> dict[str, float]:
    return {
        f"HR@{k}": hit_rate_at_k(ranks, k),
        f"MRR@{k}": mrr_at_k(ranks, k),
        f"NDCG@{k}": ndcg_at_k(ranks, k),
        "MAP": mean_average_precision(ranks),
    }
