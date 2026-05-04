"""RT2 evaluation — ranking quality for exploit-vulnerability linking."""

from __future__ import annotations

import numpy as np

from ..eval.metrics_ir import all_ranking_metrics, ranks_from_scores
from ..eval.significance import per_query_ndcg


def evaluate_ranking(
    scores: np.ndarray,
    positive_indices: np.ndarray,
    k: int = 10,
    return_per_query: bool = False,
) -> "dict[str, float] | tuple[dict[str, float], np.ndarray]":
    """Compute ranking metrics for a score matrix.

    Parameters
    ----------
    scores :
        Shape (n_queries, n_candidates) — similarity score per candidate.
    positive_indices :
        Shape (n_queries,) — index of the positive candidate per query.
    k :
        Truncation depth for HR@k, MRR@k, NDCG@k.
    return_per_query :
        When True, also return a numpy array of per-query NDCG@k values
        of shape (n_queries,).

    Returns
    -------
    metrics : dict[str, float]
        Aggregated metrics dict (HR@k, MRR@k, NDCG@k, MAP, median_rank, mean_rank).
    per_query_scores : np.ndarray, optional
        Per-query NDCG@k array of shape (n_queries,).
        Only returned when *return_per_query=True*.
    """
    ranks = ranks_from_scores(scores, positive_indices)
    metrics = all_ranking_metrics(ranks, k=k)
    metrics["median_rank"] = float(np.median(ranks))
    metrics["mean_rank"] = float(np.mean(ranks))

    if return_per_query:
        pq = per_query_ndcg(scores, positive_indices, k=k)
        return metrics, pq

    return metrics
