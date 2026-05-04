import numpy as np
from sklearn.metrics import ndcg_score

from etg.eval.metrics_ir import (
    hit_rate_at_k,
    mean_average_precision,
    mrr_at_k,
    ndcg_at_k,
    ranks_from_scores,
)


def test_ranks_from_scores_simple():
    # Two queries, 4 candidates each. Positives at index 0 in query 0 and at 2 in query 1.
    scores = np.array(
        [
            [0.9, 0.8, 0.1, 0.2],  # true positive at 0 -> rank 1
            [0.2, 0.1, 0.95, 0.5],  # true positive at 2 -> rank 1
        ]
    )
    pos = np.array([0, 2])
    ranks = ranks_from_scores(scores, pos)
    assert (ranks == np.array([1, 1])).all()


def test_ranking_metrics_match_sklearn_ndcg():
    # Build a scenario where we know the ranks exactly, then compare numpy impl to sklearn.
    ranks = np.array([1, 2, 3, 10, 20])
    our_ndcg = ndcg_at_k(ranks, 10)
    # Build y_true / y_score for sklearn.
    # For each query, y_true has a single 1 at the positive item; y_score is chosen to produce the desired rank.
    Q = len(ranks)
    C = 25
    y_true = np.zeros((Q, C), dtype=np.float64)
    y_score = np.zeros((Q, C), dtype=np.float64)
    for q, r in enumerate(ranks):
        # Give the positive at position q, and rank it at 1-indexed `r` by assigning descending scores.
        y_true[q, q] = 1.0
        # Create scores: positive gets score `C - r`, all others get descending scores excluding `C - r`.
        score = np.linspace(C, 1, C)
        # Move positive's score to the r-th highest.
        score_pos = score[r - 1]
        # Place the positive's score at index q and shift other scores.
        y_score[q] = score
        y_score[q, q] = score_pos
        # Swap index (r-1) to match (so true positive is literally ranked r).
        # Reassign: give unique decreasing values to every index, and place score_pos at index q.
        vals = np.linspace(C, 1, C).copy()
        # Set the r-1-th value to go to q, q's original to go to r-1.
        vals[q], vals[r - 1] = vals[r - 1], vals[q]
        y_score[q] = vals
    # Compute ndcg@10 with sklearn.
    sk_ndcg = ndcg_score(y_true, y_score, k=10)
    assert abs(our_ndcg - sk_ndcg) < 1e-6


def test_ranking_metrics_basic_values():
    ranks = np.array([1, 1, 1, 1])
    assert hit_rate_at_k(ranks, 10) == 1.0
    assert mrr_at_k(ranks, 10) == 1.0
    assert mean_average_precision(ranks) == 1.0
    ranks = np.array([100, 100, 100, 100])
    assert hit_rate_at_k(ranks, 10) == 0.0
    assert mrr_at_k(ranks, 10) == 0.0
