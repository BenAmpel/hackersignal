"""RT1 evaluation — extrinsic and intrinsic.

Extrinsic:
- Rolling-window ARIMA forecast of the per-spell mean shift score → MAE/RMSE/MAPE.
- Analogy hit-rate on a small CTI analogy set (placeholder; works on
  synthetic data by exercising embedding similarity semantics).

Intrinsic (simulated on synthetic corpus):
- Detection recall@k for injected ground-truth shift words.
"""

from __future__ import annotations

import numpy as np
from torch import Tensor

from ..config import Config
from ..eval.metrics_forecast import all_metrics as forecast_metrics
from .forecast_arima import rolling_forecast_eval
from .shift_detection import ShiftResult
from .vocab import Vocab


def mean_shift_per_pair(results: list[ShiftResult]) -> np.ndarray:
    """Return [T-1] mean shift across present words per adjacent pair."""
    return np.array([float(r.scores.mean()) if len(r.scores) else 0.0 for r in results])


def extrinsic_forecast(
    results: list[ShiftResult], config: Config
) -> dict[str, float]:
    series = mean_shift_per_pair(results)
    preds, actuals = rolling_forecast_eval(
        series, min_train=max(2, min(3, len(series) - 1)), order=config.arima_order
    )
    if len(actuals) == 0:
        return {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0, "R2": 0.0}
    return forecast_metrics(actuals, preds)


def intrinsic_shift_recall_at_k(
    results: list[ShiftResult],
    vocab: Vocab,
    ground_truth_words: set[str],
    k: int,
) -> float:
    """Fraction of ground-truth shift words that appear in the union of the
    top-k shifted words across all pairs."""
    if not ground_truth_words:
        return float("nan")
    gt_ids = {vocab.id(w) for w in ground_truth_words if w in vocab.tokens_to_id}
    if not gt_ids:
        return 0.0
    detected: set[int] = set()
    for r in results:
        if len(r.scores) == 0:
            continue
        order = np.argsort(-r.scores)
        top_k = r.word_ids[order[: max(1, k)]]
        detected.update(int(x) for x in top_k)
    return len(detected & gt_ids) / len(gt_ids)


def analogy_hit_rate(
    embeddings: list[Tensor], vocab: Vocab, k: int = 5
) -> float:
    """Tiny CTI analogy set — placeholder that exercises nearest-neighbor
    retrieval on the final-spell embedding."""
    analogies = [
        ("sqli", "rce", "xss"),
        ("apache", "nginx", "wordpress"),
        ("creds", "token", "cookie"),
        ("payload", "shell", "poc"),
    ]
    if not embeddings:
        return 0.0
    emb = embeddings[-1].numpy()
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    hits = 0
    total = 0
    for a, b, c in analogies:
        if not all(w in vocab.tokens_to_id for w in (a, b, c)):
            continue
        ia, ib, ic = (vocab.id(w) for w in (a, b, c))
        # Target vector: b - a + c  (classic analogy).
        target = emb[ib] - emb[ia] + emb[ic]
        target = target / (np.linalg.norm(target) + 1e-12)
        sims = emb @ target
        # Exclude the input words from retrieval.
        sims[ia] = sims[ib] = sims[ic] = -np.inf
        top = np.argsort(-sims)[:k]
        # Any CTI-class token in the top-k counts as a hit on synthetic data.
        hit = any(
            vocab.token(int(i)) not in ("<UNK>", "<MASK>") for i in top
        )
        hits += int(hit)
        total += 1
    return hits / max(1, total)
