"""Given a fine-tuned CTE, score exploit queries against a candidate pool.

For each positive pair, the candidate pool is {true vuln} ∪ {N distractors}
drawn from the full dataset. We then compute retrieval metrics on the
ranks of the true vuln.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from ..config import Config
from ..data.schemas import EVPair
from .cte_model import CTE
from .trigram_hasher import tokenize_cte


@torch.no_grad()
def _embed(model: CTE, texts: list[str], config: Config, device: torch.device) -> torch.Tensor:
    model.eval()
    ids = [tokenize_cte(t, config.trigram_buckets, config.cte_max_len) for t in texts]
    ids_t = torch.tensor(ids, dtype=torch.long, device=device)
    return model.encode_sentence(ids_t)


def rank_exploits(
    model: CTE,
    positive_pairs: list[EVPair],
    candidate_pool_texts: list[str],
    config: Config,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """For each positive pair, rank all candidate vulnerability texts.

    Returns (scores [Q, C], positive_indices [Q]) aligned to positive_pairs.
    """
    exploit_texts = [p.exploit_text for p in positive_pairs]
    positive_texts = [p.vulnerability_text for p in positive_pairs]
    # Build a stable list of candidates: unique vuln texts, positives first.
    seen: dict[str, int] = {}
    for t in positive_texts:
        if t not in seen:
            seen[t] = len(seen)
    for t in candidate_pool_texts:
        if t not in seen:
            seen[t] = len(seen)
    candidates = list(seen.keys())
    # Embed everything in batches.
    cand_vecs = []
    B = max(1, config.cte_batch_size)
    for i in range(0, len(candidates), B):
        cand_vecs.append(_embed(model, candidates[i : i + B], config, device))
    cand_matrix = torch.cat(cand_vecs, dim=0)  # [C, d]
    exp_vecs = []
    for i in range(0, len(exploit_texts), B):
        exp_vecs.append(_embed(model, exploit_texts[i : i + B], config, device))
    exp_matrix = torch.cat(exp_vecs, dim=0)  # [Q, d]
    exp_n = torch.nn.functional.normalize(exp_matrix, dim=-1)
    cand_n = torch.nn.functional.normalize(cand_matrix, dim=-1)
    scores = exp_n @ cand_n.T  # [Q, C]
    positive_indices = np.array([seen[t] for t in positive_texts], dtype=np.int64)
    return scores.cpu().numpy(), positive_indices


def sample_candidate_pool(
    pairs: list[EVPair], n_candidates: int, seed: int
) -> list[str]:
    """Sample `n_candidates` distinct vulnerability texts as distractors."""
    rng = random.Random(seed)
    uniq = list({p.vulnerability_text for p in pairs})
    rng.shuffle(uniq)
    return uniq[: max(1, min(n_candidates, len(uniq)))]
