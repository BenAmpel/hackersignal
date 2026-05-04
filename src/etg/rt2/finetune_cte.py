"""Supervised fine-tuning of the CTE for exploit-vulnerability linking.

Uses a margin-based triplet ranking loss. For each anchor exploit we
sample one positive vulnerability (the true link) and one negative,
sampled from the same time window if available (time-aware negatives
prevent trivial temporal shortcut learning).
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
import random
from datetime import timedelta
from typing import Callable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..config import Config
from ..data.schemas import EVPair
from ..logging_utils import get_logger
from .cte_model import CTE
from .ev_dataset import EVPairDataset


def _time_window_negatives(
    pairs: list[EVPair],
    *,
    window_days: int = 90,
) -> Callable[[EVPair, random.Random], EVPair]:
    """Return a selector that prefers timestamp-local negatives for each positive.

    Priority:
    1. Explicit negatives tied to the same exploit text within the time window.
    2. Any explicit negatives within the time window.
    3. Any explicit negatives globally.
    4. As a last resort, positives with a different vulnerability text.
    """
    window = timedelta(days=window_days)
    explicit_negatives = [p for p in pairs if p.label == 0]
    fallback_pairs = explicit_negatives or [p for p in pairs if p.label == 1]
    if not fallback_pairs:
        raise ValueError("finetune_cte requires at least one candidate negative or fallback pair")

    negatives_by_exploit: dict[str, list[EVPair]] = defaultdict(list)
    for pair in explicit_negatives:
        negatives_by_exploit[pair.exploit_text].append(pair)

    sorted_pairs = sorted(fallback_pairs, key=lambda pair: pair.timestamp)
    sorted_timestamps = [pair.timestamp for pair in sorted_pairs]

    def _filter_candidates(candidates: list[EVPair], positive: EVPair) -> list[EVPair]:
        return [
            pair
            for pair in candidates
            if pair.vulnerability_text != positive.vulnerability_text
        ]

    def _selector(positive: EVPair, rng: random.Random) -> EVPair:
        local_same_exploit = _filter_candidates(
            [
                pair
                for pair in negatives_by_exploit.get(positive.exploit_text, [])
                if abs(pair.timestamp - positive.timestamp) <= window
            ],
            positive,
        )
        if local_same_exploit:
            return rng.choice(local_same_exploit)

        lo = bisect_left(sorted_timestamps, positive.timestamp - window)
        hi = bisect_right(sorted_timestamps, positive.timestamp + window)
        local_window = _filter_candidates(sorted_pairs[lo:hi], positive)
        if local_window:
            return rng.choice(local_window)

        global_candidates = _filter_candidates(fallback_pairs, positive)
        if global_candidates:
            return rng.choice(global_candidates)

        return rng.choice(fallback_pairs)

    return _selector


def _triplet_loss(anchor: Tensor, pos: Tensor, neg: Tensor, margin: float = 0.2) -> Tensor:
    sim_pos = torch.nn.functional.cosine_similarity(anchor, pos, dim=-1)
    sim_neg = torch.nn.functional.cosine_similarity(anchor, neg, dim=-1)
    return torch.clamp(margin - (sim_pos - sim_neg), min=0.0).mean()


def finetune_cte(
    model: CTE,
    pairs: list[EVPair],
    config: Config,
    device: torch.device,
    *,
    use_time_aware_negatives: bool = True,
    negative_window_days: int = 90,
) -> dict[str, list[float]]:
    logger = get_logger("etg.finetune_cte")
    rng = random.Random(config.seed + 202)
    dataset = EVPairDataset(pairs, config)
    neg_selector = _time_window_negatives(
        pairs,
        window_days=negative_window_days,
    )
    fallback_pool = [p for p in pairs if p.label == 0] or [p for p in pairs if p.label == 1]
    loader = DataLoader(
        dataset,
        batch_size=config.cte_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: b,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.cte_lr)
    model.to(device).train()
    history = {"loss": []}
    for epoch in range(config.cte_finetune_epochs):
        ep_loss = 0.0
        n = 0
        for batch in loader:
            positives = [row["pos"] for row in batch]
            negatives = [
                neg_selector(pos, rng) if use_time_aware_negatives else rng.choice(fallback_pool)
                for pos in positives
            ]
            from .trigram_hasher import tokenize_cte

            def _tok(texts: list[str]) -> torch.Tensor:
                return torch.tensor(
                    [tokenize_cte(x, config.trigram_buckets, config.cte_max_len) for x in texts],
                    dtype=torch.long,
                )

            exp = _tok([pair.exploit_text for pair in positives]).to(device)
            pv = _tok([pair.vulnerability_text for pair in positives]).to(device)
            nv = _tok([pair.vulnerability_text for pair in negatives]).to(device)
            z_e = model.encode_sentence(exp)
            z_p = model.encode_sentence(pv)
            z_n = model.encode_sentence(nv)
            loss = _triplet_loss(z_e, z_p, z_n)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            ep_loss += float(loss.item())
            n += 1
        n = max(1, n)
        history["loss"].append(ep_loss / n)
        mode = "time-aware" if use_time_aware_negatives else "random"
        logger.info(
            f"CTE fine-tune epoch {epoch + 1}/{config.cte_finetune_epochs} "
            f"loss={ep_loss / n:.4f} negatives={mode}"
        )
    return history
