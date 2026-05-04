"""Torch Dataset over EVPair instances.

Provides three sampling modes:
- `mlm`        : returns a single side (exploit or vulnerability) for the
                 masked-prediction pretext task.
- `contrastive`: returns two augmented views of the same sample for SimCSE-
                 style contrastive pretraining.
- `triplet`    : returns (anchor=exploit, positive=linked vuln, negative=unlinked vuln)
                 for supervised fine-tuning.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from ..config import Config
from ..data.schemas import EVPair
from .augmentations import contrastive_views
from .trigram_hasher import batch_tokenize_cte, tokenize_cte


@dataclass
class EVBatch:
    a: torch.LongTensor
    b: torch.LongTensor | None
    c: torch.LongTensor | None  # optional third tensor (negative for triplet)
    meta: dict


class EVPretrainDataset(Dataset):
    def __init__(self, pairs: list[EVPair], config: Config, mode: str = "mlm"):
        assert mode in ("mlm", "contrastive")
        self.mode = mode
        self.config = config
        # Combine exploit + vuln text as pretraining corpus.
        texts: list[str] = []
        for p in pairs:
            texts.append(p.exploit_text)
            texts.append(p.vulnerability_text)
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        return {"text": self.texts[idx]}


def collate_mlm(batch: list[dict], config: Config, rng: random.Random) -> dict:
    texts = [b["text"] for b in batch]
    ids, lens = batch_tokenize_cte(texts, config.trigram_buckets, config.cte_max_len)
    ids_t = torch.tensor(ids, dtype=torch.long)
    labels = ids_t.clone()
    # Mask ~cte_mask_rate positions (excluding CLS, PAD).
    from .trigram_hasher import CLS_ID, MASK_ID, PAD_ID

    mask = torch.zeros_like(ids_t, dtype=torch.bool)
    for b in range(ids_t.shape[0]):
        for p in range(ids_t.shape[1]):
            if ids_t[b, p] in (PAD_ID, CLS_ID):
                continue
            if rng.random() < config.cte_mask_rate:
                mask[b, p] = True
    ids_masked = ids_t.clone()
    ids_masked[mask] = MASK_ID
    return {"input_ids": ids_masked, "labels": labels, "mlm_mask": mask, "lengths": torch.tensor(lens)}


def collate_contrastive(batch: list[dict], config: Config, rng: random.Random) -> dict:
    texts = [b["text"] for b in batch]
    view_a = [tokenize_cte(t, config.trigram_buckets, config.cte_max_len) for t in texts]
    view_b_texts = [contrastive_views(t, rng) for t in texts]
    view_b = [
        tokenize_cte(t, config.trigram_buckets, config.cte_max_len) for t in view_b_texts
    ]
    return {
        "view_a": torch.tensor(view_a, dtype=torch.long),
        "view_b": torch.tensor(view_b, dtype=torch.long),
    }


class EVPairDataset(Dataset):
    """Returns positive pairs and k negatives drawn from the dataset."""

    def __init__(self, pairs: list[EVPair], config: Config):
        self.positives = [p for p in pairs if p.label == 1]
        self.negatives = [p for p in pairs if p.label == 0]
        self.config = config
        if not self.positives:
            raise ValueError("EVPairDataset requires at least one positive")

    def __len__(self) -> int:
        return len(self.positives)

    def __getitem__(self, idx: int) -> dict:
        pos = self.positives[idx]
        return {"pos": pos, "idx": idx}


def collate_triplet(batch: list[dict], config: Config, neg_pool: list[EVPair], rng: random.Random) -> dict:
    # Build triplets: (exploit, positive vuln, sampled negative vuln).
    exploits = []
    pos_vulns = []
    neg_vulns = []
    labels = []
    for b in batch:
        p: EVPair = b["pos"]
        exploits.append(p.exploit_text)
        pos_vulns.append(p.vulnerability_text)
        n: EVPair = rng.choice(neg_pool)
        neg_vulns.append(n.vulnerability_text)
        labels.append(1)
    def _tok(xs):
        return torch.tensor(
            [tokenize_cte(x, config.trigram_buckets, config.cte_max_len) for x in xs],
            dtype=torch.long,
        )
    return {
        "exploit": _tok(exploits),
        "pos_vuln": _tok(pos_vulns),
        "neg_vuln": _tok(neg_vulns),
        "labels": torch.tensor(labels),
    }
