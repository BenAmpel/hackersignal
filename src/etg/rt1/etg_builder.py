"""Exploit Text Graph construction per time-spell.

Implements the proposal's ETG:
- Directed, weighted graph-of-words.
- Nodes = words in a global vocabulary.
- Edges = co-occurrence within a sliding window of size d (the "considers
  order") in posts from the current spell.
- Edge-weight persistence: e^t_{ij} = e^{t-1}_{ij} + δ_t, where δ_t is the
  new co-occurrence count observed at spell t. This keeps the graph at t
  reflective of the full vocabulary history, per the proposal.

The output for each spell is a compact tensor dict (ETGSnapshot) suitable
for the DGT model.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, TypedDict

import numpy as np
import torch
from torch import Tensor

from ..config import Config
from ..data.schemas import ForumPost
from .features import build_node_features, per_spell_frequencies
from .vocab import Vocab, tokenize


class ETGSnapshot(TypedDict):
    t: int
    x: Tensor               # [V, F] node features (static across spells)
    edge_index: Tensor      # [2, E] long
    edge_weight: Tensor     # [E] float
    node_mask: Tensor       # [V] bool — True if word observed at t
    freq: Tensor            # [V] long
    lap_pe: Tensor          # [V, K] (filled by laplacian_pe.attach_lap_pe)


def group_posts_by_spell(
    posts: Iterable[ForumPost], time_spells
) -> list[list[ForumPost]]:
    posts = list(posts)
    T = time_spells.T
    by_spell: list[list[ForumPost]] = [[] for _ in range(T)]
    for p in posts:
        by_spell[time_spells.assign(p.timestamp)].append(p)
    return by_spell


def _spell_cooccurrences(
    posts: list[ForumPost], vocab: Vocab, window: int, min_len: int
) -> dict[tuple[int, int], int]:
    """Count directed co-occurrences (i → j with j after i) within `window`."""
    out: dict[tuple[int, int], int] = defaultdict(int)
    for post in posts:
        ids = [vocab.id(t) for t in tokenize(post.text, min_len)]
        L = len(ids)
        for i in range(L):
            src = ids[i]
            # j in (i, i+window]
            for j in range(i + 1, min(L, i + 1 + window)):
                dst = ids[j]
                if src == dst:
                    continue
                out[(src, dst)] += 1
    return out


def build_etg_sequence(
    posts: list[ForumPost],
    posts_by_spell: list[list[ForumPost]],
    vocab: Vocab,
    config: Config,
) -> list[ETGSnapshot]:
    """Apply the persistence rule across spells → list of T snapshots."""
    V = vocab.size
    # Static node features (same across spells; per-spell info lives in node_mask/freq).
    X = build_node_features(vocab, config)
    X_t = torch.from_numpy(X)

    spell_freqs_np = per_spell_frequencies(posts_by_spell, vocab, config)
    spell_freqs = torch.from_numpy(spell_freqs_np)

    snapshots: list[ETGSnapshot] = []
    # Running accumulator of persistent edge weights.
    running: dict[tuple[int, int], float] = defaultdict(float)
    for t, spell_posts in enumerate(posts_by_spell):
        delta = _spell_cooccurrences(spell_posts, vocab, config.window_size, config.min_token_len)
        # e^t_{ij} = e^{t-1}_{ij} + δ_t
        for edge, d in delta.items():
            running[edge] += d
        # Prune edges below threshold to keep graphs tractable.
        keep = [(src, dst, w) for (src, dst), w in running.items() if w >= config.min_edge_weight]
        if not keep:
            keep = [(0, 0, 1.0)]  # self-loop to keep tensor shape valid
        edge_index = torch.tensor(
            [[src for src, _, _ in keep], [dst for _, dst, _ in keep]], dtype=torch.long
        )
        edge_weight = torch.tensor([w for _, _, w in keep], dtype=torch.float32)
        freq_t = spell_freqs[t]
        node_mask = freq_t > 0
        snap: ETGSnapshot = {
            "t": t,
            "x": X_t,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "node_mask": node_mask,
            "freq": freq_t,
            "lap_pe": torch.zeros((V, config.lap_pe_k), dtype=torch.float32),
        }
        snapshots.append(snap)
    return snapshots
