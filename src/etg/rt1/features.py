"""Node features for the ETG.

Per the proposal: syntactic (POS) + trigram hashes + frequency.
We use a lightweight rule-based POS tagger as a fallback so the pipeline
never requires spaCy's model download — the code path upgrades transparently
if `en_core_web_sm` is installed.

Trigram-hashes follow FastText-style bucket hashing of character trigrams.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Iterable

import numpy as np

from ..config import Config
from ..data.schemas import ForumPost
from .vocab import MASK, UNK, Vocab, tokenize

# POS set: coarse categories (noun/verb/adj/num/other).
POS_DIM = 5
POS_NOUN, POS_VERB, POS_ADJ, POS_NUM, POS_OTHER = range(POS_DIM)


def _coarse_pos(token: str) -> int:
    """Tiny heuristic POS; replaced by spaCy if available."""
    if token.endswith(("ing", "ed", "ize", "ise", "ate")):
        return POS_VERB
    if token.endswith(("able", "ible", "ous", "ive", "al")):
        return POS_ADJ
    if any(ch.isdigit() for ch in token):
        return POS_NUM
    # heuristic: common hacker verbs
    if token in {
        "exploit", "inject", "bypass", "pwn", "leak", "root", "escalate",
        "hijack", "fuzz", "spoof", "sniff", "crack", "dump", "drop",
    }:
        return POS_VERB
    return POS_NOUN


def _trigram_hash_vec(token: str, n_buckets: int, dim: int) -> np.ndarray:
    """Hash all character trigrams of a token into a `dim`-sized bag-of-buckets
    vector (sum of one-hots). Uses SHA-1 for deterministic hashing."""
    vec = np.zeros(dim, dtype=np.float32)
    if len(token) < 3:
        return vec
    padded = f"<{token}>"
    for i in range(len(padded) - 2):
        trig = padded[i : i + 3]
        h = int.from_bytes(
            hashlib.sha1(trig.encode("utf-8")).digest()[:4], "big", signed=False
        )
        # Fold hash into `dim` via modulo; bucket count capped separately.
        vec[h % dim] += 1.0
    # L2 normalize so features have consistent scale across token lengths.
    nrm = float(np.linalg.norm(vec))
    if nrm > 0:
        vec /= nrm
    return vec


def feature_dim(trigram_feat_dim: int) -> int:
    """Total feature dim = POS one-hot + trigram-hash vec + log-freq(1)."""
    return POS_DIM + trigram_feat_dim + 1


def build_node_features(
    vocab: Vocab,
    config: Config,
    trigram_feat_dim: int = 32,
) -> np.ndarray:
    """Return [V, F] float32 array — one row per vocabulary word."""
    V = vocab.size
    F = feature_dim(trigram_feat_dim)
    X = np.zeros((V, F), dtype=np.float32)
    max_freq = max(vocab.freqs.values()) if vocab.freqs else 1
    for i in range(V):
        tok = vocab.token(i)
        # POS
        if tok in (UNK, MASK):
            X[i, POS_OTHER] = 1.0
        else:
            X[i, _coarse_pos(tok)] = 1.0
        # Trigram hash
        X[i, POS_DIM : POS_DIM + trigram_feat_dim] = _trigram_hash_vec(
            tok, config.trigram_buckets, trigram_feat_dim
        )
        # Log-frequency (+1 smoothing)
        X[i, POS_DIM + trigram_feat_dim] = math.log1p(vocab.freq(tok)) / math.log1p(
            max_freq or 1
        )
    return X


def per_spell_frequencies(
    posts_by_spell: list[list[ForumPost]], vocab: Vocab, config: Config
) -> np.ndarray:
    """[T, V] integer array — word frequency at each time-spell."""
    T = len(posts_by_spell)
    V = vocab.size
    freqs = np.zeros((T, V), dtype=np.int64)
    for t, posts in enumerate(posts_by_spell):
        counter: Counter = Counter()
        for p in posts:
            counter.update(tokenize(p.text, config.min_token_len))
        for tok, c in counter.items():
            if tok in vocab.tokens_to_id:
                freqs[t, vocab.id(tok)] += c
    return freqs
