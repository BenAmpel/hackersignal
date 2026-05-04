"""Global vocabulary across all time-spells.

Node indices are shared across spells. A per-spell `node_mask` indicates
which words actually appear at t (others are present in the tensor but
should be ignored by attention and loss terms).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from ..config import Config
from ..data.schemas import ForumPost

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")

UNK = "<UNK>"
MASK = "<MASK>"


def tokenize(text: str, min_len: int = 2) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text) if len(tok) >= min_len]


class Vocab:
    """Minimal word → id mapping with frequency tracking."""

    def __init__(self, tokens_to_id: dict[str, int], freqs: dict[str, int]):
        self.tokens_to_id = tokens_to_id
        self.id_to_token = {v: k for k, v in tokens_to_id.items()}
        self.freqs = freqs

    def __len__(self) -> int:
        return len(self.tokens_to_id)

    def id(self, token: str) -> int:
        return self.tokens_to_id.get(token, self.tokens_to_id[UNK])

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.id(t) for t in tokens]

    def token(self, i: int) -> str:
        return self.id_to_token[i]

    def freq(self, token: str) -> int:
        return self.freqs.get(token, 0)

    @property
    def size(self) -> int:
        return len(self.tokens_to_id)


def build_vocab(posts: Iterable[ForumPost], config: Config) -> Vocab:
    counter: Counter = Counter()
    for p in posts:
        counter.update(tokenize(p.text, config.min_token_len))
    # Reserve special tokens first.
    tokens_to_id = {UNK: 0, MASK: 1}
    for word, _ in counter.most_common(config.vocab_size_cap - len(tokens_to_id)):
        tokens_to_id[word] = len(tokens_to_id)
    freqs = {UNK: 0, MASK: 0, **{w: c for w, c in counter.items() if w in tokens_to_id}}
    return Vocab(tokens_to_id, freqs)
