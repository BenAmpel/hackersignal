"""FastText-style hashing for short CTI text.

Tokens: bucket-hash each word's character trigrams AND the word itself.
For the CTE we emit a single token-id per input word that is the hash of
the padded word token — simple, fast, and handles OOV cleanly.
"""

from __future__ import annotations

import hashlib
import re

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")

PAD_ID = 0
MASK_ID = 1  # reserved (never returned by hashing below)
CLS_ID = 2   # reserved
SEP_ID = 3   # reserved
_RESERVED = 4


def _sha1_u32(s: str) -> int:
    return int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:4], "big")


def token_hash(token: str, n_buckets: int) -> int:
    t = token.lower()
    bucket = _RESERVED + (_sha1_u32(f"w::{t}") % (n_buckets - _RESERVED))
    return bucket


def tokenize_cte(text: str, n_buckets: int, max_len: int) -> list[int]:
    toks = _TOKEN_RE.findall(text)
    ids = [CLS_ID] + [token_hash(t, n_buckets) for t in toks]
    ids = ids[:max_len]
    while len(ids) < max_len:
        ids.append(PAD_ID)
    return ids


def batch_tokenize_cte(
    texts: list[str], n_buckets: int, max_len: int
) -> tuple[list[list[int]], list[int]]:
    """Returns (ids [B, L], lengths [B])."""
    ids = []
    lens = []
    for t in texts:
        seq = tokenize_cte(t, n_buckets, max_len)
        ids.append(seq)
        lens.append(sum(1 for x in seq if x != PAD_ID))
    return ids, lens
