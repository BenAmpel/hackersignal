"""Augmentations for the contrastive pretext task.

We combine two cheap augmentations:
- token masking at ~15%
- word deletion at ~10%
Both are applied at the text level; downstream dropout in the encoder
provides the SimCSE-style independent-dropout view even when the tokens
are identical.
"""

from __future__ import annotations

import random
import re

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")


def contrastive_views(text: str, rng: random.Random) -> str:
    toks = _TOKEN_RE.findall(text)
    if not toks:
        return text
    kept: list[str] = []
    for tok in toks:
        r = rng.random()
        if r < 0.10:
            continue  # delete
        if r < 0.25:
            kept.append("<mask>")
        else:
            kept.append(tok)
    if not kept:
        kept = [toks[0]]
    return " ".join(kept)
