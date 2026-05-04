"""Synthetic exploit-vulnerability pair generator.

Each positive pair is an (exploit_text, vulnerability_text) that share a
hidden CTI concept (vulnerability class + target). Negatives are sampled
from different concepts with a controllable vocabulary overlap.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Iterator

from ..config import Config
from .schemas import EVPair
from .synthetic_forum import (
    ATTACK_VERBS,
    CREDS,
    ENCODINGS,
    TARGETS,
    VERS,
    VULN_CLASSES,
)

EXPLOIT_TEMPLATES = [
    "{act} {tgt} via {cls} exploit {ver}",
    "{cls} rce on {tgt} using {enc} payload",
    "unauth {cls} in {tgt} drops {creds}",
    "{tgt} {cls} poc remote shell",
    "module for {cls} against {tgt}",
    "chain {cls} with auth bypass on {tgt}",
]

VULN_TEMPLATES = [
    "CVE-{year}-{num}: {cls} vulnerability affecting {tgt} {ver}",
    "A {cls} flaw in {tgt} allows remote attackers to {act} arbitrary code",
    "{tgt} contains a {cls} issue that permits unauthenticated {act}",
    "Improper input validation in {tgt} leads to {cls}",
    "{tgt} prior to {ver} is vulnerable to {cls} via crafted input",
    "A memory-safety {cls} in {tgt} allows denial of service",
]


def _render(rng: random.Random, template: str, concept: tuple[str, str]) -> str:
    cls, tgt = concept
    return template.format(
        cls=cls,
        tgt=tgt,
        ver=rng.choice(VERS),
        act=rng.choice(ATTACK_VERBS),
        enc=rng.choice(ENCODINGS),
        creds=rng.choice(CREDS),
        year=rng.randint(2019, 2024),
        num=rng.randint(1000, 49999),
    )


class SyntheticEV:
    """Yields EVPair instances: positives + controlled negatives."""

    def __init__(self, config: Config):
        self.config = config
        self.rng = random.Random(config.seed + 7)
        # Fixed window roughly matches synthetic forum.
        self.t_min = datetime(2023, 1, 1, tzinfo=timezone.utc)
        self.t_max = datetime(2024, 7, 1, tzinfo=timezone.utc)
        self.total_seconds = (self.t_max - self.t_min).total_seconds()
        self.concepts = [(c, t) for c in VULN_CLASSES for t in TARGETS]

    def _random_ts(self) -> datetime:
        frac = self.rng.random()
        return self.t_min + timedelta(seconds=frac * self.total_seconds)

    def __iter__(self) -> Iterator[EVPair]:
        cfg = self.config
        rng = self.rng
        for i in range(cfg.n_ev_pairs):
            concept = rng.choice(self.concepts)
            exploit_text = _render(rng, rng.choice(EXPLOIT_TEMPLATES), concept)
            vuln_text = _render(rng, rng.choice(VULN_TEMPLATES), concept)
            ts = self._random_ts()
            yield EVPair(
                exploit_text=exploit_text,
                vulnerability_text=vuln_text,
                cve_id=f"CVE-{rng.randint(2019,2024)}-{rng.randint(1000,49999)}",
                timestamp=ts,
                label=1,
            )
            # k controlled negatives per positive
            for _ in range(cfg.n_ev_negatives_per_positive):
                neg_concept = concept
                # allow overlap_rate chance of sharing one axis
                if rng.random() < cfg.overlap_rate:
                    if rng.random() < 0.5:
                        # same class, different target
                        others = [t for t in TARGETS if t != concept[1]]
                        neg_concept = (concept[0], rng.choice(others))
                    else:
                        # same target, different class
                        others = [c for c in VULN_CLASSES if c != concept[0]]
                        neg_concept = (rng.choice(others), concept[1])
                else:
                    while neg_concept == concept:
                        neg_concept = rng.choice(self.concepts)
                neg_vuln = _render(rng, rng.choice(VULN_TEMPLATES), neg_concept)
                yield EVPair(
                    exploit_text=exploit_text,
                    vulnerability_text=neg_vuln,
                    cve_id=None,
                    timestamp=ts,
                    label=0,
                )
