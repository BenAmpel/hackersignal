"""Synthetic hacker-forum corpus generator.

Design goals:
- Produce CTI-flavored text (exploit/vuln vocabulary, varied surface forms).
- Distribute posts across T time-spells.
- Inject **deterministic semantic shifts** for a small set of "shift words" —
  their co-occurring context changes between spells so shift detection has
  ground truth to recover.

The generator is seedable and reproducible.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator

from ..config import Config
from .schemas import ForumPost


# ---- Domain lexicons (hand-curated CTI flavors) ----

ATTACK_VERBS = [
    "exploit", "inject", "bypass", "pwn", "leak", "root", "escalate",
    "hijack", "fuzz", "spoof", "sniff", "crack", "dump", "drop",
]

VULN_CLASSES = [
    "sqli", "xss", "rce", "lfi", "ssrf", "deserialization", "bufferoverflow",
    "csrf", "xxe", "race", "toctou", "privesc", "auth-bypass", "path-traversal",
]

TARGETS = [
    "apache", "nginx", "wordpress", "drupal", "joomla", "jenkins",
    "tomcat", "exchange", "vcenter", "confluence", "spring", "struts",
    "kibana", "solr", "openssl", "log4j", "samba", "postgres",
]

CONTEXT_PHRASES = [
    "tested on {tgt} {ver}", "works after {act}", "drop a shell via {act}",
    "need {creds} first", "module ready for {tgt}", "poc attached",
    "{cls} triggers on {tgt}", "new {cls} chain", "0day for {tgt}",
    "payload hidden in {enc}", "ready to ship", "any {tgt} admins here",
    "see the {artifact}", "patch drops next {time}", "reverse {artifact}",
]

FILLERS = [
    "fyi", "btw", "imo", "lol", "tho", "plz", "thx", "yo", "brb", "nm",
    "kthx", "ya", "nope", "yep", "maybe", "idk", "meh",
]

ENCODINGS = ["base64", "rot13", "hex", "xor", "rc4", "aes"]
ARTIFACTS = ["binary", "pcap", "coredump", "log", "dump", "memdump"]
CREDS = ["creds", "token", "cookie", "session", "apikey", "jwt"]
VERS = ["v1", "v2", "2.3", "3.1.4", "unpatched", "legacy"]
TIMES = ["week", "month", "patch-tuesday", "release"]


# ---- Shift mechanism ----
#
# A "shift word" is a token whose **context distribution** changes after a
# chosen spell. We achieve this by re-routing its co-occurring words from
# one cluster (e.g. VULN_CLASSES) to another (e.g. TARGETS) at shift_spell.


@dataclass
class ShiftPlan:
    word: str
    shift_spell: int   # first spell where new context applies
    pre_context: list[str]
    post_context: list[str]


def _make_shift_plans(rng: random.Random, n: int, T: int) -> list[ShiftPlan]:
    """Deterministically choose words from our lexicons and reassign their context."""
    candidates = ATTACK_VERBS + VULN_CLASSES + TARGETS
    rng.shuffle(candidates)
    plans: list[ShiftPlan] = []
    clusters = {
        "verbs": ATTACK_VERBS,
        "classes": VULN_CLASSES,
        "targets": TARGETS,
        "encs": ENCODINGS,
        "creds": CREDS,
    }
    keys = list(clusters.keys())
    for w in candidates[:n]:
        pre_k, post_k = rng.sample(keys, 2)
        pre = [x for x in clusters[pre_k] if x != w][:6]
        post = [x for x in clusters[post_k] if x != w][:6]
        # shift at a spell that leaves at least one spell on each side
        spell = rng.randint(1, max(1, T - 1))
        plans.append(ShiftPlan(word=w, shift_spell=spell, pre_context=pre, post_context=post))
    return plans


def _hash_author(i: int) -> str:
    return hashlib.sha1(f"author-{i}".encode()).hexdigest()[:12]


def _make_post_text(
    rng: random.Random,
    spell: int,
    shift_lookup: dict[str, ShiftPlan],
    noise_rate: float,
    subtlety: float,
) -> str:
    # Core structure: pick 2–4 phrases, interleave shift words when applicable.
    n_phrases = rng.randint(2, 4)
    tokens: list[str] = []

    # Randomly include some shift words so that their context mirrors the plan.
    active_shifts = [
        p for p in shift_lookup.values() if rng.random() < 0.35 * (1.0 - subtlety * 0.5)
    ]
    for plan in active_shifts:
        if spell < plan.shift_spell:
            ctx = plan.pre_context
        else:
            ctx = plan.post_context
        # sprinkle word+context together
        tokens.append(plan.word)
        tokens.extend(rng.sample(ctx, k=min(3, len(ctx))))

    for _ in range(n_phrases):
        phrase = rng.choice(CONTEXT_PHRASES)
        phrase = phrase.format(
            tgt=rng.choice(TARGETS),
            ver=rng.choice(VERS),
            act=rng.choice(ATTACK_VERBS),
            cls=rng.choice(VULN_CLASSES),
            enc=rng.choice(ENCODINGS),
            creds=rng.choice(CREDS),
            artifact=rng.choice(ARTIFACTS),
            time=rng.choice(TIMES),
        )
        tokens.extend(phrase.split())

    # Filler / noise
    n_filler = rng.randint(1, 4)
    tokens.extend(rng.choices(FILLERS, k=n_filler))

    # Character-level noise (typos / case)
    if rng.random() < noise_rate:
        i = rng.randrange(len(tokens))
        w = tokens[i]
        if len(w) > 3:
            j = rng.randrange(1, len(w) - 1)
            tokens[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]  # swap two chars

    rng.shuffle(tokens)
    return " ".join(tokens)


class SyntheticForum:
    """Iterable yielding ForumPost objects per Config."""

    def __init__(self, config: Config):
        self.config = config
        self.rng = random.Random(config.seed)
        # Fixed window: roughly 18 months, split into T spells of equal length.
        self.t_min = datetime(2023, 1, 1, tzinfo=timezone.utc)
        self.t_max = datetime(2024, 7, 1, tzinfo=timezone.utc)
        self.total_seconds = (self.t_max - self.t_min).total_seconds()
        self._shift_plans = _make_shift_plans(
            self.rng, config.inject_n_shift_words, config.n_spells
        )

    @property
    def shift_plans(self) -> list[ShiftPlan]:
        return list(self._shift_plans)

    def ground_truth_shift_words(self) -> set[str]:
        return {p.word for p in self._shift_plans}

    def __iter__(self) -> Iterator[ForumPost]:
        cfg = self.config
        rng = self.rng
        lookup = {p.word: p for p in self._shift_plans}
        # posts roughly uniform across spells (+/- jitter)
        base_per_spell = cfg.n_posts // cfg.n_spells
        remainder = cfg.n_posts - base_per_spell * cfg.n_spells
        counts = [base_per_spell] * cfg.n_spells
        for i in range(remainder):
            counts[i] += 1
        post_counter = 0
        for spell, n in enumerate(counts):
            for _ in range(n):
                # Timestamp uniformly within spell window.
                spell_start_frac = spell / cfg.n_spells
                spell_end_frac = (spell + 1) / cfg.n_spells
                frac = rng.uniform(spell_start_frac, spell_end_frac)
                seconds = frac * self.total_seconds
                ts = self.t_min + timedelta(seconds=seconds)
                text = _make_post_text(
                    rng,
                    spell=spell,
                    shift_lookup=lookup,
                    noise_rate=cfg.noise_rate,
                    subtlety=cfg.shift_subtlety,
                )
                author_id = rng.randint(0, max(1, cfg.n_posts // 10))
                yield ForumPost(
                    id=f"p{post_counter:07d}",
                    text=text,
                    timestamp=ts,
                    forum_id=f"forum{rng.randrange(cfg.n_forums)}",
                    author_hash=_hash_author(author_id),
                )
                post_counter += 1
