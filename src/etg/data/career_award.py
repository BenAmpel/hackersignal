"""CAREER-award-aligned real-data helpers for RT1 and RT2.

RT1 uses curated hacker-community post corpora rather than the broad raw crawl.
RT2 uses real exploit-vulnerability pairs from ``data/ev_pairs.jsonl`` rather
than synthetic pairs.
"""

from __future__ import annotations

import heapq
import json
import logging
import random
from pathlib import Path
from typing import Iterable

from ..config import Config
from .preprocessing import PreprocessConfig, _text_fingerprint, clean_text, cti_score, is_english
from .real_ev_loader import RealEV
from .schemas import EVPair, ForumPost

log = logging.getLogger(__name__)

DEFAULT_RT1_FILES = (
    "0x00sec_posts.jsonl",
    "go4expert_posts.jsonl",
    "hackforums_posts.jsonl",
    "kaeli_hacker_posts.jsonl",
    "cve_hacker_forum_posts.jsonl",
)

DEFAULT_RT2_FILE = "ev_pairs.jsonl"


def default_rt1_paths(data_dir: str | Path = "data") -> list[Path]:
    """Return the available award-aligned RT1 community corpora."""
    root = Path(data_dir)
    out: list[Path] = []
    for name in DEFAULT_RT1_FILES:
        path = root / name
        if path.exists() and path.stat().st_size > 0:
            out.append(path)
    return out


def _iter_posts_jsonl(path: Path) -> Iterable[ForumPost]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ForumPost.from_dict(json.loads(line))
            except Exception as exc:
                log.debug("Skipping malformed post in %s:%d: %s", path, line_no, exc)


def _prepare_rt1_post(
    post: ForumPost,
    preprocess_config: PreprocessConfig,
) -> tuple[ForumPost, int] | None:
    text = clean_text(post.text)
    if not text:
        return None
    token_count = len(text.split())
    if token_count < preprocess_config.min_tokens or token_count > preprocess_config.max_tokens:
        return None
    if not is_english(text, preprocess_config):
        return None
    score = cti_score(text)
    if preprocess_config.apply_cti_filter and score < preprocess_config.min_cti_score:
        return None
    return (
        ForumPost(
            id=post.id,
            text=text,
            timestamp=post.timestamp,
            forum_id=post.forum_id,
            author_hash=post.author_hash,
        ),
        score,
    )


def _spell_index(ts, ts_min, ts_max, n_spells: int) -> int:
    span = max((ts_max - ts_min).total_seconds(), 1.0)
    frac = (ts - ts_min).total_seconds() / span
    frac = min(max(frac, 0.0), 0.999999)
    return min(n_spells - 1, int(frac * n_spells))


def load_career_rt1_posts(
    config: Config,
    *,
    paths: list[str | Path] | None = None,
    data_dir: str | Path = "data",
    preprocess_config: PreprocessConfig | None = None,
) -> list[ForumPost]:
    """Load a balanced RT1 sample from curated hacker-community corpora.

    The source files are already in ``ForumPost`` JSONL format, so we apply only
    lightweight cleaning/filtering and then keep the highest-CTI posts per time
    spell. This preserves the proposal's time-aware RT1 setup while avoiding a
    dependency on the broader raw crawl.
    """
    if preprocess_config is None:
        preprocess_config = PreprocessConfig(
            require_real_timestamp=False,
            min_ascii_ratio=0.72,
            use_langdetect=False,
            min_tokens=20,
            max_tokens=2_000,
            apply_cti_filter=False,
            min_cti_score=0,
            exact_dedup=False,
            near_dedup=False,
            sample_n=config.n_posts,
            n_spells=config.n_spells,
            sample_bias_cti=True,
            prefer_body_field=False,
        )

    resolved_paths = [Path(p) for p in paths] if paths else default_rt1_paths(data_dir)
    if not resolved_paths:
        raise FileNotFoundError(
            "No RT1 source files found. Expected one or more of: "
            + ", ".join(DEFAULT_RT1_FILES)
        )

    ts_min = None
    ts_max = None
    for path in resolved_paths:
        for post in _iter_posts_jsonl(path):
            prepared = _prepare_rt1_post(post, preprocess_config)
            if prepared is None:
                continue
            cleaned_post, _ = prepared
            ts_min = cleaned_post.timestamp if ts_min is None else min(ts_min, cleaned_post.timestamp)
            ts_max = cleaned_post.timestamp if ts_max is None else max(ts_max, cleaned_post.timestamp)

    if ts_min is None or ts_max is None:
        raise ValueError("No RT1 posts survived filtering in the configured source files")

    sample_n = preprocess_config.sample_n or config.n_posts
    n_spells = max(1, preprocess_config.n_spells)
    per_spell_target = [sample_n // n_spells] * n_spells
    for idx in range(sample_n % n_spells):
        per_spell_target[idx] += 1

    heaps: list[list[tuple[int, int, ForumPost]]] = [[] for _ in range(n_spells)]
    counter = 0

    for path in resolved_paths:
        for post in _iter_posts_jsonl(path):
            prepared = _prepare_rt1_post(post, preprocess_config)
            if prepared is None:
                continue
            cleaned_post, score = prepared
            spell = _spell_index(cleaned_post.timestamp, ts_min, ts_max, n_spells)
            # Keep a modest overflow buffer per spell, then trim deterministically.
            keep_limit = max(1, per_spell_target[spell] * 3)
            entry = (score, counter, cleaned_post)
            counter += 1
            if len(heaps[spell]) < keep_limit:
                heapq.heappush(heaps[spell], entry)
            elif entry[0] > heaps[spell][0][0]:
                heapq.heapreplace(heaps[spell], entry)

    selected: list[ForumPost] = []
    leftovers: list[tuple[int, int, ForumPost]] = []
    seen_fingerprints: set[str] = set()

    for spell, heap in enumerate(heaps):
        ranked = sorted(heap, key=lambda item: (-item[0], item[1]))
        kept_here = 0
        for item in ranked:
            fp = _text_fingerprint(item[2].text)
            if fp in seen_fingerprints:
                continue
            if kept_here < per_spell_target[spell]:
                selected.append(item[2])
                seen_fingerprints.add(fp)
                kept_here += 1
            else:
                leftovers.append(item)

    if len(selected) < sample_n:
        for _, _, post in sorted(leftovers, key=lambda item: (-item[0], item[1])):
            fp = _text_fingerprint(post.text)
            if fp in seen_fingerprints:
                continue
            selected.append(post)
            seen_fingerprints.add(fp)
            if len(selected) >= sample_n:
                break

    selected.sort(key=lambda post: post.timestamp)
    return selected


def load_career_rt2_pairs(
    config: Config,
    *,
    path: str | Path = Path("data") / DEFAULT_RT2_FILE,
    seed: int | None = None,
) -> list[EVPair]:
    """Load a deterministic, config-sized RT2 subset from real EV pairs."""
    rng = random.Random(config.seed if seed is None else seed)
    pairs = list(RealEV(path))
    positives = [pair for pair in pairs if pair.label == 1]
    negatives = [pair for pair in pairs if pair.label == 0]

    if not positives:
        raise ValueError(f"No positive EV pairs found in {path}")

    if len(positives) > config.n_ev_pairs:
        positives = rng.sample(positives, config.n_ev_pairs)

    neg_target = min(len(negatives), len(positives) * max(1, config.n_ev_negatives_per_positive))
    if neg_target:
        selected_exploits = {pair.exploit_text for pair in positives}
        exploit_matched = [pair for pair in negatives if pair.exploit_text in selected_exploits]
        rng.shuffle(exploit_matched)
        selected_negatives = exploit_matched[: min(len(exploit_matched), neg_target)]
        if len(selected_negatives) < neg_target:
            selected_set = set(selected_negatives)
            remaining = [pair for pair in negatives if pair not in selected_set]
            remainder_n = min(len(remaining), neg_target - len(selected_negatives))
            if remainder_n:
                selected_negatives.extend(rng.sample(remaining, remainder_n))
        negatives = selected_negatives

    sampled = positives + negatives
    sampled.sort(key=lambda pair: (pair.timestamp, -pair.label, pair.cve_id or ""))
    return sampled
