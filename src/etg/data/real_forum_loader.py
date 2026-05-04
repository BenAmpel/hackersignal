"""Real hacker-forum data loaders.

Two classes:

  RealForum(path)
      Thin wrapper: streams ForumPost objects from a JSONL file that already
      conforms to the ForumPost schema (id, text, timestamp, forum_id,
      author_hash).  Use this when you have a file produced by run_pipeline()
      or otherwise already cleaned.

  PreprocessedForum(raw_path, config)
      Full pipeline: reads the raw crawl JSONL (with the richer scraper schema),
      runs the preprocessing pipeline (clean → filter → dedup → sample), caches
      the result to data/preprocessed/, and then serves ForumPost objects.
      Drop-in replacement for SyntheticForum: same __iter__ and
      ground_truth_shift_words() interface.

Both classes satisfy the ForumLoader Protocol defined in schemas.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .preprocessing import PreprocessConfig, PipelineStats, run_pipeline
from .schemas import ForumPost, load_posts_jsonl


# ---------------------------------------------------------------------------
# RealForum — pass-through loader for already-cleaned data
# ---------------------------------------------------------------------------

class RealForum:
    """Stream ForumPost objects from a pre-cleaned JSONL file.

    The file must already conform to the ForumPost schema (one JSON object
    per line with keys: id, text, timestamp, forum_id, author_hash).

    This is the post-pipeline loader.  For raw crawl data, use
    PreprocessedForum instead.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Real forum data not found: {self.path}\n"
                "Point `path` to a JSONL file with one ForumPost per line, "
                "or use PreprocessedForum(raw_path=...) to run the pipeline "
                "on a raw crawl file."
            )

    def __iter__(self) -> Iterator[ForumPost]:
        yield from load_posts_jsonl(self.path)

    def ground_truth_shift_words(self) -> set[str]:
        """No synthetic ground truth for real data; returns empty set."""
        return set()


# ---------------------------------------------------------------------------
# PreprocessedForum — full pipeline from raw crawl to ForumPost stream
# ---------------------------------------------------------------------------

class PreprocessedForum:
    """Load, clean, filter, deduplicate, and sample real forum posts.

    Parameters
    ----------
    raw_path : str | Path
        Path to the raw scraper JSONL dump (data/raw_posts.jsonl).
    config : PreprocessConfig, optional
        Pipeline configuration.  If None, uses smoke-test defaults:
          - English only (ASCII ratio ≥ 0.72)
          - Real timestamps only (timestamp ≠ scraped_at)
          - 20–2000 token length window
          - Exact + near deduplication
          - Stratified sample of 2,000 posts across 5 time-spells
    cache_dir : str | Path, optional
        Directory for the processed JSONL cache file.  Defaults to the
        'preprocessed/' subdirectory next to raw_path.
    force_reprocess : bool
        If True, ignore any existing cache and re-run the pipeline.
    verbose : bool
        Print per-stage statistics.

    Usage
    -----
    # Replace SyntheticForum with one line:
    loader = PreprocessedForum("data/raw_posts.jsonl")

    # Custom config — e.g. no CTI filter, keep more posts:
    cfg = PreprocessConfig(sample_n=5000, apply_cti_filter=False)
    loader = PreprocessedForum("data/raw_posts.jsonl", config=cfg)
    """

    def __init__(
        self,
        raw_path: str | Path,
        config: PreprocessConfig | None = None,
        cache_dir: str | Path | None = None,
        force_reprocess: bool = False,
        verbose: bool = True,
    ):
        self.raw_path = Path(raw_path)
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw crawl file not found: {self.raw_path}")

        # Build a default smoke-test config if none supplied
        if config is None:
            config = _smoke_preprocess_config()
        self.config = config

        # Determine cache path
        if cache_dir is None:
            cache_dir = self.raw_path.parent / "preprocessed"
        cache_dir = Path(cache_dir)

        # Cache filename encodes key config params so changing config
        # automatically invalidates the cache.
        cache_key = _config_cache_key(config)
        self.cache_path = cache_dir / f"posts_{cache_key}.jsonl"

        if force_reprocess and self.cache_path.exists():
            self.cache_path.unlink()

        # Run the pipeline (or load from cache)
        self._posts, self.stats = run_pipeline(
            raw_path=self.raw_path,
            config=config,
            cache_path=self.cache_path,
            verbose=verbose,
        )

        if verbose:
            print(
                f"[PreprocessedForum] Serving {len(self._posts):,} posts "
                f"from {len({p.forum_id for p in self._posts})} forums "
                f"spanning {_date_range(self._posts)}"
            )

    # ------------------------------------------------------------------
    # ForumLoader protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[ForumPost]:
        yield from self._posts

    def ground_truth_shift_words(self) -> set[str]:
        """No synthetic ground truth for real data; returns empty set."""
        return set()

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._posts)

    @property
    def posts(self) -> list[ForumPost]:
        """All posts as a list (same objects as __iter__ yields)."""
        return list(self._posts)

    def forum_ids(self) -> set[str]:
        return {p.forum_id for p in self._posts}

    def date_range(self) -> str:
        return _date_range(self._posts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smoke_preprocess_config() -> PreprocessConfig:
    """Default config for smoke-test runs.

    Produces ~2,000 English posts with real timestamps, stratified
    across 5 time-spells, ranked by CTI keyword score.
    """
    return PreprocessConfig(
        require_real_timestamp=True,
        min_ascii_ratio=0.72,
        use_langdetect=True,
        min_tokens=20,
        max_tokens=2_000,
        apply_cti_filter=False,   # keep all — CTI score used for ranking only
        min_cti_score=0,
        exact_dedup=True,
        near_dedup=True,
        near_dedup_threshold=0.70,
        near_dedup_num_perm=128,
        sample_n=2_000,
        n_spells=5,
        sample_bias_cti=True,
        prefer_body_field=True,
    )


def _config_cache_key(config: PreprocessConfig) -> str:
    """Short deterministic hash of the config parameters that affect output."""
    import hashlib, json, dataclasses
    d = dataclasses.asdict(config)
    blob = json.dumps(d, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _date_range(posts: list[ForumPost]) -> str:
    if not posts:
        return "no posts"
    timestamps = [p.timestamp for p in posts]
    lo = min(timestamps).strftime("%Y-%m-%d")
    hi = max(timestamps).strftime("%Y-%m-%d")
    return f"{lo} → {hi}"
