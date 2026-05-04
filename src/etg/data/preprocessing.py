"""
Raw crawl pre-processing pipeline → ForumPost objects.

Input: a JSONL file whose rows have the raw scraper schema:
  id, forum_id, thread_url, thread_title, section, post_index, post_id,
  thread_id, text, body, timestamp, scraped_at, author_hash,
  reply_count, view_count, tags, cve_refs

Output: list[ForumPost] (same frozen dataclass used everywhere else), ready
        to be fed into the RT1/RT2 pipeline in place of SyntheticForum.

Pipeline stages (each individually toggleable via PreprocessConfig):
  1. Parse        raw JSON row → RawPost struct
  2. Clean text   HTML entities, BBCode tags, inline code, URLs, whitespace
  3. Filter       (a) timestamp validity  (b) language  (c) token length
  4. CTI score    keyword heuristic – optional hard floor
  5. Exact dedup  SHA-256 on normalised text (keeps first occurrence)
  6. Near-dedup   MinHash-LSH (Jaccard ≥ threshold treated as duplicate)
                  Falls back to exact-only if datasketch is not installed.
  7. Sample       Stratified by time-spell so temporal coverage is preserved;
                  within each spell posts are ranked by CTI score.
  8. Export       yield ForumPost (id, text=body, timestamp, forum_id,
                  author_hash) – drop-in for SyntheticForum.__iter__
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from .schemas import ForumPost
from .time_spells import TimeSpellIndex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1.  PreprocessConfig
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """All knobs for the preprocessing pipeline.

    Sensible defaults work well for the 250k-post crawl in data/raw_posts.jsonl.
    """

    # --- timestamp ---
    require_real_timestamp: bool = True
    """Drop posts where timestamp == scraped_at (scraper used fallback date)."""

    # --- language ---
    min_ascii_ratio: float = 0.72
    """Fraction of ASCII characters required to accept a post as English.
    Pure-ASCII text scores 1.0; mostly-Cyrillic scores ~0.1."""

    use_langdetect: bool = True
    """Run langdetect on borderline posts (ascii_ratio in [0.60, 0.85]).
    Gracefully skipped if langdetect is not installed."""

    # --- length ---
    min_tokens: int = 20
    """Minimum whitespace-split token count (very short posts carry no signal)."""

    max_tokens: int = 2_000
    """Maximum token count (very long posts are likely scraped pages/spam)."""

    # --- CTI relevance ---
    apply_cti_filter: bool = False
    """If True, posts with cti_score == 0 are dropped."""

    min_cti_score: int = 0
    """Hard floor on CTI keyword score when apply_cti_filter is True."""

    # --- dedup ---
    exact_dedup: bool = True
    """Remove exact duplicates (SHA-256 on normalised text)."""

    near_dedup: bool = True
    """Remove near-duplicates via MinHash-LSH (requires datasketch)."""

    near_dedup_threshold: float = 0.70
    """Jaccard similarity threshold above which two posts are considered
    duplicates; the lower-CTI-score post is discarded."""

    near_dedup_num_perm: int = 128
    """Number of MinHash permutations (higher → more accurate, slower)."""

    # --- sampling ---
    sample_n: int | None = None
    """Target post count after sampling.  None keeps everything."""

    n_spells: int = 5
    """Number of temporal buckets for stratified sampling."""

    sample_bias_cti: bool = True
    """Within each time-spell prefer higher-CTI-score posts."""

    # --- field selection ---
    prefer_body_field: bool = True
    """Use 'body' over 'text' when body is non-empty (body is cleaner – no
    prepended thread title)."""

    forum_allowlist: list[str] = field(default_factory=list)
    """If non-empty, only posts from these forum_ids are kept."""

    forum_blocklist: list[str] = field(default_factory=list)
    """Forums to skip outright (e.g. non-CTI-relevant sources)."""


# Default config used when calling preprocess() without a config argument
DEFAULT_CONFIG = PreprocessConfig()


# ---------------------------------------------------------------------------
# 2.  CTI keyword vocabulary
# ---------------------------------------------------------------------------

# Tiered CTI keywords.  Each match adds its tier weight to the score.
_CTI_TIER1 = {
    # exploit / vulnerability primitives
    "cve", "exploit", "exploiting", "exploited", "vulnerability", "vuln",
    "zero-day", "0day", "0-day", "zeroday", "rce", "lfi", "rfi", "sqli",
    "xss", "csrf", "xxe", "ssrf", "ssti", "deserialization", "bufferoverflow",
    "buffer overflow", "use-after-free", "heap spray", "rop chain", "shellcode",
    "privesc", "privilege escalation", "arbitrary code", "code execution",
    "remote code", "local privilege",
    # malware / tooling
    "malware", "ransomware", "rootkit", "backdoor", "rat", "c2", "c&c",
    "botnet", "keylogger", "trojan", "dropper", "loader", "stager",
    "payload", "implant", "beacon", "cobalt strike", "metasploit", "msfvenom",
    # disclosure / research
    "poc", "proof of concept", "disclosure", "patch", "cvss", "nvd",
    "security advisory", "security bulletin",
}

_CTI_TIER2 = {
    "injection", "bypass", "escalate", "shell", "reverse shell", "bind shell",
    "pwn", "pwned", "root", "rooted", "leaked", "breach", "breached",
    "credential", "credentials", "password dump", "hash dump", "lsass",
    "mimikatz", "pass-the-hash", "golden ticket", "silver ticket", "kerberoast",
    "lateral movement", "persistence", "persistence mechanism", "command and control",
    "exfiltration", "data exfil", "phishing", "spear phishing", "social engineering",
    "dork", "scanner", "fuzzing", "fuzz", "enumeration", "recon",
    "nmap", "shodan", "burp suite", "sqlmap", "nikto", "gobuster",
    "nuclei", "subfinder",
}

_CTI_TIER3 = {
    "hacking", "hacked", "hack", "security", "attack", "attacker",
    "penetration test", "pentest", "red team", "bug bounty", "ctf",
    "cracking", "cracked", "crack", "brute force", "bruteforce",
    "token", "cookie", "session hijack", "mitm", "wireshark", "packet",
    "firewall", "ids", "ips", "siem", "endpoint", "edr", "av bypass",
}

_ALL_CTI_KEYWORDS: list[tuple[set[str], int]] = [
    (_CTI_TIER1, 3),
    (_CTI_TIER2, 2),
    (_CTI_TIER3, 1),
]


def cti_score(text: str) -> int:
    """Return a CTI keyword relevance score (integer ≥ 0).

    Each Tier-1 keyword hit contributes +3, Tier-2 +2, Tier-3 +1.
    A score of 0 means no security-relevant keywords found.
    """
    lower = text.lower()
    score = 0
    for kw_set, weight in _ALL_CTI_KEYWORDS:
        for kw in kw_set:
            if kw in lower:
                score += weight
    return score


# ---------------------------------------------------------------------------
# 3.  Text cleaning
# ---------------------------------------------------------------------------

_RE_HTML_TAG = re.compile(r"<[^>]{1,200}>")
_RE_BBCODE = re.compile(
    r"\[(?:/?(?:b|i|u|s|url|img|code|quote|size|color|font|spoiler|table|tr|td|th|list|li|hr|center|left|right|indent))[^\]]{0,100}\]",
    re.IGNORECASE,
)
_RE_URL = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)
_RE_CODE_BLOCK = re.compile(
    r"```.*?```|`[^`\n]{1,200}`",
    re.DOTALL,
)
_RE_QUOTE_BLOCK = re.compile(
    r"^>\s*.+",  # markdown-style quote lines (anchored to line-start)
    re.MULTILINE,
)
_RE_WHITESPACE = re.compile(r"[ \t]{2,}")
_RE_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(raw: str) -> str:
    """Return a cleaned version of a forum post body.

    Steps (order matters):
      1. Unicode normalisation (NFC)
      2. HTML entity decode  (&amp; → &)
      3. Strip HTML tags
      4. Strip BBCode tags
      5. Replace URLs with '<URL>'
      6. Replace code blocks with '<CODE>'
      7. Strip markdown quote lines
      8. Collapse whitespace
    """
    if not raw:
        return ""
    # 1. NFC normalise (handles combining characters, etc.)
    text = unicodedata.normalize("NFC", raw)
    # 2. HTML entity decode
    text = html.unescape(text)
    # 3. HTML tags
    text = _RE_HTML_TAG.sub(" ", text)
    # 4. BBCode
    text = _RE_BBCODE.sub(" ", text)
    # 5. Quote lines (must run before URL/code substitution — the pattern
    #    anchors to '^>' so it won't consume '>' inside our <URL>/<CODE> tokens)
    text = _RE_QUOTE_BLOCK.sub("", text)
    # 6. URLs
    text = _RE_URL.sub(" <URL> ", text)
    # 7. Inline/block code
    text = _RE_CODE_BLOCK.sub(" <CODE> ", text)
    # 8. Whitespace
    text = _RE_WHITESPACE.sub(" ", text)
    text = _RE_BLANK_LINES.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 4.  Language detection
# ---------------------------------------------------------------------------

def _ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) < 128) / len(text)


try:
    from langdetect import detect as _langdetect_detect  # type: ignore
    from langdetect import LangDetectException  # type: ignore
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False
    _langdetect_detect = None  # type: ignore
    LangDetectException = Exception  # type: ignore


def is_english(text: str, config: PreprocessConfig) -> bool:
    """Return True if the post appears to be written in English.

    Fast path: if ascii_ratio ≥ min_ascii_ratio, accept.
    If ascii_ratio < 0.50, reject without langdetect (clearly non-Latin).
    Borderline [0.50, min_ascii_ratio): use langdetect when available.
    """
    ratio = _ascii_ratio(text)
    if ratio >= config.min_ascii_ratio:
        return True
    if ratio < 0.50:
        return False
    # Borderline — try langdetect
    if config.use_langdetect and _LANGDETECT_AVAILABLE and _langdetect_detect is not None:
        try:
            return _langdetect_detect(text[:2000]) == "en"
        except LangDetectException:
            pass
    return False  # conservative default for borderline without langdetect


# ---------------------------------------------------------------------------
# 5.  Timestamp validation
# ---------------------------------------------------------------------------

def has_real_timestamp(record: dict) -> bool:
    """Return True if the post has an authentic post timestamp.

    The scraper falls back to setting timestamp = scraped_at when it cannot
    extract the actual post date from the page.  We detect this by checking
    exact string equality (scraper copies the value verbatim).
    """
    return record.get("timestamp", "") != record.get("scraped_at", "")


def parse_timestamp(ts_str: str) -> datetime | None:
    """Parse an ISO-8601 timestamp string, returning UTC-aware datetime."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# 6.  Exact deduplication
# ---------------------------------------------------------------------------

def _text_fingerprint(text: str) -> str:
    """SHA-256 digest of lowercased, whitespace-normalised text."""
    normalised = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def exact_dedup(posts: list[_Post]) -> list[_Post]:  # type: ignore[name-defined]
    """Remove exact duplicates; keep the first occurrence."""
    seen: set[str] = set()
    out: list[_Post] = []  # type: ignore[name-defined]
    for p in posts:
        fp = _text_fingerprint(p.text)
        if fp not in seen:
            seen.add(fp)
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# 7.  Near-dedup (MinHash-LSH)
# ---------------------------------------------------------------------------

try:
    from datasketch import MinHash, MinHashLSH  # type: ignore
    _DATASKETCH_AVAILABLE = True
except ImportError:
    _DATASKETCH_AVAILABLE = False
    MinHash = None  # type: ignore
    MinHashLSH = None  # type: ignore


def _shingles(text: str, k: int = 3) -> list[bytes]:
    """Character k-shingles for MinHash."""
    s = re.sub(r"\s+", " ", text.lower())
    return [s[i : i + k].encode("utf-8") for i in range(len(s) - k + 1)]


def near_dedup(posts: list[_Post], config: PreprocessConfig) -> list[_Post]:  # type: ignore[name-defined]
    """Remove near-duplicate posts using MinHash-LSH.

    Posts are sorted by CTI score descending before dedup so that the
    higher-quality post in a near-duplicate pair is retained.
    Falls back to identity (no-op) if datasketch is not installed.
    """
    if not _DATASKETCH_AVAILABLE or not posts:
        if not _DATASKETCH_AVAILABLE:
            logger.warning(
                "datasketch not installed — skipping near-dedup. "
                "Install with: pip install datasketch"
            )
        return posts

    # Sort best-first so LSH retains the higher-CTI post
    sorted_posts = sorted(posts, key=lambda p: p.cti_score, reverse=True)

    lsh = MinHashLSH(
        threshold=config.near_dedup_threshold,
        num_perm=config.near_dedup_num_perm,
    )
    kept: list[_Post] = []  # type: ignore[name-defined]

    for idx, post in enumerate(sorted_posts):
        mh = MinHash(num_perm=config.near_dedup_num_perm)
        for shingle in _shingles(post.text):
            mh.update(shingle)
        key = f"p{idx}"
        candidates = lsh.query(mh)
        if not candidates:
            lsh.insert(key, mh)
            kept.append(post)
        # else: near-duplicate of an already-kept post → discard

    return kept


# ---------------------------------------------------------------------------
# 8.  Stratified temporal sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    posts: list[_Post],  # type: ignore[name-defined]
    n: int,
    n_spells: int,
    bias_cti: bool = True,
) -> list[_Post]:  # type: ignore[name-defined]
    """Draw up to *n* posts with balanced temporal coverage.

    The surviving posts are split into *n_spells* uniform time buckets.
    We allocate quota proportional to each bucket's population, then draw
    the highest-CTI-score posts within each bucket.

    If n ≥ len(posts), all posts are returned unchanged.
    """
    if n >= len(posts):
        return posts

    if not posts:
        return []

    tsi = TimeSpellIndex.from_posts(posts, n_spells)
    buckets: list[list[_Post]] = [[] for _ in range(n_spells)]  # type: ignore[name-defined]
    for p in posts:
        idx = tsi.assign(p.timestamp)
        buckets[idx].append(p)

    total = len(posts)
    selected: list[_Post] = []  # type: ignore[name-defined]

    for bucket in buckets:
        if not bucket:
            continue
        quota = max(1, round(n * len(bucket) / total))
        if bias_cti:
            bucket_sorted = sorted(bucket, key=lambda p: p.cti_score, reverse=True)
        else:
            bucket_sorted = bucket
        selected.extend(bucket_sorted[:quota])

    # trim / top-up to exactly n (rounding may overshoot slightly)
    if len(selected) > n:
        if bias_cti:
            selected = sorted(selected, key=lambda p: p.cti_score, reverse=True)[:n]
        else:
            selected = selected[:n]

    return selected


# ---------------------------------------------------------------------------
# 9.  Internal working struct
# ---------------------------------------------------------------------------

@dataclass
class _Post:
    """Internal struct that carries extra fields during pipeline processing."""

    id: str
    text: str
    timestamp: datetime
    forum_id: str
    author_hash: str
    cti_score: int = 0

    def to_forum_post(self) -> ForumPost:
        return ForumPost(
            id=self.id,
            text=self.text,
            timestamp=self.timestamp,
            forum_id=self.forum_id,
            author_hash=self.author_hash,
        )


# ---------------------------------------------------------------------------
# 10.  Pipeline statistics
# ---------------------------------------------------------------------------

@dataclass
class PipelineStats:
    total_read: int = 0
    after_allowlist: int = 0
    after_timestamp_filter: int = 0
    after_language_filter: int = 0
    after_length_filter: int = 0
    after_cti_filter: int = 0
    after_exact_dedup: int = 0
    after_near_dedup: int = 0
    after_sample: int = 0

    def report(self) -> str:
        lines = [
            "=== Preprocessing pipeline statistics ===",
            f"  Read from disk          : {self.total_read:>8,}",
            f"  After forum filter      : {self.after_allowlist:>8,}",
            f"  After timestamp filter  : {self.after_timestamp_filter:>8,}",
            f"  After language filter   : {self.after_language_filter:>8,}",
            f"  After length filter     : {self.after_length_filter:>8,}",
            f"  After CTI filter        : {self.after_cti_filter:>8,}",
            f"  After exact dedup       : {self.after_exact_dedup:>8,}",
            f"  After near-dedup        : {self.after_near_dedup:>8,}",
            f"  Final (after sample)    : {self.after_sample:>8,}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 11.  Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    raw_path: str | Path,
    config: PreprocessConfig | None = None,
    cache_path: str | Path | None = None,
    verbose: bool = True,
) -> tuple[list[ForumPost], PipelineStats]:
    """Run the full preprocessing pipeline and return (posts, stats).

    Parameters
    ----------
    raw_path:
        Path to the raw JSONL crawl file (one JSON object per line).
    config:
        PreprocessConfig instance; defaults to PreprocessConfig() when None.
    cache_path:
        If supplied and the file exists, load from cache instead of
        re-processing.  If supplied and the file does not exist, write the
        cleaned posts to this path after processing (for fast re-runs).
    verbose:
        Print a progress summary to stdout.

    Returns
    -------
    (posts, stats) where posts is a list[ForumPost] ready for the pipeline.
    """
    if config is None:
        config = PreprocessConfig()

    raw_path = Path(raw_path)
    stats = PipelineStats()

    # ---- cache hit ----
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            if verbose:
                print(f"[preprocess] Loading from cache: {cache_path}")
            from .schemas import load_posts_jsonl
            posts = load_posts_jsonl(cache_path)
            stats.total_read = len(posts)
            stats.after_sample = len(posts)
            return posts, stats

    # ---- Stage 1: parse + clean + score ----
    working: list[_Post] = []

    with raw_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Bad JSON at line %d — skipped", line_no)
                continue

            stats.total_read += 1

            # Forum filter (applied first — cheapest)
            fid = record.get("forum_id", "")
            if config.forum_allowlist and fid not in config.forum_allowlist:
                continue
            if fid in config.forum_blocklist:
                continue

            # Pick text field
            body = (record.get("body") or "").strip()
            text_raw = (record.get("text") or "").strip()
            raw_text = body if (config.prefer_body_field and body) else text_raw
            if not raw_text:
                continue

            cleaned = clean_text(raw_text)
            if not cleaned:
                continue

            # Timestamp
            ts_str = record.get("timestamp", "")
            sa_str = record.get("scraped_at", "")
            if config.require_real_timestamp and ts_str == sa_str:
                continue

            ts = parse_timestamp(ts_str)
            if ts is None:
                # fall back to scraped_at
                ts = parse_timestamp(sa_str)
            if ts is None:
                continue

            working.append(
                _Post(
                    id=record.get("id", f"row{line_no}"),
                    text=cleaned,
                    timestamp=ts,
                    forum_id=fid,
                    author_hash=record.get("author_hash", ""),
                    cti_score=cti_score(cleaned),
                )
            )

    stats.after_allowlist = len(working)

    # After timestamp filter already applied inline above
    stats.after_timestamp_filter = len(working)

    # ---- Stage 2: language filter ----
    working = [p for p in working if is_english(p.text, config)]
    stats.after_language_filter = len(working)

    # ---- Stage 3: length filter ----
    working = [
        p for p in working
        if config.min_tokens <= len(p.text.split()) <= config.max_tokens
    ]
    stats.after_length_filter = len(working)

    # ---- Stage 4: CTI score filter ----
    if config.apply_cti_filter:
        working = [p for p in working if p.cti_score >= config.min_cti_score]
    stats.after_cti_filter = len(working)

    if not working:
        logger.warning("No posts survived filtering — check PreprocessConfig settings.")
        return [], stats

    # ---- Stage 5: exact dedup ----
    if config.exact_dedup:
        working = exact_dedup(working)
    stats.after_exact_dedup = len(working)

    # ---- Stage 6: near-dedup ----
    if config.near_dedup:
        working = near_dedup(working, config)
    stats.after_near_dedup = len(working)

    # ---- Stage 7: stratified temporal sample ----
    if config.sample_n is not None and config.sample_n < len(working):
        working = stratified_sample(
            working,
            n=config.sample_n,
            n_spells=config.n_spells,
            bias_cti=config.sample_bias_cti,
        )
    stats.after_sample = len(working)

    # ---- Stage 8: convert to ForumPost ----
    posts = [p.to_forum_post() for p in working]

    # ---- cache write ----
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        from .schemas import dump_jsonl
        dump_jsonl(posts, cache_path)
        if verbose:
            print(f"[preprocess] Wrote cleaned posts to: {cache_path}")

    if verbose:
        print(stats.report())

    return posts, stats


# ---------------------------------------------------------------------------
# 12.  Convenience: quick profile of a raw file (no full processing)
# ---------------------------------------------------------------------------

def profile_raw_file(
    raw_path: str | Path,
    n_sample: int = 10_000,
) -> dict:
    """Quick non-destructive audit of a raw JSONL file.

    Reads up to *n_sample* lines and returns a summary dict with:
      - total_lines (estimated from sample ratio)
      - pct_real_timestamp
      - pct_ascii_english
      - median_token_len, p5_token_len, p95_token_len
      - top_forums  (Counter)
      - timestamp_year_counts  (Counter)
      - pct_with_cve_refs
      - pct_cti_nonzero  (at least one CTI keyword hit)
    """
    from collections import Counter
    import statistics

    raw_path = Path(raw_path)
    rows_read = 0
    real_ts = 0
    ascii_ok = 0
    token_lens: list[int] = []
    forums: Counter = Counter()
    year_counts: Counter = Counter()
    cve_count = 0
    cti_nonzero = 0

    with raw_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if rows_read >= n_sample:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows_read += 1

            if has_real_timestamp(record):
                real_ts += 1

            body = (record.get("body") or record.get("text") or "")
            cleaned = clean_text(body)
            toks = len(cleaned.split())
            token_lens.append(toks)

            if _ascii_ratio(cleaned) >= 0.70:
                ascii_ok += 1

            forums[record.get("forum_id", "?")] += 1

            ts = parse_timestamp(record.get("timestamp", ""))
            if ts:
                year_counts[ts.year] += 1

            if record.get("cve_refs"):
                cve_count += 1

            if cti_score(cleaned) > 0:
                cti_nonzero += 1

    def pct(n: int) -> float:
        return round(100.0 * n / max(rows_read, 1), 1)

    return {
        "rows_sampled": rows_read,
        "pct_real_timestamp": pct(real_ts),
        "pct_ascii_english": pct(ascii_ok),
        "median_token_len": round(statistics.median(token_lens)) if token_lens else 0,
        "p5_token_len": sorted(token_lens)[max(0, int(0.05 * len(token_lens)))] if token_lens else 0,
        "p95_token_len": sorted(token_lens)[min(len(token_lens) - 1, int(0.95 * len(token_lens)))] if token_lens else 0,
        "top_forums": dict(forums.most_common(15)),
        "timestamp_year_counts": dict(sorted(year_counts.items())),
        "pct_with_cve_refs": pct(cve_count),
        "pct_cti_nonzero": pct(cti_nonzero),
    }
