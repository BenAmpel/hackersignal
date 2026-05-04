"""Orchestration pipeline: forum index → detect software → scrape → write JSONL.

Usage (Python):
    from etg.scraper.pipeline import run
    run(output="data/real_posts.jsonl", max_threads=50)

    # Scrape 5 forums in parallel
    run(output="data/real_posts.jsonl", forum_workers=5)

Usage (CLI):
    python -m etg.scraper.cli --help
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, Optional

from etg.data.schemas import ForumPost, dump_jsonl
from etg.scraper.detector import ForumKind, detect
from etg.scraper.forum_index import ForumEntry, online_entries
from etg.scraper.models import RawPost
from etg.scraper.scrapers.base import make_fetcher, resp_html, _is_dead_host_error
from etg.scraper.scrapers import (
    BaseScraper,
    DiscourseScraper,
    GenericScraper,
    InvisionScraper,
    MyBBScraper,
    PhpBBScraper,
    XenForoScraper,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scraper factory
# ---------------------------------------------------------------------------

_TOR_PROXY = "socks5://127.0.0.1:9050"

# ---------------------------------------------------------------------------
# Per-software-type depth tuning
#
# Structured forums (XenForo, Invision, phpBB, MyBB) expose clean pagination
# and have high post density — go deeper.  Discourse uses a JSON API so
# pagination is essentially free.  Generic/unknown sites are scraped shallowly
# to avoid wasted requests on non-forum pages.
# ---------------------------------------------------------------------------
_KIND_OVERRIDES: dict[ForumKind, dict] = {
    ForumKind.XENFORO:  {"max_threads": 10_000, "max_list_pages": 500, "request_delay": 1.5},
    ForumKind.INVISION: {"max_threads": 10_000, "max_list_pages": 500, "request_delay": 1.5},
    ForumKind.PHPBB:    {"max_threads": 10_000, "max_list_pages": 500, "request_delay": 1.5},
    ForumKind.MYBB:     {"max_threads": 10_000, "max_list_pages": 500, "request_delay": 1.5},
    ForumKind.DISCOURSE:{"max_threads": 10_000, "max_list_pages": 500, "request_delay": 1.0},
    ForumKind.UNKNOWN:  {"max_threads":  2_000, "max_list_pages": 100, "request_delay": 2.0},
}

_CRAWL_MODE_OVERRIDES: dict[str, dict] = {
    "deep": {},
    "broad": {
        "max_posts_per_thread": 25,
    },
}


def _make_scraper(
    kind: ForumKind,
    proxy: Optional[str] = None,
    max_threads: Optional[int] = None,
    max_list_pages: Optional[int] = None,
    request_delay: Optional[float] = None,
    cookies=None,
) -> BaseScraper:
    cls = {
        ForumKind.XENFORO: XenForoScraper,
        ForumKind.PHPBB: PhpBBScraper,
        ForumKind.MYBB: MyBBScraper,
        ForumKind.DISCOURSE: DiscourseScraper,
        ForumKind.INVISION: InvisionScraper,
    }.get(kind, GenericScraper)

    scraper = cls(proxy=proxy, cookies=cookies)

    # Apply per-kind defaults, then allow explicit overrides from caller
    overrides = _KIND_OVERRIDES.get(kind, _KIND_OVERRIDES[ForumKind.UNKNOWN])
    scraper.max_threads   = max_threads   or overrides["max_threads"]
    scraper.max_list_pages= max_list_pages or overrides["max_list_pages"]
    scraper.request_delay = request_delay  or overrides["request_delay"]
    return scraper


def _load_forum_profiles(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        log.warning("Forum profile file not found: %s", p)
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to parse forum profile file %s: %s", p, exc)
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        profiles = data.get("profiles")
        if isinstance(profiles, list):
            return [item for item in profiles if isinstance(item, dict)]
    return []


def _profile_matches(entry: ForumEntry, profile: dict) -> bool:
    patterns = profile.get("forums") or profile.get("match") or []
    if isinstance(patterns, str):
        patterns = [patterns]
    haystacks = [entry.forum_id.lower(), entry.name.lower(), entry.url.lower()]
    return any(
        isinstance(pattern, str) and pattern.strip().lower() in haystacks[0]
        or isinstance(pattern, str) and any(pattern.strip().lower() in hay for hay in haystacks)
        for pattern in patterns
    )


def _resolve_entry_access(
    entry: ForumEntry,
    *,
    default_proxy: Optional[str],
    default_cookies,
    profiles: list[dict],
) -> dict:
    resolved = {
        "proxy": default_proxy,
        "cookies": default_cookies,
        "crawl_mode": None,
        "max_posts_per_thread": None,
        "max_thread_pages": None,
    }
    for profile in profiles:
        if not _profile_matches(entry, profile):
            continue
        if profile.get("proxy"):
            resolved["proxy"] = profile["proxy"]
        if profile.get("cookies"):
            resolved["cookies"] = profile["cookies"]
        if profile.get("crawl_mode"):
            resolved["crawl_mode"] = profile["crawl_mode"]
        if profile.get("max_posts_per_thread") is not None:
            resolved["max_posts_per_thread"] = profile["max_posts_per_thread"]
        if profile.get("max_thread_pages") is not None:
            resolved["max_thread_pages"] = profile["max_thread_pages"]
    return resolved


def _resolve_entry_runtime(
    entry: ForumEntry,
    *,
    default_proxy: Optional[str],
    default_cookies,
    profiles: list[dict],
    crawl_mode: str,
    max_posts_per_thread: Optional[int],
    max_thread_pages: Optional[int],
) -> dict:
    access = _resolve_entry_access(
        entry,
        default_proxy=default_proxy,
        default_cookies=default_cookies,
        profiles=profiles,
    )
    crawl_options = _resolve_crawl_options(
        crawl_mode=access["crawl_mode"] or crawl_mode,
        max_posts_per_thread=(
            access["max_posts_per_thread"]
            if access["max_posts_per_thread"] is not None
            else max_posts_per_thread
        ),
        max_thread_pages=(
            access["max_thread_pages"]
            if access["max_thread_pages"] is not None
            else max_thread_pages
        ),
    )
    return {
        "proxy": access["proxy"],
        "cookies": access["cookies"],
        "max_posts_per_thread": crawl_options["max_posts_per_thread"],
        "max_thread_pages": crawl_options["max_thread_pages"],
    }


def _resolve_crawl_options(
    *,
    crawl_mode: str,
    max_posts_per_thread: Optional[int],
    max_thread_pages: Optional[int],
) -> dict[str, Optional[int]]:
    overrides = _CRAWL_MODE_OVERRIDES.get(crawl_mode, {})
    return {
        "max_posts_per_thread": (
            max_posts_per_thread
            if max_posts_per_thread is not None
            else overrides.get("max_posts_per_thread")
        ),
        "max_thread_pages": max_thread_pages,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _load_checkpoint(path: Path) -> set[str]:
    """Return set of forum_ids already scraped."""
    if path.exists():
        with path.open() as f:
            return set(json.load(f))
    return set()


def _save_checkpoint(path: Path, done: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(sorted(done), f, indent=2)


# ---------------------------------------------------------------------------
# Per-forum worker (runs in its own thread when forum_workers > 1)
# ---------------------------------------------------------------------------

_DEAD = "__DEAD__"   # sentinel return value


def _scrape_one_forum(
    entry: ForumEntry,
    *,
    proxy: Optional[str],
    max_threads: Optional[int],
    max_list_pages: Optional[int],
    request_delay: Optional[float],
    max_posts_per_thread: Optional[int],
    max_thread_pages: Optional[int],
    cookies=None,
    write_lock: threading.Lock,
    fout,
    fraw,
    all_posts: list,
) -> tuple[str, int]:
    """Detect, scrape, and stream-write one forum.

    Returns ``(forum_id, post_count)``.  ``post_count == -1`` signals a dead
    host so the caller can checkpoint it without counting it as scraped data.
    """
    log.info("▶ %s  %s", entry.name, entry.url)

    # --- detect ---
    kind = detect(entry.url)
    if kind == ForumKind.UNKNOWN:
        try:
            resp = make_fetcher().get(entry.url, timeout=12, proxy=proxy)
            kind = detect(entry.url, resp_html(resp))
            time.sleep(2.0)
        except Exception as exc:
            if _is_dead_host_error(exc):
                log.info("Dead host %s — skipping", entry.name)
                return entry.forum_id, -1
            log.warning("Detection fetch %s: %s", entry.url, exc)

    log.info("  [%s] Detected as: %s", entry.name, kind.name)
    scraper = _make_scraper(
        kind,
        proxy=proxy,
        max_threads=max_threads,
        max_list_pages=max_list_pages,
        request_delay=request_delay,
        cookies=cookies,
    )
    if max_thread_pages is not None:
        scraper.max_thread_pages = max_thread_pages

    # --- scrape ---
    post_count = 0
    try:
        for raw_post in scraper.scrape_forum(
            entry,
            max_posts_per_thread=max_posts_per_thread,
        ):
            forum_post = raw_post.to_forum_post()
            with write_lock:
                all_posts.append(forum_post)
                fout.write(json.dumps(forum_post.to_dict()) + "\n")
                fout.flush()
                if fraw:
                    fraw.write(json.dumps(raw_post.to_dict()) + "\n")
                    fraw.flush()
            post_count += 1
    except Exception as exc:
        log.error("Error scraping %s: %s", entry.name, exc, exc_info=True)

    log.info("  Collected %d posts from %s", post_count, entry.name)
    return entry.forum_id, post_count


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------


def run(
    output: str | Path = "data/real_posts.jsonl",
    raw_output: str | Path | None = "data/raw_posts.jsonl",
    index_source=None,
    include_onion: bool = False,
    forum_filter: list[str] | None = None,
    max_threads: Optional[int] = None,
    max_list_pages: Optional[int] = None,
    request_delay: Optional[float] = None,
    max_posts_per_thread: Optional[int] = None,
    max_thread_pages: Optional[int] = None,
    crawl_mode: str = "deep",
    checkpoint: str | Path | None = "data/.scrape_checkpoint.json",
    tor_proxy: str = _TOR_PROXY,
    log_level: str = "INFO",
    forum_workers: int = 1,
    cookies=None,
    forum_profile_file: str | Path | None = None,
) -> list[ForumPost]:
    """Scrape forums from the deepdarkCTI index and write JSONL outputs.

    Parameters
    ----------
    output:
        Path for the minimal ``ForumPost`` JSONL consumed by the ETG pipeline.
    raw_output:
        Path for the full ``RawPost`` metadata JSONL (None = skip).
    index_source:
        None → fetch live forum.md; str URL or Path → custom source.
    include_onion:
        Whether to attempt .onion sites (requires a running Tor daemon).
    forum_filter:
        If given, only scrape forums whose ``forum_id`` or ``name`` matches
        one of these strings (case-insensitive substring match).
    max_threads:
        Override max threads per forum. None = use per-software-type defaults.
    max_list_pages:
        Override thread-list page depth. None = use per-software-type defaults.
    request_delay:
        Override seconds between requests. None = use per-software-type defaults.
    max_posts_per_thread:
        Optional cap on posts yielded from any one thread. Useful for broad
        coverage runs where giant threads would otherwise dominate collection.
    max_thread_pages:
        Optional cap on paginated pages fetched inside a single thread.
    crawl_mode:
        ``"deep"`` keeps the classic behaviour. ``"broad"`` prioritises more
        unique-thread coverage by applying crawl caps such as per-thread limits.
    checkpoint:
        Path to a JSON checkpoint file; already-done forums are skipped on resume.
        Set to None to disable checkpointing.
    tor_proxy:
        SOCKS5 proxy URL for .onion sites. Default: ``socks5://127.0.0.1:9050``
    forum_workers:
        Number of forums to scrape in parallel (default 1 = sequential).
        4–6 is a practical sweet spot; too many risks IP bans across sites.

    Returns
    -------
    list[ForumPost]
        All posts collected during this run (also written to *output*).
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = Path(raw_output) if raw_output else None
    if raw_path:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(checkpoint) if checkpoint else None

    done_ids: set[str] = _load_checkpoint(ckpt_path) if ckpt_path else set()
    ckpt_lock = threading.Lock()
    write_lock = threading.Lock()
    profiles = _load_forum_profiles(forum_profile_file)
    crawl_options = _resolve_crawl_options(
        crawl_mode=crawl_mode,
        max_posts_per_thread=max_posts_per_thread,
        max_thread_pages=max_thread_pages,
    )

    # Collect entries
    entries = online_entries(index_source)
    if not include_onion:
        entries = [e for e in entries if not e.is_onion]
    if forum_filter:
        filt = [f.lower() for f in forum_filter]
        entries = [
            e for e in entries
            if any(f in e.forum_id.lower() or f in e.name.lower() for f in filt)
        ]

    pending = [e for e in entries if e.forum_id not in done_ids]
    skipped = len(entries) - len(pending)
    log.info(
        "Forums to scrape: %d  (skipping %d checkpointed)  workers=%d",
        len(pending), skipped, forum_workers,
    )

    all_posts: list[ForumPost] = []

    # Open output files in append mode so we can resume
    fout = output_path.open("a", encoding="utf-8")
    fraw = raw_path.open("a", encoding="utf-8") if raw_path else None

    def _checkpoint(forum_id: str) -> None:
        with ckpt_lock:
            done_ids.add(forum_id)
            if ckpt_path:
                _save_checkpoint(ckpt_path, done_ids)

    try:
        if forum_workers <= 1:
            # ── sequential (original behaviour) ──────────────────────────
            for entry in pending:
                runtime = _resolve_entry_runtime(
                    entry,
                    default_proxy=tor_proxy if entry.is_onion else None,
                    default_cookies=cookies,
                    profiles=profiles,
                    crawl_mode=crawl_mode,
                    max_posts_per_thread=crawl_options["max_posts_per_thread"],
                    max_thread_pages=crawl_options["max_thread_pages"],
                )
                fid, count = _scrape_one_forum(
                    entry,
                    proxy=runtime["proxy"],
                    max_threads=max_threads,
                    max_list_pages=max_list_pages,
                    request_delay=request_delay,
                    max_posts_per_thread=runtime["max_posts_per_thread"],
                    max_thread_pages=runtime["max_thread_pages"],
                    cookies=runtime["cookies"],
                    write_lock=write_lock,
                    fout=fout,
                    fraw=fraw,
                    all_posts=all_posts,
                )
                _checkpoint(fid)
        else:
            # ── parallel ─────────────────────────────────────────────────
            with ThreadPoolExecutor(max_workers=forum_workers,
                                    thread_name_prefix="etg-forum") as pool:
                runtime_by_forum_id = {
                    entry.forum_id: _resolve_entry_runtime(
                        entry,
                        default_proxy=tor_proxy if entry.is_onion else None,
                        default_cookies=cookies,
                        profiles=profiles,
                        crawl_mode=crawl_mode,
                        max_posts_per_thread=crawl_options["max_posts_per_thread"],
                        max_thread_pages=crawl_options["max_thread_pages"],
                    )
                    for entry in pending
                }
                future_map = {
                    pool.submit(
                        _scrape_one_forum,
                        entry,
                        proxy=runtime_by_forum_id[entry.forum_id]["proxy"],
                        max_threads=max_threads,
                        max_list_pages=max_list_pages,
                        request_delay=request_delay,
                        max_posts_per_thread=runtime_by_forum_id[entry.forum_id]["max_posts_per_thread"],
                        max_thread_pages=runtime_by_forum_id[entry.forum_id]["max_thread_pages"],
                        cookies=runtime_by_forum_id[entry.forum_id]["cookies"],
                        write_lock=write_lock,
                        fout=fout,
                        fraw=fraw,
                        all_posts=all_posts,
                    ): entry
                    for entry in pending
                }
                for future in as_completed(future_map):
                    entry = future_map[future]
                    try:
                        fid, count = future.result()
                    except Exception as exc:
                        log.error("Unhandled worker error for %s: %s", entry.name, exc)
                        fid = entry.forum_id
                    _checkpoint(fid)
    finally:
        fout.close()
        if fraw:
            fraw.close()

    log.info("Done. Total posts this run: %d → %s", len(all_posts), output_path)
    return all_posts


# ---------------------------------------------------------------------------
# Convenience iterator (no file I/O)
# ---------------------------------------------------------------------------


def iter_posts(
    index_source=None,
    include_onion: bool = False,
    forum_filter: list[str] | None = None,
    **kwargs,
) -> Iterator[tuple[RawPost, ForumPost]]:
    """Yield (RawPost, ForumPost) pairs without writing to disk."""
    entries = online_entries(index_source)
    if not include_onion:
        entries = [e for e in entries if not e.is_onion]
    if forum_filter:
        filt = [f.lower() for f in forum_filter]
        entries = [
            e for e in entries
            if any(f in e.forum_id.lower() or f in e.name.lower() for f in filt)
        ]

    tor_proxy = kwargs.get("tor_proxy", _TOR_PROXY)
    request_delay = kwargs.get("request_delay", 1.5)
    max_threads = kwargs.get("max_threads", 200)
    max_list_pages = kwargs.get("max_list_pages", 10)
    crawl_mode = kwargs.get("crawl_mode", "deep")
    max_posts_per_thread = kwargs.get("max_posts_per_thread")
    max_thread_pages = kwargs.get("max_thread_pages")
    profiles = _load_forum_profiles(kwargs.get("forum_profile_file"))
    crawl_options = _resolve_crawl_options(
        crawl_mode=crawl_mode,
        max_posts_per_thread=max_posts_per_thread,
        max_thread_pages=max_thread_pages,
    )

    for entry in entries:
        runtime = _resolve_entry_runtime(
            entry,
            default_proxy=tor_proxy if entry.is_onion else None,
            default_cookies=kwargs.get("cookies"),
            profiles=profiles,
            crawl_mode=crawl_mode,
            max_posts_per_thread=crawl_options["max_posts_per_thread"],
            max_thread_pages=crawl_options["max_thread_pages"],
        )
        kind = detect(entry.url)
        if kind == ForumKind.UNKNOWN:
            try:
                resp = make_fetcher().get(entry.url, timeout=20, proxy=runtime["proxy"])
                kind = detect(entry.url, resp_html(resp))
            except Exception:
                pass

        scraper = _make_scraper(kind, proxy=runtime["proxy"],
                                max_threads=max_threads,
                                max_list_pages=max_list_pages,
                                request_delay=request_delay,
                                cookies=runtime["cookies"])
        if runtime["max_thread_pages"] is not None:
            scraper.max_thread_pages = runtime["max_thread_pages"]
        for raw_post in scraper.scrape_forum(
            entry,
            max_posts_per_thread=runtime["max_posts_per_thread"],
        ):
            yield raw_post, raw_post.to_forum_post()
    crawl_options = _resolve_crawl_options(
        crawl_mode=crawl_mode,
        max_posts_per_thread=max_posts_per_thread,
    )
