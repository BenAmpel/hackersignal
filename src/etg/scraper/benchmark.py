"""Small live benchmark runner for forum scraper coverage checks."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Optional

from etg.scraper.detector import ForumKind, detect
from etg.scraper.forum_index import ForumEntry, online_entries
from etg.scraper.pipeline import _make_scraper
from etg.scraper.scrapers.base import make_fetcher, resp_html

log = logging.getLogger(__name__)


def _resolve_entries(index_source, include_onion: bool, forum_filter: list[str] | None) -> list[ForumEntry]:
    entries = online_entries(index_source)
    if not include_onion:
        entries = [entry for entry in entries if not entry.is_onion]
    if forum_filter:
        needles = [needle.lower() for needle in forum_filter]
        entries = [
            entry for entry in entries
            if any(needle in entry.forum_id.lower() or needle in entry.name.lower() for needle in needles)
        ]
    return entries


def _detect_kind(entry: ForumEntry, proxy: Optional[str]) -> ForumKind:
    kind = detect(entry.url)
    if kind != ForumKind.UNKNOWN:
        return kind
    try:
        resp = make_fetcher().get(entry.url, timeout=12, proxy=proxy)
    except Exception as exc:
        log.debug("Benchmark detect fetch error for %s: %s", entry.url, exc)
        return ForumKind.UNKNOWN
    return detect(entry.url, resp_html(resp))


def _benchmark_one(
    entry: ForumEntry,
    *,
    proxy: Optional[str],
    max_thread_urls: int,
    max_posts: int,
    max_list_pages: Optional[int],
    request_delay: Optional[float],
    cookies=None,
) -> dict[str, Any]:
    kind = _detect_kind(entry, proxy)
    scraper = _make_scraper(
        kind,
        proxy=proxy,
        max_threads=max_thread_urls,
        max_list_pages=max_list_pages,
        request_delay=request_delay,
        cookies=cookies,
    )

    thread_candidates: list[str] = []
    seen_threads: set[str] = set()
    status = "ok"
    error = ""
    try:
        for thread_url in scraper.thread_urls(entry.url):
            if thread_url in seen_threads:
                continue
            seen_threads.add(thread_url)
            thread_candidates.append(thread_url)
            if len(thread_candidates) >= max_thread_urls:
                break
    except Exception as exc:
        status = "discovery_error"
        error = str(exc)

    posts = 0
    threads_scraped = 0
    sections: set[str] = set()
    if status == "ok":
        for thread_url in thread_candidates:
            thread_posts = 0
            try:
                for post in scraper.scrape_thread(thread_url, entry):
                    if len(post.body) < scraper.min_body_len:
                        continue
                    posts += 1
                    thread_posts += 1
                    if post.section:
                        sections.add(post.section)
                    if posts >= max_posts:
                        break
            except Exception as exc:
                status = "thread_error"
                error = str(exc)
                break
            if thread_posts:
                threads_scraped += 1
            if posts >= max_posts:
                break

    if status == "ok" and scraper._dead:
        status = "dead_host"
    elif status == "ok" and not thread_candidates:
        status = "zero_threads"
    elif status == "ok" and posts == 0:
        status = "zero_posts"

    return {
        "forum_id": entry.forum_id,
        "name": entry.name,
        "url": entry.url,
        "detected_kind": kind.name,
        "status": status,
        "thread_candidates": len(thread_candidates),
        "threads_scraped": threads_scraped,
        "posts": posts,
        "sections": sorted(sections)[:8],
        "error": error,
    }


def _render_markdown(results: list[dict[str, Any]], *, max_thread_urls: int, max_posts: int) -> str:
    lines = [
        "# deepdarkCTI scraper benchmark",
        "",
        f"- thread candidate cap per forum: {max_thread_urls}",
        f"- post cap per benchmark run: {max_posts}",
        "",
        "| Forum | Kind | Status | Thread candidates | Productive threads | Posts | Sections |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        sections = ", ".join(result["sections"]) if result["sections"] else "-"
        lines.append(
            "| {name} | {kind} | {status} | {candidates} | {scraped} | {posts} | {sections} |".format(
                name=result["name"],
                kind=result["detected_kind"],
                status=result["status"],
                candidates=result["thread_candidates"],
                scraped=result["threads_scraped"],
                posts=result["posts"],
                sections=sections,
            )
        )
        if result["error"]:
            lines.append(f"|  |  | error: `{result['error']}` |  |  |  |  |")
    return "\n".join(lines) + "\n"


def run_benchmark(
    *,
    markdown_output: str | Path,
    json_output: str | Path | None = None,
    index_source=None,
    include_onion: bool = False,
    forum_filter: list[str] | None = None,
    tor_proxy: str = "socks5://127.0.0.1:9050",
    max_thread_urls: int = 3,
    max_posts: int = 20,
    max_list_pages: Optional[int] = 3,
    per_forum_timeout: int = 60,
    request_delay: Optional[float] = None,
    cookies=None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    entries = _resolve_entries(index_source, include_onion, forum_filter)
    for entry in entries:
        proxy = tor_proxy if entry.is_onion else None
        log.info("Benchmarking %s (%s)", entry.name, entry.url)
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(
            _benchmark_one,
            entry,
            proxy=proxy,
            max_thread_urls=max_thread_urls,
            max_posts=max_posts,
            max_list_pages=max_list_pages,
            request_delay=request_delay,
            cookies=cookies,
        )
        try:
            results.append(future.result(timeout=per_forum_timeout))
        except FutureTimeoutError:
            future.cancel()
            results.append(
                {
                    "forum_id": entry.forum_id,
                    "name": entry.name,
                    "url": entry.url,
                    "detected_kind": "UNKNOWN",
                    "status": "timeout",
                    "thread_candidates": 0,
                    "threads_scraped": 0,
                    "posts": 0,
                    "sections": [],
                    "error": f"Timed out after {per_forum_timeout}s",
                }
            )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    markdown_path = Path(markdown_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        _render_markdown(results, max_thread_urls=max_thread_urls, max_posts=max_posts),
        encoding="utf-8",
    )
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
