"""Collector for go4expert.com — XenForo 1.x security/tech forum.

Uses requests with browser-like headers (no JS rendering required).

URL structure
-------------
Forum list:          https://www.go4expert.com/forums/
Thread list:         https://www.go4expert.com/forums/{slug}/
Thread list page N:  https://www.go4expert.com/forums/{slug}/page-{N}/
Thread:              https://www.go4expert.com/forums/{slug}-t{id}/
Thread page N:       https://www.go4expert.com/forums/{slug}-t{id}/page-{N}/

HTML structure (XenForo 1.x)
-----------------------------
Thread list items:  <li id="thread-NNNNN" …>
Thread link:        <h3 class="title"><a href="forums/{slug}-t{id}/">
Posts:              <li class="message" data-author="username">
Post body:          <blockquote class="messageText SelectQuoteContainer …">
Post date:          <span class="DateTime" title="Mon DD, YYYY at HH:MM AM">
Thread title:       <h1> first occurrence
Last page:          <div class="PageNav" data-last="N">
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

import requests

from etg.collectors._common import author_hash, parse_date, post_id

log = logging.getLogger(__name__)

FORUM_ID = "go4expert"
BASE = "https://www.go4expert.com"

# Subforums most relevant for security/exploit research
_SECURITY_SUBFORUMS = [
    "forums/ethical-hacking-forum/",
    "forums/ransomware-support/",
    "forums/linux-forum/",
    "forums/unix-forum/",
    "forums/shell-script/",
    "forums/operating-system-forum/",
    "forums/engineering-concepts/",
    "forums/windows-forum/",
    "forums/programming-forum/",
    "forums/web-development-forum/",
    "forums/database-forum/",
    "forums/assembly-language-programming-forum/",
]

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s{2,}")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
}

# Date format used in span.DateTime title attribute: "Jun 28, 2015 at 11:58 AM"
_DATE_RE = re.compile(
    r"(\w+ \d{1,2},\s+\d{4})\s+at\s+(\d{1,2}:\d{2}\s+[AP]M)"
)


def _strip_html(html: str) -> str:
    """Remove HTML tags and decode entities; collapse whitespace."""
    for ent, ch in (
        ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
        ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " "), ("&#160;", " "),
    ):
        html = html.replace(ent, ch)
    text = _TAG_RE.sub(" ", html)
    return _SPACE_RE.sub(" ", text).strip()


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    return s


def _get(session: requests.Session, url: str, delay: float = 1.5) -> str | None:
    """Fetch *url* and return HTML string, or None on failure."""
    try:
        r = session.get(url, timeout=20, allow_redirects=True)
        if r.status_code == 200:
            return r.text
        log.warning("go4expert: HTTP %d for %s", r.status_code, url)
        return None
    except requests.RequestException as exc:
        log.warning("go4expert: request error for %s: %s", url, exc)
        return None
    finally:
        time.sleep(delay)


def _last_page(html: str) -> int:
    """Return the last page number from a PageNav element, or 1 if not found."""
    m = re.search(r'<div[^>]+class="PageNav"[^>]+data-last="(\d+)"', html)
    return int(m.group(1)) if m else 1


def _thread_links_from_page(html: str) -> list[str]:
    """Extract thread slugs (relative paths) from a forum listing page."""
    # Links appear as href="forums/{slug}-t{id}/" relative to BASE
    return re.findall(r'href="(forums/[a-z0-9\-]+t\d+/)"', html)


def _iter_subforum_threads(
    session: requests.Session,
    subforum_slug: str,  # e.g. "forums/ethical-hacking-forum/"
    max_pages: int = 500,
    delay: float = 1.5,
) -> list[str]:
    """Yield all thread relative paths in a subforum, following pagination."""
    threads: list[str] = []
    seen_slugs: set[str] = set()

    for page_num in range(1, max_pages + 1):
        if page_num == 1:
            url = f"{BASE}/{subforum_slug}"
        else:
            # XenForo 1.x pagination: no trailing slash on page-N URLs
            slug_base = subforum_slug.rstrip("/")
            url = f"{BASE}/{slug_base}/page-{page_num}"

        html = _get(session, url, delay=delay)
        if html is None:
            break

        page_threads = _thread_links_from_page(html)
        new_this_page = 0
        for slug in page_threads:
            if slug not in seen_slugs:
                seen_slugs.add(slug)
                threads.append(slug)
                new_this_page += 1

        last = _last_page(html)
        log.debug(
            "go4expert: %s page %d/%d — %d threads (+%d new)",
            subforum_slug, page_num, last, len(threads), new_this_page,
        )

        if page_num >= last:
            break

    return threads


def _parse_posts_from_page(html: str, thread_url: str, page_offset: int = 0):
    """Yield (post_data_dict) for each post on a thread HTML page."""
    # Find all post containers: <li id="post-NNNNN" class="message" data-author="...">
    post_pattern = re.compile(
        r'<li[^>]+id="post-(\d+)"[^>]+class="message[^"]*"[^>]+data-author="([^"]*)"[^>]*>'
        r'(.*?)'
        r'(?=<li[^>]+id="post-\d+"|</ol>)',
        re.DOTALL,
    )

    thread_title_m = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.DOTALL)
    thread_title = _strip_html(thread_title_m.group(1)) if thread_title_m else ""

    for idx, m in enumerate(post_pattern.finditer(html)):
        post_num = m.group(1)
        author = unescape(m.group(2).strip())
        post_html = m.group(3)

        # Extract date from span.DateTime title attribute
        date_m = re.search(r'<span[^>]+class="DateTime"[^>]+title="([^"]+)"', post_html)
        date_str = ""
        if date_m:
            raw = date_m.group(1).strip()
            dm = _DATE_RE.match(raw)
            if dm:
                date_str = f"{dm.group(1)} {dm.group(2)}"

        # Extract body from blockquote.messageText
        body_m = re.search(
            r'<blockquote[^>]+class="messageText[^"]*"[^>]*>(.*?)</blockquote>',
            post_html, re.DOTALL,
        )
        body = _strip_html(body_m.group(1)) if body_m else ""

        if len(body) < 10:
            continue

        yield {
            "post_num": post_num,
            "author": author or "unknown",
            "date_str": date_str,
            "body": body,
            "thread_title": thread_title,
            "index": page_offset + idx,
        }


def _iter_thread_posts(
    session: requests.Session,
    thread_slug: str,  # e.g. "forums/password-hacking-t12345/"
    max_pages: int = 200,
    delay: float = 1.5,
):
    """Yield post dicts for all posts across all pages of a thread."""
    post_offset = 0
    thread_url = f"{BASE}/{thread_slug}"

    for page_num in range(1, max_pages + 1):
        if page_num == 1:
            url = thread_url
        else:
            # XenForo 1.x: thread pagination uses page-N without trailing slash
            thread_base = thread_url.rstrip("/")
            url = f"{thread_base}/page-{page_num}"

        html = _get(session, url, delay=delay)
        if html is None:
            break

        page_posts = list(_parse_posts_from_page(html, thread_url, post_offset))
        if not page_posts:
            break

        for p in page_posts:
            p["thread_url"] = thread_url
            yield p

        post_offset += len(page_posts)

        last = _last_page(html)
        if page_num >= last:
            break


def _make_forum_post(p: dict) -> tuple:
    """Return (ForumPost-dict, cve-list) from a parsed post dict."""
    thread_url = p["thread_url"]
    author = p["author"]
    body = p["body"]
    thread_title = p["thread_title"]
    post_num = p["post_num"]

    full_text = f"{thread_title}\n\n{body}" if p["index"] == 0 and thread_title else body

    ts = parse_date(p["date_str"])
    pid = post_id(FORUM_ID, f"{thread_url}::{post_num}")
    ahash = author_hash(FORUM_ID, author)

    cves = list({m.upper() for m in _CVE_RE.findall(full_text)})

    post_dict = {
        "id": pid,
        "text": full_text,
        "timestamp": ts.isoformat(),
        "forum_id": FORUM_ID,
        "author_hash": ahash,
    }
    return post_dict, cves


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect(
    output: Path = Path("data/go4expert_posts.jsonl"),
    cve_index_output: Path = Path("data/go4expert_cve_index.jsonl"),
    subforums: list[str] | None = None,
    max_list_pages: int = 500,
    max_thread_pages: int = 200,
    request_delay: float = 1.5,
    log_every: int = 100,
) -> tuple[int, int]:
    """Scrape go4expert.com and write ForumPost + CVE-index JSONL.

    Parameters
    ----------
    subforums:
        List of subforum slugs (relative to BASE) to scrape.
        Defaults to :data:`_SECURITY_SUBFORUMS`.
    max_list_pages:
        Maximum thread-list pages to follow per subforum.
    max_thread_pages:
        Maximum pages to fetch per thread.
    request_delay:
        Seconds to sleep between requests.
    log_every:
        Log progress every N posts.

    Returns
    -------
    (post_count, cve_index_count)
    """
    output = Path(output)
    cve_index_output = Path(cve_index_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cve_index_output.parent.mkdir(parents=True, exist_ok=True)

    if subforums is None:
        subforums = _SECURITY_SUBFORUMS

    session = _make_session()
    seen_thread_slugs: set[str] = set()
    seen_post_ids: set[str] = set()
    post_count = 0
    cve_index_count = 0

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        for subforum_slug in subforums:
            log.info("go4expert: scanning subforum %s", subforum_slug)
            thread_slugs = _iter_subforum_threads(
                session, subforum_slug,
                max_pages=max_list_pages,
                delay=request_delay,
            )
            new_threads = [s for s in thread_slugs if s not in seen_thread_slugs]
            seen_thread_slugs.update(new_threads)
            log.info(
                "go4expert: %s — %d threads (%d new)",
                subforum_slug, len(thread_slugs), len(new_threads),
            )

            for t_slug in new_threads:
                for post_data in _iter_thread_posts(
                    session, t_slug,
                    max_pages=max_thread_pages,
                    delay=request_delay,
                ):
                    post_dict, cves = _make_forum_post(post_data)
                    pid = post_dict["id"]

                    if pid in seen_post_ids:
                        continue
                    seen_post_ids.add(pid)

                    post_fh.write(json.dumps(post_dict) + "\n")
                    post_fh.flush()
                    post_count += 1

                    for cve in cves:
                        entry = {
                            "exploit_id": pid,
                            "cve_id": cve,
                            "exploit_text": post_dict["text"][:500],
                            "title": post_data.get("thread_title", ""),
                            "published": post_dict["timestamp"],
                            "source": FORUM_ID,
                        }
                        cve_fh.write(json.dumps(entry) + "\n")
                        cve_fh.flush()
                        cve_index_count += 1

                    if post_count % log_every == 0:
                        log.info(
                            "go4expert: %d posts, %d CVE refs so far",
                            post_count, cve_index_count,
                        )

    log.info(
        "go4expert: done — %d posts, %d CVE index entries → %s",
        post_count, cve_index_count, output,
    )
    return post_count, cve_index_count
