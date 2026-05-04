"""Collector for forum.antichat.xyz — Russian-language security forum (vBulletin).

Antichat is a long-running Russian hacking and security forum.  The public
content is accessible without login (vBulletin ``SECURITYTOKEN = "guest"``).

Thread discovery strategy
--------------------------
Two complementary approaches are combined:

1. **RSS feeds** — ``external.php?type=RSS2&forumids={fid}`` returns up to
   ~50 recent items per subforum.  Used for quick coverage of recent threads.

2. **Subforum pagination** — ``forumdisplay.php?f={fid}&page={N}`` provides
   historical thread discovery.  Thread IDs are extracted from ``id="tNNNN"``
   attributes on ``<td>`` elements.

Both sources feed the same thread-scraping pipeline.

Thread / post HTML structure (vBulletin 3.x/4.x hybrid)
---------------------------------------------------------
Post container:  ``<table … class="tborder" … id="postNNNNN">``
Post date:       ``<td colspan=2 class="thead">`` — text ``DD.MM.YYYY, HH:MM``
Username:        ``<div id="postmenu_{pid}"> … <a … class="bigusername …">NAME</a>``
Post body:       ``<div id="post_message_{pid}">…</div>``
Thread pagination: ``showthread.php?t={tid}&page={N}``

Security-relevant subforums
----------------------------
  74  Уязвимости (Vulnerabilities)
  41  Безопасность (Security)
  89  Избранное (Featured)
 110  Проверка на уязвимости (Vulnerability testing)
  94  Реверсинг (Reversing)
  24  C/C++/Delphi/.NET/Asm
  37  PHP/Perl/MySQL/JS
  30  Статьи (Articles)

Encoding
--------
The forum is UTF-8.  ``response.encoding = 'utf-8'`` is set explicitly.
Cyrillic text is preserved as-is in output JSONL (``ensure_ascii=False``).
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from xml.etree import ElementTree

import requests

from etg.collectors._common import author_hash, parse_date, post_id

log = logging.getLogger(__name__)

FORUM_ID = "antichat"
BASE = "https://forum.antichat.xyz"

# Security-relevant subforum IDs → human-readable name (Russian)
_SECURITY_SUBFORUMS: dict[int, str] = {
    74:  "Уязвимости",
    41:  "Безопасность",
    89:  "Избранное",
    110: "Проверка на уязвимости",
    94:  "Реверсинг",
    24:  "C/C++/Delphi/.NET/Asm",
    37:  "PHP/Perl/MySQL/JS",
    30:  "Статьи",
}

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"[ \t]{2,}")

# vBulletin post container: id="post494" (no underscore)
_POST_SPLIT_RE = re.compile(
    r'(?=<table[^>]+class="tborder"[^>]+id="post\d+)', re.IGNORECASE
)
_POST_ID_RE = re.compile(r'id="post(\d+)"', re.IGNORECASE)

# Date: DD.MM.YYYY, HH:MM  (inside thead cell)
_DATE_RE = re.compile(r'(\d{2})\.(\d{2})\.(\d{4}),\s*(\d{2}:\d{2})')

# Username inside postmenu div
_USER_RE = re.compile(
    r'<div\s+id="postmenu_\d+"[^>]*>.*?'
    r'<a[^>]+class="bigusername[^"]*"[^>]*>([^<]+)</a>',
    re.DOTALL | re.IGNORECASE,
)

# Thread ID from forumdisplay td id="tNNNNNN"
_THREAD_TD_RE = re.compile(r'<td[^>]+id="t(\d+)"', re.IGNORECASE)
# Thread title link inside the thread td
_THREAD_TITLE_RE = re.compile(
    r'<a\s+href="(?:showthread\.php\?t=\d+|thread\d+\.html)"[^>]*>([^<]+)</a>',
    re.IGNORECASE,
)

# Next-page check for thread pagination
_NEXT_PAGE_RE = re.compile(r'showthread\.php\?t=\d+&(?:amp;)?page=(\d+)', re.IGNORECASE)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    "Referer": f"{BASE}/",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    return s


def _get(session: requests.Session, url: str, delay: float = 1.5) -> str | None:
    """Fetch *url*; return HTML text (UTF-8) or None on failure."""
    try:
        r = session.get(url, timeout=30, allow_redirects=True)
        r.encoding = "utf-8"
        if r.status_code == 200:
            return r.text
        log.warning("antichat: HTTP %d for %s", r.status_code, url)
        return None
    except requests.RequestException as exc:
        log.warning("antichat: request error %s: %s", url, exc)
        return None
    finally:
        time.sleep(delay)


def _strip_html(html: str) -> str:
    """Remove HTML tags, decode entities, collapse whitespace.

    Preserves Cyrillic and other non-ASCII characters (UTF-8).
    """
    # Common HTML entity substitutions (order matters: &amp; last for safety)
    for old, new in (
        ("<br>", "\n"), ("<br/>", "\n"), ("<br />", "\n"),
        ("&nbsp;", " "), ("&#160;", " "),
        ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'"),
        ("&amp;", "&"),
    ):
        html = html.replace(old, new)
    text = _TAG_RE.sub(" ", html)
    # Collapse repeated horizontal whitespace but preserve newlines
    lines = [_SPACE_RE.sub(" ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def _parse_vb_date(raw: str) -> datetime:
    """Parse antichat date format ``DD.MM.YYYY, HH:MM`` → UTC datetime."""
    m = _DATE_RE.search(raw)
    if m:
        day, month, year, hhmm = m.group(1), m.group(2), m.group(3), m.group(4)
        try:
            dt = datetime.strptime(f"{day}.{month}.{year} {hhmm}", "%d.%m.%Y %H:%M")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return parse_date(raw)


# ---------------------------------------------------------------------------
# RSS-based thread discovery
# ---------------------------------------------------------------------------


def _thread_ids_from_rss(
    session: requests.Session,
    fid: int,
    delay: float = 1.5,
) -> list[tuple[int, str]]:
    """Return ``[(thread_id, title), …]`` from the subforum RSS feed."""
    if fid == 0:
        url = f"{BASE}/external.php?type=RSS2"
    else:
        url = f"{BASE}/external.php?type=RSS2&forumids={fid}"

    html = _get(session, url, delay=delay)
    if not html:
        return []

    # vBulletin may wrap RSS in outer HTML — extract the <rss> element
    rss_m = re.search(r"(<rss\b.*?</rss>)", html, re.DOTALL)
    if not rss_m:
        log.debug("antichat: no <rss> in feed for fid=%d", fid)
        return []

    try:
        root = ElementTree.fromstring(rss_m.group(1))
    except ElementTree.ParseError as exc:
        log.warning("antichat: RSS parse error fid=%d: %s", fid, exc)
        return []

    results: list[tuple[int, str]] = []
    for item in root.iter("item"):
        link_el = item.find("link")
        if link_el is None or not link_el.text:
            continue
        link = link_el.text.strip()
        tid_m = re.search(r"showthread\.php\?(?:t=)?(\d+)", link)
        if not tid_m:
            # Try slug-style: thread494.html
            tid_m = re.search(r"thread(\d+)\.html", link)
        if not tid_m:
            continue
        tid = int(tid_m.group(1))
        title_el = item.find("title")
        title = (title_el.text or "").strip() if title_el is not None else ""
        results.append((tid, title))

    log.debug("antichat: RSS fid=%d → %d threads", fid, len(results))
    return results


# ---------------------------------------------------------------------------
# Subforum thread-list scraping
# ---------------------------------------------------------------------------


def _iter_subforum_threads(
    session: requests.Session,
    fid: int,
    max_pages: int = 200,
    delay: float = 1.5,
) -> list[tuple[int, str]]:
    """Yield ``(thread_id, title)`` pairs from subforum listing pages.

    Iterates ``forumdisplay.php?f={fid}&page={N}`` until no thread IDs are
    found or ``max_pages`` is reached.
    """
    results: list[tuple[int, str]] = []
    for page in range(1, max_pages + 1):
        url = f"{BASE}/forumdisplay.php?f={fid}&page={page}"
        html = _get(session, url, delay=delay)
        if html is None:
            break

        # Find all thread td elements: id="tNNNNNN"
        tid_matches = _THREAD_TD_RE.findall(html)
        if not tid_matches:
            log.debug("antichat: subforum %d page %d — no thread IDs, stopping", fid, page)
            break

        for tid_str in tid_matches:
            tid = int(tid_str)
            # Try to find the matching title link — search in a window after each td
            title = f"thread-{tid}"
            results.append((tid, title))

        log.debug("antichat: subforum %d page %d → %d threads", fid, page, len(tid_matches))

    return results


# ---------------------------------------------------------------------------
# Thread page parsing
# ---------------------------------------------------------------------------


def _parse_posts_from_page(html: str, thread_url: str) -> list[dict]:
    """Extract posts from one HTML page of a thread.

    Returns a list of dicts with keys:
        ``post_id``, ``author``, ``date_str``, ``text``, ``url``
    """
    posts: list[dict] = []

    # Thread title for context
    title_m = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
    thread_title = ""
    if title_m:
        raw = unescape(title_m.group(1)).strip()
        # Strip site suffix like " - AntiChat Forum"
        thread_title = re.sub(r"\s*[-|]\s*AntiChat.*$", "", raw, flags=re.IGNORECASE).strip()

    # Split HTML into per-post sections on vBulletin post table boundaries
    sections = _POST_SPLIT_RE.split(html)

    for section in sections:
        # Post numeric ID from table id="postNNNN"
        pid_m = _POST_ID_RE.search(section)
        if not pid_m:
            continue
        raw_pid = pid_m.group(1)

        # Date from thead cell
        date_str = ""
        thead_m = re.search(
            r'<td[^>]+colspan[^>]*class="thead"[^>]*>(.*?)(?:</td>|#\d)',
            section, re.DOTALL | re.IGNORECASE,
        )
        if thead_m:
            date_str = _strip_html(thead_m.group(1)).strip()
        if not date_str:
            # Fallback: search raw text for DD.MM.YYYY pattern
            date_fallback = _DATE_RE.search(section)
            if date_fallback:
                date_str = date_fallback.group(0)

        # Username
        author = "unknown"
        user_m = _USER_RE.search(section)
        if user_m:
            author = unescape(user_m.group(1).strip())
            author = _strip_html(author)

        # Post body: <div id="post_message_{pid}">…</div>
        body_m = re.search(
            rf'<div\s+id="post_message_{raw_pid}"[^>]*>(.*?)</div>',
            section, re.DOTALL | re.IGNORECASE,
        )
        if not body_m:
            # Fallback: any post_message div
            body_m = re.search(
                r'<div\s+id="post_message_\d+"[^>]*>(.*?)</div>',
                section, re.DOTALL | re.IGNORECASE,
            )
        body = _strip_html(body_m.group(1)) if body_m else ""

        if len(body) < 5:
            continue

        posts.append({
            "post_id": raw_pid,
            "author": author,
            "date_str": date_str,
            "text": body,
            "thread_title": thread_title,
            "url": thread_url,
        })

    return posts


def _iter_thread_posts(
    session: requests.Session,
    tid: int,
    max_pages: int = 200,
    delay: float = 1.5,
):
    """Yield post dicts for all pages of thread *tid*."""
    base_url = f"{BASE}/showthread.php?t={tid}"

    html = _get(session, base_url, delay=delay)
    if html is None:
        return

    # Discover total pages from navigation links
    page_links = _NEXT_PAGE_RE.findall(html)
    last_page = max((int(p) for p in page_links), default=1)
    last_page = min(last_page, max_pages)

    for page_num in range(1, last_page + 1):
        if page_num == 1:
            page_html = html
            url = base_url
        else:
            url = f"{BASE}/showthread.php?t={tid}&page={page_num}"
            page_html = _get(session, url, delay=delay)
            if page_html is None:
                break

        posts = _parse_posts_from_page(page_html, base_url)
        if not posts and page_num > 1:
            break

        for p in posts:
            p["page"] = page_num
            yield p


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect(
    output: Path = Path("data/antichat_posts.jsonl"),
    cve_index_output: Path = Path("data/antichat_cve_index.jsonl"),
    request_delay: float = 1.5,
    max_pages: int = 200,
) -> tuple[int, int]:
    """Scrape forum.antichat.xyz security subforums and write JSONL output.

    Parameters
    ----------
    output:
        Path for the posts JSONL file.
    cve_index_output:
        Path for the CVE-index JSONL file.
    request_delay:
        Seconds to sleep between HTTP requests.
    max_pages:
        Maximum pages to fetch per subforum (thread listing) and per thread.

    Returns
    -------
    (post_count, cve_index_count)
        Number of posts written and number of CVE-index entries written.
    """
    output = Path(output)
    cve_index_output = Path(cve_index_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cve_index_output.parent.mkdir(parents=True, exist_ok=True)

    session = _make_session()
    seen_thread_ids: set[int] = set()
    seen_post_ids: set[str] = set()
    post_count = 0
    cve_index_count = 0

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        for fid, fname in _SECURITY_SUBFORUMS.items():
            log.info("antichat: collecting subforum %d (%s)", fid, fname)

            # --- 1. RSS-based recent threads ---
            rss_threads = _thread_ids_from_rss(session, fid, delay=request_delay)
            for tid, title in rss_threads:
                if tid not in seen_thread_ids:
                    seen_thread_ids.add(tid)

            # --- 2. Historical threads via forumdisplay pagination ---
            html_threads = _iter_subforum_threads(
                session, fid, max_pages=max_pages, delay=request_delay
            )
            for tid, title in html_threads:
                if tid not in seen_thread_ids:
                    seen_thread_ids.add(tid)

        log.info("antichat: discovered %d unique threads total", len(seen_thread_ids))

        # Re-collect RSS for titles (already done above; now process all threads)
        # Build a combined list: re-fetch RSS titles for each subforum
        thread_list: list[tuple[int, str]] = []
        seen_for_list: set[int] = set()

        for fid, fname in _SECURITY_SUBFORUMS.items():
            rss_threads = _thread_ids_from_rss(session, fid, delay=request_delay)
            for tid, title in rss_threads:
                if tid not in seen_for_list:
                    seen_for_list.add(tid)
                    thread_list.append((tid, title or f"thread-{tid}"))

            html_threads = _iter_subforum_threads(
                session, fid, max_pages=max_pages, delay=request_delay
            )
            for tid, title in html_threads:
                if tid not in seen_for_list:
                    seen_for_list.add(tid)
                    thread_list.append((tid, title or f"thread-{tid}"))

        log.info("antichat: scraping %d threads", len(thread_list))

        for thread_num, (tid, thread_title_hint) in enumerate(thread_list):
            log.debug(
                "antichat: thread %d/%d (t=%d)",
                thread_num + 1, len(thread_list), tid,
            )
            for post_data in _iter_thread_posts(
                session, tid, max_pages=max_pages, delay=request_delay
            ):
                raw_pid = post_data["post_id"]
                author = post_data["author"]
                body = post_data["text"]
                thread_title = post_data.get("thread_title") or thread_title_hint

                # Prefix first post with thread title for better CVE context
                if post_data.get("page", 1) == 1:
                    full_text = f"{thread_title}\n\n{body}" if thread_title else body
                else:
                    full_text = body

                date_str = post_data.get("date_str", "")
                ts = _parse_vb_date(date_str) if date_str else datetime.now(tz=timezone.utc)

                pid = post_id(FORUM_ID, f"thread::{tid}::{raw_pid}")
                if pid in seen_post_ids:
                    continue
                seen_post_ids.add(pid)

                ahash = author_hash(FORUM_ID, author)
                cves = list({m.upper() for m in _CVE_RE.findall(full_text)})

                post_dict = {
                    "id": pid,
                    "text": full_text,
                    "timestamp": ts.isoformat(),
                    "forum_id": FORUM_ID,
                    "author_hash": ahash,
                }
                post_fh.write(json.dumps(post_dict, ensure_ascii=False) + "\n")
                post_count += 1

                for cve in cves:
                    entry = {
                        "exploit_id": pid,
                        "cve_id": cve,
                        "exploit_text": full_text[:500],
                        "title": thread_title,
                        "published": ts.isoformat(),
                        "source": FORUM_ID,
                    }
                    cve_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    cve_index_count += 1

                if post_count % 500 == 0:
                    log.info(
                        "antichat: %d posts, %d CVE refs written so far",
                        post_count, cve_index_count,
                    )

    log.info(
        "antichat: done — %d posts, %d CVE index entries → %s",
        post_count, cve_index_count, output,
    )
    return post_count, cve_index_count


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Scrape forum.antichat.xyz security subforums"
    )
    parser.add_argument("--output", default="data/antichat_posts.jsonl")
    parser.add_argument("--cve-index", default="data/antichat_cve_index.jsonl")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between requests")
    parser.add_argument("--max-pages", type=int, default=200, help="Max pages per subforum/thread")
    args = parser.parse_args()

    posts, cves = collect(
        output=Path(args.output),
        cve_index_output=Path(args.cve_index),
        request_delay=args.delay,
        max_pages=args.max_pages,
    )
    print(f"Done: {posts} posts, {cves} CVE index entries")
