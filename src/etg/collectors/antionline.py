"""Collector for antionline.com — vBulletin 4.x computer security forum.

Thread discovery strategy
--------------------------
The forum thread list (``forumdisplay.php``) renders thread links via
JavaScript (AJAX), so plain HTTP requests return no thread links.  Instead we
use the site's built-in RSS feeds, which are statically served:

    https://www.antionline.com/external.php?type=RSS2&forumids={id}

Each feed returns up to ~50 recent items.  Combined with per-subforum feeds
across all security-related forums this gives broad coverage of recent content.

Historical antionline posts are already covered by the ``hackerExploits.h5``
dataset (≈277 k posts), so this collector focuses on **new posts** not yet in
the pipeline.

Thread content HTML structure (vBulletin 4.x)
----------------------------------------------
Post container:  <div id="post_{post_id}">
Post body:       <blockquote class="postcontent restore "> (strip HTML)
Author:          <a class="username …"><strong>username</strong></a>
Date:            <span class="date">Month DDth/rd/st/nd, YYYY, …</span>
Pagination:      link rel="next" (or ``showthread.php?t={tid}&page={n}``)

Security subforums (by forum ID):
  55  Security Discussions     83  Security News
  56  IDS & Scanner            52  AntiVirus
  78  Wireless Security        57  Firewall & Honeypot
  45  Newbie Security          71  Spyware/Adware
  35  Microsoft Security       39  Mac Security
  40  Network Security         82  Phishing & Cyber Scams
  46  Computer Forensics       67  Web Security
  60  Cryptography             47  Programming Security
  37  Misc Security
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
from xml.etree import ElementTree

import requests

from etg.collectors._common import author_hash, parse_date, post_id

log = logging.getLogger(__name__)

FORUM_ID = "antionline"
BASE = "https://www.antionline.com"

# Security-focused forum IDs (from the main page forum listing)
_SECURITY_FORUM_IDS: list[int] = [
    55,  # Security Discussions
    83,  # Security News
    56,  # IDS & Scanner Discussions
    52,  # AntiVirus Discussions
    78,  # Wireless Security
    57,  # Firewall & Honeypot Discussions
    45,  # Newbie Security Questions
    71,  # Spyware / Adware
    35,  # Microsoft Security Discussions
    39,  # Mac Security Discussions
    40,  # Network Security Discussions
    82,  # Phishing and Cyber Scams
    46,  # Computer Forensics
    67,  # Web Security
    60,  # Cryptography / Steganography
    47,  # Programming Security
    37,  # Miscellaneous Security Discussions
]

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s{2,}")

# vBulletin 4.x date format: "March 21st, 2026, 12:35 PM"
# The ordinal suffix varies: st, nd, rd, th
_DATE_RE = re.compile(
    r"(\w+)\s+(\d{1,2})(?:st|nd|rd|th),\s+(\d{4})(?:,\s+(\d{1,2}:\d{2}\s+[AP]M))?",
    re.IGNORECASE,
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.antionline.com/",
}


def _strip_html(html: str) -> str:
    """Remove HTML tags and decode common entities; collapse whitespace."""
    for ent, ch in (
        ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
        ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " "), ("&#160;", " "),
        ("<br>", "\n"), ("<br/>", "\n"), ("<br />", "\n"),
    ):
        html = html.replace(ent, ch)
    text = _TAG_RE.sub(" ", html)
    return _SPACE_RE.sub(" ", text).strip()


def _parse_vb_date(raw: str) -> datetime:
    """Parse vBulletin 4-style date string → UTC datetime."""
    raw = re.sub(r"\s+", " ", raw.strip())
    m = _DATE_RE.search(raw)
    if m:
        month, day, year = m.group(1), m.group(2), m.group(3)
        time_str = m.group(4) or "12:00 AM"
        try:
            dt = datetime.strptime(f"{month} {day} {year} {time_str}", "%B %d %Y %I:%M %p")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return parse_date(raw)


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    return s


def _get(session: requests.Session, url: str, delay: float = 1.5) -> str | None:
    """Fetch *url*; return HTML string or None on failure."""
    try:
        r = session.get(url, timeout=25, allow_redirects=True)
        if r.status_code == 200:
            return r.text
        log.warning("antionline: HTTP %d for %s", r.status_code, url)
        return None
    except requests.RequestException as exc:
        log.warning("antionline: request error %s: %s", url, exc)
        return None
    finally:
        time.sleep(delay)


# ---------------------------------------------------------------------------
# RSS-based thread discovery
# ---------------------------------------------------------------------------


def _thread_ids_from_rss(
    session: requests.Session,
    forum_id: int,
    delay: float = 1.5,
) -> list[tuple[int, str, str]]:
    """Return list of (thread_id, thread_url, description_html) from forum RSS.

    The RSS gives us the latest ~50 threads per subforum.
    """
    url = f"{BASE}/external.php?type=RSS2&forumids={forum_id}"
    html = _get(session, url, delay=delay)
    if not html:
        return []

    results: list[tuple[int, str, str]] = []
    try:
        # RSS is wrapped in outer <html><body> tags (vBulletin quirk)
        # Strip those before XML parsing
        rss_match = re.search(r"(<rss.*</rss>)", html, re.DOTALL)
        if not rss_match:
            log.debug("antionline: no <rss> in RSS for forum %d", forum_id)
            return []
        root = ElementTree.fromstring(rss_match.group(1))
    except ElementTree.ParseError as exc:
        log.warning("antionline: RSS parse error for forum %d: %s", forum_id, exc)
        return []

    ns = {"content": "http://purl.org/rss/1.0/modules/content/"}

    for item in root.iter("item"):
        link_el = item.find("link")
        if link_el is None or not link_el.text:
            continue
        link = link_el.text.strip()
        # Extract thread ID from URL pattern:
        # showthread.php?296872-slug&goto=newpost  OR  showthread.php?t=296872
        tid_m = re.search(r"showthread\.php\?(?:t=)?(\d+)", link)
        if not tid_m:
            continue
        tid = int(tid_m.group(1))
        # Use content:encoded if available, else description
        body_el = item.find("content:encoded", ns)
        if body_el is None:
            body_el = item.find("description")
        desc = (body_el.text or "") if body_el is not None else ""
        results.append((tid, link, desc))

    log.debug(
        "antionline: RSS forum %d → %d threads", forum_id, len(results)
    )
    return results


# ---------------------------------------------------------------------------
# Thread page scraping
# ---------------------------------------------------------------------------


def _parse_thread_page(html: str, thread_url: str) -> list[dict]:
    """Extract all posts from one page of a vBulletin thread.

    Returns a list of post-dicts with keys:
        post_id, author, date_str, body, thread_title
    """
    posts: list[dict] = []

    # Thread title
    title_m = re.search(r"<title>([^<]+)</title>", html)
    thread_title = ""
    if title_m:
        raw = unescape(title_m.group(1)).strip()
        # Remove site suffix e.g. "Thread Title - Antionline Forums…"
        thread_title = re.sub(r"\s*-\s*Antionline.*$", "", raw, flags=re.IGNORECASE).strip()

    # Split HTML into per-post sections.
    # vBulletin 4.x uses <li id="post_NNN"> as the post container.
    post_sections = re.split(r'(?=<li[^>]+id="post_\d+)', html)

    for section in post_sections:
        # Post ID
        post_id_m = re.match(r'<li[^>]+id="post_(\d+)"', section)
        if not post_id_m:
            continue
        post_num = post_id_m.group(1)

        # Author — inside username_container > a.username > strong
        author_m = re.search(
            r'class="username[^"]*"[^>]*>(?:<[^>]+>)*\s*([^<\s][^<]+?)\s*(?:</[^>]+>)*</a>',
            section,
        )
        if not author_m:
            author_m = re.search(r'<strong>([^<]+)</strong>', section)
        author = unescape(author_m.group(1).strip()) if author_m else "unknown"
        # Remove any remaining tags from author
        author = _strip_html(author)

        # Date — span.date
        date_m = re.search(r'<span[^>]+class="date"[^>]*>(.*?)</span>', section, re.DOTALL)
        date_str = ""
        if date_m:
            # Remove inner <span class="time"> wrapper to get combined text
            date_raw = re.sub(r"<span[^>]+>", "", date_m.group(1))
            date_raw = re.sub(r"</span>", "", date_raw)
            date_str = unescape(date_raw).strip().replace("\xa0", " ")

        # Body — blockquote.postcontent (class may have trailing space)
        body_m = re.search(
            r'<blockquote[^>]+class="postcontent[^"]*"[^>]*>(.*?)</blockquote>',
            section, re.DOTALL,
        )
        body = _strip_html(body_m.group(1)) if body_m else ""

        if len(body) < 10:
            continue

        posts.append({
            "post_id": post_num,
            "author": author,
            "date_str": date_str,
            "body": body,
            "thread_title": thread_title,
            "thread_url": thread_url,
        })

    return posts


def _iter_thread_posts(
    session: requests.Session,
    thread_id: int,
    max_pages: int = 200,
    delay: float = 1.5,
):
    """Yield post dicts for all pages of a thread."""
    base_url = f"{BASE}/showthread.php?t={thread_id}"

    # First page also sets the thread title and discovers total pages
    html = _get(session, base_url, delay=delay)
    if html is None:
        return

    # Find total pages: look for "Page 1 of N" or navigation links
    last_page = 1
    pages_m = re.search(r"Page\s+\d+\s+of\s+(\d+)", html, re.IGNORECASE)
    if pages_m:
        last_page = int(pages_m.group(1))
    else:
        # Try rel="last" link or highest page link
        page_links = re.findall(r"showthread\.php\?t=\d+&amp;page=(\d+)", html)
        if page_links:
            last_page = max(int(p) for p in page_links)

    last_page = min(last_page, max_pages)

    for page_num in range(1, last_page + 1):
        if page_num == 1:
            url = base_url
            page_html = html  # already fetched
        else:
            url = f"{BASE}/showthread.php?t={thread_id}&page={page_num}"
            page_html = _get(session, url, delay=delay)
            if page_html is None:
                break

        posts = _parse_thread_page(page_html, base_url)
        if not posts:
            break

        for p in posts:
            p["page"] = page_num
            yield p


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect(
    output: Path = Path("data/antionline_posts.jsonl"),
    cve_index_output: Path = Path("data/antionline_cve_index.jsonl"),
    forum_ids: list[int] | None = None,
    max_thread_pages: int = 200,
    request_delay: float = 2.0,
    log_every: int = 100,
) -> tuple[int, int]:
    """Scrape antionline.com (security subforums) via RSS + thread fetching.

    Parameters
    ----------
    forum_ids:
        vBulletin forum IDs to collect from.
        Defaults to :data:`_SECURITY_FORUM_IDS`.
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

    if forum_ids is None:
        forum_ids = _SECURITY_FORUM_IDS

    session = _make_session()
    seen_thread_ids: set[int] = set()
    seen_post_ids: set[str] = set()
    post_count = 0
    cve_index_count = 0

    # --- collect all thread IDs via RSS ---
    all_threads: list[tuple[int, str, str]] = []

    # Per-forum RSS feeds (security subforums)
    for fid in forum_ids:
        log.info("antionline: fetching RSS for forum %d", fid)
        threads = _thread_ids_from_rss(session, fid, delay=request_delay)
        for tid, url, desc in threads:
            if tid not in seen_thread_ids:
                seen_thread_ids.add(tid)
                all_threads.append((tid, url, desc))

    # Fall back to the global RSS if per-forum feeds are empty
    if not all_threads:
        log.info("antionline: per-forum RSS empty, trying global RSS feed")
        global_threads = _thread_ids_from_rss(session, 0, delay=request_delay)
        # forumids=0 is ignored by vBulletin; we use `type=RSS2` without forumids
        # Redo with the global URL directly
        html = _get(session, f"{BASE}/external.php?type=RSS2", delay=request_delay)
        if html:
            try:
                rss_match = re.search(r"(<rss.*</rss>)", html, re.DOTALL)
                if rss_match:
                    from xml.etree import ElementTree
                    root = ElementTree.fromstring(rss_match.group(1))
                    ns = {"content": "http://purl.org/rss/1.0/modules/content/"}
                    for item in root.iter("item"):
                        link_el = item.find("link")
                        if link_el is None or not link_el.text:
                            continue
                        link = link_el.text.strip()
                        tid_m = re.search(r"showthread\.php\?(?:t=)?(\d+)", link)
                        if not tid_m:
                            continue
                        tid = int(tid_m.group(1))
                        body_el = item.find("content:encoded", ns)
                        if body_el is None:
                            body_el = item.find("description")
                        desc = (body_el.text or "") if body_el is not None else ""
                        if tid not in seen_thread_ids:
                            seen_thread_ids.add(tid)
                            all_threads.append((tid, link, desc))
            except Exception as exc:
                log.warning("antionline: global RSS parse error: %s", exc)

    log.info("antionline: discovered %d unique threads from RSS feeds", len(all_threads))

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        # --- also write RSS description-derived posts (quick coverage) ---
        for tid, thread_url, desc_html in all_threads:
            if not desc_html:
                continue
            body = _strip_html(desc_html)
            if len(body) < 20:
                continue
            pid = post_id(FORUM_ID, f"rss::{tid}::0")
            ahash = author_hash(FORUM_ID, "rss_author")
            cves = list({m.upper() for m in _CVE_RE.findall(body)})
            ts = datetime.now(tz=timezone.utc)
            post_dict = {
                "id": pid,
                "text": body,
                "timestamp": ts.isoformat(),
                "forum_id": FORUM_ID,
                "author_hash": ahash,
            }
            if pid not in seen_post_ids:
                seen_post_ids.add(pid)
                post_fh.write(json.dumps(post_dict) + "\n")
                post_count += 1
                for cve in cves:
                    entry = {
                        "exploit_id": pid,
                        "cve_id": cve,
                        "exploit_text": body[:500],
                        "title": "",
                        "published": ts.isoformat(),
                        "source": FORUM_ID,
                    }
                    cve_fh.write(json.dumps(entry) + "\n")
                    cve_index_count += 1

        # --- scrape each thread for full content ---
        for thread_num, (tid, thread_url, _) in enumerate(all_threads):
            log.info(
                "antionline: thread %d/%d (t=%d)",
                thread_num + 1, len(all_threads), tid,
            )
            for post_data in _iter_thread_posts(
                session, tid,
                max_pages=max_thread_pages,
                delay=request_delay,
            ):
                post_num = post_data["post_id"]
                author = post_data["author"]
                body = post_data["body"]
                thread_title = post_data.get("thread_title", "")
                idx = post_data.get("page", 1) * 100 + hash(post_num) % 100

                full_text = (
                    f"{thread_title}\n\n{body}"
                    if post_data.get("page", 1) == 1 and thread_title and idx < 2
                    else body
                )

                ts = _parse_vb_date(post_data["date_str"])
                pid = post_id(FORUM_ID, f"thread::{tid}::{post_num}")
                ahash = author_hash(FORUM_ID, author)
                cves = list({m.upper() for m in _CVE_RE.findall(full_text)})

                post_dict = {
                    "id": pid,
                    "text": full_text,
                    "timestamp": ts.isoformat(),
                    "forum_id": FORUM_ID,
                    "author_hash": ahash,
                }

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
                        "exploit_text": full_text[:500],
                        "title": thread_title,
                        "published": ts.isoformat(),
                        "source": FORUM_ID,
                    }
                    cve_fh.write(json.dumps(entry) + "\n")
                    cve_fh.flush()
                    cve_index_count += 1

                if post_count % log_every == 0:
                    log.info(
                        "antionline: %d posts, %d CVE refs so far",
                        post_count, cve_index_count,
                    )

    log.info(
        "antionline: done — %d posts, %d CVE index entries → %s",
        post_count, cve_index_count, output,
    )
    return post_count, cve_index_count
