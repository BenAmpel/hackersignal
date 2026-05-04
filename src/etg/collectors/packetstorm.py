"""Scraper for https://packetstorm.news — public security advisory/exploit archive.

The site moved from packetstormsecurity.com to packetstorm.news and introduced
a TOS acceptance gate.  We handle this with a one-time POST to /tos/ and then
use the resulting `tos` session cookie for all subsequent requests.

URL structure (post-migration):
  listing:  /files/exploit/{page}    (25 entries per page)
  file:     /files/id/{id}/
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Iterator

import requests

from etg.collectors._common import author_hash, parse_date, post_id
from etg.data.schemas import ForumPost

log = logging.getLogger(__name__)

FORUM_ID = "packetstorm"
BASE = "https://packetstorm.news"
FILE_ID_RE = re.compile(r"/files/id/(\d+)/")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ---------------------------------------------------------------------------
# Session / TOS helpers
# ---------------------------------------------------------------------------


def _make_session() -> requests.Session:
    """Return a requests.Session with browser-like headers."""
    s = requests.Session()
    s.headers.update(_HEADERS)
    return s


def _accept_tos(session: requests.Session) -> bool:
    """Perform the one-time TOS acceptance POST and set the `tos` cookie.

    PacketStorm redirects new visitors to /tos/?redir=<b64>.  We GET that
    page, extract the CSRF token + redir value, then POST acceptance back to
    /tos/.  This sets ``tos=<date>`` in the session cookie jar.

    Returns True if the cookie is now set (acceptance succeeded or was already
    present), False on error.
    """
    if "tos" in session.cookies:
        return True

    # Trigger the TOS redirect by hitting the exploit listing
    try:
        r = session.get(
            f"{BASE}/files/exploit/1",
            timeout=30,
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        log.warning("PacketStorm TOS: failed to reach site: %s", exc)
        return False

    if "/tos/" not in r.url:
        # No TOS gate — already accepted or site changed
        log.debug("PacketStorm: no TOS redirect (url=%s)", r.url)
        return True

    # Extract CSRF token and redir value
    csrf_match = re.search(r'name="csrf"\s+value="([^"]+)"', r.text)
    redir_match = re.search(r'name="redir"\s+value="([^"]+)"', r.text)
    if not csrf_match or not redir_match:
        log.warning("PacketStorm TOS: CSRF/redir not found in TOS page")
        return False

    # POST acceptance
    try:
        session.post(
            f"{BASE}/tos/",
            data={
                "csrf": csrf_match.group(1),
                "redir": redir_match.group(1),
                "go": "Accept",
            },
            headers={"Referer": r.url, "Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        log.warning("PacketStorm TOS: POST failed: %s", exc)
        return False

    if "tos" in session.cookies:
        log.info("PacketStorm: TOS accepted, cookie=%s", session.cookies["tos"])
        return True

    log.warning("PacketStorm: TOS POST did not set cookie; trying direct cookie injection")
    # Fallback: inject a known-good cookie value
    session.cookies.set("tos", "20250912", domain="packetstorm.news")
    return True


# ---------------------------------------------------------------------------
# Pagination — yield individual file-page URLs
# ---------------------------------------------------------------------------


def _file_urls(
    session: requests.Session,
    category: str = "exploit",
    max_pages: int = 500,
) -> Iterator[str]:
    """Yield /files/id/{n}/ URLs from the category listing pages."""
    seen: set[str] = set()

    for page_num in range(1, max_pages + 1):
        url = f"{BASE}/files/{category}/{page_num}"
        log.debug("Fetching listing page %d: %s", page_num, url)

        try:
            resp = session.get(url, timeout=30, allow_redirects=True)
        except requests.RequestException as exc:
            log.warning("Error fetching listing page %s: %s", url, exc)
            break

        if resp.status_code != 200 or "/404" in resp.url:
            log.warning(
                "Listing page %s → %s (HTTP %s) — stopping",
                url,
                resp.url,
                resp.status_code,
            )
            break

        new_count = 0
        for fid in FILE_ID_RE.findall(resp.text):
            full = f"{BASE}/files/id/{fid}/"
            if full not in seen:
                seen.add(full)
                new_count += 1
                yield full

        if new_count == 0:
            log.debug("No new URLs on page %d — stopping pagination", page_num)
            break

        if page_num < max_pages:
            time.sleep(1)


# ---------------------------------------------------------------------------
# Individual file page scraper
# ---------------------------------------------------------------------------


_LOCKOUT_STRINGS = ("LOCKED OUT", "24 hour lockout", "eightysixed")


def _is_blocked(resp: requests.Response) -> bool:
    """Return True if the response is a rate-limit/lockout page."""
    if "/eightysixed/" in resp.url or "/404" in resp.url:
        return True
    if resp.status_code == 200 and any(s in resp.text for s in _LOCKOUT_STRINGS):
        return True
    return False


def _scrape_file(
    session: requests.Session,
    url: str,
    max_retries: int = 3,
    retry_delay: float = 30.0,
) -> ForumPost | None:
    """Fetch and parse an individual Packet Storm file page."""
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=30, allow_redirects=True)
        except requests.RequestException as exc:
            log.warning("Error fetching %s: %s", url, exc)
            return None

        if resp.status_code != 200:
            log.debug("HTTP %s for %s", resp.status_code, url)
            return None

        if _is_blocked(resp):
            if attempt < max_retries - 1:
                wait = retry_delay * (2 ** attempt)
                log.warning(
                    "PacketStorm blocked on %s (attempt %d/%d) — sleeping %.0fs",
                    url, attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
                continue
            else:
                log.warning("PacketStorm blocked after %d retries for %s", max_retries, url)
                return None

        break  # successful, non-blocked response
    else:
        return None

    html = resp.text

    # --- uid ---
    m = FILE_ID_RE.search(url)
    if not m:
        log.debug("Cannot extract file ID from URL: %s", url)
        return None
    uid = m.group(1)

    # --- title ---
    title = ""
    # New structure: <td class="cardheadbg">TITLE</td> inside .fretrot
    t_match = re.search(r'class="cardheadbg">\s*(.*?)\s*</td>', html, re.DOTALL)
    if t_match:
        title = unescape(re.sub(r"<[^>]+>", "", t_match.group(1))).strip()
    if not title:
        # Fallback: <title> tag minus site name
        title_tag = re.search(r"<title>(.*?)</title>", html, re.DOTALL)
        if title_tag:
            raw = unescape(title_tag.group(1)).strip()
            title = re.sub(r"\s*[-|]\s*Packet Storm\s*$", "", raw, flags=re.IGNORECASE).strip()

    # --- date (Posted: YYYY-MM-DD) ---
    date_str = ""
    posted_match = re.search(r"Posted:</td>\s*<td>(\d{4}-\d{2}-\d{2})</td>", html, re.IGNORECASE)
    if posted_match:
        date_str = posted_match.group(1)

    ts: datetime
    if date_str:
        ts = parse_date(date_str)
    else:
        ts = datetime.now(tz=timezone.utc)

    # --- author (Source(s): <a>NAME</a>) ---
    author = ""
    src_match = re.search(
        r"Source\(s\):</td><td>(.*?)</td>", html, re.DOTALL | re.IGNORECASE
    )
    if src_match:
        names = re.findall(r"<a[^>]*>([^<]+)</a>", src_match.group(1))
        author = ", ".join(unescape(n.strip()) for n in names)
    if not author:
        # Fallback: look for class="author" link
        auth_el = re.search(r'class="author"[^>]*>.*?<a[^>]*>(.*?)</a>', html, re.DOTALL)
        if auth_el:
            author = unescape(re.sub(r"<[^>]+>", "", auth_el.group(1))).strip()

    # --- body text ---
    # The exploit source lives in a <pre> block as HTML-escaped text
    pre_match = re.search(r"<pre[^>]*>(.*?)</pre>", html, re.DOTALL)
    body_text = ""
    if pre_match:
        inner = pre_match.group(1)
        # Strip any nested HTML tags (e.g. <div> wrappers inside pre)
        inner_clean = re.sub(r"<[^>]+>", "", inner)
        body_text = unescape(inner_clean).strip()

    # Extract CVE references (deduplicated, from the metadata section)
    # Prefer CVEs from the metadata table; fall back to scanning full HTML
    cves_raw = re.findall(r"CVE-\d{4}-\d+", html)
    cves = list(dict.fromkeys(c.upper() for c in cves_raw))
    if cves:
        log.debug("CVEs found in %s: %s", url, cves)

    full_text = f"{title}\n\n{body_text[:8000]}".strip()

    if len(full_text) < 40:
        log.debug("Skipping %s — full_text too short (%d chars)", url, len(full_text))
        return None

    post = ForumPost(
        id=post_id("packetstorm", uid),
        text=full_text,
        timestamp=ts,
        forum_id=FORUM_ID,
        author_hash=author_hash("packetstorm", author or "unknown"),
    )
    # Attach CVEs as a non-schema attribute for the collect() loop to use
    post._cves = cves  # type: ignore[attr-defined]
    post._uid = uid  # type: ignore[attr-defined]
    post._title = title  # type: ignore[attr-defined]
    return post


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect(
    output: Path = Path("data/packetstorm_posts.jsonl"),
    cve_index_output: Path = Path("data/packetstorm_cve_index.jsonl"),
    category: str = "exploit",
    max_pages: int = 500,
    request_delay: float = 5.0,
    log_every: int = 200,
) -> tuple[int, int]:
    """Scrape Packet Storm and write ForumPost + CVE-index records as JSONL.

    Returns ``(post_count, cve_index_count)``.
    """
    output = Path(output)
    cve_index_output = Path(cve_index_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cve_index_output.parent.mkdir(parents=True, exist_ok=True)

    session = _make_session()

    # Accept TOS before scraping
    if not _accept_tos(session):
        log.error("PacketStorm: could not accept TOS — aborting collection")
        return 0, 0

    post_count = 0
    cve_index_count = 0

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        for file_url in _file_urls(session, category=category, max_pages=max_pages):
            post = _scrape_file(session, file_url)

            if post is None or len(post.text) < 40:
                time.sleep(request_delay)
                continue

            post_fh.write(json.dumps(post.to_dict()) + "\n")
            post_fh.flush()
            post_count += 1

            # Write CVE index entries
            cves: list[str] = getattr(post, "_cves", [])
            uid: str = getattr(post, "_uid", "")
            title: str = getattr(post, "_title", "")
            for cve in cves:
                entry = {
                    "exploit_id": uid,
                    "cve_id": cve,
                    "exploit_text": post.text[:500],
                    "title": title,
                    "published": post.timestamp.isoformat(),
                    "source": "packetstorm",
                }
                cve_fh.write(json.dumps(entry) + "\n")
                cve_index_count += 1
            if cves:
                cve_fh.flush()

            if post_count % log_every == 0:
                log.info(
                    "packetstorm: wrote %d posts / %d CVE entries (latest: %s)",
                    post_count, cve_index_count, file_url,
                )

            time.sleep(request_delay)

    log.info(
        "packetstorm: finished — %d posts / %d CVE index entries written to %s",
        post_count, cve_index_count, output,
    )
    return post_count, cve_index_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collect()
