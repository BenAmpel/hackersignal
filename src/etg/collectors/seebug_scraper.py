"""Collector for seebug.org — Chinese vulnerability database.

Seebug hosts 58 000+ vulnerability entries (SSV IDs) with CVE linkage, severity,
and titles.  Detail pages are behind a login wall, so this collector harvests only
the **public listing pages** (`/vuldb/vulnerabilities?page=N`), which expose:

  - SSV ID and URL
  - Publication date (YYYY-MM-DD)
  - Severity (critical / high / medium / low)
  - Title
  - CVE IDs (one or more, from the icon tooltip attribute)

Because the detail page is gated, we synthesise the post text from the metadata
available on the listing:

    "[Seebug SSV-{ssv_id}] {title}\\nSeverity: {severity}\\nDate: {date}\\nCVE: {cves}"

This is sufficient for CVE-linkage indexing, which is the primary value of Seebug
in the ETG pipeline (~58 k entries, high CVE density).

Pagination
----------
``GET /vuldb/vulnerabilities?page=N`` (N starts at 1).  Stop when a page returns
no ``<tr>`` rows inside ``<tbody>``, or when ``max_pages`` is reached.

Site notes
----------
- The site is Chinese-language; headers include ``Accept-Language: zh-CN``.
- Rate-limit: default 2 s between requests.
- ~2942 pages × 20 items = ~58 840 entries total (as of early 2026).
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import timezone
from pathlib import Path

import requests

from etg.collectors._common import author_hash, parse_date, post_id

log = logging.getLogger(__name__)

FORUM_ID = "seebug"
BASE = "https://www.seebug.org"

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)

# --- listing-page regexes ---
# SSV row anchor:  <a href="/vuldb/ssvid-12345">
_SSV_HREF_RE = re.compile(r'/vuldb/ssvid-(\d+)')
# Date column: <td class="text-center datetime hidden-sm hidden-xs">2026-02-03</td>
_DATE_RE = re.compile(
    r'<td[^>]+class="[^"]*datetime[^"]*"[^>]*>\s*(\d{4}-\d{2}-\d{2})\s*</td>'
)
# Severity: <div class="vul-level high">
_SEVERITY_RE = re.compile(r'class="vul-level\s+(\w+)"')
# Title: <a class="vul-title" title="FULL TITLE"
_TITLE_RE = re.compile(r'<a[^>]+class="vul-title"[^>]+title="([^"]+)"')
# CVE icon tooltip: <i class="fa fa-id-card" data-original-title="CVE-XXXX/CVE-YYYY">
_CVE_ICON_RE = re.compile(
    r'class="fa fa-id-card"[^>]+data-original-title="([^"]+)"'
)
# Detect tbody rows
_TBODY_ROW_RE = re.compile(r'<tr\b', re.IGNORECASE)
_TBODY_RE = re.compile(r'<tbody[^>]*>(.*?)</tbody>', re.DOTALL | re.IGNORECASE)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": f"{BASE}/vuldb/vulnerabilities",
}


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    # Seed session cookie by hitting the homepage first — Seebug issues a
    # __jsluid_s cookie that must be present to avoid 403s on listing pages.
    try:
        s.get(f"{BASE}/", timeout=15)
        time.sleep(1.0)
    except Exception:
        pass
    return s


def _get(session: requests.Session, url: str, delay: float = 2.0,
         referer: str | None = None) -> str | None:
    """Fetch *url*; return HTML text or None on failure."""
    hdrs = {"Referer": referer} if referer else {}
    try:
        r = session.get(url, timeout=30, allow_redirects=True, headers=hdrs)
        r.encoding = "utf-8"
        if r.status_code == 200:
            return r.text
        if r.status_code == 403:
            # One retry after a longer back-off — re-seed the cookie
            log.warning("seebug: 403 for %s — re-seeding session and retrying", url)
            time.sleep(10)
            try:
                session.get(f"{BASE}/", timeout=15)
                time.sleep(3)
            except Exception:
                pass
            r2 = session.get(url, timeout=30, allow_redirects=True, headers=hdrs)
            r2.encoding = "utf-8"
            if r2.status_code == 200:
                return r2.text
            log.warning("seebug: retry still returned %d for %s, stopping", r2.status_code, url)
            return None
        log.warning("seebug: HTTP %d for %s", r.status_code, url)
        return None
    except requests.RequestException as exc:
        log.warning("seebug: request error %s: %s", url, exc)
        return None
    finally:
        time.sleep(delay)


def _parse_listing_rows(html: str) -> list[dict]:
    """Extract vulnerability metadata rows from a listing page HTML.

    Returns a list of dicts with keys:
        ssv_id, title, date, severity, cves
    """
    results: list[dict] = []

    # Work only inside tbody to avoid false matches in headers/footers
    tbody_m = _TBODY_RE.search(html)
    if not tbody_m:
        return results
    tbody = tbody_m.group(1)

    # Split into individual rows
    rows = re.split(r'(?=<tr\b)', tbody, flags=re.IGNORECASE)

    for row in rows:
        if not row.strip():
            continue

        # SSV ID — find the first /vuldb/ssvid-NNN link
        ssv_m = _SSV_HREF_RE.search(row)
        if not ssv_m:
            continue
        ssv_id = int(ssv_m.group(1))

        # Title
        title_m = _TITLE_RE.search(row)
        title = title_m.group(1).strip() if title_m else f"SSV-{ssv_id}"

        # Date
        date_m = _DATE_RE.search(row)
        date_str = date_m.group(1) if date_m else ""

        # Severity
        sev_m = _SEVERITY_RE.search(row)
        severity = sev_m.group(1).lower() if sev_m else "unknown"

        # CVEs from icon tooltip
        cves: list[str] = []
        cve_icon_m = _CVE_ICON_RE.search(row)
        if cve_icon_m:
            raw_cves = cve_icon_m.group(1)
            # Multiple CVEs separated by "/"
            for part in raw_cves.split("/"):
                part = part.strip()
                if _CVE_RE.match(part):
                    cves.append(part.upper())

        # Also scan the full row for any stray CVE mentions
        for m in _CVE_RE.finditer(row):
            cve = m.group(0).upper()
            if cve not in cves:
                cves.append(cve)

        results.append({
            "ssv_id": ssv_id,
            "title": title,
            "date": date_str,
            "severity": severity,
            "cves": cves,
        })

    return results


def _iter_listing_pages(
    session: requests.Session,
    max_pages: int = 3000,
    delay: float = 2.0,
):
    """Yield vulnerability metadata dicts from Seebug listing pages.

    Yields dicts with keys: ``ssv_id``, ``title``, ``date``, ``severity``, ``cves``.
    Stops when a page returns no rows or ``max_pages`` is reached.
    """
    for page in range(1, max_pages + 1):
        url = f"{BASE}/vuldb/vulnerabilities?page={page}"
        prev_url = f"{BASE}/vuldb/vulnerabilities?page={page - 1}" if page > 1 else f"{BASE}/"
        log.debug("seebug: fetching listing page %d", page)

        html = _get(session, url, delay=delay, referer=prev_url)
        if html is None:
            log.warning("seebug: failed to fetch page %d, stopping", page)
            break

        rows = _parse_listing_rows(html)
        if not rows:
            log.info("seebug: page %d returned no rows — done", page)
            break

        log.debug("seebug: page %d → %d rows", page, len(rows))
        yield from rows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect(
    output: Path = Path("data/seebug_posts.jsonl"),
    cve_index_output: Path = Path("data/seebug_cve_index.jsonl"),
    request_delay: float = 2.0,
    max_pages: int = 3000,
) -> tuple[int, int]:
    """Scrape Seebug vulnerability listing pages and write JSONL output.

    Parameters
    ----------
    output:
        Path for the posts JSONL file.
    cve_index_output:
        Path for the CVE-index JSONL file.
    request_delay:
        Seconds to sleep between HTTP requests.
    max_pages:
        Maximum listing pages to fetch (each page has ~20 rows).

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
    seen_ids: set[int] = set()
    post_count = 0
    cve_index_count = 0

    # Constant author hash — no per-author info on listing pages
    _ahash = author_hash(FORUM_ID, "seebug")

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        for row in _iter_listing_pages(session, max_pages=max_pages, delay=request_delay):
            ssv_id = row["ssv_id"]
            if ssv_id in seen_ids:
                continue
            seen_ids.add(ssv_id)

            title = row["title"]
            date_str = row["date"]
            severity = row["severity"]
            cves = row["cves"]

            # Build synthetic post text from available metadata
            cve_part = ", ".join(cves) if cves else "N/A"
            text = (
                f"[Seebug SSV-{ssv_id}] {title}\n"
                f"Severity: {severity}\n"
                f"Date: {date_str}\n"
                f"CVE: {cve_part}"
            )

            ts = parse_date(date_str) if date_str else __import__("datetime").datetime.now(
                tz=timezone.utc
            )
            pid = post_id(FORUM_ID, str(ssv_id))

            post_dict = {
                "id": pid,
                "text": text,
                "timestamp": ts.isoformat(),
                "forum_id": FORUM_ID,
                "author_hash": _ahash,
            }
            post_fh.write(json.dumps(post_dict, ensure_ascii=False) + "\n")
            post_count += 1

            for cve in cves:
                entry = {
                    "exploit_id": pid,
                    "cve_id": cve,
                    "exploit_text": text[:500],
                    "title": title,
                    "published": ts.isoformat(),
                    "source": FORUM_ID,
                }
                cve_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                cve_index_count += 1

            if post_count % 500 == 0:
                log.info(
                    "seebug: %d posts, %d CVE refs written so far",
                    post_count, cve_index_count,
                )

    log.info(
        "seebug: done — %d posts, %d CVE index entries → %s",
        post_count, cve_index_count, output,
    )
    return post_count, cve_index_count


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Scrape seebug.org vulnerability listings")
    parser.add_argument("--output", default="data/seebug_posts.jsonl")
    parser.add_argument("--cve-index", default="data/seebug_cve_index.jsonl")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between requests")
    parser.add_argument("--max-pages", type=int, default=3000, help="Max listing pages to fetch")
    args = parser.parse_args()

    posts, cves = collect(
        output=Path(args.output),
        cve_index_output=Path(args.cve_index),
        request_delay=args.delay,
        max_pages=args.max_pages,
    )
    print(f"Done: {posts} posts, {cves} CVE index entries")
