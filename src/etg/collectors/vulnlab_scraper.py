"""Scraper for https://www.vulnerability-lab.com — Vulnerability Lab advisories.

The site exposes advisories via paginated listing pages::

    /show.php?cat={category}&page={N}

Categories: ``webapp``, ``mobile``, ``remote``, ``local``

Each listing row (``<tr class="submit">``) contains:
- Advisory ID extracted from ``href="get_content.php?id=NNNN"``
- Advisory title from the link text
- Date from the second table column
- Author from the last table column

Detail pages at ``/get_content.php?id=NNNN`` return **plain text** formatted
with section headers separated by ``===`` dividers.  The site was last updated
July 2023 and contains ~1,138 historical advisories.

Note: The site does not consistently include CVE IDs, but the full text is
scanned for any ``CVE-YYYY-NNNNN`` patterns.

Output
------
Posts JSONL:
    {"id": pid, "text": text, "timestamp": iso_str,
     "forum_id": "vulnerability_lab", "author_hash": ahash}

CVE index JSONL:
    {"exploit_id": pid, "cve_id": cve, "exploit_text": text[:500],
     "title": title, "published": iso_str, "source": "vulnerability_lab"}
"""

from __future__ import annotations

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

FORUM_ID = "vulnerability_lab"
BASE = "https://www.vulnerability-lab.com"

CATEGORIES = ["webapp", "mobile", "remote", "local"]

# Advisory ID from listing href
_ADV_ID_RE = re.compile(r'href="get_content\.php\?id=(\d+)"', re.IGNORECASE)

# CVE pattern for full-text scanning
CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)

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
# Session helpers
# ---------------------------------------------------------------------------


def _make_session() -> requests.Session:
    """Return a requests.Session with browser-like headers."""
    s = requests.Session()
    s.headers.update(_HEADERS)
    return s


# ---------------------------------------------------------------------------
# Listing page scraping
# ---------------------------------------------------------------------------


def _strip_tags(html_fragment: str) -> str:
    """Remove all HTML tags and unescape entities."""
    return unescape(re.sub(r"<[^>]+>", "", html_fragment)).strip()


def _parse_listing_row(row_html: str) -> dict | None:
    """Parse a single ``<tr class="submit">`` row from a listing page.

    Returns a dict with ``id``, ``title``, ``date``, ``author`` keys, or
    ``None`` if the advisory ID cannot be extracted.
    """
    # Extract advisory ID from href
    id_m = _ADV_ID_RE.search(row_html)
    if not id_m:
        return None
    adv_id = id_m.group(1)

    # Extract title from the anchor tag text
    title_m = re.search(
        r'href="get_content\.php\?id=\d+"[^>]*>(.*?)</a>',
        row_html,
        re.DOTALL | re.IGNORECASE,
    )
    title = _strip_tags(title_m.group(1)) if title_m else ""

    # Extract table cells (td elements)
    cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.DOTALL | re.IGNORECASE)

    # The second cell (index 1) typically contains the date
    date_str = ""
    if len(cells) >= 2:
        raw_date = _strip_tags(cells[1])
        if re.search(r"\d{4}", raw_date):
            date_str = raw_date

    # The last cell typically contains the author
    author = ""
    if cells:
        author = _strip_tags(cells[-1])

    return {
        "id": adv_id,
        "title": title,
        "date": date_str,
        "author": author,
    }


def _iter_listing(
    session: requests.Session,
    category: str,
    max_pages: int = 50,
    request_delay: float = 2.0,
) -> list[dict]:
    """Iterate listing pages for *category* and return advisory metadata rows.

    Stops when a page returns no ``<tr class="submit">`` rows or after
    *max_pages* pages.
    """
    rows: list[dict] = []
    seen_ids: set[str] = set()

    for page_num in range(1, max_pages + 1):
        url = f"{BASE}/show.php?cat={category}&page={page_num}"
        log.debug("vulnlab: fetching listing %s page %d", category, page_num)

        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("vulnlab: HTTP error fetching listing %s page %d: %s", category, page_num, exc)
            break

        # Find all submit rows
        submit_rows = re.findall(
            r'<tr[^>]+class="submit"[^>]*>(.*?)</tr>',
            resp.text,
            re.DOTALL | re.IGNORECASE,
        )

        if not submit_rows:
            log.debug("vulnlab: no submit rows on %s page %d — stopping", category, page_num)
            break

        new_count = 0
        for row_html in submit_rows:
            parsed = _parse_listing_row(row_html)
            if parsed is None:
                continue
            if parsed["id"] not in seen_ids:
                seen_ids.add(parsed["id"])
                parsed["category"] = category
                rows.append(parsed)
                new_count += 1

        log.debug("vulnlab: %s page %d — %d new rows", category, page_num, new_count)

        if new_count == 0:
            log.debug("vulnlab: no new rows on %s page %d — stopping", category, page_num)
            break

        if page_num < max_pages:
            time.sleep(request_delay)

    log.info("vulnlab: category '%s' — collected %d advisory entries", category, len(rows))
    return rows


# ---------------------------------------------------------------------------
# Detail page parsing
# ---------------------------------------------------------------------------


def _parse_kv_line(text: str, key: str) -> str:
    """Extract value from a ``Key:    value`` style plain-text line."""
    m = re.search(
        r"^" + re.escape(key) + r"\s*:\s*(.+)$",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def _scrape_detail(session: requests.Session, meta: dict) -> dict | None:
    """Fetch and parse a Vulnerability Lab advisory detail page.

    Parameters
    ----------
    meta:
        Advisory metadata dict from the listing (keys: ``id``, ``title``,
        ``date``, ``author``, ``category``).

    Returns
    -------
    dict with ``post``, ``cves``, ``title``, ``timestamp``, ``pid`` keys,
    or ``None`` on error.
    """
    adv_id: str = meta["id"]
    url = f"{BASE}/get_content.php?id={adv_id}"

    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("vulnlab: HTTP error fetching advisory %s: %s", adv_id, exc)
        return None

    # The content is plain text embedded in HTML — extract the body text
    # Try to get the text content from the page
    html = resp.text

    # Strip all HTML tags to get the plain text content
    plain = _strip_tags(html)

    # Alternatively, if there's a <pre> or <code> block, prefer that
    pre_m = re.search(r"<pre[^>]*>(.*?)</pre>", html, re.DOTALL | re.IGNORECASE)
    if pre_m:
        plain = _strip_tags(pre_m.group(1))

    if not plain.strip():
        log.debug("vulnlab: empty content for advisory %s", adv_id)
        return None

    # --- Split on === dividers to extract sections ---
    sections = re.split(r"\n={3,}\n", plain)

    # Extract title from the "Document Title:" section (first meaningful block)
    doc_title = ""
    title_m = re.search(r"Document Title\s*:\s*\n+(.+?)(?:\n|$)", plain, re.IGNORECASE)
    if title_m:
        doc_title = title_m.group(1).strip()
    if not doc_title:
        doc_title = meta.get("title", "")

    # --- Extract key-value metadata fields ---
    release_date = _parse_kv_line(plain, "Release Date")
    cvss_score = _parse_kv_line(plain, "CVSS Score")
    vuln_class = _parse_kv_line(plain, "Vulnerability Class")
    affected = _parse_kv_line(plain, "Affected Product(s)")

    # --- Extract description and PoC sections ---
    description = ""
    poc = ""

    # Identify section headers and their content
    # Sections are separated by === lines; look for known section names
    for i, section in enumerate(sections):
        section_stripped = section.strip()
        lower_sec = section_stripped.lower()
        if "technical details" in lower_sec or "description" in lower_sec:
            # Next section after this header contains the description
            if i + 1 < len(sections):
                description = sections[i + 1].strip()
        elif "proof of concept" in lower_sec or "poc" in lower_sec:
            if i + 1 < len(sections):
                poc = sections[i + 1].strip()

    # --- Determine timestamp ---
    # Prefer the Release Date from content; fall back to listing date.
    # Guard against stray section-divider strings (===, ---) leaking in.
    def _is_date_like(s: str) -> bool:
        return bool(s) and bool(re.search(r"\d{4}", s)) and not re.fullmatch(r"[=\-]+", s)

    ts_str = release_date if _is_date_like(release_date) else meta.get("date", "")
    if _is_date_like(ts_str):
        try:
            dt = parse_date(ts_str)
        except Exception:
            dt = datetime.now(tz=timezone.utc)
    else:
        dt = datetime.now(tz=timezone.utc)

    # --- Assemble post text (cap at 8000 chars) ---
    parts = [doc_title]
    if vuln_class:
        parts.append(f"\nClass: {vuln_class}")
    if affected:
        parts.append(f"Product: {affected}")
    if cvss_score:
        parts.append(f"CVSS: {cvss_score}")
    if description:
        parts.append(f"\n{description}")
    if poc:
        parts.append(f"\n{poc}")
    text = "\n".join(parts).strip()[:8000]

    # --- Extract CVEs via regex scan ---
    cves: list[str] = list(dict.fromkeys(
        m.upper() for m in CVE_RE.findall(plain)
    ))

    author: str = meta.get("author", "") or "unknown"
    pid = post_id(FORUM_ID, adv_id)
    ahash = author_hash(FORUM_ID, author)

    return {
        "post": {
            "id": pid,
            "text": text,
            "timestamp": dt.isoformat(),
            "forum_id": FORUM_ID,
            "author_hash": ahash,
        },
        "cves": cves,
        "title": doc_title,
        "timestamp": dt.isoformat(),
        "pid": pid,
        "advisory_id": adv_id,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect(
    output: Path = Path("data/vulnlab_posts.jsonl"),
    cve_index_output: Path = Path("data/vulnlab_cve_index.jsonl"),
    request_delay: float = 2.0,
    max_pages_per_cat: int = 50,
) -> tuple[int, int]:
    """Scrape Vulnerability Lab advisories and write posts + CVE-index as JSONL.

    Iterates all four categories (webapp, mobile, remote, local) and all
    available listing pages.  Advisory IDs are deduplicated globally so that
    advisories appearing in multiple categories are only written once.

    Parameters
    ----------
    output:
        Destination file for post JSONL records.
    cve_index_output:
        Destination file for CVE index JSONL records.
    request_delay:
        Seconds to sleep between HTTP requests.
    max_pages_per_cat:
        Maximum listing pages to fetch per category (safety cap).

    Returns
    -------
    tuple[int, int]
        ``(post_count, cve_index_count)``
    """
    output = Path(output)
    cve_index_output = Path(cve_index_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cve_index_output.parent.mkdir(parents=True, exist_ok=True)

    session = _make_session()

    # --- Phase 1: Collect all advisory metadata from listing pages ---
    all_meta: list[dict] = []
    seen_ids: set[str] = set()

    for category in CATEGORIES:
        cat_rows = _iter_listing(
            session,
            category,
            max_pages=max_pages_per_cat,
            request_delay=request_delay,
        )
        for row in cat_rows:
            if row["id"] not in seen_ids:
                seen_ids.add(row["id"])
                all_meta.append(row)

    log.info("vulnlab: %d unique advisory IDs collected across all categories", len(all_meta))

    # --- Phase 2: Scrape each detail page ---
    post_count = 0
    cve_index_count = 0

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        for meta in all_meta:
            time.sleep(request_delay)

            result = _scrape_detail(session, meta)
            if result is None:
                continue

            if not result["post"]["text"].strip():
                log.debug("vulnlab: skipping advisory %s — empty text", meta["id"])
                continue

            # Write post record
            post_fh.write(json.dumps(result["post"]) + "\n")
            post_fh.flush()
            post_count += 1

            # Write CVE index entries (only when CVEs found via regex)
            for cve in result["cves"]:
                cve_entry = {
                    "exploit_id": result["pid"],
                    "cve_id": cve,
                    "exploit_text": result["post"]["text"][:500],
                    "title": result["title"],
                    "published": result["timestamp"],
                    "source": FORUM_ID,
                }
                cve_fh.write(json.dumps(cve_entry) + "\n")
                cve_index_count += 1
            if result["cves"]:
                cve_fh.flush()

            if post_count % 50 == 0:
                log.info(
                    "vulnlab: wrote %d posts / %d CVE entries (latest id: %s)",
                    post_count,
                    cve_index_count,
                    meta["id"],
                )

    log.info(
        "vulnlab: finished — %d posts / %d CVE index entries written to %s",
        post_count,
        cve_index_count,
        output,
    )
    return post_count, cve_index_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collect()
