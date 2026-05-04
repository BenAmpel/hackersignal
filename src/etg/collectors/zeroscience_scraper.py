"""Scraper for https://www.zeroscience.mk — ZeroScience Lab security advisories.

The site is a JavaScript SPA whose full advisory index is embedded in
``/index.html`` as a JS variable::

    var ADV = [
      {id:"ZSL-2026-5986", title:"...", date:"12.04.2026", sev:"high",
       local:"advisories/ZSL-2026-5986.html"},
      ...
    ]

Each advisory's detail page is a static HTML file at
``https://www.zeroscience.mk/advisories/ZSL-YYYY-NNNN.html``.

Output
------
Posts JSONL:
    {"id": pid, "text": text, "timestamp": iso_str, "forum_id": "zeroscience",
     "author_hash": ahash}

CVE index JSONL:
    {"exploit_id": pid, "cve_id": cve, "exploit_text": text[:500],
     "title": title, "published": iso_str, "source": "zeroscience"}
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

from etg.collectors._common import author_hash, post_id

log = logging.getLogger(__name__)

FORUM_ID = "zeroscience"
BASE = "https://www.zeroscience.mk"

# Regex to locate the ADV array in the SPA's index.html
_ADV_RE = re.compile(r"(?:var|const|let)\s+ADV\s*=\s*(\[.*?\]);", re.DOTALL)

# CVE pattern — used both on href attributes and full advisory text
CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)

# CVE href pattern: https://www.cve.org/CVERecord?id=CVE-XXXX
_CVE_HREF_RE = re.compile(r'href="https://www\.cve\.org/CVERecord\?id=(CVE-[^"]+)"', re.IGNORECASE)

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
# Index parsing
# ---------------------------------------------------------------------------


def _fetch_adv_index(session: requests.Session) -> list[dict]:
    """Fetch ``/index.html`` and extract the ``ADV`` JS array.

    Returns a list of dicts with keys ``id``, ``title``, ``date``, ``sev``,
    ``local``.  Returns an empty list on any error.
    """
    url = f"{BASE}/index.html"
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("zeroscience: failed to fetch index: %s", exc)
        return []

    m = _ADV_RE.search(resp.text)
    if not m:
        log.error("zeroscience: ADV variable not found in index.html")
        return []

    raw_js = m.group(1)
    # JS object keys are unquoted — quote bare keys for JSON
    json_str = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r'"\1":', raw_js)
    # JS allows \' inside double-quoted strings; JSON does not — strip the backslash
    json_str = json_str.replace("\\'", "'")
    try:
        entries = json.loads(json_str)
    except json.JSONDecodeError as exc:
        log.error("zeroscience: failed to JSON-parse ADV array: %s", exc)
        return []

    log.info("zeroscience: found %d entries in ADV index", len(entries))
    return entries


# ---------------------------------------------------------------------------
# Detail page parsing helpers
# ---------------------------------------------------------------------------


def _strip_tags(html_fragment: str) -> str:
    """Remove all HTML tags and unescape entities."""
    return unescape(re.sub(r"<[^>]+>", "", html_fragment)).strip()


def _extract_info_card(html: str, label: str) -> str:
    """Extract the ``.info-card-value`` text for a given ``.info-card-label``.

    Looks for a block like::

        <div class="info-card">
          ...
          <span class="info-card-label">Release Date</span>
          <span class="info-card-value">...</span>
          ...
        </div>

    Returns the stripped plain text value, or ``""`` if not found.
    """
    # Match the info-card block containing the requested label
    pattern = re.compile(
        r'<[^>]+class="info-card"[^>]*>(.*?)</[^>]+>',
        re.DOTALL | re.IGNORECASE,
    )
    for card_m in pattern.finditer(html):
        card = card_m.group(1)
        # Check if this card contains the requested label text
        if re.search(
            r'class="info-card-label"[^>]*>\s*' + re.escape(label) + r'\s*<',
            card,
            re.IGNORECASE,
        ):
            val_m = re.search(
                r'class="info-card-value"[^>]*>(.*?)(?:</[a-z]+>|$)',
                card,
                re.DOTALL | re.IGNORECASE,
            )
            if val_m:
                return _strip_tags(val_m.group(1))
    return ""


def _extract_adv_section(html: str, section_title: str) -> str:
    """Extract paragraph text from an ``.adv-sec`` block with a given title.

    Sections look like::

        <div class="adv-sec">
          <span class="adv-sec-t">Summary</span>
          <p class="adv-body-p">...</p>
        </div>

    Returns the stripped text of all matching ``<p class="adv-body-p">``
    elements, joined by newlines.  Returns ``""`` if not found.
    """
    # Locate the adv-sec block with the matching title
    sec_pattern = re.compile(
        r'<[^>]+class="adv-sec"[^>]*>(.*?)</div>',
        re.DOTALL | re.IGNORECASE,
    )
    for sec_m in sec_pattern.finditer(html):
        block = sec_m.group(1)
        if re.search(
            r'class="adv-sec-t"[^>]*>\s*' + re.escape(section_title) + r'\s*<',
            block,
            re.IGNORECASE,
        ):
            paragraphs = re.findall(
                r'<p[^>]*class="adv-body-p"[^>]*>(.*?)</p>',
                block,
                re.DOTALL | re.IGNORECASE,
            )
            return "\n".join(_strip_tags(p) for p in paragraphs).strip()
    return ""


def _scrape_advisory(session: requests.Session, entry: dict) -> dict | None:
    """Fetch and parse a single ZeroScience advisory detail page.

    Parameters
    ----------
    entry:
        One element from the ``ADV`` index array with keys ``id``, ``title``,
        ``date``, ``sev``, ``local``.

    Returns
    -------
    dict with keys ``post``, ``cves``, ``title``, ``timestamp``, or ``None``
    on error.
    """
    advisory_id: str = entry.get("id", "")
    title: str = entry.get("title", "")
    sev: str = entry.get("sev", "")
    date_str: str = entry.get("date", "")
    local: str = entry.get("local", "")

    if not local:
        log.debug("zeroscience: no 'local' path for %s — skipping", advisory_id)
        return None

    url = f"{BASE}/{local}"

    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("zeroscience: HTTP error for %s (%s): %s", advisory_id, url, exc)
        return None

    html = resp.text

    # --- Parse timestamp ---
    if date_str:
        try:
            dt = datetime.strptime(date_str, "%d.%m.%Y").replace(tzinfo=timezone.utc)
        except ValueError:
            log.debug("zeroscience: cannot parse date %r for %s", date_str, advisory_id)
            dt = datetime.now(tz=timezone.utc)
    else:
        dt = datetime.now(tz=timezone.utc)

    # --- Extract structured fields from info-card blocks ---
    vendor = _extract_info_card(html, "Vendor")
    affected = _extract_info_card(html, "Affected Version")

    # --- Extract summary and description sections ---
    summary = _extract_adv_section(html, "Summary")
    description = _extract_adv_section(html, "Description")

    # --- Assemble post text ---
    parts = [title]
    if sev:
        parts.append(f"\nSeverity: {sev}")
    if vendor:
        parts.append(f"Vendor: {vendor}")
    if affected:
        parts.append(f"Affected: {affected}")
    if summary:
        parts.append(f"\n{summary}")
    if description:
        parts.append(f"\n{description}")
    text = "\n".join(parts).strip()

    # --- Extract CVEs ---
    # First: from explicit CVE href links (most reliable)
    cves: list[str] = list(dict.fromkeys(
        m.upper() for m in _CVE_HREF_RE.findall(html)
    ))
    # Also scan the full HTML for any additional CVE-YYYY-NNNN references
    for cve in CVE_RE.findall(html):
        cve_upper = cve.upper()
        if cve_upper not in cves:
            cves.append(cve_upper)

    pid = post_id(FORUM_ID, advisory_id)
    ahash = author_hash(FORUM_ID, "zeroscience")

    return {
        "post": {
            "id": pid,
            "text": text,
            "timestamp": dt.isoformat(),
            "forum_id": FORUM_ID,
            "author_hash": ahash,
        },
        "cves": cves,
        "title": title,
        "timestamp": dt.isoformat(),
        "pid": pid,
        "advisory_id": advisory_id,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect(
    output: Path = Path("data/zeroscience_posts.jsonl"),
    cve_index_output: Path = Path("data/zeroscience_cve_index.jsonl"),
    request_delay: float = 1.5,
) -> tuple[int, int]:
    """Scrape ZeroScience Lab advisories and write posts + CVE-index as JSONL.

    Parameters
    ----------
    output:
        Destination file for ``ForumPost``-shaped JSONL records.
    cve_index_output:
        Destination file for CVE index JSONL records.
    request_delay:
        Seconds to sleep between HTTP requests (be polite to the server).

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

    adv_index = _fetch_adv_index(session)
    if not adv_index:
        log.error("zeroscience: empty advisory index — aborting")
        return 0, 0

    post_count = 0
    cve_index_count = 0

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        for entry in adv_index:
            time.sleep(request_delay)

            result = _scrape_advisory(session, entry)
            if result is None:
                continue

            # Write post record
            post_fh.write(json.dumps(result["post"]) + "\n")
            post_fh.flush()
            post_count += 1

            # Write CVE index entries
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

            if post_count % 100 == 0:
                log.info(
                    "zeroscience: wrote %d posts / %d CVE entries (latest: %s)",
                    post_count,
                    cve_index_count,
                    result["advisory_id"],
                )

    log.info(
        "zeroscience: finished — %d posts / %d CVE index entries written to %s",
        post_count,
        cve_index_count,
        output,
    )
    return post_count, cve_index_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collect()
