"""Scraper for the Full Disclosure security mailing list archive at https://seclists.org/fulldisclosure/"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from scrapling import Fetcher

from etg.collectors._common import author_hash, parse_date, post_id
from etg.data.schemas import ForumPost

log = logging.getLogger(__name__)

FORUM_ID = "full_disclosure"
BASE = "https://seclists.org/fulldisclosure"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
START_YEAR = 2002

# Matches href values that are purely numeric (individual message links)
_MSG_HREF_RE = re.compile(r"^\d+$")

# PGP signature block markers
_PGP_BEGIN_RE = re.compile(r"-----BEGIN PGP", re.IGNORECASE)
_PGP_END_RE = re.compile(r"-----END PGP", re.IGNORECASE)

# Email header lines to strip from body
_HEADER_LINE_RE = re.compile(
    r"^(From|To|Cc|Date|Subject|Message-ID|Reply-To|MIME-Version|Content-Type|"
    r"Content-Transfer-Encoding|In-Reply-To|References|X-[\w-]+)\s*:",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _html(resp) -> str:
    """Decode response body safely."""
    encoding = getattr(resp, "encoding", None) or "utf-8"
    return resp.body.decode(encoding, errors="replace")


def _strip_body(raw_text: str) -> str:
    """Clean a mailing-list message body.

    - Strip leading email header lines (From/To/Date/Subject/…).
    - Strip quoted lines (starting with '>').
    - Strip PGP signature blocks.
    - Collapse excessive blank lines.
    """
    lines = raw_text.splitlines()

    # Strip leading header lines
    start = 0
    for i, line in enumerate(lines):
        if _HEADER_LINE_RE.match(line):
            start = i + 1
        elif line.strip() == "" and i == start:
            # blank separator line between headers and body
            start = i + 1
            break
        elif i > 0 and not _HEADER_LINE_RE.match(line):
            break

    lines = lines[start:]

    # Strip PGP blocks
    cleaned: list[str] = []
    in_pgp = False
    for line in lines:
        if _PGP_BEGIN_RE.search(line):
            in_pgp = True
            continue
        if in_pgp:
            if _PGP_END_RE.search(line):
                in_pgp = False
            continue
        # Strip quoted lines
        if line.strip().startswith(">"):
            continue
        cleaned.append(line)

    # Collapse runs of more than 2 blank lines
    result: list[str] = []
    blank_run = 0
    for line in cleaned:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                result.append(line)
        else:
            blank_run = 0
            result.append(line)

    return "\n".join(result).strip()


def _extract_header_field(pre_text: str, field: str) -> str:
    """Extract the value of *field* from email-style headers in *pre_text*."""
    pattern = re.compile(
        r"^" + re.escape(field) + r"\s*:\s*(.+?)(?=\n\S|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(pre_text)
    if m:
        return " ".join(m.group(1).split())  # normalise whitespace
    return ""


def _clean_author(raw: str) -> str:
    """Strip angle-bracket email addresses and decode RFC 2047 encoded words minimally."""
    # Remove <email@address>
    raw = re.sub(r"<[^>]+>", "", raw).strip()
    # Remove (comment) sections
    raw = re.sub(r"\([^)]*\)", "", raw).strip()
    # Remove RFC 2047 encoded words =?charset?encoding?text?= (leave decoded text as-is)
    raw = re.sub(r"=\?[^?]+\?[BbQq]\?[^?]+\?=", "", raw).strip()
    return raw or "unknown"


# ---------------------------------------------------------------------------
# Month index page → message URLs
# ---------------------------------------------------------------------------


def _month_urls(fetcher: Fetcher, year: int, month: str) -> list[str]:
    """Return individual message URLs for a given year/month archive page."""
    url = f"{BASE}/{year}/{month}/"
    try:
        resp = fetcher.get(url, timeout=15)
        if resp.status != 200:
            log.debug("Month page %s returned HTTP %s", url, resp.status)
            return []
    except Exception as exc:
        log.debug("Error fetching month page %s: %s", url, exc)
        return []

    urls: list[str] = []
    for a in resp.css("a[href]"):
        href = a.attrib["href"]
        if _MSG_HREF_RE.match(href.strip("/")):
            # href is a bare number like "1" or "42"
            urls.append(f"{BASE}/{year}/{month}/{href.strip('/')}")

    return urls


# ---------------------------------------------------------------------------
# Individual message scraper
# ---------------------------------------------------------------------------


def _scrape_message(fetcher: Fetcher, url: str) -> ForumPost | None:
    """Fetch and parse one Full Disclosure message page."""
    try:
        resp = fetcher.get(url, timeout=15)
        if resp.status != 200:
            log.debug("Non-200 (%s) for %s", resp.status, url)
            return None
    except Exception as exc:
        log.warning("Error fetching %s: %s", url, exc)
        return None

    # --- uid from URL ---
    # URL pattern: .../fulldisclosure/2023/Jan/42
    parts = url.rstrip("/").split("/")
    uid = parts[-1] if parts else None
    if not uid or not uid.isdigit():
        # Try to extract any trailing number
        m = re.search(r"/(\d+)/?$", url)
        uid = m.group(1) if m else url  # fall back to full URL as uid

    # --- subject ---
    subject = ""
    h1_els = resp.css("h1")
    if h1_els:
        subject = h1_els[0].get_all_text().strip()
    if not subject:
        title_els = resp.css("title")
        if title_els:
            raw = title_els[0].get_all_text().strip()
            subject = re.sub(r"\s*-\s*Full Disclosure\s*$", "", raw, flags=re.IGNORECASE).strip()

    # --- find the main <pre> block ---
    pre_text = ""
    pre_els = resp.css("pre")
    if pre_els:
        # Take the longest <pre> block as the message body
        pre_text = max((el.get_all_text() for el in pre_els), key=len, default="")

    # --- extract From / Date from headers in the pre block ---
    raw_from = _extract_header_field(pre_text, "From")
    raw_date = _extract_header_field(pre_text, "Date")

    author = _clean_author(raw_from)

    ts: datetime
    if raw_date:
        ts = parse_date(raw_date)
    else:
        ts = datetime.now(tz=timezone.utc)

    # --- clean body ---
    cleaned_body = _strip_body(pre_text)

    if len(cleaned_body) < 40:
        log.debug("Skipping %s — cleaned body too short (%d chars)", url, len(cleaned_body))
        return None

    full_text = f"{subject}\n\n{cleaned_body[:6000]}".strip()

    return ForumPost(
        id=post_id("full_disclosure", uid),
        text=full_text,
        timestamp=ts,
        forum_id=FORUM_ID,
        author_hash=author_hash("full_disclosure", author or "unknown"),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect(
    output: Path = Path("data/fulldisclosure_posts.jsonl"),
    start_year: int = START_YEAR,
    end_year: int | None = None,
    request_delay: float = 1.0,
    log_every: int = 500,
) -> int:
    """Scrape Full Disclosure archives and write ForumPost records to *output* as JSONL.

    Returns the number of posts written.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if end_year is None:
        end_year = datetime.now(tz=timezone.utc).year

    fetcher = Fetcher()
    post_count = 0

    with output.open("w", encoding="utf-8") as fh:
        for year in range(start_year, end_year + 1):
            for month in MONTHS:
                msg_urls = _month_urls(fetcher, year, month)
                log.debug("fulldisclosure: %d/%s — %d message URLs", year, month, len(msg_urls))

                for msg_url in msg_urls:
                    try:
                        post = _scrape_message(fetcher, msg_url)
                    except Exception as exc:
                        log.warning("Unhandled error scraping %s: %s", msg_url, exc)
                        post = None

                    if post is not None and len(post.text) >= 40:
                        fh.write(json.dumps(post.to_dict()) + "\n")
                        fh.flush()
                        post_count += 1

                        if post_count % log_every == 0:
                            log.info(
                                "fulldisclosure: wrote %d posts (latest: %s)",
                                post_count,
                                msg_url,
                            )

                    time.sleep(request_delay)

                # Brief pause between months to be polite
                time.sleep(0.5)

    log.info("fulldisclosure: finished — %d posts written to %s", post_count, output)
    return post_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collect()
