"""Collect public security advisories from the GitHub Advisory Database (GHSA).

Fetches all published advisories and emits:
- A ForumPost JSONL file (one post per advisory, text = summary + description).
- A CVE index JSONL file mapping exploit IDs to CVE identifiers.

Pagination uses cursor-based ``after`` parameter (link header).

Rate limits:
  - Unauthenticated: 60 req/hr
  - Authenticated (GITHUB_TOKEN): 5,000 req/hr

Recommended: export GITHUB_TOKEN=<personal-access-token> for faster collection.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from datetime import datetime, timezone

import requests

from etg.collectors._common import author_hash, parse_date, post_id
from etg.data.schemas import ForumPost

log = logging.getLogger(__name__)

FORUM_ID = "github_advisory"
API_BASE = "https://api.github.com/advisories"
PER_PAGE = 100  # max allowed

_HEADERS_BASE = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "ETG-Research/1.0",
}


def _make_session(github_token: str | None = None) -> requests.Session:
    s = requests.Session()
    headers = dict(_HEADERS_BASE)
    token = github_token or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
        log.info("GitHub Advisories: using authenticated requests (5,000 req/hr limit)")
    else:
        log.info(
            "GitHub Advisories: unauthenticated (60 req/hr limit). "
            "Set GITHUB_TOKEN env var for faster collection."
        )
    s.headers.update(headers)
    return s


def _fetch_page(
    session: requests.Session,
    next_url: str | None,
    max_retries: int = 5,
) -> tuple[list[dict], str | None]:
    """Fetch one page of advisories.  Returns (items, next_page_url).

    Uses the full URL from the Link header to avoid cursor double-encoding.
    """
    url = next_url or f"{API_BASE}?per_page={PER_PAGE}"

    for attempt in range(max_retries):
        try:
            # Use the pre-built URL directly to avoid requests double-encoding the cursor
            resp = session.get(url, timeout=60)

            if resp.status_code == 429 or resp.status_code == 403:
                # Rate limited
                retry_after = int(resp.headers.get("Retry-After", "60"))
                reset_at = resp.headers.get("X-RateLimit-Reset")
                if reset_at:
                    wait = max(0, int(reset_at) - int(time.time())) + 5
                else:
                    wait = retry_after
                log.warning(
                    "GitHub rate limit hit (HTTP %s) — sleeping %ds",
                    resp.status_code, wait,
                )
                time.sleep(wait)
                continue

            resp.raise_for_status()
            items = resp.json()

            # Extract next page URL directly from Link header (avoids double-encoding)
            next_cursor = None
            link_hdr = resp.headers.get("Link", "")
            next_match = re.search(r'<([^>]+)>;\s*rel="next"', link_hdr)
            if next_match:
                next_cursor = next_match.group(1)

            remaining = resp.headers.get("X-RateLimit-Remaining", "?")
            log.debug(
                "GitHub Advisories: fetched %d items, rate-limit remaining=%s, next_cursor=%s",
                len(items), remaining, bool(next_cursor),
            )

            return items, next_cursor

        except (requests.RequestException, ValueError) as exc:
            wait = 2.0 * (2 ** attempt)
            if attempt < max_retries - 1:
                log.warning(
                    "GitHub request failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, max_retries, exc, wait,
                )
                time.sleep(wait)
            else:
                log.error("GitHub request failed after %d attempts: %s", max_retries, exc)
                raise

    raise RuntimeError("_fetch_page: exhausted retries")


def collect(
    output: Path = Path("data/github_advisory_posts.jsonl"),
    cve_index_output: Path = Path("data/github_advisory_cve_index.jsonl"),
    github_token: str | None = None,
    log_every: int = 1000,
) -> tuple[int, int]:
    """Fetch all GitHub Security Advisories and write ForumPost + CVE-index JSONL.

    Parameters
    ----------
    output:
        Destination JSONL for ForumPost records.
    cve_index_output:
        Destination JSONL mapping GHSA IDs → CVE IDs.
    github_token:
        Personal access token.  Falls back to ``GITHUB_TOKEN`` env var.
    log_every:
        Emit a progress log every *log_every* records written.

    Returns
    -------
    tuple[int, int]
        ``(post_count, cve_index_count)``
    """
    output = Path(output)
    cve_index_output = Path(cve_index_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cve_index_output.parent.mkdir(parents=True, exist_ok=True)

    session = _make_session(github_token)

    # Delay between pages:
    # - Authenticated: 5000 req/hr = ~0.72s/req; use 0.8s to be safe
    # - Unauthenticated: 60 req/hr = 60s/req; use 62s
    token = github_token or os.environ.get("GITHUB_TOKEN")
    page_delay = 0.8 if token else 62.0

    post_count = 0
    cve_index_count = 0
    next_url: str | None = None  # None → use default first-page URL

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        page_num = 0
        while True:
            page_num += 1
            items, next_url = _fetch_page(session, next_url)

            if not items:
                log.info("GitHub Advisories: empty page — done at page %d", page_num)
                break

            for adv in items:
                try:
                    ghsa_id: str = adv.get("ghsa_id", "")
                    cve_id: str | None = adv.get("cve_id")
                    summary: str = adv.get("summary", "") or ""
                    description: str = adv.get("description", "") or ""
                    severity: str = adv.get("severity", "") or ""
                    published_str: str = adv.get("published_at", "") or ""

                    if not ghsa_id:
                        continue

                    # Build rich text: summary + severity badge + description
                    parts = [summary]
                    if severity:
                        parts.append(f"[Severity: {severity.upper()}]")
                    if description and description != summary:
                        parts.append(description)
                    text = "\n\n".join(parts)[:6000]

                    if len(text) < 20:
                        continue

                    ts = parse_date(published_str) if published_str else datetime.now(tz=timezone.utc)

                    post = ForumPost(
                        id=post_id("github_advisory", ghsa_id),
                        text=text,
                        timestamp=ts,
                        forum_id=FORUM_ID,
                        author_hash=author_hash("github_advisory", "github"),
                    )
                    post_fh.write(json.dumps(post.to_dict()) + "\n")
                    post_count += 1

                    # CVE index entry (only if CVE ID is known)
                    if cve_id:
                        cve_entry = {
                            "exploit_id": ghsa_id,
                            "cve_id": cve_id.upper(),
                            "title": summary,
                            "published": ts.isoformat(),
                            "source": "github_advisory",
                        }
                        cve_fh.write(json.dumps(cve_entry) + "\n")
                        cve_index_count += 1

                    if post_count % log_every == 0:
                        post_fh.flush()
                        cve_fh.flush()
                        log.info(
                            "GitHub Advisories: written %d posts / %d CVE index entries (page %d)",
                            post_count, cve_index_count, page_num,
                        )

                except Exception as exc:
                    log.warning("GitHub Advisories: skipping entry %s: %s", adv.get("ghsa_id"), exc)
                    continue

            if not next_url:
                log.info("GitHub Advisories: no more pages after page %d", page_num)
                break
            time.sleep(page_delay)

    log.info(
        "GitHub Advisories: done. posts=%d, cve_index=%d",
        post_count, cve_index_count,
    )
    return post_count, cve_index_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collect()
