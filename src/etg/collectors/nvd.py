"""Collect CVE records from the NIST NVD REST API v2.0.

Fetches all published CVEs year-by-year and emits:
- A ForumPost JSONL file (one post per CVE, text = "CVE-ID: description").
- A CVE dictionary JSONL file with structured metadata.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from etg.collectors._common import author_hash, parse_date, post_id
from etg.data.schemas import ForumPost

log = logging.getLogger(__name__)

NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
FORUM_ID = "nvd_cve"
RESULTS_PER_PAGE = 2000


def _fetch_with_retry(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    headers: dict[str, str],
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> dict[str, Any]:
    """GET *url* with *params*, retrying on failure with exponential back-off."""
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=60)
            if resp.status_code == 429:
                # Respect Retry-After (minimum 35s — NVD rate window is 30s)
                raw_ra = resp.headers.get("Retry-After", "35")
                try:
                    retry_after = max(int(raw_ra), 35)
                except (ValueError, TypeError):
                    retry_after = 35
                log.warning("NVD 429 rate-limit — sleeping %ds", retry_after)
                time.sleep(retry_after)
                # Don't count this as a retry attempt
                attempt = max(attempt - 1, 0)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            wait = base_delay * (2 ** attempt)
            if attempt < max_retries - 1:
                log.warning(
                    "NVD request failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                log.error("NVD request failed after %d attempts: %s", max_retries, exc)
                raise
    # unreachable
    raise RuntimeError("_fetch_with_retry: exhausted retries")


def _extract_cvss(metrics: dict[str, Any]) -> float | None:
    """Return the highest-priority CVSS base score available in *metrics*."""
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        entries = metrics.get(key)
        if entries:
            try:
                return float(entries[0]["cvssData"]["baseScore"])
            except (KeyError, IndexError, TypeError, ValueError):
                pass
    return None


def collect(
    output: Path = Path("data/nvd_posts.jsonl"),
    cve_dict_output: Path = Path("data/nvd_cve_dict.jsonl"),
    api_key: str | None = None,
    start_year: int = 2002,
    end_year: int | None = None,
    log_every: int = 2000,
) -> tuple[int, int]:
    """Fetch CVE records from NVD and write ForumPost + CVE-dict JSONL.

    Parameters
    ----------
    output:
        Destination JSONL path for ForumPost records.
    cve_dict_output:
        Destination JSONL path for structured CVE metadata.
    api_key:
        Optional NVD API key (raises rate limit from 5/30 s to 50/30 s).
    start_year:
        First publication year to collect.
    end_year:
        Last publication year (inclusive). Defaults to the current year.
    log_every:
        Emit a progress log every *log_every* records written.

    Returns
    -------
    tuple[int, int]
        ``(post_count, cve_dict_count)``
    """
    output = Path(output)
    cve_dict_output = Path(cve_dict_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cve_dict_output.parent.mkdir(parents=True, exist_ok=True)

    if end_year is None:
        end_year = datetime.now(tz=timezone.utc).year

    # NVD rate limits: 50 req/30 s with key, 5 req/30 s without
    delay = 0.7 if api_key else 6.1
    headers: dict[str, str] = {"apiKey": api_key} if api_key else {}

    post_count = 0
    cve_dict_count = 0

    session = requests.Session()

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_dict_output.open("w", encoding="utf-8") as dict_fh,
    ):
        # NVD date-filtered queries return 404 for many year ranges in API v2.
        # Paginate through all CVEs without date filter, then apply year filter locally.
        log.info("NVD: fetching all CVEs (years %d–%d) via full pagination", start_year, end_year)
        start_index = 0

        while True:
            params: dict[str, Any] = {
                "startIndex": start_index,
                "resultsPerPage": RESULTS_PER_PAGE,
            }

            data = _fetch_with_retry(session, NVD_API, params, headers)

            vulnerabilities = data.get("vulnerabilities", [])
            total_results = int(data.get("totalResults", 0))

            for vuln in vulnerabilities:
                try:
                    cve_obj = vuln["cve"]
                    cve_id: str = cve_obj["id"]

                    # Filter by publication year locally
                    published_str = cve_obj.get("published", "")
                    pub_year = int(published_str[:4]) if published_str else 0
                    if pub_year < start_year or pub_year > end_year:
                        continue

                    # English description
                    desc = next(
                        (
                            d["value"]
                            for d in cve_obj.get("descriptions", [])
                            if d.get("lang") == "en"
                        ),
                        "",
                    )

                    published = parse_date(cve_obj.get("published", ""))
                    cvss_score = _extract_cvss(cve_obj.get("metrics", {}))

                    text = f"{cve_id}: {desc}"[:3000]

                    post = ForumPost(
                        id=post_id("nvd", cve_id),
                        text=text,
                        timestamp=published,
                        forum_id=FORUM_ID,
                        author_hash=author_hash("nvd", "nist"),
                    )
                    post_fh.write(json.dumps(post.to_dict()) + "\n")
                    post_count += 1

                    cve_entry = {
                        "cve_id": cve_id,
                        "description": desc,
                        "published": published.isoformat(),
                        "cvss": cvss_score,
                    }
                    dict_fh.write(json.dumps(cve_entry) + "\n")
                    cve_dict_count += 1

                    if post_count % log_every == 0:
                        post_fh.flush()
                        dict_fh.flush()
                        log.info(
                            "NVD: written %d posts / %d CVE dict entries",
                            post_count,
                            cve_dict_count,
                        )

                except Exception as exc:
                    log.warning("NVD: skipping malformed entry: %s", exc)
                    continue

            # Advance to next page after processing all results in this batch
            if not vulnerabilities:
                log.info("NVD: no more results at startIndex=%d — done", start_index)
                break

            start_index += RESULTS_PER_PAGE
            if start_index >= total_results:
                log.info("NVD: reached end (startIndex=%d >= total=%d)", start_index, total_results)
                break
            time.sleep(delay)
            log.info("NVD: paginating… startIndex=%d / %d", start_index, total_results)

    log.info(
        "NVD: done. posts=%d, cve_dict=%d", post_count, cve_dict_count
    )
    return post_count, cve_dict_count
