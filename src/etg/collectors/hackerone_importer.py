"""Importer for the HackerOne disclosed bug-bounty reports dataset.

Data source: HuggingFace dataset saved at ``data/external/hackerone_disclosed_reports/``

The dataset contains publicly disclosed HackerOne vulnerability reports.  Each
record represents a single report with metadata about the reporter, the affected
company (team), and the weakness category.  The main report body is stored in
``vulnerability_information``.

All three dataset splits (train, test, validation) are processed.

Outputs
-------
- Forum posts JSONL: one record per report with non-trivial vulnerability text.
- CVE index JSONL: one record per (report, CVE) pair found in the combined text.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from etg.collectors._common import author_hash, parse_date, post_id

log = logging.getLogger(__name__)

FORUM_ID = "hackerone"

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)

_MAX_VULN_INFO = 10_000  # characters


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_str(value: object, default: str = "") -> str:
    """Return *value* as a stripped string, or *default* if null-ish."""
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def _reporter_username(reporter: object) -> str:
    """Extract the reporter username from the reporter field (dict or None)."""
    if not isinstance(reporter, dict):
        return "unknown"
    username = reporter.get("username") or reporter.get("name") or ""
    return _safe_str(username) or "unknown"


def _weakness_name(weakness: object) -> str:
    """Extract the weakness name from the weakness field (dict or None)."""
    if not isinstance(weakness, dict):
        return "Unknown"
    name = weakness.get("name") or ""
    return _safe_str(name) or "Unknown"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect(
    data_dir: Path = Path("data/external/hackerone_disclosed_reports"),
    output: Path = Path("data/hackerone_posts.jsonl"),
    cve_index_output: Path = Path("data/hackerone_cve_index.jsonl"),
    log_every: int = 500,
) -> tuple[int, int]:
    """Import HackerOne disclosed reports and write ForumPost + CVE-index JSONL.

    Parameters
    ----------
    data_dir:
        Directory containing the HuggingFace dataset saved with
        ``dataset.save_to_disk()``.
    output:
        Destination JSONL path for forum post records.
    cve_index_output:
        Destination JSONL path for the CVE reference index.
    log_every:
        Log a progress message every *log_every* posts written.

    Returns
    -------
    tuple[int, int]
        ``(post_count, cve_index_count)``
    """
    from datasets import load_from_disk  # type: ignore[import]

    data_dir = Path(data_dir)
    output = Path(output)
    cve_index_output = Path(cve_index_output)

    if not data_dir.exists():
        log.error("hackerone: dataset directory not found: %s", data_dir)
        return 0, 0

    output.parent.mkdir(parents=True, exist_ok=True)
    cve_index_output.parent.mkdir(parents=True, exist_ok=True)

    log.info("hackerone: loading dataset from %s", data_dir)
    ds = load_from_disk(str(data_dir))

    post_count = 0
    cve_index_count = 0

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        for split in ds.keys():
            log.info("hackerone: processing split '%s' (%d records)", split, len(ds[split]))

            for record in ds[split]:
                # --- Vulnerability information ---
                vuln_info = _safe_str(record.get("vulnerability_information"))
                if len(vuln_info) < 20:
                    continue

                # Truncate very long bodies
                if len(vuln_info) > _MAX_VULN_INFO:
                    vuln_info = vuln_info[:_MAX_VULN_INFO]

                # --- Metadata ---
                title = _safe_str(record.get("title"), default="(no title)")
                weakness_name = _weakness_name(record.get("weakness"))
                reporter_username = _reporter_username(record.get("reporter"))

                # --- Composed text ---
                text = f"[{weakness_name}] {title}\n\n{vuln_info}"

                # --- IDs ---
                pid = post_id(FORUM_ID, str(record.get("id", "")))
                ahash = author_hash(FORUM_ID, reporter_username)

                # --- Timestamp: prefer disclosed_at, fall back to created_at ---
                ts_str = _safe_str(record.get("disclosed_at")) or _safe_str(
                    record.get("created_at")
                )
                ts = parse_date(ts_str)

                # --- Write post record ---
                post_dict = {
                    "id": pid,
                    "text": text,
                    "timestamp": ts.isoformat(),
                    "forum_id": FORUM_ID,
                    "author_hash": ahash,
                }
                post_fh.write(json.dumps(post_dict) + "\n")
                post_count += 1

                # --- CVE index ---
                cves = list(dict.fromkeys(m.upper() for m in _CVE_RE.findall(text)))
                cve_title = f"{title} ({weakness_name})"
                for cve in cves:
                    entry = {
                        "exploit_id": pid,
                        "cve_id": cve,
                        "exploit_text": text[:500],
                        "title": cve_title,
                        "published": ts.isoformat(),
                        "source": FORUM_ID,
                    }
                    cve_fh.write(json.dumps(entry) + "\n")
                    cve_index_count += 1

                if post_count % log_every == 0:
                    log.info(
                        "hackerone: wrote %d posts / %d CVE entries so far",
                        post_count,
                        cve_index_count,
                    )

            post_fh.flush()
            cve_fh.flush()

    log.info(
        "hackerone: done — %d posts, %d CVE index entries → %s",
        post_count,
        cve_index_count,
        output,
    )
    return post_count, cve_index_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collect()
