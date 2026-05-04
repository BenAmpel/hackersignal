"""Importer for HackForums crawl data — gzip-compressed JSON-lines archive.

Data source: ``data/external/hackforums/hackforums-out-fix.jl.*.gz``

Each ``.gz`` file contains one JSON record per line.  Every record represents a
crawled HackForums thread page and carries a ``features.items`` list of
individual posts.  Items may be repeated across records (the same post appears
on multiple page offsets of the same thread), so we deduplicate by ``item_id``.

Outputs
-------
- Forum posts JSONL: one record per unique post.
- CVE index JSONL: one record per (post, CVE) pair found in the post text.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from pathlib import Path

from etg.collectors._common import author_hash, parse_date, post_id

log = logging.getLogger(__name__)

FORUM_ID = "hackforums"

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_records(gz_path: Path):
    """Yield parsed JSON records from a single gzip JSON-lines file."""
    try:
        with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    log.debug("hackforums: JSON decode error in %s: %s", gz_path.name, exc)
    except EOFError:
        log.warning("hackforums: truncated archive %s — partial data used", gz_path.name)
    except Exception as exc:
        log.warning("hackforums: error reading %s: %s", gz_path.name, exc)


def _extract_items(record: dict) -> list[dict]:
    """Return the list of post items from a crawl record, or empty list."""
    try:
        return record["features"]["items"]
    except (KeyError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect(
    data_dir: Path = Path("data/external/hackforums"),
    output: Path = Path("data/hackforums_posts.jsonl"),
    cve_index_output: Path = Path("data/hackforums_cve_index.jsonl"),
    log_every: int = 1000,
) -> tuple[int, int]:
    """Import HackForums crawl archives and write ForumPost + CVE-index JSONL.

    Parameters
    ----------
    data_dir:
        Directory containing ``hackforums-out-fix.jl.*.gz`` files.
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
    data_dir = Path(data_dir)
    output = Path(output)
    cve_index_output = Path(cve_index_output)

    output.parent.mkdir(parents=True, exist_ok=True)
    cve_index_output.parent.mkdir(parents=True, exist_ok=True)

    gz_files = sorted(data_dir.glob("hackforums-out-fix.jl.*.gz"))
    if not gz_files:
        log.warning("hackforums: no .gz files found in %s", data_dir)
        return 0, 0

    log.info("hackforums: found %d archive file(s) in %s", len(gz_files), data_dir)

    # Global deduplication sets
    seen_pids: set[str] = set()
    seen_item_ids: set[str] = set()
    # Track which thread_ids have already had their title prepended
    seen_thread_ids: set[str] = set()

    post_count = 0
    cve_index_count = 0

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        for gz_path in gz_files:
            log.info("hackforums: processing %s", gz_path.name)

            for record in _iter_records(gz_path):
                items = _extract_items(record)

                for item in items:
                    item_id = str(item.get("item_id", "")).strip()
                    if not item_id:
                        continue

                    # Deduplicate at item_id level before computing pid
                    if item_id in seen_item_ids:
                        continue
                    seen_item_ids.add(item_id)

                    # --- Content ---
                    content = item.get("content", "") or ""
                    content = content.strip()
                    if len(content) < 10:
                        continue

                    thread_id = str(item.get("thread_id", "")).strip()
                    thread_name = (item.get("thread_name", "") or "").strip()

                    # Prepend thread title for the first post of each thread
                    is_first = thread_id and thread_id not in seen_thread_ids
                    if is_first and thread_name:
                        text = f"{thread_name}\n\n{content}"
                        seen_thread_ids.add(thread_id)
                    else:
                        text = content
                        if thread_id:
                            seen_thread_ids.add(thread_id)

                    # --- IDs ---
                    pid = post_id(FORUM_ID, item_id)
                    if pid in seen_pids:
                        continue
                    seen_pids.add(pid)

                    # --- Author ---
                    author_raw = item.get("author") or {}
                    if isinstance(author_raw, dict):
                        author_name = (
                            author_raw.get("name")
                            or author_raw.get("author_id")
                            or "unknown"
                        )
                    else:
                        author_name = str(author_raw) or "unknown"
                    ahash = author_hash(FORUM_ID, author_name)

                    # --- Timestamp ---
                    created_at = item.get("created_at", "") or ""
                    ts = parse_date(created_at)

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
                    cves = list(dict.fromkeys(
                        m.upper() for m in _CVE_RE.findall(text)
                    ))
                    title = thread_name if is_first else ""
                    for cve in cves:
                        entry = {
                            "exploit_id": pid,
                            "cve_id": cve,
                            "exploit_text": text[:500],
                            "title": title,
                            "published": ts.isoformat(),
                            "source": FORUM_ID,
                        }
                        cve_fh.write(json.dumps(entry) + "\n")
                        cve_index_count += 1

                    if post_count % log_every == 0:
                        log.info(
                            "hackforums: wrote %d posts / %d CVE entries so far",
                            post_count,
                            cve_index_count,
                        )

            # Flush after each archive file
            post_fh.flush()
            cve_fh.flush()

    log.info(
        "hackforums: done — %d posts, %d CVE index entries → %s",
        post_count,
        cve_index_count,
        output,
    )
    return post_count, cve_index_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collect()
