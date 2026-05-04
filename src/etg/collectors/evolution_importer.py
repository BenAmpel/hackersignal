"""Importer for the Evolution dark-web cryptomarket forum dataset (Zenodo).

Reference
---------
A large-scale longitudinal structured dataset of the dark web cryptomarket
Evolution (2014–2015). Zenodo record 10171217.

Data layout (within ``evolution/forum/``)
-----------------------------------------
post.tsv     tid  pid  seq_id  year  month  day  time  uid  text  signature
topic.tsv    fid  tid  first_uid  scrape_id  title  posts  ...
forum.tsv    fid  scrape_id  category  title  ...

Post text is HTML (``<p>…</p>``); we strip tags before storing.
There are duplicate scrape_id snapshots in topic.tsv so we take the latest
title per tid (highest scrape_id).  Posts are deduplicated by pid.

Outputs
-------
- Forum posts JSONL: one record per unique post.
- CVE index JSONL: one record per (post, CVE) pair found in the post text.
"""

from __future__ import annotations

import json
import logging
import re
from html import unescape
from pathlib import Path

import pandas as pd

from etg.collectors._common import author_hash, parse_date, post_id

log = logging.getLogger(__name__)

FORUM_ID = "evolution_darkweb"

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s{2,}")


def _strip_html(html: str) -> str:
    """Remove HTML tags, decode entities, collapse whitespace."""
    html = unescape(html)
    text = _TAG_RE.sub(" ", html)
    return _SPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect(
    data_dir: Path = Path("data/external/zenodo/evolution"),
    output: Path = Path("data/evolution_posts.jsonl"),
    cve_index_output: Path = Path("data/evolution_cve_index.jsonl"),
    log_every: int = 10000,
) -> tuple[int, int]:
    """Import the Evolution forum TSV files and write ForumPost + CVE-index JSONL.

    Parameters
    ----------
    data_dir:
        Directory containing the extracted Evolution dataset (has a ``forum/``
        subdirectory with ``post.tsv``, ``topic.tsv``, etc.).
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

    forum_dir = data_dir / "forum"
    post_tsv = forum_dir / "post.tsv"
    topic_tsv = forum_dir / "topic.tsv"

    if not post_tsv.exists():
        log.error("evolution: post.tsv not found at %s", post_tsv)
        return 0, 0

    output.parent.mkdir(parents=True, exist_ok=True)
    cve_index_output.parent.mkdir(parents=True, exist_ok=True)

    # ---- Build tid → title mapping from topic.tsv ----
    tid_to_title: dict[str, str] = {}
    if topic_tsv.exists():
        log.info("evolution: loading topic titles from %s", topic_tsv)
        try:
            topics = pd.read_csv(
                topic_tsv,
                sep="\t",
                encoding="utf-8",
                encoding_errors="replace",
                dtype=str,
                low_memory=False,
            )
            # Deduplicate by tid, keep latest scrape_id entry
            if "scrape_id" in topics.columns:
                topics["scrape_id"] = pd.to_numeric(topics["scrape_id"], errors="coerce").fillna(0)
                topics = topics.sort_values("scrape_id").drop_duplicates("tid", keep="last")
            for _, row in topics.iterrows():
                tid = str(row.get("tid", "")).strip()
                title = str(row.get("title", "")).strip()
                if tid and title and title != "nan":
                    tid_to_title[tid] = title
            log.info("evolution: loaded %d thread titles", len(tid_to_title))
        except Exception as exc:
            log.warning("evolution: could not load topic.tsv: %s", exc)

    # ---- Process posts ----
    log.info("evolution: reading posts from %s", post_tsv)

    seen_pids: set[str] = set()
    seen_thread_ids: set[str] = set()
    post_count = 0
    cve_index_count = 0

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        reader = pd.read_csv(
            post_tsv,
            sep="\t",
            encoding="utf-8",
            encoding_errors="replace",
            dtype=str,
            low_memory=False,
            chunksize=50000,
        )

        for chunk in reader:
            for _, row in chunk.iterrows():
                pid_raw = str(row.get("pid", "")).strip()
                tid_raw = str(row.get("tid", "")).strip()
                uid_raw = str(row.get("uid", "")).strip()
                text_raw = str(row.get("text", "")).strip()

                # Skip empty posts
                if not text_raw or text_raw == "nan" or len(text_raw) < 5:
                    continue

                # Deduplicate by pid
                if pid_raw in seen_pids:
                    continue
                seen_pids.add(pid_raw)

                # Strip HTML
                body = _strip_html(text_raw)
                if len(body) < 10:
                    continue

                # Compose text with thread title for first post of each thread
                thread_title = tid_to_title.get(tid_raw, "")
                is_first = tid_raw and tid_raw not in seen_thread_ids
                if is_first and thread_title:
                    text = f"{thread_title}\n\n{body}"
                    seen_thread_ids.add(tid_raw)
                else:
                    text = body
                    if tid_raw:
                        seen_thread_ids.add(tid_raw)

                # Build timestamp from year/month/day/time columns
                year = row.get("year", "")
                month = row.get("month", "")
                day = row.get("day", "")
                time_str = row.get("time", "")
                date_str = ""
                if all(v and str(v) != "nan" for v in [year, month, day]):
                    date_str = f"{year}-{int(month):02d}-{int(day):02d}"
                    if time_str and str(time_str) != "nan":
                        date_str += f" {time_str}"
                ts = parse_date(date_str)

                pid = post_id(FORUM_ID, pid_raw)
                ahash = author_hash(FORUM_ID, uid_raw or "unknown")

                post_dict = {
                    "id": pid,
                    "text": text,
                    "timestamp": ts.isoformat(),
                    "forum_id": FORUM_ID,
                    "author_hash": ahash,
                }
                post_fh.write(json.dumps(post_dict) + "\n")
                post_count += 1

                # CVE index
                cves = list(dict.fromkeys(m.upper() for m in _CVE_RE.findall(text)))
                title_for_index = thread_title if is_first else ""
                for cve in cves:
                    entry = {
                        "exploit_id": pid,
                        "cve_id": cve,
                        "exploit_text": text[:500],
                        "title": title_for_index,
                        "published": ts.isoformat(),
                        "source": FORUM_ID,
                    }
                    cve_fh.write(json.dumps(entry) + "\n")
                    cve_index_count += 1

                if post_count % log_every == 0:
                    log.info(
                        "evolution: wrote %d posts / %d CVE entries so far",
                        post_count, cve_index_count,
                    )

        post_fh.flush()
        cve_fh.flush()

    log.info(
        "evolution: done — %d posts, %d CVE index entries → %s",
        post_count, cve_index_count, output,
    )
    return post_count, cve_index_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collect()
