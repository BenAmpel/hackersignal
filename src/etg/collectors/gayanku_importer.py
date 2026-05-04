"""Importer for the Gayanku dark-web forum dataset — CSV format.

Data source: ``data/external/gayanku/Clearnedup_ALL_7.csv``

The CSV contains posts scraped from multiple dark-web markets and forums.
Each row carries a ``Forum Name`` column; this importer maps that column
to a canonical ``forum_id`` and writes **one JSONL file per forum** so
that per-forum statistics and cross-source deduplication work correctly.

Forum mapping
-------------
  "Silk Road 1"            → gayanku_silk_road_1
  "Silk Road 2"            → gayanku_silk_road_2
  "Agora"                  → gayanku_agora
  "Evolution"              → gayanku_evolution
  "Reddit Darknet Markets" → gayanku_reddit_darknet_markets
  "Reddit Silk Road"       → gayanku_reddit_silk_road
  "Black Market Reloaded"  → gayanku_black_market_reloaded

All forum_ids carry a ``gayanku_`` prefix so they don't overwrite
independently-sourced files (e.g. ``evolution_posts.jsonl`` collected
from Zenodo).  Cross-source deduplication merges them later.

Usernames are already pseudonymised/hashed in the source file and are
stored as-is.  No unique post ID is present, so one is derived from a
SHA-256 hash of forum name, thread title, username, and datetime.

Columns
-------
(unnamed index), f_index, Forum Name, Thread Title, Username, User Type,
Post Content, datetime

Outputs (per forum, inside *output_dir*)
-----------------------------------------
- ``{forum_id}_posts.jsonl``      — one record per non-empty post row
- ``{forum_id}_cve_index.jsonl``  — one record per (post, CVE) pair
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import IO

import pandas as pd

from etg.collectors._common import parse_date, post_id

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Forum name → canonical forum_id mapping
# ---------------------------------------------------------------------------

#: Maps the raw ``Forum Name`` column values to normalised ``forum_id`` strings.
#: All IDs carry a ``gayanku_`` prefix to avoid colliding with independently
#: sourced files for the same communities.
FORUM_NAME_MAP: dict[str, str] = {
    "Silk Road 1":            "gayanku_silk_road_1",
    "Silk Road 2":            "gayanku_silk_road_2",
    "Agora":                  "gayanku_agora",
    "Evolution":              "gayanku_evolution",
    "Reddit Darknet Markets": "gayanku_reddit_darknet_markets",
    "Reddit Silk Road":       "gayanku_reddit_silk_road",
    "Black Market Reloaded":  "gayanku_black_market_reloaded",
}

# Legacy single-file ID kept for backward-compatibility when callers pass
# explicit ``output`` / ``cve_index_output`` paths instead of ``output_dir``.
_LEGACY_FORUM_ID = "gayanku_darkweb"

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


def _normalize_forum_name(raw: str) -> str:
    """Return a canonical ``forum_id`` for a CSV *Forum Name* value.

    Falls back to a slugified version of the name if not in the known map.
    """
    cleaned = raw.strip()
    if cleaned in FORUM_NAME_MAP:
        return FORUM_NAME_MAP[cleaned]
    slug = re.sub(r"[^a-z0-9]+", "_", cleaned.lower()).strip("_")
    return f"gayanku_{slug}" if slug else "gayanku_unknown"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect(
    csv_path: Path = Path("data/external/gayanku/Clearnedup_ALL_7.csv"),
    output_dir: Path | None = None,
    output: Path = Path("data/gayanku_posts.jsonl"),
    cve_index_output: Path = Path("data/gayanku_cve_index.jsonl"),
    log_every: int = 10_000,
    chunksize: int = 50_000,
) -> tuple[int, int]:
    """Import the Gayanku dark-web CSV and write per-forum ForumPost + CVE-index JSONL.

    Parameters
    ----------
    csv_path:
        Path to ``Clearnedup_ALL_7.csv``.
    output_dir:
        **Preferred** destination directory.  One ``{forum_id}_posts.jsonl``
        and one ``{forum_id}_cve_index.jsonl`` file is created here for each
        distinct forum found in the CSV.  When supplied, ``output`` and
        ``cve_index_output`` are ignored.
    output:
        *Legacy* single-file destination (used only when ``output_dir`` is
        ``None``).  All posts are written here with ``forum_id`` set to the
        per-row canonical ID (not the old flat ``gayanku_darkweb`` value).
    cve_index_output:
        *Legacy* single-file CVE index destination (used only when
        ``output_dir`` is ``None``).
    log_every:
        Log a progress message every *log_every* rows processed.
    chunksize:
        Number of CSV rows to read per pandas chunk.

    Returns
    -------
    tuple[int, int]
        ``(post_count, cve_index_count)``
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        log.error("gayanku: CSV not found: %s", csv_path)
        return 0, 0

    per_forum_mode = output_dir is not None

    if per_forum_mode:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Lazily-opened file handles, keyed by forum_id
        _post_fhs: dict[str, IO[str]] = {}
        _cve_fhs:  dict[str, IO[str]] = {}
    else:
        output = Path(output)
        cve_index_output = Path(cve_index_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        cve_index_output.parent.mkdir(parents=True, exist_ok=True)

    post_count = 0
    cve_index_count = 0
    seen_pids: set[str] = set()

    def _get_fh(fhs: dict[str, IO[str]], forum_id: str, suffix: str) -> IO[str]:
        """Return (opening if necessary) the file handle for *forum_id*."""
        if forum_id not in fhs:
            assert output_dir is not None
            path = output_dir / f"{forum_id}_{suffix}.jsonl"
            fhs[forum_id] = path.open("w", encoding="utf-8")
            log.info("gayanku: opened %s", path)
        return fhs[forum_id]

    try:
        if not per_forum_mode:
            post_fh_legacy  = output.open("w", encoding="utf-8")
            cve_fh_legacy   = cve_index_output.open("w", encoding="utf-8")
        else:
            post_fh_legacy = None  # type: ignore[assignment]
            cve_fh_legacy  = None  # type: ignore[assignment]

        reader = pd.read_csv(
            csv_path,
            encoding="utf-8",
            encoding_errors="replace",
            low_memory=False,
            chunksize=chunksize,
        )

        for chunk in reader:
            for _, row in chunk.iterrows():
                # --- Post content ---
                raw_content = row.get("Post Content", "")
                if pd.isna(raw_content):
                    continue
                post_content = str(raw_content).strip()
                if len(post_content) < 10:
                    continue

                # --- Metadata fields ---
                raw_forum_name = str(row.get("Forum Name", "") or "").strip()
                thread_title   = str(row.get("Thread Title", "") or "").strip()
                username       = str(row.get("Username", "") or "").strip()
                datetime_str   = str(row.get("datetime", "") or "").strip()

                # --- Per-row forum_id (canonical) ---
                forum_id = _normalize_forum_name(raw_forum_name)

                # --- Composed text ---
                text = f"[{raw_forum_name}] {thread_title}\n\n{post_content}"

                # --- Post ID ---
                uid = f"{raw_forum_name}::{thread_title}::{username}::{datetime_str}"
                pid = post_id(forum_id, uid)

                if pid in seen_pids:
                    continue
                seen_pids.add(pid)

                # --- Timestamp ---
                ts = parse_date(datetime_str)

                # --- Author hash ---
                ahash = username if username else "unknown"

                # --- Write post record ---
                post_dict = {
                    "id":          pid,
                    "text":        text,
                    "timestamp":   ts.isoformat(),
                    "forum_id":    forum_id,
                    "author_hash": ahash,
                }
                line = json.dumps(post_dict) + "\n"

                if per_forum_mode:
                    _get_fh(_post_fhs, forum_id, "posts").write(line)
                else:
                    post_fh_legacy.write(line)

                post_count += 1

                # --- CVE index ---
                cves = list(dict.fromkeys(
                    m.upper() for m in _CVE_RE.findall(text)
                ))
                for cve in cves:
                    entry = {
                        "exploit_id":   pid,
                        "cve_id":       cve,
                        "exploit_text": text[:500],
                        "title":        thread_title,
                        "published":    ts.isoformat(),
                        "source":       forum_id,
                    }
                    cve_line = json.dumps(entry) + "\n"
                    if per_forum_mode:
                        _get_fh(_cve_fhs, forum_id, "cve_index").write(cve_line)
                    else:
                        cve_fh_legacy.write(cve_line)
                    cve_index_count += 1

                if post_count % log_every == 0:
                    log.info(
                        "gayanku: wrote %d posts / %d CVE entries so far",
                        post_count,
                        cve_index_count,
                    )

            # Flush after each chunk
            if per_forum_mode:
                for fh in _post_fhs.values():
                    fh.flush()
                for fh in _cve_fhs.values():
                    fh.flush()
            else:
                post_fh_legacy.flush()
                cve_fh_legacy.flush()

    finally:
        if per_forum_mode:
            for fh in _post_fhs.values():
                fh.close()
            for fh in _cve_fhs.values():
                fh.close()
            log.info(
                "gayanku: done — %d posts, %d CVE index entries across %d forums → %s/",
                post_count,
                cve_index_count,
                len(_post_fhs),
                output_dir,
            )
        else:
            post_fh_legacy.close()
            cve_fh_legacy.close()
            log.info(
                "gayanku: done — %d posts, %d CVE index entries → %s",
                post_count,
                cve_index_count,
                output,
            )

    return post_count, cve_index_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import argparse

    p = argparse.ArgumentParser(description="Import Gayanku darkweb CSV into per-forum JSONL files.")
    p.add_argument("--csv", default="data/external/gayanku/Clearnedup_ALL_7.csv",
                   help="Path to Clearnedup_ALL_7.csv")
    p.add_argument("--output-dir", default="data/",
                   help="Output directory for per-forum JSONL files (default: data/)")
    args = p.parse_args()

    n_posts, n_cve = collect(
        csv_path=Path(args.csv),
        output_dir=Path(args.output_dir),
    )
    print(f"Posts: {n_posts:,}  CVE entries: {n_cve:,}")
