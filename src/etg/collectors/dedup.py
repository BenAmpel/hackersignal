"""Cross-file deduplication for ETG post JSONL files.

Deduplication strategy
----------------------
Posts are fingerprinted on ``(canonical_forum_id, text_fingerprint)`` where
``text_fingerprint`` is an MD5 of the first 400 normalised characters.

Using the forum ID in the key prevents false-positive deduplication across
genuinely distinct communities that happen to share boilerplate text (e.g.
standard header/footer templates).  Forum alias normalisation (see
``_FORUM_ALIASES``) ensures that the *same* community scrapped from two
independent sources is treated as a single forum for dedup purposes.

When two posts share the same ``(forum_id, text)`` fingerprint, the one from
the **higher-priority source** is kept and the other is dropped.

Source priority (higher number = preferred):
  10  Freshly scraped hacker forums: antionline, go4expert, cipherspace,
      hackforums, antichat, crackingarena
   8  CISA KEV (confirmed-exploited, high-signal CVEs)
   7  Full Disclosure, GitHub Advisories, PacketStorm
   6  ExploitDB (git-repo version — most complete)
   5  ZeroDay.today
   4  NVD (vulnerability descriptions — not exploit text per se)
   3  DTL exploits.h5 / PublicExploits.h5 (older ExploitDB snapshots)
   2  Zenodo dumps: evolution_posts.jsonl
   1  Everything else (gayanku derived, synthetic, unknown)

Usage
-----
CLI::

    python -m etg.collectors.cli --deduplicate
    python -m etg.collectors.cli --deduplicate --dedup-dry-run

Library::

    from etg.collectors.dedup import deduplicate_posts
    stats = deduplicate_posts(Path("data"))
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Forum alias normalisation
# ---------------------------------------------------------------------------

#: Maps variant ``forum_id`` values to a single canonical ID used *only* for
#: deduplication fingerprinting.  The original ``forum_id`` is preserved in
#: written output; this mapping only affects which posts are considered
#: duplicates of each other.
_FORUM_ALIASES: dict[str, str] = {
    # Gayanku CSV re-scrapes of communities that also have standalone files
    "gayanku_evolution":              "evolution",
    # If a future scraper names the same markets differently, add aliases here.
}


def _canonical_forum(forum_id: str) -> str:
    """Return the canonical forum ID for dedup purposes."""
    return _FORUM_ALIASES.get(forum_id, forum_id)


# ---------------------------------------------------------------------------
# Source priority table
# ---------------------------------------------------------------------------

_HIGH_PRIORITY_FORUMS = {
    "antionline", "go4expert", "cipherspace", "hackforums", "antichat",
    "crackingarena", "breachforums", "exploitforums", "nulledbb",
}

_PRIORITY: dict[str, int] = {
    # real hacker forums (freshly scraped or well-curated datasets)
    **{f: 10 for f in _HIGH_PRIORITY_FORUMS},
    # CVE-confirmed exploited
    "cisa_kev": 8,
    # security mailing lists / advisories
    "fulldisclosure": 7,
    "github_advisory": 7,
    "packetstorm": 7,
    # exploit DBs
    "exploitdb": 6,
    # 0day
    "zeroday_today": 5,
    "zeroday": 5,
    # vulnerability descriptions
    "nvd": 4,
    "nvd_cve": 4,
    # older H5 snapshots / Zenodo dumps
    "dtl_exploits": 3,
    "public_exploits": 3,
    "evolution": 2,       # Zenodo dump — good, but lower than freshly scraped
    # synthetic
    "nvd_cve_analysis": 1,
}

_DEFAULT_PRIORITY = 1


def _source_priority(forum_id: str) -> int:
    """Return the priority score for a ``forum_id``.

    Uses the *canonical* forum ID (after alias resolution) so that all
    sources of the same community compete on equal footing.
    Checks exact match first, then prefix match against known sources.
    """
    if not forum_id:
        return _DEFAULT_PRIORITY
    fid = _canonical_forum(forum_id.lower())
    if fid in _PRIORITY:
        return _PRIORITY[fid]
    # Prefix checks (e.g. "hacker_hackforums" or any unknown variant)
    for key, score in _PRIORITY.items():
        if fid.startswith(key) or key.startswith(fid):
            return score
    return _DEFAULT_PRIORITY


def _text_fingerprint(text: str) -> str:
    """MD5 of the first 400 chars of normalised text."""
    t = re.sub(r"\s+", " ", text[:400].lower()).strip()
    return hashlib.md5(t.encode()).hexdigest()


def _post_key(forum_id: str, text: str) -> str:
    """Composite dedup key: ``canonical_forum_id:text_fingerprint``.

    Using both fields prevents false-positive deduplication when two
    different communities share boilerplate text.
    """
    return f"{_canonical_forum(forum_id)}:{_text_fingerprint(text)}"


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------

def _discover_post_files(data_dir: Path) -> list[Path]:
    """Return all ``*_posts.jsonl`` files in *data_dir*, sorted alphabetically.

    Lower file-index (earlier in sort) wins ties during dedup.  Alphabetical
    order naturally puts freshly scraped files (e.g. ``antionline_posts.jsonl``)
    before aggregated H5-derived files (``hacker_exploits_posts.jsonl``,
    ``kaeli_hacker_posts.jsonl``), which is the desired priority.
    """
    return sorted(data_dir.glob("*_posts.jsonl"))


# ---------------------------------------------------------------------------
# Core deduplication
# ---------------------------------------------------------------------------

def deduplicate_posts(
    data_dir: Path = Path("data"),
    post_glob: str = "*_posts.jsonl",
    dry_run: bool = False,
    min_text_len: int = 20,
) -> dict[str, dict]:
    """Deduplicate all ``*_posts.jsonl`` files in *data_dir*.

    Algorithm
    ---------
    1. Read every post from every file; compute ``(canonical_forum_id,
       text_fingerprint)`` key.
    2. For each key that appears in multiple records, keep the post from
       the highest-priority source; mark the rest as duplicates.
    3. Rewrite each file (unless *dry_run*) with duplicates removed.

    Returns a stats dict::

        {
          "total_posts_before": int,
          "total_posts_after":  int,
          "total_removed":      int,
          "per_file": {
            "<filename>": {"before": int, "removed": int, "after": int}
          }
        }
    """
    data_dir = Path(data_dir)
    post_files = sorted(data_dir.glob(post_glob))

    if not post_files:
        log.warning("dedup: no files matching %s in %s", post_glob, data_dir)
        return {}

    log.info("dedup: scanning %d files in %s …", len(post_files), data_dir)

    # ---- Pass 1: index every post ----------------------------------------
    # key → list of (priority, file_index, line_index)
    key_map: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    # file_index → list of raw JSON strings
    all_lines: list[list[str]] = []

    for fi, fpath in enumerate(post_files):
        lines: list[str] = []
        with fpath.open(encoding="utf-8") as f:
            for li, line in enumerate(f):
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    text = d.get("text", "")
                    if len(text) < min_text_len:
                        lines.append(line)
                        continue
                    fid  = d.get("forum_id", "")
                    key  = _post_key(fid, text)
                    prio = _source_priority(fid)
                    key_map[key].append((prio, fi, li))
                    lines.append(line)
                except (json.JSONDecodeError, Exception):
                    lines.append(line)
        all_lines.append(lines)

    log.info("dedup: indexed %d unique (forum, text) keys across %d files",
             len(key_map), len(post_files))

    # ---- Pass 2: decide which posts to keep ---------------------------------
    # For each key group, keep the entry with the highest priority.
    # Ties: keep the lowest (fi, li) = earliest alphabetically-sorted file.
    keep_set: set[tuple[int, int]] = set()
    dup_count = 0

    for key, entries in key_map.items():
        if len(entries) == 1:
            keep_set.add((entries[0][1], entries[0][2]))
            continue
        best_prio = max(e[0] for e in entries)
        winners   = [e for e in entries if e[0] == best_prio]
        winner    = min(winners, key=lambda e: (e[1], e[2]))
        keep_set.add((winner[1], winner[2]))
        dup_count += len(entries) - 1

    log.info("dedup: %d duplicate posts identified", dup_count)

    # ---- Pass 3: rewrite files  -------------------------------------------
    stats: dict[str, dict] = {"per_file": {}}
    total_before = 0
    total_after  = 0

    for fi, fpath in enumerate(post_files):
        lines = all_lines[fi]
        before = len(lines)
        kept_lines = [line for li, line in enumerate(lines) if (fi, li) in keep_set]

        removed     = before - len(kept_lines)
        total_before += before
        total_after  += len(kept_lines)

        stats["per_file"][fpath.name] = {
            "before":  before,
            "removed": removed,
            "after":   len(kept_lines),
        }

        if removed > 0:
            log.info("dedup: %s — %d → %d posts (%d removed)",
                     fpath.name, before, len(kept_lines), removed)

        if not dry_run and removed > 0:
            tmp = fpath.with_suffix(".jsonl.dedup_tmp")
            with tmp.open("w", encoding="utf-8") as out:
                for l in kept_lines:
                    out.write(l + "\n")
            tmp.replace(fpath)

    stats["total_posts_before"] = total_before
    stats["total_posts_after"]  = total_after
    stats["total_removed"]      = total_before - total_after

    log.info(
        "dedup: complete. before=%d, after=%d, removed=%d%s",
        total_before, total_after, stats["total_removed"],
        " (DRY RUN — no files written)" if dry_run else "",
    )
    return stats


# ---------------------------------------------------------------------------
# CLI entry point (also called from etg.collectors.cli)
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Deduplicate ETG post JSONL files.")
    p.add_argument("--data-dir", default="data", help="Directory containing *_posts.jsonl files.")
    p.add_argument("--dry-run", action="store_true", help="Report duplicates without modifying files.")
    args = p.parse_args()

    stats = deduplicate_posts(Path(args.data_dir), dry_run=args.dry_run)

    print(f"\nTotal before : {stats.get('total_posts_before', 0):>10,}")
    print(f"Total after  : {stats.get('total_posts_after', 0):>10,}")
    print(f"Removed      : {stats.get('total_removed', 0):>10,}")
    print()
    print(f"{'File':<55} {'Before':>8} {'Removed':>8} {'After':>8}")
    print("-" * 83)
    for fname, s in sorted(stats.get("per_file", {}).items()):
        if s["removed"] > 0:
            print(f"{fname:<55} {s['before']:>8,} {s['removed']:>8,} {s['after']:>8,}")


if __name__ == "__main__":
    main()
