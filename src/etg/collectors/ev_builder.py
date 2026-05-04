"""Build EVPair JSONL by joining Exploit-DB CVE index with NVD CVE dict.

Positive pairs: exploit text matched to the NVD description of its CVE(s).
Negative pairs: exploit text matched to a randomly chosen *unrelated* CVE
description (not one the exploit already references).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from etg.collectors._common import parse_date
from etg.data.schemas import EVPair

log = logging.getLogger(__name__)


def build(
    exploitdb_cve_index: Path = Path("data/exploitdb_cve_index.jsonl"),
    nvd_cve_dict: Path = Path("data/nvd_cve_dict.jsonl"),
    output: Path = Path("data/ev_pairs.jsonl"),
    neg_ratio: float = 1.0,
    seed: int = 42,
    extra_cve_indexes: list[Path] | None = None,
) -> tuple[int, int]:
    """Join exploit–CVE links with NVD descriptions to produce EVPair JSONL.

    Parameters
    ----------
    exploitdb_cve_index:
        JSONL produced by ``exploitdb.collect``.
    nvd_cve_dict:
        JSONL produced by ``nvd.collect``.
    output:
        Destination JSONL path for EVPair records.
    neg_ratio:
        Number of negative samples per positive sample.
    seed:
        RNG seed for reproducible negative sampling.
    extra_cve_indexes:
        Additional CVE index JSONL files (e.g. from ``zeroday.collect``).
        Each row must have ``exploit_id``, ``cve_id``, and optionally
        ``title`` / ``published``.  The exploit text is taken from the
        ``title`` field when no ``exploit_text`` key is present.

    Returns
    -------
    tuple[int, int]
        ``(pos_count, neg_count)``
    """
    exploitdb_cve_index = Path(exploitdb_cve_index)
    nvd_cve_dict = Path(nvd_cve_dict)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Load NVD CVE dictionary ---
    log.info("Loading NVD CVE dict from %s", nvd_cve_dict)
    nvd_dict: dict[str, dict] = {}
    with nvd_cve_dict.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                nvd_dict[entry["cve_id"]] = entry
            except (json.JSONDecodeError, KeyError) as exc:
                log.debug("Skipping malformed NVD dict line: %s", exc)

    log.info("Loaded %d CVE entries from NVD dict", len(nvd_dict))

    # --- 2. Load all exploit entries from CVE index(es) ---
    index_files: list[Path] = [exploitdb_cve_index]
    if extra_cve_indexes:
        index_files.extend(Path(p) for p in extra_cve_indexes)

    exploit_entries: list[dict] = []
    for idx_path in index_files:
        if not idx_path.exists():
            log.warning("CVE index not found, skipping: %s", idx_path)
            continue
        log.info("Loading exploit CVE index from %s", idx_path)
        with idx_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    # Normalise zeroday-style rows to the exploitdb schema
                    if "exploit_text" not in row and "title" in row:
                        row["exploit_text"] = row.get("title", "")
                    if "cves" not in row and "cve_id" in row:
                        row["cves"] = [row["cve_id"]]
                    if "timestamp" not in row and "published" in row:
                        row["timestamp"] = row["published"]
                    exploit_entries.append(row)
                except json.JSONDecodeError as exc:
                    log.debug("Skipping malformed exploit index line: %s", exc)

    log.info("Loaded %d exploit CVE-index entries total", len(exploit_entries))

    if not exploit_entries:
        log.warning("No exploit entries found; output will be empty.")
        output.touch()
        return 0, 0

    all_cve_ids = list(nvd_dict.keys())
    if not all_cve_ids:
        log.warning("NVD dict is empty; no pairs can be built.")
        output.touch()
        return 0, 0

    rng = random.Random(seed)
    pos_count = 0

    with output.open("w", encoding="utf-8") as out_fh:
        # --- 3. Positive pairs ---
        for entry in exploit_entries:
            exploit_text: str = entry.get("exploit_text", "")
            entry_cves: list[str] = entry.get("cves", [])
            ts = parse_date(entry.get("timestamp", ""))

            matched_cves = [c for c in entry_cves if c in nvd_dict]
            for cve_id in matched_cves:
                vuln_text = nvd_dict[cve_id].get("description", "")
                pair = EVPair(
                    exploit_text=exploit_text,
                    vulnerability_text=vuln_text,
                    cve_id=cve_id,
                    timestamp=ts,
                    label=1,
                )
                out_fh.write(json.dumps(pair.to_dict()) + "\n")
                pos_count += 1

        log.info("Written %d positive EVPairs", pos_count)

        # --- 4. Negative pairs ---
        neg_count_target = int(pos_count * neg_ratio)
        neg_count = 0

        # Build a fast lookup: post_id -> set of its CVE ids, for negative sampling
        exploit_cve_sets: list[tuple[dict, set[str]]] = [
            (e, set(e.get("cves", []))) for e in exploit_entries
        ]

        attempts = 0
        max_attempts = neg_count_target * 20  # safety cap

        while neg_count < neg_count_target and attempts < max_attempts:
            attempts += 1
            entry, entry_cve_set = rng.choice(exploit_cve_sets)
            exploit_text = entry.get("exploit_text", "")
            ts = parse_date(entry.get("timestamp", ""))

            # Pick a CVE not linked to this exploit
            neg_cve_id = rng.choice(all_cve_ids)
            if neg_cve_id in entry_cve_set:
                continue

            vuln_text = nvd_dict[neg_cve_id].get("description", "")
            pair = EVPair(
                exploit_text=exploit_text,
                vulnerability_text=vuln_text,
                cve_id=neg_cve_id,
                timestamp=ts,
                label=0,
            )
            out_fh.write(json.dumps(pair.to_dict()) + "\n")
            neg_count += 1

        if neg_count < neg_count_target:
            log.warning(
                "Could only generate %d/%d negative pairs (hit attempt cap)",
                neg_count,
                neg_count_target,
            )

    log.info(
        "ev_builder: done. positive=%d, negative=%d, total=%d",
        pos_count,
        neg_count,
        pos_count + neg_count,
    )
    return pos_count, neg_count
