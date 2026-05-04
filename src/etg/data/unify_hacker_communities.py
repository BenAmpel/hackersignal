"""Build a unified exact-deduped hacker-community corpus JSONL."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .preprocessing import clean_text, parse_timestamp


INCLUDED_FILES = [
    "0x00sec_posts.jsonl",
    "antionline_posts.jsonl",
    "crackingarena_posts.jsonl",
    "cve_hacker_forum_posts.jsonl",
    "deepdarkcti_public_broad_2026-04-18_raw.jsonl",
    "evolution_posts.jsonl",
    "fulldisclosure_posts.jsonl",
    "gayanku_posts.jsonl",
    "go4expert_posts.jsonl",
    "hacker_exploits_posts.jsonl",
    "hackersploit_posts.jsonl",
    "hackthebox_posts.jsonl",
    "hackforums_posts.jsonl",
    "kaeli_hacker_posts.jsonl",
    "parrotsec_posts.jsonl",
    "seebug_posts.jsonl",
    "vulnlab_posts.jsonl",
    "zeroday_posts.jsonl",
    "zeroscience_posts.jsonl",
]

CONTEXT_FILES = [
    "cisa_kev_posts.jsonl",
    "cvefixes_posts.jsonl",
    "exploitdb_hf_posts.jsonl",
    "exploitdb_posts.jsonl",
    "github_advisory_posts.jsonl",
    "hackerone_posts.jsonl",
    "nvd_posts.jsonl",
    "packetstorm_posts.jsonl",
    "public_exploits_posts.jsonl",
]

NEURIPS_EXPANSION_FILES = INCLUDED_FILES + CONTEXT_FILES

_SOURCE_LAYER_BY_FILE = {
    # Community discourse
    **{name: "hacker_community" for name in INCLUDED_FILES},
    # Disclosure/advisory/reference context
    "cisa_kev_posts.jsonl": "exploitation_reference",
    "cvefixes_posts.jsonl": "fix_commit_reference",
    "exploitdb_hf_posts.jsonl": "exploit_qa_reference",
    "exploitdb_posts.jsonl": "exploit_archive",
    "github_advisory_posts.jsonl": "advisory_reference",
    "hackerone_posts.jsonl": "bug_bounty_disclosure",
    "nvd_posts.jsonl": "vulnerability_reference",
    "packetstorm_posts.jsonl": "exploit_archive",
    "public_exploits_posts.jsonl": "exploit_archive",
}


@dataclass(frozen=True)
class UnifiedStats:
    input_rows: int = 0
    written_rows: int = 0
    duplicate_rows: int = 0
    skipped_empty_text: int = 0
    files_seen: int = 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalised_text_fingerprint(text: str) -> bytes:
    normalised = " ".join(text.lower().split())
    return hashlib.sha256(normalised.encode("utf-8")).digest()


def _unified_id(record: dict) -> str:
    parts = [
        record.get("source_dataset", ""),
        record.get("source_file", ""),
        record.get("source_record_id", ""),
        record.get("forum_id", ""),
        record.get("timestamp", ""),
        record.get("author_hash", ""),
    ]
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:24]


def _base_record(
    *,
    source_file: str,
    source_dataset: str,
    source_format: str,
    obj: dict,
) -> dict:
    timestamp = obj.get("timestamp") or ""
    if timestamp:
        dt = parse_timestamp(timestamp)
        if dt is not None:
            timestamp = dt.isoformat()

    text_raw = (obj.get("body") or obj.get("text") or "").strip()
    cleaned = clean_text(text_raw)

    record = {
        "unified_id": "",
        "source_dataset": source_dataset,
        "source_file": source_file,
        "source_format": source_format,
        "source_layer": _SOURCE_LAYER_BY_FILE.get(Path(source_file).name, "unknown"),
        "source_record_id": str(obj.get("id", "")),
        "forum_id": obj.get("forum_id", ""),
        "timestamp": timestamp,
        "author_hash": obj.get("author_hash", ""),
        "text": cleaned,
        "text_raw": text_raw,
        "thread_url": obj.get("thread_url"),
        "thread_title": obj.get("thread_title"),
        "section": obj.get("section"),
        "post_index": obj.get("post_index"),
        "post_id": obj.get("post_id"),
        "thread_id": obj.get("thread_id"),
        "scraped_at": obj.get("scraped_at"),
        "reply_count": obj.get("reply_count"),
        "view_count": obj.get("view_count"),
        "tags": obj.get("tags", []),
        "cve_refs": obj.get("cve_refs", []),
        "dataset_generated_at": _now_iso(),
    }
    record["unified_id"] = _unified_id(record)
    return record


def iter_unified_records(data_dir: str | Path, files: Iterable[str] = INCLUDED_FILES):
    data_dir = Path(data_dir)
    for name in files:
        path = data_dir / name
        if not path.exists() or path.stat().st_size == 0:
            continue
        source_dataset = path.stem
        source_format = "raw" if source_dataset.endswith("_raw") else "forumpost"
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield _base_record(
                    source_file=str(path),
                    source_dataset=source_dataset,
                    source_format=source_format,
                    obj=obj,
                )


def build_unified_hacker_community_dataset(
    *,
    data_dir: str | Path = "data",
    output_path: str | Path = "data/unified_hacker_communities.jsonl",
    manifest_path: str | Path = "data/unified_hacker_communities_manifest.json",
    dedupe_db_path: str | Path = "data/unified_hacker_communities_seen.sqlite3",
    files: Iterable[str] = INCLUDED_FILES,
) -> dict:
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)
    dedupe_db_path = Path(dedupe_db_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    dedupe_db_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()
    if dedupe_db_path.exists():
        dedupe_db_path.unlink()

    stats = {
        "generated_at": _now_iso(),
        "output_path": str(output_path),
        "dedupe_db_path": str(dedupe_db_path),
        "included_files": [],
        "input_rows": 0,
        "written_rows": 0,
        "duplicate_rows": 0,
        "skipped_empty_text": 0,
        "source_rows": {},
        "source_written_rows": {},
        "unique_forums": set(),
    }

    conn = sqlite3.connect(dedupe_db_path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("CREATE TABLE seen (fp BLOB PRIMARY KEY)")

    try:
        with output_path.open("w", encoding="utf-8") as out:
            for record in iter_unified_records(data_dir, files):
                source = record["source_dataset"]
                if source not in stats["included_files"]:
                    stats["included_files"].append(source)
                stats["input_rows"] += 1
                stats["source_rows"][source] = stats["source_rows"].get(source, 0) + 1

                if not record["text"]:
                    stats["skipped_empty_text"] += 1
                    continue

                fp = _normalised_text_fingerprint(record["text"])
                cur = conn.execute("INSERT OR IGNORE INTO seen(fp) VALUES (?)", (fp,))
                if cur.rowcount == 0:
                    stats["duplicate_rows"] += 1
                    continue

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["written_rows"] += 1
                stats["source_written_rows"][source] = stats["source_written_rows"].get(source, 0) + 1
                if record["forum_id"]:
                    stats["unique_forums"].add(record["forum_id"])
        conn.commit()
    finally:
        conn.close()

    stats["unique_forums"] = sorted(stats["unique_forums"])
    stats["unique_forum_count"] = len(stats["unique_forums"])
    manifest_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats
