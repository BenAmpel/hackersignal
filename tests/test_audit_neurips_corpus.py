import csv
import json
from pathlib import Path

from etg.data.audit_neurips_corpus import audit_neurips_corpus


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_audit_neurips_corpus_writes_forum_category_tables(tmp_path: Path):
    src = tmp_path / "corpus.jsonl"
    _write_jsonl(
        src,
        [
            {
                "unified_id": "a",
                "source_layer": "hacker_community",
                "source_dataset": "forum_a_posts",
                "forum_id": "forum_a",
                "section": "exploits",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "text": "Exploit writeup for CVE-2026-12345 with payload details",
            },
            {
                "unified_id": "b",
                "source_layer": "hacker_community",
                "source_dataset": "forum_a_posts",
                "forum_id": "forum_a",
                "section": "exploits",
                "timestamp": "2026-01-02T00:00:00+00:00",
                "text": "Exploit writeup for CVE-2026-12345 with payload details",
            },
            {
                "unified_id": "c",
                "source_layer": "vulnerability_reference",
                "source_dataset": "nvd_posts",
                "forum_id": "nvd_cve",
                "timestamp": "2026-01-03T00:00:00+00:00",
                "text": "NVD description for buffer overflow vulnerability",
            },
        ],
    )

    report = audit_neurips_corpus(
        src,
        output_prefix=tmp_path / "audit",
        sample_rate=1.0,
        near_hamming_threshold=3,
    )

    assert report["total_rows"] == 3
    assert report["cve_rows"] == 2
    assert report["near_duplicate_sample_rows"] == 1
    assert report["new_source_rows"] == {
        "hackersploit": 0,
        "parrotsec": 0,
        "hackthebox": 0,
    }

    category_path = tmp_path / "audit_forum_category_counts.csv"
    rows = list(csv.DictReader(category_path.open(encoding="utf-8")))
    assert rows[0]["forum_id"] == "forum_a"
    assert rows[0]["section"] == "exploits"
    assert rows[0]["rows"] == "2"

    layer_path = tmp_path / "audit_layer_counts.csv"
    layer_rows = list(csv.DictReader(layer_path.open(encoding="utf-8")))
    assert layer_rows[0]["source_layer"] == "hacker_community"
    assert layer_rows[0]["rows"] == "2"
