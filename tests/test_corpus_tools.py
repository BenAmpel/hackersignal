import json
from pathlib import Path

from etg.data.corpus_tools import audit_raw_corpus, build_filtered_corpus
from etg.data.preprocessing import PreprocessConfig
from etg.data.schemas import load_posts_jsonl


def _write_raw(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_audit_raw_corpus_counts_duplicates_and_coverage(tmp_path: Path):
    raw_path = tmp_path / "raw.jsonl"
    rows = [
        {
            "id": "p1",
            "forum_id": "alpha",
            "thread_url": "https://forum.example/t/1",
            "section": "market",
            "body": "Working exploit kit for CVE-2024-1234 with detailed setup",
            "timestamp": "2026-04-18T08:00:00+00:00",
            "scraped_at": "2026-04-18T08:05:00+00:00",
            "cve_refs": ["CVE-2024-1234"],
        },
        {
            "id": "p2",
            "forum_id": "alpha",
            "thread_url": "https://forum.example/t/2",
            "section": "market",
            "body": "Working exploit kit for CVE-2024-1234 with detailed setup",
            "timestamp": "2026-04-18T08:06:00+00:00",
            "scraped_at": "2026-04-18T08:06:00+00:00",
            "cve_refs": [],
        },
        {
            "id": "p3",
            "forum_id": "beta",
            "thread_url": "https://forum.example/t/3",
            "section": "chat",
            "body": "Privet mir eto test soobschenie dlya foruma",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "scraped_at": "2025-01-02T00:00:00+00:00",
            "cve_refs": [],
        },
    ]
    _write_raw(raw_path, rows)

    report = audit_raw_corpus(raw_path)

    assert report["total_rows"] == 3
    assert report["unique_forums"] == 2
    assert report["unique_threads"] == 3
    assert report["exact_duplicate_rows"] == 1
    assert report["rows_with_real_timestamp"] == 2
    assert report["rows_with_cve_refs"] == 1
    assert report["rows_cti_nonzero"] == 2
    assert report["top_forums"][0] == ("alpha", 2)


def test_build_filtered_corpus_writes_deduped_posts_and_reports(tmp_path: Path):
    raw_path = tmp_path / "raw.jsonl"
    out_path = tmp_path / "filtered.jsonl"
    audit_json = tmp_path / "audit.json"
    audit_md = tmp_path / "audit.md"
    rows = [
        {
            "id": "p1",
            "forum_id": "alpha",
            "thread_url": "https://forum.example/t/1",
            "body": "Working exploit kit for CVE-2024-1234 with detailed setup",
            "timestamp": "2026-04-18T08:00:00+00:00",
            "scraped_at": "2026-04-18T08:05:00+00:00",
            "author_hash": "a1",
        },
        {
            "id": "p2",
            "forum_id": "alpha",
            "thread_url": "https://forum.example/t/2",
            "body": "Working exploit kit for CVE-2024-1234 with detailed setup",
            "timestamp": "2026-04-18T08:10:00+00:00",
            "scraped_at": "2026-04-18T08:10:00+00:00",
            "author_hash": "a2",
        },
        {
            "id": "p3",
            "forum_id": "beta",
            "thread_url": "https://forum.example/t/3",
            "body": "Privet mir eto test soobschenie dlya foruma s exploit details",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "scraped_at": "2025-01-02T00:00:00+00:00",
            "author_hash": "b1",
        },
    ]
    _write_raw(raw_path, rows)

    config = PreprocessConfig(
        require_real_timestamp=False,
        min_ascii_ratio=0.0,
        use_langdetect=False,
        min_tokens=1,
        max_tokens=100,
        exact_dedup=True,
        near_dedup=False,
    )
    report = build_filtered_corpus(
        raw_path,
        out_path,
        config=config,
        audit_json=audit_json,
        audit_markdown=audit_md,
    )

    posts = load_posts_jsonl(out_path)
    assert len(posts) == 2
    assert report["filtered_posts"] == 2
    assert report["pipeline_stats"]["after_exact_dedup"] == 2
    assert audit_json.exists()
    assert audit_md.exists()
    assert "Filtered Corpus" in audit_md.read_text(encoding="utf-8")
