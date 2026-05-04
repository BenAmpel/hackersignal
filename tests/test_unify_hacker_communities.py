import json
from pathlib import Path

from etg.data.unify_hacker_communities import build_unified_hacker_community_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_unified_hacker_community_dataset_dedupes_and_preserves_columns(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_jsonl(
        data_dir / "forum_a_posts.jsonl",
        [
            {
                "id": "a1",
                "forum_id": "forum_a",
                "text": "Exploit release for CVE-2024-1234 with proof of concept",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "author_hash": "u1",
            },
            {
                "id": "a2",
                "forum_id": "forum_a",
                "text": "Exploit release for CVE-2024-1234 with proof of concept",
                "timestamp": "2026-01-02T00:00:00+00:00",
                "author_hash": "u2",
            },
        ],
    )
    _write_jsonl(
        data_dir / "forum_b_raw.jsonl",
        [
            {
                "id": "b1",
                "forum_id": "forum_b",
                "body": "Fresh stealer logs and panel access for sale",
                "timestamp": "2026-02-01T00:00:00+00:00",
                "author_hash": "u3",
                "thread_url": "https://forum.example/thread/1",
                "thread_title": "Logs for sale",
                "section": "market",
                "post_index": 0,
                "post_id": "11",
                "thread_id": "22",
                "scraped_at": "2026-02-02T00:00:00+00:00",
            }
        ],
    )

    out = tmp_path / "unified.jsonl"
    manifest = tmp_path / "manifest.json"
    seen = tmp_path / "seen.sqlite3"
    stats = build_unified_hacker_community_dataset(
        data_dir=data_dir,
        output_path=out,
        manifest_path=manifest,
        dedupe_db_path=seen,
        files=["forum_a_posts.jsonl", "forum_b_raw.jsonl"],
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert stats["input_rows"] == 3
    assert stats["written_rows"] == 2
    assert stats["duplicate_rows"] == 1
    assert stats["unique_forum_count"] == 2
    assert rows[0]["source_dataset"] == "forum_a_posts"
    assert rows[0]["source_format"] == "forumpost"
    assert rows[0]["source_layer"] == "unknown"
    assert rows[1]["source_format"] == "raw"
    assert rows[1]["thread_url"] == "https://forum.example/thread/1"
    assert rows[1]["section"] == "market"
