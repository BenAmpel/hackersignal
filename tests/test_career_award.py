from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from etg.config import Config
from etg.data.career_award import load_career_rt1_posts, load_career_rt2_pairs
from etg.data.schemas import EVPair, ForumPost
from etg.rt2.finetune_cte import _time_window_negatives


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_forum_post_accepts_unified_id():
    post = ForumPost.from_dict(
        {
            "unified_id": "u1",
            "text": "ctf exploit writeup with payload details",
            "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat(),
            "forum_id": "0x00sec",
            "author_hash": "abc",
        }
    )
    assert post.id == "u1"


def test_load_career_rt1_posts_balances_spells(tmp_path):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for idx in range(12):
        rows.append(
            ForumPost(
                id=f"p{idx}",
                text=(
                    "exploit payload privilege escalation walkthrough with shellcode, "
                    "credential dumping, remote code execution, persistence, post "
                    "exploitation tradecraft, lateral movement, phishing lures, "
                    "session hijacking, ransomware staging, and analyst notes for "
                    f"career award alignment example {idx}"
                ),
                timestamp=base + timedelta(days=idx * 10),
                forum_id="hackforums",
                author_hash=f"a{idx}",
            ).to_dict()
        )
    src = tmp_path / "hackforums_posts.jsonl"
    _write_jsonl(src, rows)

    config = replace(Config.tiny(), n_posts=6, n_spells=3)
    posts = load_career_rt1_posts(config, paths=[src])

    assert len(posts) == config.n_posts
    assert posts == sorted(posts, key=lambda post: post.timestamp)
    assert len({post.id for post in posts}) == len(posts)


def test_load_career_rt2_pairs_respects_config_limit(tmp_path):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for idx in range(20):
        rows.append(
            EVPair(
                exploit_text=f"exploit {idx}",
                vulnerability_text=f"vuln {idx}",
                cve_id=f"CVE-2024-{idx}",
                timestamp=base + timedelta(days=idx),
                label=1,
            ).to_dict()
        )
        rows.append(
            EVPair(
                exploit_text=f"exploit {idx}",
                vulnerability_text=f"negative vuln {idx}",
                cve_id=f"CVE-2024-{100 + idx}",
                timestamp=base + timedelta(days=idx),
                label=0,
            ).to_dict()
        )
    src = tmp_path / "ev_pairs.jsonl"
    _write_jsonl(src, rows)

    config = replace(Config.tiny(), n_ev_pairs=5, n_ev_negatives_per_positive=1)
    pairs = load_career_rt2_pairs(config, path=src, seed=7)

    assert sum(1 for pair in pairs if pair.label == 1) == 5
    assert sum(1 for pair in pairs if pair.label == 0) == 5


def test_time_aware_negative_selector_prefers_local_same_exploit():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    positive = EVPair(
        exploit_text="exploit a",
        vulnerability_text="vuln a",
        cve_id="CVE-2024-1",
        timestamp=base,
        label=1,
    )
    same_exploit_negative = EVPair(
        exploit_text="exploit a",
        vulnerability_text="vuln b",
        cve_id="CVE-2024-2",
        timestamp=base + timedelta(days=2),
        label=0,
    )
    far_negative = EVPair(
        exploit_text="exploit z",
        vulnerability_text="vuln z",
        cve_id="CVE-2024-3",
        timestamp=base + timedelta(days=400),
        label=0,
    )

    selector = _time_window_negatives([positive, same_exploit_negative, far_negative], window_days=90)
    picked = selector(positive, __import__("random").Random(13))

    assert picked == same_exploit_negative
