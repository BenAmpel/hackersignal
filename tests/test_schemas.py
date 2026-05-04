from datetime import datetime, timezone

from etg.data.schemas import EVPair, ForumPost


def test_forum_post_round_trip():
    p = ForumPost(
        id="p1",
        text="sqli exploit",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        forum_id="f0",
        author_hash="abc",
    )
    d = p.to_dict()
    p2 = ForumPost.from_dict(d)
    assert p == p2


def test_ev_pair_round_trip():
    ev = EVPair(
        exploit_text="inject payload",
        vulnerability_text="CVE-2023-1 sqli on apache",
        cve_id="CVE-2023-1",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        label=1,
    )
    d = ev.to_dict()
    ev2 = EVPair.from_dict(d)
    assert ev == ev2
