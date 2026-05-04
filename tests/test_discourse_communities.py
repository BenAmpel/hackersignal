import json
from pathlib import Path

from etg.collectors import discourse_communities as dc


def test_collect_discourse_community_writes_posts_and_cve_index(monkeypatch, tmp_path: Path):
    community = dc.DiscourseCommunity(
        forum_id="example_discourse",
        base_url="https://forum.example",
        label="Example",
        include_category_slugs=("exploits",),
    )

    def fake_get_json(session, url, *, request_delay, params=None):
        if url.endswith("/categories.json"):
            return {
                "category_list": {
                    "categories": [
                        {"id": 1, "slug": "general", "name": "General"},
                        {"id": 2, "slug": "exploits", "name": "Exploits"},
                    ]
                }
            }
        if url.endswith("/c/exploits/2/l/latest.json"):
            return {
                "topic_list": {
                    "topics": [
                        {
                            "id": 99,
                            "slug": "poc-drop",
                            "title": "PoC drop",
                            "created_at": "2026-01-01T00:00:00.000Z",
                            "views": 7,
                        }
                    ],
                    "more_topics_url": None,
                }
            }
        if url.endswith("/t/99.json"):
            return {
                "id": 99,
                "slug": "poc-drop",
                "title": "PoC drop",
                "posts_count": 2,
                "views": 7,
                "tags": ["cve"],
                "post_stream": {
                    "stream": [501, 502],
                    "posts": [
                        {
                            "id": 501,
                            "post_number": 1,
                            "username": "alice",
                            "created_at": "2026-01-01T00:00:00.000Z",
                            "cooked": "<p>Exploit details for CVE-2026-12345.</p>",
                        }
                    ],
                },
            }
        if "/t/99/posts.json?" in url:
            return {
                "post_stream": {
                    "posts": [
                        {
                            "id": 502,
                            "post_number": 2,
                            "username": "bob",
                            "created_at": "2026-01-01T00:05:00.000Z",
                            "cooked": "<p>Mitigation notes.</p>",
                        }
                    ]
                }
            }
        return None

    monkeypatch.setattr(dc, "_get_json", fake_get_json)
    monkeypatch.setattr(dc.time, "sleep", lambda _: None)

    output = tmp_path / "posts.jsonl"
    cve_output = tmp_path / "cves.jsonl"
    post_count, cve_count = dc.collect_discourse_community(
        community,
        output=output,
        cve_index_output=cve_output,
        request_delay=0,
    )

    posts = [json.loads(line) for line in output.read_text().splitlines()]
    cves = [json.loads(line) for line in cve_output.read_text().splitlines()]

    assert post_count == 2
    assert cve_count == 1
    assert posts[0]["forum_id"] == "example_discourse"
    assert posts[0]["thread_url"] == "https://forum.example/t/poc-drop/99"
    assert posts[0]["thread_title"] == "PoC drop"
    assert posts[0]["section"] == "Exploits"
    assert posts[0]["tags"] == ["cve"]
    assert "CVE-2026-12345" in posts[0]["text"]
    assert cves[0]["cve_id"] == "CVE-2026-12345"


def test_selected_categories_respects_include_and_exclude():
    community = dc.DiscourseCommunity(
        forum_id="x",
        base_url="https://x.example",
        label="X",
        include_category_slugs=("content", "ctfs"),
        exclude_category_slugs=("off-topic",),
    )
    categories = [
        {"slug": "content", "path_slug": "content"},
        {"slug": "ctfs", "path_slug": "ctfs"},
        {"slug": "off-topic", "path_slug": "off-topic"},
        {"slug": "support", "path_slug": "support"},
    ]

    selected = dc._selected_categories(community, categories)

    assert [category["slug"] for category in selected] == ["content", "ctfs"]
