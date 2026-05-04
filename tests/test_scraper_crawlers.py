import time
from unittest.mock import patch

from etg.scraper.scrapers.base import canonical_thread_url, extract_redirect_targets
from etg.scraper.scrapers.generic import GenericScraper
from etg.scraper.scrapers.discourse import DiscourseScraper
from etg.scraper.scrapers.invision import InvisionScraper
from etg.scraper.scrapers.mybb import MyBBScraper
from etg.scraper.scrapers.phpbb import PhpBBScraper
from etg.scraper.scrapers.xenforo import XenForoScraper
from etg.scraper.benchmark import run_benchmark
from etg.scraper.models import ForumEntry


def test_canonical_thread_url_normalizes_common_forum_variants():
    assert (
        canonical_thread_url(
            "https://forum.example/threads/topic-name.42/unread?page=3#post-999"
        )
        == "https://forum.example/threads/topic-name.42/"
    )
    assert (
        canonical_thread_url(
            "https://forum.example/viewtopic.php?f=9&t=123&start=50#p777"
        )
        == "https://forum.example/viewtopic.php?f=9&t=123"
    )
    assert (
        canonical_thread_url(
            "https://forum.example/showthread.php?tid=88&page=4&highlight=foo"
        )
        == "https://forum.example/showthread.php?tid=88"
    )
    assert (
        canonical_thread_url(
            "https://forum.example/topic/321-sample-thread/page/2/?view=getlastpost"
        )
        == "https://forum.example/topic/321-sample-thread/"
    )


def test_discourse_thread_urls_include_category_timelines(monkeypatch):
    scraper = DiscourseScraper()

    def fake_get_json(url: str):
        if url.endswith("/latest.json?page=0"):
            return {
                "topic_list": {
                    "topics": [{"id": 11}],
                }
            }
        if url.endswith("/latest.json?page=1"):
            return {"topic_list": {"topics": []}}
        if url.endswith("/categories.json"):
            return {
                "category_list": {
                    "categories": [
                        {"id": 5, "slug": "exploits", "name": "Exploits"},
                    ]
                }
            }
        if url.endswith("/c/exploits/5.json"):
            return {
                "topic_list": {
                    "topics": [{"id": 22}, {"id": 11}],
                }
            }
        if url.endswith("/c/exploits/5.json?page=1"):
            return {"topic_list": {"topics": []}}
        return None

    monkeypatch.setattr(scraper, "_get_json", fake_get_json)

    urls = list(scraper.thread_urls("https://forum.example"))

    assert urls == [
        "https://forum.example/t/11.json",
        "https://forum.example/t/22.json",
    ]


def test_discourse_thread_urls_include_nested_categories(monkeypatch):
    scraper = DiscourseScraper()

    def fake_get_json(url: str):
        if url.endswith("/latest.json?page=0"):
            return {"topic_list": {"topics": []}}
        if url.endswith("/categories.json"):
            return {
                "category_list": {
                    "categories": [
                        {
                            "id": 5,
                            "slug": "security",
                            "subcategory_list": [
                                {"id": 6, "slug": "exploits"},
                            ],
                        },
                    ]
                }
            }
        if url.endswith("/c/security/5.json"):
            return {"topic_list": {"topics": []}}
        if url.endswith("/c/security/exploits/6.json"):
            return {"topic_list": {"topics": [{"id": 33}]}}
        if url.endswith("/c/security/exploits/6.json?page=1"):
            return {"topic_list": {"topics": []}}
        return None

    monkeypatch.setattr(scraper, "_get_json", fake_get_json)

    assert list(scraper.thread_urls("https://forum.example")) == [
        "https://forum.example/t/33.json",
    ]


def test_extract_redirect_targets_finds_js_redirects():
    html = '<script>window.location.href="/lander"</script>'
    assert extract_redirect_targets("https://forum.example/index.php", html) == [
        "https://forum.example/lander"
    ]


def test_xenforo_thread_urls_use_backup_discovery_html(monkeypatch):
    scraper = XenForoScraper()

    class _Resp:
        status = 200

        def __init__(self, html: str):
            self.body = html.encode()

        def css(self, selector):
            return []

    responses = {
        "https://forum.example": _Resp("<html><body><a href='/forums/general.1/'>General</a></body></html>"),
        "https://forum.example/whats-new/posts/": _Resp(
            "<html><body><a href='/threads/interesting.42/'>Interesting</a></body></html>"
        ),
    }

    def fake_get(url, **kwargs):
        return responses.get(url)

    monkeypatch.setattr(scraper, "get", fake_get)
    urls = list(scraper.thread_urls("https://forum.example"))
    assert "https://forum.example/threads/interesting.42/" in urls


def test_generic_thread_urls_follow_placeholder_redirect_and_sitemap(monkeypatch):
    scraper = GenericScraper()

    class _Resp:
        status = 200

        def __init__(self, html: str):
            self.body = html.encode()

        def css(self, selector):
            return []

    responses = {
        "https://forum.example": _Resp(
            '<html><head><script>window.location.href="/lander"</script></head></html>'
        ),
        "https://forum.example/lander": _Resp(
            '<html><body><a href="/sitemap.xml">Sitemap</a></body></html>'
        ),
        "https://forum.example/sitemap.xml": _Resp(
            '<urlset><url><loc>https://forum.example/topic/99-hidden-thread/</loc></url></urlset>'
        ),
    }

    def fake_get(url, **kwargs):
        return responses.get(url)

    monkeypatch.setattr(scraper, "get", fake_get)
    urls = list(scraper.thread_urls("https://forum.example"))
    assert "https://forum.example/topic/99-hidden-thread/" in urls


def test_generic_scrape_thread_preserves_paginated_thread_urls(monkeypatch):
    scraper = GenericScraper()
    entry = ForumEntry(name="Example", url="https://forum.example", status="ONLINE")

    class _El:
        def __init__(self, html: str = "", href: str = ""):
            self.html_content = html
            self.attrib = {"href": href} if href else {}

        def css(self, selector):
            return []

        def get_all_text(self):
            return self.html_content

    class _Resp:
        status = 200

        def __init__(self, body: str, next_href: str = ""):
            self.body = body.encode()
            self.next_href = next_href

        def css(self, selector):
            if selector == ".post-content":
                return [_El("<div>Exploit discussion with enough content to keep.</div>")]
            if selector == "a[rel='next']" and self.next_href:
                return [_El(href=self.next_href)]
            if selector in {"h1", "title"}:
                return [_El("Thread title")]
            return []

    responses = {
        "https://forum.example/topic/99-hidden-thread/": _Resp(
            "<html></html>", "/topic/99-hidden-thread/page/2/"
        ),
        "https://forum.example/topic/99-hidden-thread/page/2/": _Resp("<html></html>"),
    }

    monkeypatch.setattr(scraper, "get", lambda url, **kwargs: responses.get(url))

    posts = list(scraper.scrape_thread("https://forum.example/topic/99-hidden-thread/", entry))
    assert len(posts) == 2


def test_benchmark_timeout_marks_forum(monkeypatch, tmp_path):
    entry = ForumEntry(name="Slow", url="https://slow.example", status="ONLINE")

    def fake_benchmark_one(*args, **kwargs):
        time.sleep(0.2)
        return {}

    with (
        patch("etg.scraper.benchmark.online_entries", return_value=[entry]),
        patch("etg.scraper.benchmark._benchmark_one", side_effect=fake_benchmark_one),
    ):
        results = run_benchmark(
            markdown_output=tmp_path / "slow.md",
            json_output=tmp_path / "slow.json",
            per_forum_timeout=0.01,
        )

    assert results[0]["status"] == "timeout"


def test_invision_thread_urls_use_backup_discovery_html(monkeypatch):
    scraper = InvisionScraper()

    class _Resp:
        status = 200

        def __init__(self, html: str):
            self.body = html.encode()

        def css(self, selector):
            return []

    responses = {
        "https://forum.example": _Resp("<html><body><a href='/discover/'>Discover</a></body></html>"),
        "https://forum.example/discover/": _Resp(
            "<html><body><a href='/topic/123-interesting-thread/'>Interesting</a></body></html>"
        ),
    }

    def fake_get(url, **kwargs):
        return responses.get(url)

    monkeypatch.setattr(scraper, "get", fake_get)
    urls = list(scraper.thread_urls("https://forum.example"))
    assert "https://forum.example/topic/123-interesting-thread/" in urls


def test_phpbb_thread_urls_use_raw_href_fallback(monkeypatch):
    scraper = PhpBBScraper()

    class _Resp:
        status = 200

        def __init__(self, html: str):
            self.body = html.encode()

        def css(self, selector):
            return []

    responses = {
        "https://forum.example": _Resp(
            '<a href="/viewforum.php?f=2">Exploits</a>'
        ),
        "https://forum.example/viewforum.php?f=2": _Resp(
            '<a href="/viewtopic.php?f=2&t=123&start=25#p456">PoC</a>'
        ),
    }

    monkeypatch.setattr(scraper, "get", lambda url, **kwargs: responses.get(url))

    assert list(scraper.thread_urls("https://forum.example")) == [
        "https://forum.example/viewtopic.php?f=2&t=123",
    ]


def test_mybb_thread_urls_use_raw_href_fallback(monkeypatch):
    scraper = MyBBScraper()

    class _Resp:
        status = 200

        def __init__(self, html: str):
            self.body = html.encode()

        def css(self, selector):
            return []

    responses = {
        "https://forum.example": _Resp(
            '<a href="/forumdisplay.php?fid=7">Marketplace</a>'
        ),
        "https://forum.example/forumdisplay.php?fid=7": _Resp(
            '<a href="/showthread.php?tid=88&page=2#pid_9">Thread</a>'
        ),
    }

    monkeypatch.setattr(scraper, "get", lambda url, **kwargs: responses.get(url))

    assert list(scraper.thread_urls("https://forum.example")) == [
        "https://forum.example/showthread.php?tid=88",
    ]
