"""Unit tests for etg.scraper — no network calls."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from etg.scraper.detector import ForumKind, detect
from etg.scraper.benchmark import run_benchmark
from etg.scraper.forum_index import _dedupe_entries, _parse_markdown
from etg.scraper.models import ForumEntry, RawPost
from etg.scraper.scrapers.base import BaseScraper, strip_html, resp_html
from etg.scraper.scrapers.invision import InvisionScraper
from etg.scraper.scrapers.mybb import _breadcrumb_section, _extract_post_body, _parse_mybb_date
from etg.scraper.pipeline import _load_forum_profiles, _resolve_crawl_options, _resolve_entry_runtime


# ---------------------------------------------------------------------------
# ForumEntry
# ---------------------------------------------------------------------------


def test_forum_entry_basic():
    e = ForumEntry(name="BHF", url="https://bhf.pro", status="ONLINE")
    assert e.is_online
    assert not e.is_onion
    assert e.forum_id == "bhf"


def test_forum_entry_offline():
    e = ForumEntry(name="OLD FORUM", url="https://old.io", status="OFFLINE")
    assert not e.is_online


def test_forum_entry_onion():
    e = ForumEntry(
        name="DARK",
        url="http://abc123abc123abc123abc123abc123abc123abc123abc123abc123abc1234.onion",
        status="ONLINE",
    )
    assert e.is_onion
    assert e.is_online


def test_forum_entry_forum_id_normalisation():
    e = ForumEntry(name="BREACH FORUMS (Deep)", url="https://x.com", status="ONLINE")
    fid = e.forum_id
    # Must be lowercase ASCII slug with underscores only
    assert fid == "breach_forums_deep"


# ---------------------------------------------------------------------------
# forum.md parser
# ---------------------------------------------------------------------------

_SAMPLE_MD = """\
| Name | Status | Description |
| ------ | ------ | ------ |
|[BHF](https://bhf.pro)| ONLINE | |
|[BREACHED (Dark)](http://breached65.onion)| OFFLINE | |
|[CRACKING](https://cracking.org)| ONLINE | |
|[ALPHV Forum](https://alphv.pro) | ONLINE | |
"""


def test_parse_markdown_count():
    entries = _parse_markdown(_SAMPLE_MD)
    assert len(entries) == 4


def test_parse_markdown_fields():
    entries = _parse_markdown(_SAMPLE_MD)
    bhf = next(e for e in entries if e.name == "BHF")
    assert bhf.url == "https://bhf.pro"
    assert bhf.status == "ONLINE"
    assert bhf.is_online
    assert not bhf.is_onion


def test_parse_markdown_onion():
    entries = _parse_markdown(_SAMPLE_MD)
    breached = next(e for e in entries if "BREACHED" in e.name)
    assert breached.is_onion
    assert not breached.is_online


def test_online_entries_dedupes_same_community_by_host_or_id():
    entries = [
        ForumEntry(name="HackForums", url="https://hackforums.net/index.php", status="ONLINE"),
        ForumEntry(name="HackForums", url="https://hackforums.net", status="ONLINE"),
        ForumEntry(name="0x00sec", url="https://www.0x00sec.org/", status="ONLINE"),
        ForumEntry(name="0x00sec Mirror", url="https://0x00sec.org", status="ONLINE"),
        ForumEntry(name="Unique", url="https://unique.example", status="ONLINE"),
    ]
    deduped = _dedupe_entries(entries)
    assert [e.name for e in deduped] == ["HackForums", "0x00sec", "Unique"]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def test_detect_phpbb_from_url():
    assert detect("https://example.com/viewtopic.php?t=1") == ForumKind.PHPBB
    assert detect("https://example.com/viewforum.php?f=2") == ForumKind.PHPBB


def test_detect_mybb_from_url():
    assert detect("https://example.com/showthread.php?tid=1") == ForumKind.MYBB
    assert detect("https://example.com/forumdisplay.php?fid=5") == ForumKind.MYBB


def test_detect_xenforo_from_html():
    html = '<html><meta name="generator" content="XenForo 2.2.14"></html>'
    assert detect("https://bhf.pro/", html) == ForumKind.XENFORO


def test_detect_phpbb_from_html():
    html = '<html><meta name="generator" content="phpBB 3.3.11"></html>'
    assert detect("https://example.com/", html) == ForumKind.PHPBB


def test_detect_mybb_from_html():
    html = "<html><script>var MyBB = {}</script></html>"
    assert detect("https://example.com/", html) == ForumKind.MYBB


def test_detect_discourse_from_html():
    html = "<html><script>Discourse.SiteSettings = {}</script></html>"
    assert detect("https://forum.example.com/", html) == ForumKind.DISCOURSE


def test_detect_invision_from_html():
    html = "<html><script>ips.setSetting('ipb_url_filter_option', 'none')</script></html>"
    assert detect("https://forum.example.com/", html) == ForumKind.INVISION


def test_detect_unknown():
    assert detect("https://weirdcustom.net/") == ForumKind.UNKNOWN
    assert detect("https://weirdcustom.net/", "<html><body>Hi</body></html>") == ForumKind.UNKNOWN


def test_parse_mybb_date_hackforums_format():
    ts = _parse_mybb_date("Dec 9, 2025 12:33 AM (This post was last modified: Dec 9, 2025 12:34 AM by user.)")
    assert ts == datetime(2025, 12, 9, 0, 33, tzinfo=timezone.utc)


def test_mybb_breadcrumb_section_ignores_empty_icon_links():
    resp = MagicMock()
    els = []
    for text, href in [
        ("", "https://hackforums.net/index.php"),
        ("Hack Forums", "https://hackforums.net/index.php"),
        ("", "forumdisplay.php?fid=105"),
        ("Marketplace", "forumdisplay.php?fid=105"),
        ("", "forumdisplay.php?fid=450"),
        ("Bazaar", "forumdisplay.php?fid=450"),
        ("", "forumdisplay.php?fid=402"),
        ("Promotional Advertising", "forumdisplay.php?fid=402"),
    ]:
        el = MagicMock()
        el.get_all_text.return_value = text
        el.attrib = {"href": href}
        els.append(el)
    resp.css.return_value = els
    assert _breadcrumb_section(resp) == "Bazaar / Promotional Advertising"


def test_mybb_extract_post_body_skips_antibot_body_fallback():
    div = MagicMock()
    div.css.return_value = []
    resp = MagicMock()
    body_el = MagicMock()
    body_el.html_content = "<body>Just a moment... verify you are human</body>"
    resp.css.side_effect = lambda sel: [body_el] if sel == "body" else []
    assert _extract_post_body(div, resp) == ""


def test_mybb_extract_post_body_uses_generic_content_fallback():
    div = MagicMock()
    div.css.return_value = []
    resp = MagicMock()
    content_el = MagicMock()
    content_el.html_content = "<div id='content'><p>LeakRadar helps identify stolen credentials quickly.</p></div>"
    resp.css.side_effect = lambda sel: [content_el] if sel == "#content" else []
    assert "LeakRadar helps identify stolen credentials" in _extract_post_body(div, resp)


class _FakeScraper(BaseScraper):
    def __init__(self):
        super().__init__()
        self.min_body_len = 1

    def thread_urls(self, forum_url):
        yield "https://forum.example/t1"
        yield "https://forum.example/t2"
        yield "https://forum.example/t3"

    def scrape_thread(self, thread_url, entry):
        if thread_url.endswith("t2"):
            return
        yield RawPost(
            forum_id=entry.forum_id,
            thread_url=thread_url,
            post_index=0,
            raw_author="alice",
            body=f"body for {thread_url}",
            scraped_at=_NOW,
            thread_title=thread_url.rsplit("/", 1)[-1],
        )


def test_scrape_forum_max_thread_urls_caps_scan_budget():
    scraper = _FakeScraper()
    entry = ForumEntry(name="Example", url="https://forum.example", status="ONLINE")
    posts = list(scraper.scrape_forum(entry, max_thread_urls=2))
    assert len(posts) == 1
    assert posts[0].thread_url.endswith("t1")


def test_scrape_forum_max_posts_per_thread_caps_giant_threads():
    class _ManyPostsScraper(_FakeScraper):
        def scrape_thread(self, thread_url, entry):
            for idx in range(5):
                yield RawPost(
                    forum_id=entry.forum_id,
                    thread_url=thread_url,
                    post_index=idx,
                    raw_author="alice",
                    body=f"body {idx}",
                    scraped_at=_NOW,
                    thread_title="many",
                )

    scraper = _ManyPostsScraper()
    entry = ForumEntry(name="Example", url="https://forum.example", status="ONLINE")
    posts = list(scraper.scrape_forum(entry, max_thread_urls=1, max_posts_per_thread=2))
    assert len(posts) == 2
    assert [post.post_index for post in posts] == [0, 1]


def test_resolve_crawl_options_broad_defaults_to_per_thread_cap():
    assert _resolve_crawl_options(
        crawl_mode="broad",
        max_posts_per_thread=None,
        max_thread_pages=None,
    ) == {
        "max_posts_per_thread": 25,
        "max_thread_pages": None,
    }


def test_resolve_crawl_options_explicit_override_beats_mode_default():
    assert _resolve_crawl_options(
        crawl_mode="broad",
        max_posts_per_thread=7,
        max_thread_pages=None,
    ) == {
        "max_posts_per_thread": 7,
        "max_thread_pages": None,
    }


def test_resolve_crawl_options_supports_thread_page_cap():
    assert _resolve_crawl_options(
        crawl_mode="deep",
        max_posts_per_thread=None,
        max_thread_pages=3,
    ) == {
        "max_posts_per_thread": None,
        "max_thread_pages": 3,
    }


def test_load_forum_profiles_from_json_file(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"profiles": [{"forums": ["blackhatworld"], "proxy": "http://p1"}]}))
    profiles = _load_forum_profiles(path)
    assert profiles == [{"forums": ["blackhatworld"], "proxy": "http://p1"}]


def test_resolve_entry_runtime_applies_profile_override():
    entry = ForumEntry(name="BLACKHATWORLD", url="https://blackhatworld.com", status="ONLINE")
    runtime = _resolve_entry_runtime(
        entry,
        default_proxy=None,
        default_cookies=None,
        profiles=[{
            "forums": ["blackhatworld"],
            "proxy": "http://proxy.example:8080",
            "max_posts_per_thread": 11,
            "max_thread_pages": 2,
        }],
        crawl_mode="broad",
        max_posts_per_thread=25,
        max_thread_pages=5,
    )
    assert runtime["proxy"] == "http://proxy.example:8080"
    assert runtime["max_posts_per_thread"] == 11
    assert runtime["max_thread_pages"] == 2


def test_base_get_seeds_homepage_once_before_page_fetch():
    scraper = _FakeScraper()
    calls = []

    def fake_fetch(url, **kw):
        calls.append((url, kw))
        resp = MagicMock()
        resp.status = 200
        return resp

    scraper._fetch_once = fake_fetch
    scraper.get("https://forum.example/showthread.php?tid=1")
    scraper.get("https://forum.example/showthread.php?tid=2")

    assert calls[0][0] == "https://forum.example/"
    assert calls[1][0] == "https://forum.example/showthread.php?tid=1"
    assert calls[2][0] == "https://forum.example/showthread.php?tid=2"
    assert calls[1][1]["headers"]["Referer"] == "https://forum.example/"


def test_invision_thread_urls_prioritise_topics_before_deep_subforums():
    scraper = InvisionScraper()

    def make_link(href: str):
        el = MagicMock()
        el.attrib = {"href": href}
        return el

    root = MagicMock()
    root.css.side_effect = lambda sel: (
        [
            make_link("/forum/10-security/"),
            make_link("/topic/123-interesting-thread/"),
        ] if sel == "a[href]" else []
    )
    subforum = MagicMock()
    subforum.css.side_effect = lambda sel: (
        [make_link("/topic/456-deeper-thread/")] if sel == "a[href]" else []
    )

    responses = {
        "https://bits.example/forum/": root,
        "https://bits.example/forum/10-security/": subforum,
    }
    scraper.get = lambda url, **kwargs: responses.get(url)

    urls = list(scraper.thread_urls("https://bits.example/forum/"))
    assert urls[0] == "https://bits.example/topic/123-interesting-thread/"
    assert "https://bits.example/topic/456-deeper-thread/" in urls


class _BenchmarkScraper:
    min_body_len = 1

    def __init__(self):
        self._dead = False

    def thread_urls(self, forum_url):
        if "alpha" in forum_url:
            yield "https://alpha.example/thread/1"
            yield "https://alpha.example/thread/2"
        else:
            if False:
                yield ""

    def scrape_thread(self, thread_url, entry):
        yield RawPost(
            forum_id=entry.forum_id,
            thread_url=thread_url,
            post_index=0,
            raw_author="alice",
            body=f"body for {thread_url}",
            scraped_at=_NOW,
            thread_title=entry.name,
            section="market" if "alpha" in thread_url else "",
        )


def test_run_benchmark_writes_reports(tmp_path):
    entries = [
        ForumEntry(name="Alpha", url="https://alpha.example", status="ONLINE"),
        ForumEntry(name="Beta", url="https://beta.example", status="ONLINE"),
    ]

    with (
        patch("etg.scraper.benchmark.online_entries", return_value=entries),
        patch("etg.scraper.benchmark._detect_kind", return_value=ForumKind.MYBB),
        patch("etg.scraper.benchmark._make_scraper", return_value=_BenchmarkScraper()),
    ):
        md_path = tmp_path / "benchmark.md"
        json_path = tmp_path / "benchmark.json"
        results = run_benchmark(
            markdown_output=md_path,
            json_output=json_path,
            max_thread_urls=2,
            max_posts=5,
        )

    assert len(results) == 2
    assert md_path.exists()
    assert json_path.exists()
    assert "Alpha" in md_path.read_text()
    assert '"forum_id": "alpha"' in json_path.read_text()


# ---------------------------------------------------------------------------
# RawPost
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def _make_raw(post_index=0, **kwargs) -> RawPost:
    defaults = dict(
        forum_id="bhf_pro",
        thread_url="https://bhf.pro/threads/test-exploit.12345/",
        post_index=post_index,
        raw_author="h4cker_one",
        body="Here is a working PoC for CVE-2024-1234 on Windows.",
        scraped_at=_NOW,
        thread_title="New zero-day dropped" if post_index == 0 else "",
        thread_id="12345",
        reply_count=42,
        view_count=1337,
    )
    defaults.update(kwargs)
    return RawPost(**defaults)


def test_rawpost_unique_id_stable():
    r1 = _make_raw()
    r2 = _make_raw()
    assert r1.unique_id == r2.unique_id


def test_rawpost_unique_id_differs_by_index():
    r0 = _make_raw(post_index=0)
    r1 = _make_raw(post_index=1)
    assert r0.unique_id != r1.unique_id


def test_rawpost_author_hash_anonymised():
    r = _make_raw()
    assert r.author_hash != "h4cker_one"
    expected = hashlib.sha256(b"bhf_pro::h4cker_one").hexdigest()
    assert r.author_hash == expected


def test_rawpost_author_hash_forum_scoped():
    """Same username on different forums → different hash."""
    r_bhf = _make_raw(forum_id="bhf_pro")
    r_bwf = _make_raw(forum_id="breachforums")
    assert r_bhf.author_hash != r_bwf.author_hash


def test_rawpost_cve_extraction():
    r = _make_raw(body="PoC for CVE-2024-1234 and cve-2023-9999 and random text.")
    cves = r.cve_refs
    assert "CVE-2024-1234" in cves
    assert "CVE-2023-9999" in cves


def test_rawpost_cve_in_title():
    r = _make_raw(thread_title="Exploit for CVE-2025-0001", body="no cve here")
    assert "CVE-2025-0001" in r.cve_refs


def test_rawpost_full_text_opening_post():
    r = _make_raw(post_index=0, thread_title="Thread Title", body="Body text.")
    assert r.full_text() == "Thread Title\n\nBody text."


def test_rawpost_full_text_reply():
    r = _make_raw(post_index=1, thread_title="Thread Title", body="Reply body.")
    assert r.full_text() == "Reply body."


def test_rawpost_timestamp_fallback():
    r = _make_raw(timestamp=None)
    # Falls back to scraped_at
    assert r.effective_timestamp == _NOW


def test_rawpost_timestamp_explicit():
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    r = _make_raw(timestamp=ts)
    assert r.effective_timestamp == ts


def test_rawpost_to_forum_post():
    r = _make_raw()
    fp = r.to_forum_post()
    assert fp.id == r.unique_id
    assert fp.forum_id == r.forum_id
    assert fp.author_hash == r.author_hash
    assert fp.timestamp == r.effective_timestamp
    assert fp.text == r.full_text()


def test_rawpost_to_forum_post_schema_roundtrip():
    from etg.data.schemas import ForumPost
    r = _make_raw()
    fp = r.to_forum_post()
    d = fp.to_dict()
    fp2 = ForumPost.from_dict(d)
    assert fp2.id == fp.id
    assert fp2.forum_id == fp.forum_id
    assert fp2.timestamp.tzinfo is not None


def test_rawpost_to_dict_extended_fields():
    r = _make_raw(section="Exploits", tags=["zero-day", "RCE"])
    d = r.to_dict()
    assert d["section"] == "Exploits"
    assert d["tags"] == ["zero-day", "RCE"]
    assert "CVE-2024-1234" in d["cve_refs"]
    assert "raw_author" not in d   # must never leak raw username


# ---------------------------------------------------------------------------
# strip_html
# ---------------------------------------------------------------------------


def test_strip_html_basic():
    assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"


def test_strip_html_entities():
    # &lt;p&gt; decodes to <p>, which is then stripped as a tag — expected.
    assert strip_html("&amp; &lt;p&gt; &quot;test&quot;") == '& "test"'
    assert strip_html("&nbsp;hello&nbsp;") == "hello"


def test_strip_html_collapses_whitespace():
    result = strip_html("<p>  lots   of   space  </p>")
    assert "  " not in result


def test_strip_html_empty():
    assert strip_html("") == ""


# ---------------------------------------------------------------------------
# resp_html
# ---------------------------------------------------------------------------


def test_resp_html_from_bytes():
    mock = MagicMock()
    mock.body = b"<html>hello</html>"
    mock.encoding = "utf-8"
    assert resp_html(mock) == "<html>hello</html>"


def test_resp_html_from_text_fallback():
    mock = MagicMock()
    mock.body = b""
    mock.text = "<html>fallback</html>"
    assert resp_html(mock) == "<html>fallback</html>"


def test_resp_html_none():
    assert resp_html(None) == ""


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def test_pipeline_scraper_factory():
    from etg.scraper.pipeline import _make_scraper
    from etg.scraper.scrapers import (
        XenForoScraper, PhpBBScraper, MyBBScraper,
        DiscourseScraper, InvisionScraper, GenericScraper,
    )
    mapping = {
        ForumKind.XENFORO: XenForoScraper,
        ForumKind.PHPBB: PhpBBScraper,
        ForumKind.MYBB: MyBBScraper,
        ForumKind.DISCOURSE: DiscourseScraper,
        ForumKind.INVISION: InvisionScraper,
        ForumKind.UNKNOWN: GenericScraper,
    }
    for kind, expected_cls in mapping.items():
        s = _make_scraper(kind)
        assert isinstance(s, expected_cls), f"{kind} → expected {expected_cls.__name__}"


def test_pipeline_scraper_factory_params():
    from etg.scraper.pipeline import _make_scraper
    s = _make_scraper(ForumKind.XENFORO, max_threads=50, max_list_pages=3, request_delay=2.0)
    assert s.max_threads == 50
    assert s.max_list_pages == 3
    assert s.request_delay == 2.0
