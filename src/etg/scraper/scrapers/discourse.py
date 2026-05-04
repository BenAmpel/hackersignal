"""Discourse scraper — uses the public JSON API.

Endpoints used (no auth required for public forums):
  /latest.json?page=N     → paginated list of recent topics
  /c/{slug}/{id}.json     → category topic list
  /t/{id}.json            → full topic with all posts
  /t/{id}/{page}.json     → paginated posts within a topic

The JSON API is significantly more reliable than HTML scraping for Discourse.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Generator

from etg.scraper.models import ForumEntry, RawPost
from .base import BaseScraper, strip_html, utcnow

log = logging.getLogger(__name__)


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _base(url: str) -> str:
    """Return scheme://host from any URL."""
    from urllib.parse import urlparse
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


class DiscourseScraper(BaseScraper):
    """Scraper for Discourse-powered forums using their JSON API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._category_map: dict[int, str] | None = None

    def _get_json(self, url: str) -> dict | list | None:
        resp = self.get(url, headers={"Accept": "application/json"})
        if resp is None:
            return None
        try:
            return json.loads(resp.body.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, AttributeError):
            # Fallback: try resp.text
            try:
                text = resp.text if hasattr(resp, "text") else resp.text
                return json.loads(text)
            except Exception:
                return None

    def _load_category_map(self, base: str) -> dict[int, str]:
        if self._category_map is not None:
            return self._category_map

        mapping: dict[int, str] = {}
        data = self._get_json(f"{base}/categories.json")

        def visit_category(cat: dict, prefix: str = "") -> None:
            cid = cat.get("id")
            slug = cat.get("slug") or cat.get("name")
            if isinstance(cid, int) and slug:
                slug = str(slug)
                mapping[cid] = f"{prefix}/{slug}" if prefix else slug
                for child in cat.get("subcategory_list") or []:
                    if isinstance(child, dict):
                        visit_category(child, mapping[cid])

        if isinstance(data, dict):
            categories = data.get("category_list", {}).get("categories", [])
            for cat in categories:
                if isinstance(cat, dict):
                    visit_category(cat)
        self._category_map = mapping
        return mapping

    def thread_urls(self, forum_url: str) -> Generator[str, None, None]:
        """Yield synthetic 'API URLs' for topics (used in scrape_thread)."""
        base = _base(forum_url)
        seen: set[int] = set()

        def _yield_topics(data: dict | list | None) -> Generator[str, None, None]:
            if not data or not isinstance(data, dict):
                return
            topics = data.get("topic_list", {}).get("topics", [])
            for topic in topics:
                tid = topic.get("id")
                if tid and tid not in seen:
                    seen.add(tid)
                    yield f"{base}/t/{tid}.json"

        # Recent topics
        for page in range(self.max_list_pages):
            data = self._get_json(f"{base}/latest.json?page={page}")
            yielded = False
            for topic_url in _yield_topics(data):
                yielded = True
                yield topic_url
            if not yielded:
                break

        # Category timelines catch forums whose interesting content sits outside
        # the site-wide "latest" feed.
        category_map = self._load_category_map(base)
        for cid, slug in category_map.items():
            for page in range(self.max_list_pages):
                url = f"{base}/c/{slug}/{cid}.json"
                if page:
                    url = f"{url}?page={page}"
                data = self._get_json(url)
                yielded = False
                for topic_url in _yield_topics(data):
                    yielded = True
                    yield topic_url
                if not yielded:
                    break

    def scrape_thread(
        self, thread_url: str, entry: ForumEntry
    ) -> Generator[RawPost, None, None]:
        """thread_url is actually a /t/{id}.json API URL."""
        base = _base(thread_url)
        # Extract numeric topic id from the API URL
        tid_m = re.search(r"/t/(\d+)\.json", thread_url)
        if not tid_m:
            return
        tid = tid_m.group(1)

        # Fetch first chunk to get total post count
        data = self._get_json(thread_url)
        if not data or not isinstance(data, dict):
            return

        thread_title = data.get("title", "")
        category_id = data.get("category_id", "")
        reply_count = data.get("posts_count", 0) - 1
        view_count = data.get("views", 0)
        tags: list[str] = data.get("tags") or []
        slug = data.get("slug") or tid
        category_map = self._load_category_map(base)
        section = category_map.get(category_id, str(category_id))

        # HTML canonical URL for the thread
        canonical = f"{base}/t/{slug}/{tid}"

        def _posts_from_chunk(chunk: dict) -> Generator[RawPost, None, None]:
            stream = chunk.get("post_stream", {})
            posts = stream.get("posts", [])
            for p in posts:
                post_number = p.get("post_number", 0)
                raw_author = p.get("username", "unknown")
                cooked = p.get("cooked", "")   # HTML body
                body = strip_html(cooked)
                if not body.strip():
                    continue

                ts = _parse_iso(p.get("created_at"))

                yield RawPost(
                    forum_id=entry.forum_id,
                    thread_url=canonical,
                    post_index=post_number - 1,
                    raw_author=raw_author,
                        body=body,
                        scraped_at=utcnow(),
                        thread_title=thread_title,
                        section=section,
                    post_id=str(p.get("id", "")),
                    thread_id=tid,
                    reply_count=reply_count,
                    view_count=view_count,
                    tags=tags,
                    timestamp=ts,
                )

        yield from _posts_from_chunk(data)

        # Discourse paginates via /t/{id}/{page}.json (page >= 2)
        total_pages = data.get("post_stream", {}).get("stream", [])
        posts_per_page = 20
        if len(total_pages) > posts_per_page:
            max_pg = min(
                (len(total_pages) // posts_per_page) + 1,
                self.max_thread_pages,
            )
            for pg in range(2, max_pg + 1):
                chunk = self._get_json(f"{base}/t/{tid}/{pg}.json")
                if chunk:
                    yield from _posts_from_chunk(chunk)
