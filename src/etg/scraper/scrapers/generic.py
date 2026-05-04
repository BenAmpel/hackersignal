"""Generic / fallback scraper for unknown forum software.

Strategy:
  1. On the forum index page, collect links that look like thread listings
     (URLs containing /thread/, /topic/, /showthread, /viewtopic, etc.)
  2. On each thread page extract the largest contiguous text blocks as posts,
     using common content-area selectors and heuristics.
  3. Author and timestamp are best-effort: look for common patterns.

This scraper trades precision for coverage — it will collect *something* from
most forum-style pages even without knowing their exact software.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Generator
from urllib.parse import urlparse

from etg.scraper.models import ForumEntry, RawPost
from .base import (
    BaseScraper,
    abs_url,
    canonical_thread_url,
    extract_hrefs,
    extract_redirect_targets,
    is_placeholder_html,
    resp_html,
    strip_html,
    utcnow,
)

log = logging.getLogger(__name__)

# Regexes to identify thread-like URLs
_THREAD_URL_RE = re.compile(
    r"/(thread|threads|topic|topics|showthread|showtopic|viewtopic|post|discussion|t)/",
    re.I,
)
_THREAD_PARAM_RE = re.compile(r"[?&](tid|t|topic_id|thread_id|showtopic|p)=\d+", re.I)
_LISTING_HINT_RE = re.compile(
    r"(forum|board|category|categories|subforum|forumdisplay\.php|viewforum\.php|fid=|f=)",
    re.I,
)
_RECENT_HINT_RE = re.compile(r"(recent|latest|new|whats-new|feed|rss|sitemap)", re.I)

# Common content-area selectors (ordered by specificity)
_CONTENT_SELECTORS = [
    "[id^='post_message_']",
    "[id^='post-content-']",
    "[data-role='commentContent']",
    ".ipsType_richText", ".cPost_contentWrap",
    ".messageText", ".messageContent", ".message-content",
    ".bbWrapper", ".post_message", ".post_message_legacy",
    ".post-content", ".postcontent", ".post_content",
    ".message-content", ".message_body", ".post-body",
    ".postbody", ".post-text", ".post_text",
    ".entry-content", ".content", ".body",
    "article", "section.post",
]

# Common author selectors
_AUTHOR_SELECTORS = [
    ".bigusername", ".poster", ".poster_author",
    "[data-user-id] .username", "[data-member-id] a",
    ".username", ".author", ".post-author", ".postauthor",
    ".user-name", ".member", ".nick",
    "a[href*='/user/']", "a[href*='/member/']", "a[href*='/profile/']",
]

# Common timestamp selectors
_TIME_SELECTORS = [
    "[data-time]", "[data-timestamp]",
    "time[datetime]",
    ".post-date", ".postdate", ".post_date",
    ".date", ".timestamp", ".time",
    "abbr[title]",
]

_DATE_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?)"
    r"|\b(\d{1,2}[/ -]\d{1,2}[/ -]\d{2,4})"
    r"|\b(\d{1,2}\s+\w+\s+\d{4})"
)


def _extract_timestamp(div) -> datetime | None:
    """Try to extract a datetime from known selectors and text patterns."""
    for sel in _TIME_SELECTORS:
        els = div.css(sel)
        if not els:
            continue
        el = els[0]
        # <time datetime="...">
        dt_attr = el.attrib.get("datetime") or el.attrib.get("title", "")
        if dt_attr:
            try:
                return datetime.fromisoformat(dt_attr.replace("Z", "+00:00"))
            except ValueError:
                pass
        # Text content
        text = el.get_all_text().strip()
        m = _DATE_RE.search(text)
        if m:
            raw = (m.group(1) or m.group(2) or m.group(3)).strip()
            for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%d %H:%M", "%Y-%m-%d",
                        "%d/%m/%Y", "%m/%d/%Y",
                        "%d %B %Y", "%d %b %Y"]:
                try:
                    return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
    return None


def _pick_post_divs(resp) -> list:
    """Choose the selector that most plausibly represents repeated post bodies."""
    best: list = []
    best_score = -1
    for sel in _CONTENT_SELECTORS:
        els = list(resp.css(sel))
        if not els:
            continue
        scored = []
        for el in els:
            text = strip_html(el.html_content or "")
            if len(text) >= 20:
                scored.append(el)
        if not scored:
            continue
        score = min(len(scored), 8) * 100 + sum(
            min(len(strip_html(el.html_content or "")), 300) for el in scored[:8]
        )
        if score > best_score:
            best = scored
            best_score = score
    return best


def _next_listing_url(resp, current_url: str) -> str | None:
    selectors = [
        "a[rel='next']",
        ".pagination .next a",
        ".pageNav-jump--next",
        "li.next a",
        "a.pagination_next",
        ".ipsPagination_next a",
    ]
    for sel in selectors:
        els = resp.css(sel)
        if els:
            href = els[0].attrib.get("href", "")
            if href:
                return abs_url(current_url, href)
    return None


class GenericScraper(BaseScraper):
    """Best-effort scraper for unknown forum software."""

    def _backup_discovery_urls(self, forum_url: str) -> list[str]:
        return [
            abs_url(forum_url, "/sitemap.xml"),
            abs_url(forum_url, "/feed"),
            abs_url(forum_url, "/rss"),
            abs_url(forum_url, "/latest"),
            abs_url(forum_url, "/recent"),
            abs_url(forum_url, "/search.php?do=getnew"),
            abs_url(forum_url, "/index.php"),
        ]

    def thread_urls(self, forum_url: str) -> Generator[str, None, None]:
        visited: set[str] = set()
        base_host = urlparse(forum_url).netloc
        queue: list[tuple[str, int]] = [(forum_url, 0)]
        seen_threads: set[str] = set()
        listing_budget = max(self.max_list_pages * 4, 12)
        seeded_backups = False

        while queue and len(visited) < listing_budget:
            list_url, depth = queue.pop(0)
            if list_url in visited:
                continue
            visited.add(list_url)

            resp = self.get(list_url)
            if resp is None:
                continue
            html = resp_html(resp)

            if html:
                for redirect_url in extract_redirect_targets(list_url, html):
                    if urlparse(redirect_url).netloc == base_host and redirect_url not in visited:
                        queue.append((redirect_url, depth))

                if depth == 0 and (is_placeholder_html(html) or not resp.css("a[href]")):
                    for backup_url in self._backup_discovery_urls(forum_url):
                        if backup_url not in visited:
                            queue.append((backup_url, depth + 1))
                    seeded_backups = True

            for el in resp.css("a[href]"):
                href = el.attrib.get("href", "")
                full = abs_url(list_url, href)
                if urlparse(full).netloc != base_host:
                    continue

                if _THREAD_URL_RE.search(full) or _THREAD_PARAM_RE.search(full):
                    canonical = canonical_thread_url(full)
                    if canonical not in seen_threads:
                        seen_threads.add(canonical)
                        yield canonical
                    continue

                if depth >= 2 or full == forum_url or full in visited:
                    continue
                if _LISTING_HINT_RE.search(full):
                    queue.append((full, depth + 1))

            if html:
                for full in extract_hrefs(html, list_url):
                    if urlparse(full).netloc != base_host:
                        continue
                    if _THREAD_URL_RE.search(full) or _THREAD_PARAM_RE.search(full):
                        canonical = canonical_thread_url(full)
                        if canonical not in seen_threads:
                            seen_threads.add(canonical)
                            yield canonical
                    elif (
                        not seeded_backups
                        and depth <= 1
                        and _RECENT_HINT_RE.search(full)
                        and full not in visited
                    ):
                        queue.append((full, depth + 1))
                seeded_backups = True

            next_url = _next_listing_url(resp, list_url)
            if next_url and next_url not in visited:
                queue.append((next_url, depth))

    def scrape_thread(
        self, thread_url: str, entry: ForumEntry
    ) -> Generator[RawPost, None, None]:
        page_url: str | None = thread_url
        page_num = 0
        global_idx = 0
        visited_pages: set[str] = set()

        while page_url and page_num < self.max_thread_pages:
            page_key = page_url.split("#", 1)[0]
            if page_key in visited_pages:
                break
            visited_pages.add(page_key)

            resp = self.get(page_url)
            if resp is None:
                break

            # Thread title from <h1> or <title>
            h1s = resp.css("h1")
            thread_title = h1s[0].get_all_text().strip() if h1s else ""
            if not thread_title:
                titles = resp.css("title")
                thread_title = titles[0].get_all_text().strip() if titles else ""

            post_divs = _pick_post_divs(resp)
            if not post_divs:
                post_divs = list(resp.css("article") or resp.css("div.message"))

            if not post_divs and page_num == 0:
                body_els = resp.css("body")
                body = strip_html(body_els[0].html_content or "") if body_els else ""
                if body.strip():
                    yield RawPost(
                        forum_id=entry.forum_id,
                        thread_url=thread_url,
                        post_index=0,
                        raw_author="unknown",
                        body=body[:4000],
                        scraped_at=utcnow(),
                        thread_title=thread_title,
                    )
                return

            for div in post_divs:
                body = strip_html(div.html_content or "")
                if not body.strip() or len(body) < 20:
                    continue

                raw_author = "unknown"
                for sel in _AUTHOR_SELECTORS:
                    auth_els = div.css(sel)
                    if auth_els:
                        raw_author = auth_els[0].get_all_text().strip()
                        break

                timestamp = _extract_timestamp(div)

                yield RawPost(
                    forum_id=entry.forum_id,
                    thread_url=thread_url,
                    post_index=global_idx,
                    raw_author=raw_author,
                    body=body,
                    scraped_at=utcnow(),
                    thread_title=thread_title,
                    timestamp=timestamp,
                )
                global_idx += 1

            page_url = _next_listing_url(resp, page_url)
            page_num += 1
