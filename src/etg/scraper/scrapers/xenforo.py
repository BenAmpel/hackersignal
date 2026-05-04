"""XenForo scraper (v1 and v2).

Forum structure:
  /forums/                        -> board index (list of sub-forums)
  /forums/{name}.{id}/            -> thread list, paginated via ?page=N
  /threads/{name}.{id}/           -> thread page, paginated via page-N/ suffix
  /threads/{name}.{id}/page-2/

Key selectors:
  Thread list   .structItem--thread .structItem-title a[data-preview-url]
  Posts         article.message
  Post body     .message-body .bbWrapper
  Author        a.username[data-user-id]
  Timestamp     time[datetime]
  Thread title  h1.p-title-value
  View count    .pairs--justified dd (label contains "Views")
  Reply count   same, label contains "Replies"
  Tags          .tagList a.tagItem
  Thread ID     URL pattern: /threads/name.DIGITS/
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Generator
from urllib.parse import urljoin, urlparse

from etg.scraper.models import ForumEntry, RawPost
from .base import (
    BaseScraper,
    abs_url,
    canonical_thread_url,
    extract_hrefs,
    extract_redirect_targets,
    resp_html,
    strip_html,
    utcnow,
)

log = logging.getLogger(__name__)

_THREAD_ID_RE = re.compile(r"/threads/[^/]+\.(\d+)")
_POST_ID_RE = re.compile(r"post-(\d+)")
_PAGE_RE = re.compile(r"<link[^>]+rel=['\"]next['\"][^>]+href=['\"]([^'\"]+)['\"]")
_THREAD_HINT_RE = re.compile(r"/(threads|posts|goto/post)\b|[?&](threads?|t)=\d+", re.I)


class XenForoScraper(BaseScraper):
    """Scraper for XenForo 1.x / 2.x installations."""

    def _backup_discovery_urls(self, forum_url: str) -> list[str]:
        return [
            abs_url(forum_url, "/whats-new/posts/"),
            abs_url(forum_url, "/whats-new/"),
            abs_url(forum_url, "/recent-threads/"),
            abs_url(forum_url, "/find-new/posts"),
            abs_url(forum_url, "/forums/-/index.rss"),
            abs_url(forum_url, "/sitemap.xml"),
        ]

    def _yield_thread_links_from_html(self, html: str, current_url: str) -> Generator[str, None, None]:
        for full in extract_hrefs(html, current_url):
            if _THREAD_HINT_RE.search(full):
                yield canonical_thread_url(full)

    def thread_urls(self, forum_url: str) -> Generator[str, None, None]:
        """Collect thread URLs from the forum index, following sub-forums."""
        visited_lists: set[str] = set()
        queue = [forum_url]
        seen_threads: set[str] = set()
        seeded_backups = False

        while queue:
            list_url = queue.pop(0)
            if list_url in visited_lists:
                continue
            visited_lists.add(list_url)

            resp = self.get(list_url)
            if resp is None:
                continue

            html = resp_html(resp)
            if html:
                for redirect_url in extract_redirect_targets(list_url, html):
                    if redirect_url not in visited_lists:
                        queue.append(redirect_url)

            # Yield thread links from this page
            thread_count = 0
            for el in resp.css(".structItem--thread .structItem-title a"):
                href = el.attrib.get("href", "")
                if "/threads/" in href:
                    canonical = canonical_thread_url(abs_url(list_url, href))
                    if canonical not in seen_threads:
                        seen_threads.add(canonical)
                        yield canonical
                        thread_count += 1

            if thread_count == 0 and html:
                for canonical in self._yield_thread_links_from_html(html, list_url):
                    if canonical not in seen_threads:
                        seen_threads.add(canonical)
                        yield canonical
                        thread_count += 1

            # Queue sub-forum links (prefer shallow traversal, but allow one nested hop)
            if list_url == forum_url or "/forums/" in list_url:
                for el in resp.css(".node--forum a.node-title"):
                    href = el.attrib.get("href", "")
                    if href:
                        queue.append(abs_url(list_url, href))
                if list_url == forum_url and thread_count == 0 and not seeded_backups:
                    for backup_url in self._backup_discovery_urls(forum_url):
                        if backup_url not in visited_lists:
                            queue.append(backup_url)
                    seeded_backups = True
            elif not seeded_backups:
                for backup_url in self._backup_discovery_urls(forum_url):
                    if backup_url not in visited_lists:
                        queue.append(backup_url)
                seeded_backups = True

            # Follow pagination on thread-list pages
            pages_followed = 0
            while pages_followed < self.max_list_pages - 1:
                next_el = resp.css(".pageNav-jump--next")
                if not next_el:
                    break
                next_href = next_el[0].attrib.get("href", "")
                if not next_href:
                    break
                next_url = abs_url(list_url, next_href)
                if next_url in visited_lists:
                    break
                visited_lists.add(next_url)
                resp = self.get(next_url)
                if resp is None:
                    break
                html = resp_html(resp)
                for el in resp.css(".structItem--thread .structItem-title a"):
                    href = el.attrib.get("href", "")
                    if "/threads/" in href:
                        canonical = canonical_thread_url(abs_url(next_url, href))
                        if canonical not in seen_threads:
                            seen_threads.add(canonical)
                            yield canonical
                if html:
                    for canonical in self._yield_thread_links_from_html(html, next_url):
                        if canonical not in seen_threads:
                            seen_threads.add(canonical)
                            yield canonical
                pages_followed += 1

    def scrape_thread(
        self, thread_url: str, entry: ForumEntry
    ) -> Generator[RawPost, None, None]:
        """Yield one RawPost per post across all pages of a thread."""
        page_url: str | None = thread_url
        page_num = 0

        while page_url and page_num < self.max_thread_pages:
            resp = self.get(page_url)
            if resp is None:
                break

            # Thread-level metadata (re-read each page in case first page
            # has the canonical values)
            title_els = resp.css("h1.p-title-value")
            thread_title = title_els[0].get_all_text().strip() if title_els else ""

            # Extract thread/section metadata once
            section = ""
            crumb_els = resp.css(".p-breadcrumbs li a")
            if len(crumb_els) >= 2:
                section = crumb_els[-1].get_all_text().strip()

            tid_m = _THREAD_ID_RE.search(page_url)
            thread_id = tid_m.group(1) if tid_m else ""

            reply_str = ""
            view_str = ""
            for pair in resp.css(".pairs--justified"):
                label = (pair.css("dt") or [None])[0]
                value = (pair.css("dd") or [None])[0]
                if label and value:
                    ltext = label.get_all_text().strip().lower()
                    vtext = value.get_all_text().strip().replace(",", "")
                    if "repl" in ltext:
                        reply_str = vtext
                    elif "view" in ltext:
                        view_str = vtext

            tags: list[str] = [
                t.get_all_text().strip()
                for t in resp.css(".tagList .tagItem")
            ]

            # Per-post elements
            for idx, article in enumerate(resp.css("article.message")):
                post_id = ""
                pid_m = _POST_ID_RE.search(article.attrib.get("id", ""))
                if pid_m:
                    post_id = pid_m.group(1)

                author_els = article.css("a.username")
                raw_author = (
                    author_els[0].get_all_text().strip()
                    if author_els else "unknown"
                )

                time_els = article.css("time[datetime]")
                timestamp: datetime | None = None
                if time_els:
                    try:
                        timestamp = datetime.fromisoformat(
                            time_els[0].attrib["datetime"].replace("Z", "+00:00")
                        )
                    except (ValueError, KeyError):
                        pass

                body_els = article.css(".message-body .bbWrapper")
                if not body_els:
                    body_els = article.css(".message-body")
                body = strip_html(body_els[0].html_content or "") if body_els else ""

                if not body.strip():
                    continue

                # global post index across all pages
                global_post_idx = page_num * 20 + idx

                yield RawPost(
                    forum_id=entry.forum_id,
                    thread_url=thread_url,
                    post_index=global_post_idx,
                    raw_author=raw_author,
                    body=body,
                    scraped_at=utcnow(),
                    thread_title=thread_title,
                    section=section,
                    post_id=post_id,
                    thread_id=thread_id,
                    reply_count=int(reply_str) if reply_str.isdigit() else 0,
                    view_count=int(view_str) if view_str.isdigit() else 0,
                    tags=tags,
                    timestamp=timestamp,
                )

            # Pagination within thread
            next_els = resp.css(".pageNav-jump--next")
            if not next_els:
                break
            next_href = next_els[0].attrib.get("href", "")
            if not next_href:
                break
            page_url = abs_url(page_url, next_href)
            page_num += 1
