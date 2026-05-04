"""Invision Community (IPB / IPS) scraper.

URL patterns:
  Forum index  /forums/
  Thread list  /forums/{name}-{id}/   or  /forum/{id}-{name}/
  Thread       /topic/{id}-{name}/    or  ?topic={id}.0
  Next page    /topic/{id}-{name}/page/2/  or  ?topic={id}.{start}

Key selectors:
  Thread list  h4.ipsType_sectionHead a, .ipsDataItem_title a
  Posts        article[data-commentid], .ipsComment_content
  Post body    .ipsType_richText, .cPost_contentWrap
  Author       [data-member-id] a.ipsType_break, strong.ipsType_break
  Timestamp    time[datetime]
  Thread title h1 span.ipsType_break, h1.ipsType_pageTitle
  View count   span[data-role="replyCount"] sibling
"""

from __future__ import annotations

import logging
import re
from collections import deque
from datetime import datetime, timezone
from typing import Generator
from urllib.parse import urljoin

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

_TID_RE = re.compile(r"[?/]topic/(\d+)", re.I)
_CID_RE = re.compile(r'data-commentid=["\'](\d+)["\']')
# IPS uses index.php?/forum/ID-name/ or /forums/ID-name/ or /topic/ID-name/
_FORUM_URL_RE = re.compile(r"\?/forum/\d+|/forums?/\d+|/forum/\d+", re.I)
_TOPIC_URL_RE = re.compile(r"\?/topic/\d+|/topics?/\d+|/topic/\d+", re.I)


class InvisionScraper(BaseScraper):
    """Scraper for Invision Community (IPBoard) forums."""

    #: Maximum number of sub-forum listing pages to enqueue — prevents a
    #: single large site from consuming the entire run budget.
    max_subforums: int = 500
    #: Maximum depth of forum/category traversal from the entry page.
    max_subforum_depth: int = 2

    def _backup_discovery_urls(self, forum_url: str) -> list[str]:
        return [
            abs_url(forum_url, "/discover/"),
            abs_url(forum_url, "/discover/unread/"),
            abs_url(forum_url, "/index.php?/discover/"),
            abs_url(forum_url, "/index.php?/discover/unread/"),
            abs_url(forum_url, "/search/"),
            abs_url(forum_url, "/sitemap.xml"),
            abs_url(forum_url, "/rss/"),
            abs_url(forum_url, "/feed/"),
        ]

    def _yield_topic_links_from_html(self, html: str, current_url: str) -> Generator[str, None, None]:
        for full in extract_hrefs(html, current_url):
            if _TOPIC_URL_RE.search(full):
                yield canonical_thread_url(full)

    def thread_urls(self, forum_url: str) -> Generator[str, None, None]:
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(forum_url, 0)])
        subforum_count = 0
        seen_threads: set[str] = set()
        seeded_backups = False

        while queue:
            list_url, depth = queue.popleft()
            if list_url in visited:
                continue
            visited.add(list_url)

            resp = self.get(list_url)
            if resp is None:
                continue
            html = resp_html(resp)
            if html:
                for redirect_url in extract_redirect_targets(list_url, html):
                    if redirect_url not in visited:
                        queue.appendleft((redirect_url, depth))

            topic_urls: list[str] = []
            subforum_urls: list[str] = []
            for el in resp.css("a[href]"):
                href = el.attrib.get("href", "")
                if not href:
                    continue
                full = abs_url(list_url, href)
                if _TOPIC_URL_RE.search(href):
                    topic_urls.append(canonical_thread_url(full))
                elif (
                    _FORUM_URL_RE.search(href)
                    and full not in visited
                    and depth < self.max_subforum_depth
                    and subforum_count < self.max_subforums
                ):
                    subforum_urls.append(full)

            for topic_url in topic_urls:
                if topic_url not in seen_threads:
                    seen_threads.add(topic_url)
                    yield topic_url

            if not topic_urls and html:
                for topic_url in self._yield_topic_links_from_html(html, list_url):
                    if topic_url not in seen_threads:
                        seen_threads.add(topic_url)
                        yield topic_url

            for subforum_url in subforum_urls:
                queue.append((subforum_url, depth + 1))
                subforum_count += 1

            if depth == 0 and not seeded_backups and not topic_urls:
                for backup_url in self._backup_discovery_urls(forum_url):
                    if backup_url not in visited:
                        queue.append((backup_url, 1))
                seeded_backups = True

            # Pagination on the listing page
            pages = 0
            while pages < self.max_list_pages - 1:
                next_els = resp.css(".ipsPagination_next a")
                if not next_els:
                    break
                href = next_els[0].attrib.get("href", "")
                if not href:
                    break
                next_url = abs_url(list_url, href)
                if next_url in visited:
                    break
                visited.add(next_url)
                resp = self.get(next_url)
                if resp is None:
                    break
                html = resp_html(resp)
                topic_urls = []
                for el in resp.css("a[href]"):
                    href = el.attrib.get("href", "")
                    if href and _TOPIC_URL_RE.search(href):
                        topic_urls.append(canonical_thread_url(abs_url(next_url, href)))
                for topic_url in topic_urls:
                    if topic_url not in seen_threads:
                        seen_threads.add(topic_url)
                        yield topic_url
                if html:
                    for topic_url in self._yield_topic_links_from_html(html, next_url):
                        if topic_url not in seen_threads:
                            seen_threads.add(topic_url)
                            yield topic_url
                pages += 1

    def scrape_thread(
        self, thread_url: str, entry: ForumEntry
    ) -> Generator[RawPost, None, None]:
        page_url: str | None = thread_url
        page_num = 0

        while page_url and page_num < self.max_thread_pages:
            resp = self.get(page_url)
            if resp is None:
                break

            # Title
            title_els = (
                resp.css("h1 span.ipsType_break")
                or resp.css("h1.ipsType_pageTitle")
                or resp.css("h1")
            )
            thread_title = title_els[0].get_all_text().strip() if title_els else ""

            tid_m = _TID_RE.search(page_url)
            thread_id = tid_m.group(1) if tid_m else ""

            crumbs = resp.css("li.ipsBreadcrumb_entry a")
            section = crumbs[-1].get_all_text().strip() if crumbs else ""

            # Stats
            reply_count = 0
            view_count = 0
            rc_els = resp.css("span[data-role='replyCount']")
            if rc_els:
                try:
                    reply_count = int(rc_els[0].get_all_text().strip().replace(",", ""))
                except ValueError:
                    pass

            for idx, article in enumerate(resp.css(
                "article[data-commentid], div.cPost_contentWrap"
            )):
                cid = article.attrib.get("data-commentid", "")
                if not cid:
                    cid_m = _CID_RE.search(str(article.attrib))
                    cid = cid_m.group(1) if cid_m else ""

                # Author
                auth_els = (
                    article.css("[data-member-id] a.ipsType_break")
                    or article.css("strong.ipsType_break")
                    or article.css(".ipsComment_author a")
                )
                raw_author = (
                    auth_els[0].get_all_text().strip() if auth_els else "unknown"
                )

                # Timestamp
                time_els = article.css("time[datetime]")
                timestamp: datetime | None = None
                if time_els:
                    try:
                        timestamp = datetime.fromisoformat(
                            time_els[0].attrib["datetime"].replace("Z", "+00:00")
                        )
                    except (ValueError, KeyError):
                        pass

                # Body
                body_els = (
                    article.css(".ipsType_richText")
                    or article.css(".cPost_contentWrap")
                    or article.css(".ipsComment_content")
                )
                body = strip_html(body_els[0].html_content or "") if body_els else ""
                if not body.strip():
                    continue

                yield RawPost(
                    forum_id=entry.forum_id,
                    thread_url=thread_url,
                    post_index=page_num * 25 + idx,
                    raw_author=raw_author,
                    body=body,
                    scraped_at=utcnow(),
                    thread_title=thread_title,
                    section=section,
                    post_id=cid,
                    thread_id=thread_id,
                    reply_count=reply_count,
                    view_count=view_count,
                    timestamp=timestamp,
                )

            # Pagination
            next_els = resp.css(".ipsPagination_next a")
            if not next_els:
                break
            href = next_els[0].attrib.get("href", "")
            if not href:
                break
            page_url = abs_url(page_url, href)
            page_num += 1
