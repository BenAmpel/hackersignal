"""phpBB scraper (v3.x).

URL patterns:
  Forum list   index.php  (or root)
  Thread list  viewforum.php?f={id}[&start=N]
  Thread       viewtopic.php?t={id}[&start=N]

Key selectors:
  Thread list  ul#topiclist li.row dt a.topictitle
  Posts        div#pagecontent div.post  (or div[id^="p"])
  Post body    div.content
  Author       p.author span.username, strong.username-coloured
  Timestamp    p.author time, or text like "Posted: Mon Apr 01, 2024"
  Thread title h2 a.topic-title, or #page-body h2
  Reply count  dd.posts in thread stats
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
    resp_html,
    strip_html,
    utcnow,
)

log = logging.getLogger(__name__)

_START_RE = re.compile(r"start=(\d+)")
_TID_RE = re.compile(r"[?&]t=(\d+)")
_FID_RE = re.compile(r"[?&]f=(\d+)")
_PID_RE = re.compile(r"#p(\d+)|id=['\"]p(\d+)['\"]")

_DATE_FMTS = [
    "%a %b %d, %Y %I:%M %p",   # Mon Jan 01, 2024 12:00 pm
    "%a %b %d, %Y %H:%M",       # Mon Jan 01, 2024 12:00
    "%d %b %Y %H:%M",
    "%d %b %Y",
]


def _parse_phpbb_date(text: str) -> datetime | None:
    text = re.sub(r"\s+", " ", text.strip())
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _paginate(base_url: str, start: int, per_page: int = 25) -> str:
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}start={start}"


class PhpBBScraper(BaseScraper):
    """Scraper for phpBB 3.x forums."""

    def thread_urls(self, forum_url: str) -> Generator[str, None, None]:
        visited: set[str] = set()
        forum_queue: list[str] = [forum_url]
        seen_threads: set[str] = set()
        base_host = urlparse(forum_url).netloc

        while forum_queue:
            list_url = forum_queue.pop(0)
            if list_url in visited:
                continue
            visited.add(list_url)

            resp = self.get(list_url)
            if resp is None:
                continue
            html = resp_html(resp)

            # Sub-forum links
            for el in resp.css("a.forumtitle"):
                href = el.attrib.get("href", "")
                if "viewforum.php" in href:
                    forum_queue.append(abs_url(list_url, href))

            # Thread links on this page
            for el in resp.css("a.topictitle"):
                href = el.attrib.get("href", "")
                if "viewtopic.php" in href or "t=" in href:
                    canonical = canonical_thread_url(abs_url(list_url, href))
                    if canonical not in seen_threads:
                        seen_threads.add(canonical)
                        yield canonical

            if html:
                for full in extract_hrefs(html, list_url):
                    if urlparse(full).netloc != base_host:
                        continue
                    if "viewtopic.php" in full or "t=" in full:
                        canonical = canonical_thread_url(full)
                        if canonical not in seen_threads:
                            seen_threads.add(canonical)
                            yield canonical
                    elif "viewforum.php" in full and full not in visited:
                        forum_queue.append(full)

            # Pagination on thread-list
            pages = 0
            while pages < self.max_list_pages - 1:
                next_els = resp.css("li.next a, a.arrow.right")
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
                for el in resp.css("a.topictitle"):
                    href = el.attrib.get("href", "")
                    if "viewtopic.php" in href or "t=" in href:
                        canonical = canonical_thread_url(abs_url(next_url, href))
                        if canonical not in seen_threads:
                            seen_threads.add(canonical)
                            yield canonical
                if html:
                    for full in extract_hrefs(html, next_url):
                        if urlparse(full).netloc != base_host:
                            continue
                        if "viewtopic.php" in full or "t=" in full:
                            canonical = canonical_thread_url(full)
                            if canonical not in seen_threads:
                                seen_threads.add(canonical)
                                yield canonical
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

            # Thread title
            title_els = (
                resp.css("h2 a.topic-title")
                or resp.css("h2.topic-title")
                or resp.css("h1")
            )
            thread_title = title_els[0].get_all_text().strip() if title_els else ""

            tid_m = _TID_RE.search(page_url)
            thread_id = tid_m.group(1) if tid_m else ""

            # Section breadcrumb
            crumbs = resp.css("#nav-breadcrumbs li a, ol.breadcrumb li a")
            section = crumbs[-1].get_all_text().strip() if crumbs else ""

            # Stat bar
            stat_text = ""
            stat_els = resp.css("dl.postprofile dd") or resp.css(".topic-stats")
            if stat_els:
                stat_text = " ".join(e.get_all_text() for e in stat_els)
            reply_m = re.search(r"(\d+)\s*repl", stat_text, re.I)
            view_m = re.search(r"(\d+)\s*view", stat_text, re.I)
            reply_count = int(reply_m.group(1)) if reply_m else 0
            view_count = int(view_m.group(1)) if view_m else 0

            post_divs = resp.css("div[id^='p']") or resp.css("div.post")
            for idx, div in enumerate(post_divs):
                pid_m = _PID_RE.search(div.attrib.get("id", ""))
                post_id = pid_m.group(1) or pid_m.group(2) if pid_m else ""

                author_els = (
                    div.css("span.username-coloured")
                    or div.css("span.username")
                    or div.css("strong.username")
                    or div.css("a[href*='memberlist']")
                )
                raw_author = (
                    author_els[0].get_all_text().strip()
                    if author_els else "unknown"
                )

                # Timestamp
                time_els = div.css("time[datetime]")
                timestamp: datetime | None = None
                if time_els:
                    try:
                        timestamp = datetime.fromisoformat(
                            time_els[0].attrib["datetime"].replace("Z", "+00:00")
                        )
                    except (ValueError, KeyError):
                        pass
                else:
                    # Fallback: parse text like "Posted: Mon Jan 01, 2024 12:00 pm"
                    author_ps = div.css("p.author")
                    if author_ps:
                        txt = author_ps[0].get_all_text()
                        posted_m = re.search(r"Posted:\s*(.+)", txt)
                        if posted_m:
                            timestamp = _parse_phpbb_date(posted_m.group(1))

                body_els = div.css("div.content, div.postbody")
                body = strip_html(body_els[0].html_content or "") if body_els else ""
                if not body.strip():
                    continue

                yield RawPost(
                    forum_id=entry.forum_id,
                    thread_url=thread_url,
                    post_index=page_num * 10 + idx,
                    raw_author=raw_author,
                    body=body,
                    scraped_at=utcnow(),
                    thread_title=thread_title,
                    section=section,
                    post_id=post_id,
                    thread_id=thread_id,
                    reply_count=reply_count,
                    view_count=view_count,
                    timestamp=timestamp,
                )

            # Pagination
            next_els = resp.css("li.next a, a.arrow.right")
            if not next_els:
                break
            href = next_els[0].attrib.get("href", "")
            if not href:
                break
            page_url = abs_url(page_url, href)
            page_num += 1
