"""MyBB scraper (1.x).

URL patterns:
  Forum list   index.php (or root)
  Thread list  forumdisplay.php?fid={id}[&page=N]
  Thread       showthread.php?tid={id}[&page=N]

Key selectors:
  Thread list  .threads table tbody tr td.forumdisplay_threadtitle a (normal mode)
               OR .threadlist .inline_row td.forumdisplay_threadtitle a
  Posts        .post  (div or table row with class "post")
  Post body    .post_body  /  div[id^="pid_"] .post_body
  Author       .post_author strong a, .largetext a
  Timestamp    span.post_date, .post_date
  Thread title #thead h1, .thead h1, page title
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

_TID_RE = re.compile(r"[?&]tid=(\d+)")
_FID_RE = re.compile(r"[?&]fid=(\d+)")
_PID_RE = re.compile(r"pid_(\d+)|post_(\d+)")

_DATE_FMTS = [
    "%b %d, %Y %I:%M %p",    # Dec 9, 2025 12:33 AM
    "%B %d, %Y %I:%M %p",    # December 9, 2025 12:33 AM
    "%m-%d-%Y, %I:%M %p",    # 01-01-2024, 12:00 AM
    "%m-%d-%Y %I:%M %p",
    "%Y-%m-%d, %H:%M",
    "%Y-%m-%d %H:%M",
    "%d %b %Y, %I:%M %p",
    "%d %b %Y %I:%M %p",
    "%d %b %Y",
]


def _parse_mybb_date(text: str) -> datetime | None:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s*\(This post was last modified:.*$", "", text, flags=re.I)
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _breadcrumb_section(resp) -> str:
    labels = []
    for el in resp.css("#breadcrumb a, .breadcrumb a"):
        text = el.get_all_text().strip()
        href = el.attrib.get("href", "")
        if not text:
            continue
        if "index.php" in href or text.lower() == "hack forums":
            continue
        labels.append(text)
    return " / ".join(labels[-2:]) if labels else ""


def _extract_post_body(div, resp) -> str:
    selectors = [
        ".post_body",
        ".post_content",
        ".scaleimages",
        ".content",
        ".message",
        ".post",
        "article",
        "main",
    ]
    for sel in selectors:
        body_els = div.css(sel)
        if body_els:
            body = strip_html(body_els[0].html_content or "")
            if body.strip():
                return body

    # Some heavily customized MyBB themes collapse content into generic page
    # wrappers; pick the largest likely content block rather than dropping it.
    candidates = []
    for sel in ["#content", ".content", "main", "body"]:
        for el in resp.css(sel):
            text = strip_html(el.html_content or "")
            if text.strip():
                candidates.append(text)
    if candidates:
        body = max(candidates, key=len)
        anti_bot_markers = [
            "attention required",
            "just a moment",
            "verify you are human",
            "cloudflare",
        ]
        if not any(marker in body.lower() for marker in anti_bot_markers):
            return body
    return ""


class MyBBScraper(BaseScraper):
    """Scraper for MyBB 1.x forums."""

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
            for el in resp.css(
                "a.forumtitle, "
                "a[href*='forumdisplay.php?fid='], "
                ".subforum-text a[href*='fid='], "
                ".td-float-left > strong > a[href*='fid=']"
            ):
                href = el.attrib.get("href", "")
                if "forumdisplay.php" in href or "fid=" in href:
                    forum_queue.append(abs_url(list_url, href))

            # Thread links
            for el in resp.css(
                ".forumdisplay_threadtitle a, "
                ".thread_title a, "
                "a.subject_new, a.subject_old, "
                "a[href*='showthread.php?tid=']"
            ):
                href = el.attrib.get("href", "")
                if "showthread.php" in href or "tid=" in href:
                    canonical = canonical_thread_url(abs_url(list_url, href))
                    if canonical not in seen_threads:
                        seen_threads.add(canonical)
                        yield canonical

            if html:
                for full in extract_hrefs(html, list_url):
                    if urlparse(full).netloc != base_host:
                        continue
                    if "showthread.php" in full or "tid=" in full:
                        canonical = canonical_thread_url(full)
                        if canonical not in seen_threads:
                            seen_threads.add(canonical)
                            yield canonical
                    elif (
                        ("forumdisplay.php" in full or "fid=" in full)
                        and full not in visited
                    ):
                        forum_queue.append(full)

            # Pagination on thread list
            pages = 0
            while pages < self.max_list_pages - 1:
                next_els = resp.css("a.pagination_next, span.pagination_next a")
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
                for el in resp.css(
                    ".forumdisplay_threadtitle a, "
                    "a.subject_new, a.subject_old, "
                    "a[href*='showthread.php?tid=']"
                ):
                    href = el.attrib.get("href", "")
                    if "showthread.php" in href or "tid=" in href:
                        canonical = canonical_thread_url(abs_url(next_url, href))
                        if canonical not in seen_threads:
                            seen_threads.add(canonical)
                            yield canonical
                if html:
                    for full in extract_hrefs(html, next_url):
                        if urlparse(full).netloc != base_host:
                            continue
                        if "showthread.php" in full or "tid=" in full:
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
                resp.css("#thead h1")
                or resp.css(".thead h1")
                or resp.css("h1")
            )
            thread_title = title_els[0].get_all_text().strip() if title_els else ""

            tid_m = _TID_RE.search(page_url)
            thread_id = tid_m.group(1) if tid_m else ""

            section = _breadcrumb_section(resp)

            # Thread stats bar (if present)
            stat_text = ""
            stat_els = resp.css(".thread_info, .thread_stats, .stats_table")
            if stat_els:
                stat_text = stat_els[0].get_all_text()
            reply_m = re.search(r"(\d[\d,]*)\s*repl", stat_text, re.I)
            view_m = re.search(r"(\d[\d,]*)\s*view", stat_text, re.I)
            reply_count = int(reply_m.group(1).replace(",", "")) if reply_m else 0
            view_count = int(view_m.group(1).replace(",", "")) if view_m else 0

            # Posts — MyBB wraps each post in a div with id="pid_NNN"
            post_divs = resp.css("div[id^='pid_']") or resp.css("div.post")
            for idx, div in enumerate(post_divs):
                pid_m = _PID_RE.search(div.attrib.get("id", ""))
                post_id = (pid_m.group(1) or pid_m.group(2)) if pid_m else ""

                author_els = (
                    div.css(".post_author strong a")
                    or div.css(".largetext a")
                    or div.css("strong.post_author")
                )
                raw_author = (
                    author_els[0].get_all_text().strip()
                    if author_els else "unknown"
                )

                # Timestamp
                timestamp: datetime | None = None
                date_els = div.css("span.post_date, .post_date")
                if date_els:
                    timestamp = _parse_mybb_date(date_els[0].get_all_text())

                body = _extract_post_body(div, resp)
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
            next_els = resp.css("a.pagination_next")
            if not next_els:
                break
            href = next_els[0].attrib.get("href", "")
            if not href:
                break
            page_url = abs_url(page_url, href)
            page_num += 1
