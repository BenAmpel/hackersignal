"""Detect forum software from a live page response.

Returns one of the ForumKind enum values so the pipeline can route to the
correct scraper implementation.
"""

from __future__ import annotations

import re
from enum import Enum, auto
from urllib.parse import urlparse


class ForumKind(Enum):
    XENFORO = auto()
    PHPBB = auto()
    MYBB = auto()
    DISCOURSE = auto()
    INVISION = auto()
    UNKNOWN = auto()


# ------------------------------------------------------------------
# URL-based heuristics (fast path, no HTTP needed)
# ------------------------------------------------------------------

_URL_SIGNALS: list[tuple[re.Pattern, ForumKind]] = [
    # phpBB — viewforum.php / viewtopic.php in path
    (re.compile(r"view(forum|topic)\.php", re.I), ForumKind.PHPBB),
    # MyBB — forumdisplay.php / showthread.php
    (re.compile(r"(forumdisplay|showthread)\.php", re.I), ForumKind.MYBB),
    # Invision Community
    (re.compile(r"\?app=forums", re.I), ForumKind.INVISION),
]


def _from_url(url: str) -> ForumKind | None:
    for pattern, kind in _URL_SIGNALS:
        if pattern.search(url):
            return kind
    return None


# ------------------------------------------------------------------
# HTML-based heuristics (applied to page source)
# ------------------------------------------------------------------

_HTML_SIGNALS: list[tuple[re.Pattern, ForumKind]] = [
    # XenForo — generator meta or body class
    (re.compile(r"xenforo", re.I), ForumKind.XENFORO),
    # phpBB — meta generator or canonical footer
    (re.compile(r"phpBB", re.I), ForumKind.PHPBB),
    # MyBB
    (re.compile(r"\bMyBB\b|\bmybb\b"), ForumKind.MYBB),
    # Discourse
    (re.compile(r"Discourse\.SiteSettings|data-discourse", re.I), ForumKind.DISCOURSE),
    # Invision Community (IPB / IPS)
    (re.compile(r"ips\.setSetting|ips\.setting|Invision Community|ipb_url_filter", re.I), ForumKind.INVISION),
]


def detect(url: str, html: str | None = None) -> ForumKind:
    """Return the forum kind for *url*, optionally using raw *html*.

    Call with just the URL for a cheap guess (suitable for early routing).
    Pass the fetched HTML for a more reliable result.
    """
    # 1. URL fast-path
    url_kind = _from_url(url)
    if url_kind is not None:
        return url_kind

    if not html:
        return ForumKind.UNKNOWN

    # 2. HTML signals (first match wins; order matters)
    for pattern, kind in _HTML_SIGNALS:
        if pattern.search(html):
            return kind

    # 3. Discourse: check for /latest.json availability or meta tag
    if '"application_name":"Discourse"' in html or "discourse_meta" in html:
        return ForumKind.DISCOURSE

    return ForumKind.UNKNOWN
