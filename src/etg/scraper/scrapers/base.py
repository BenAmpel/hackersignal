"""Abstract base scraper shared by all forum-software implementations."""

from __future__ import annotations

import http.cookiejar
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from scrapling import Fetcher

from etg.scraper.models import ForumEntry, RawPost

log = logging.getLogger(__name__)

# Common HTML-entity / tag stripper
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s{2,}")
_HTML_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": '"', "&#39;": "'", "&nbsp;": " ",
}

# Alternate referers to rotate through on 403 retry
_REFERERS = [
    "https://www.google.com/",
    "https://duckduckgo.com/",
    "https://t.co/",
    "https://www.reddit.com/",
]

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# curl error fragments that indicate a truly dead host (DNS failure / port closed).
# NOTE: timeouts (Connection timed out, Operation timed out) are NOT included —
# a slow or rate-limiting server is not dead; we log and skip the individual page.
_DEAD_HOST_ERRORS = (
    "Could not resolve host",
    "Failed to connect to",
    "Connection refused",
)

_THREAD_QUERY_KEEP = {
    "t", "tid", "thread", "thread_id", "topic", "topic_id", "showtopic",
    "f", "fid", "forum", "forum_id", "id",
}

_BLOCKED_PAGE_MARKERS = (
    "just a moment",
    "verify you are human",
    "checking your browser",
    "cf-browser-verification",
    "why do i have to complete a captcha",
    "attention required! | cloudflare",
)

_PLACEHOLDER_PAGE_MARKERS = (
    "window.location.href=\"/lander\"",
    "window.location.href='/lander'",
    "<title>redirecting...</title>",
)

_HREF_RE = re.compile(r"""href=["']([^"']+)["']""", re.I)
_LOC_RE = re.compile(r"<loc>([^<]+)</loc>", re.I)
_JS_REDIRECT_RE = re.compile(
    r"""window\.location(?:\.href)?\s*=\s*["']([^"']+)["']""",
    re.I,
)


def strip_html(html: str) -> str:
    """Remove tags and decode common HTML entities."""
    for ent, ch in _HTML_ENTITIES.items():
        html = html.replace(ent, ch)
    text = _TAG_RE.sub(" ", html)
    return _SPACE_RE.sub(" ", text).strip()


def resp_html(resp) -> str:
    """Return decoded HTML string from a Scrapling Response.

    Scrapling 0.4.x stores raw bytes in ``resp.body``; ``resp.text`` is
    an empty string.  This helper normalises both.
    """
    if resp is None:
        return ""
    body = getattr(resp, "body", None)
    if isinstance(body, (bytes, bytearray)) and body:
        enc = getattr(resp, "encoding", None)
        enc = enc if isinstance(enc, str) and enc else "utf-8"
        return body.decode(enc, errors="replace")
    text = getattr(resp, "text", None)
    return text if isinstance(text, str) and text else ""


def abs_url(base: str, href: str) -> str:
    """Resolve *href* relative to *base*."""
    return urljoin(base, href)


def canonical_thread_url(url: str) -> str:
    """Collapse common forum thread URL variants to a stable first-page URL.

    This prevents re-crawling the same discussion via unread, page-N, anchor,
    or tracker variants that are common across XenForo/phpBB/MyBB/Invision and
    generic forum links.
    """
    parsed = urlparse(url)
    path = re.sub(r"/+", "/", parsed.path)
    path = re.sub(r"/(?:unread|latest|new-post|last-post)/?$", "/", path, flags=re.I)
    path = re.sub(r"/page-\d+/?$", "/", path, flags=re.I)
    path = re.sub(r"/page/\d+/?$", "/", path, flags=re.I)
    path = re.sub(r"/+$", "/", path) if path != "/" else path

    query_pairs = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() in _THREAD_QUERY_KEEP:
            query_pairs.append((key, value))
    query = urlencode(query_pairs, doseq=True)

    return urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        path,
        "",
        query,
        "",
    ))


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def make_fetcher() -> Fetcher:
    """Construct a Scrapling fetcher without hitting its deprecated __init__."""
    return object.__new__(Fetcher)


def _normalise_proxies(proxy: Optional[str | list[str]]) -> list[str]:
    """Return a de-duplicated proxy preference list.

    Supports a single proxy string, comma/semicolon-separated strings, or a
    concrete list. When no explicit proxy is supplied, we also honour
    ``ETG_HTTP_PROXY`` and ``ETG_BACKUP_PROXIES`` as optional clearnet fallbacks.
    """
    raw_values: list[str] = []
    if isinstance(proxy, str) and proxy.strip():
        raw_values.extend(re.split(r"[;,]", proxy))
    elif isinstance(proxy, list):
        raw_values.extend(proxy)
    else:
        env_proxy = os.getenv("ETG_HTTP_PROXY", "").strip()
        env_backups = os.getenv("ETG_BACKUP_PROXIES", "").strip()
        if env_proxy:
            raw_values.append(env_proxy)
        if env_backups:
            raw_values.extend(re.split(r"[;,]", env_backups))

    seen: set[str] = set()
    proxies: list[str] = []
    for value in raw_values:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            proxies.append(value)
    return proxies


def is_blocked_html(html: str) -> bool:
    lowered = html.lower()
    return any(marker in lowered for marker in _BLOCKED_PAGE_MARKERS)


def is_placeholder_html(html: str) -> bool:
    lowered = html.lower()
    return any(marker in lowered for marker in _PLACEHOLDER_PAGE_MARKERS)


def extract_redirect_targets(url: str, html: str) -> list[str]:
    targets: list[str] = []
    for match in _JS_REDIRECT_RE.findall(html):
        targets.append(abs_url(url, match))
    return targets


def extract_hrefs(html: str, base_url: str) -> list[str]:
    hrefs = [abs_url(base_url, href) for href in _HREF_RE.findall(html)]
    hrefs.extend(abs_url(base_url, href) for href in _LOC_RE.findall(html))
    return hrefs


def _is_dead_host_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(fragment in msg for fragment in _DEAD_HOST_ERRORS)


class BaseScraper(ABC):
    """Contract every per-software scraper must satisfy.

    Subclasses implement:
      - ``thread_urls(forum_url)``  → paginated list of thread URLs
      - ``scrape_thread(thread_url, entry)``  → yields RawPost objects

    The pipeline calls both in sequence.
    """

    #: Seconds to sleep between page requests.
    request_delay: float = 2.0
    #: Maximum thread-list pages to paginate through per subforum.
    max_list_pages: int = 5
    #: Maximum pages to paginate within a single thread.
    max_thread_pages: int = 5
    #: Maximum threads to collect per forum.
    max_threads: int = 100
    #: HTTP timeout (seconds) per request.
    connect_timeout: int = 15
    #: Minimum post body length (chars) to keep — filters navigation noise.
    min_body_len: int = 40

    def __init__(
        self,
        proxy: Optional[str] = None,
        cookies: Optional[dict | Path | str] = None,
    ):
        """
        Parameters
        ----------
        proxy:
            Optional SOCKS5/HTTP proxy URL (e.g. ``"socks5://127.0.0.1:9050"``).
        cookies:
            Session cookies for authenticated scraping.  Accepted formats:

            * ``dict`` — ``{"session": "abc123", "cf_clearance": "xyz"}``
            * ``str`` / ``Path`` — path to a Netscape cookie file (exported
              from your browser with the "Cookie-Editor" extension using
              *Export → Netscape* format).  One cookie per line:
              ``domain \\t flag \\t path \\t secure \\t expiry \\t name \\t value``
        """
        self._proxies = _normalise_proxies(proxy)
        self._proxy = self._proxies[0] if self._proxies else None
        self._cookies: dict = self._load_cookies(cookies)
        self._fetcher = make_fetcher()
        self._dead: bool = False     # set True after a dead-host error
        self._seeded_hosts: set[str] = set()

    @staticmethod
    def _load_cookies(cookies) -> dict:
        """Normalise *cookies* to a plain dict."""
        if cookies is None:
            return {}
        if isinstance(cookies, dict):
            return cookies
        # Path or str — treat as Netscape cookie file
        path = Path(cookies)
        if not path.exists():
            log.warning("Cookie file not found: %s", path)
            return {}
        jar = http.cookiejar.MozillaCookieJar(str(path))
        try:
            jar.load(ignore_discard=True, ignore_expires=True)
        except Exception as exc:
            log.warning("Failed to load cookie file %s: %s", path, exc)
            return {}
        return {c.name: c.value for c in jar}

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _home_url(url: str) -> str:
        parsed = urlparse(url)
        return urlunparse((parsed.scheme, parsed.netloc, "/", "", "", ""))

    def _default_headers(self, url: str) -> dict[str, str]:
        return {
            "User-Agent": _DEFAULT_USER_AGENT,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Upgrade-Insecure-Requests": "1",
            "Referer": self._home_url(url),
        }

    def _seed_host(self, url: str) -> None:
        """Warm up the forum homepage once so anti-bot/cookie checks are likelier to pass."""
        home_url = self._home_url(url)
        if home_url in self._seeded_hosts:
            return
        self._seeded_hosts.add(home_url)

        kw: dict = {
            "timeout": min(self.connect_timeout, 10),
            "headers": self._default_headers(home_url),
        }
        if self._proxy:
            kw["proxy"] = self._proxy
        if self._cookies:
            kw["cookies"] = dict(self._cookies)
        self._fetch_once(home_url, **kw)

    def _proxy_candidates(self) -> list[Optional[str]]:
        candidates = list(self._proxies)
        if None not in candidates:
            candidates.append(None)
        if self._proxy in candidates:
            candidates.remove(self._proxy)
            candidates.insert(0, self._proxy)
        return candidates

    def get(self, url: str, **kwargs):
        """Fetch *url* with automatic 403 UA-rotation retry.

        Returns a Scrapling Response or None on failure.
        Sets ``self._dead = True`` if the host is unreachable so the
        pipeline can bail out early.
        """
        if self._dead:
            return None

        if kwargs.pop("_seed", True):
            self._seed_host(url)

        kw: dict = {}
        if self._cookies:
            # Merge caller-supplied cookies with session cookies (caller wins)
            merged = dict(self._cookies)
            merged.update(kwargs.pop("cookies", {}) or {})
            kw["cookies"] = merged
        headers = self._default_headers(url)
        headers.update(kwargs.pop("headers", {}) or {})
        kw["headers"] = headers
        kw.update(kwargs)
        kw.setdefault("timeout", self.connect_timeout)

        last_resp = None
        for proxy in self._proxy_candidates():
            if proxy:
                kw["proxy"] = proxy
            else:
                kw.pop("proxy", None)

            resp = self._fetch_once(url, **kw)
            last_resp = resp
            if resp is None:
                continue
            if resp.status == 403:
                log.debug("403 on %s via %s — retrying with alternate referers", url, proxy or "direct")
                time.sleep(self.request_delay)
                for referer in _REFERERS[1:]:
                    kw2 = dict(kw)
                    headers = dict(kw2.pop("headers", {}) or {})
                    headers["Referer"] = referer
                    kw2["headers"] = headers
                    resp2 = self._fetch_once(url, **kw2)
                    if resp2 is not None and resp2.status < 400:
                        self._proxy = proxy
                        time.sleep(self.request_delay)
                        return resp2
                    time.sleep(0.5)
                continue
            if resp.status >= 400:
                last_resp = resp
                continue

            html = resp_html(resp)
            if html and is_blocked_html(html):
                log.debug("Blocked interstitial on %s via %s", url, proxy or "direct")
                last_resp = resp
                continue

            self._proxy = proxy
            time.sleep(self.request_delay)
            return resp

        if last_resp is not None and last_resp.status >= 400:
            log.warning("HTTP %s for %s", last_resp.status, url)
        elif last_resp is not None and is_blocked_html(resp_html(last_resp)):
            log.warning("Blocked by anti-bot interstitial for %s", url)
        elif last_resp is not None and is_placeholder_html(resp_html(last_resp)):
            log.warning("Placeholder/redirect page for %s", url)
        return None

    def _fetch_once(self, url: str, **kw):
        try:
            return self._fetcher.get(url, **kw)
        except Exception as exc:
            if _is_dead_host_error(exc):
                log.info("Dead host %s — marking forum unreachable", url)
                self._dead = True
            else:
                log.debug("Fetch error %s → %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def thread_urls(self, forum_url: str) -> Generator[str, None, None]:
        """Yield thread URLs from paginated thread-list pages."""
        ...

    @abstractmethod
    def scrape_thread(
        self, thread_url: str, entry: ForumEntry
    ) -> Generator[RawPost, None, None]:
        """Yield one RawPost per post within a thread page."""
        ...

    # ------------------------------------------------------------------
    # Convenience: iterate all posts for a forum
    # ------------------------------------------------------------------

    def scrape_forum(
        self,
        entry: ForumEntry,
        *,
        max_thread_urls: int | None = None,
        max_posts: int | None = None,
        max_posts_per_thread: int | None = None,
    ) -> Generator[RawPost, None, None]:
        seen_threads: set[str] = set()
        productive_threads = 0
        scanned_threads = 0
        yielded_posts = 0
        for thread_url in self.thread_urls(entry.url):
            if self._dead:
                log.info("Forum %s marked dead — stopping", entry.name)
                break
            if thread_url in seen_threads:
                continue
            seen_threads.add(thread_url)
            scanned_threads += 1
            if max_thread_urls is not None and scanned_threads > max_thread_urls:
                log.info(
                    "Reached max_thread_urls=%d for %s",
                    max_thread_urls,
                    entry.name,
                )
                break
            posts_yielded = 0
            for post in self.scrape_thread(thread_url, entry):
                if len(post.body) >= self.min_body_len:
                    yield post
                    posts_yielded += 1
                    yielded_posts += 1
                    if (
                        max_posts_per_thread is not None
                        and posts_yielded >= max_posts_per_thread
                    ):
                        break
                    if max_posts is not None and yielded_posts >= max_posts:
                        return
            if posts_yielded:
                productive_threads += 1
            if productive_threads >= self.max_threads:
                log.info("Reached max_threads=%d for %s", self.max_threads, entry.name)
                break
