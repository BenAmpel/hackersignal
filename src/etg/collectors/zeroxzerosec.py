"""Collector for forum.0x00sec.org — Discourse-based security/hacking forum.

Endpoint overview
-----------------
1. ``GET /categories.json``
   Returns ``{"category_list": {"categories": [...]}}`` with id/slug/name/topic_count.

2. ``GET /c/{slug}/{id}/l/latest.json?page=N``
   Paginated topic list. 30 topics/page. Stop when
   ``topic_list.more_topics_url`` is null.

3. ``GET /t/{topic_id}.json``
   Full topic with first ~20 posts plus the full stream of post IDs.
   ``post_stream.posts`` → initial posts;
   ``post_stream.stream`` → all post IDs in the topic.

4. ``GET /t/{topic_id}/posts.json?post_ids[]=X&post_ids[]=Y…``
   Fetch remaining posts in chunks of ≤20.

Rate limit (unauthenticated): 60 req/min → default delay of 1.0 s.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import requests

from etg.collectors._common import author_hash, parse_date, post_id

log = logging.getLogger(__name__)

FORUM_ID = "0x00sec"
BASE = "https://forum.0x00sec.org"

# All security-relevant categories (id, slug)
_CATEGORIES: list[tuple[int, str]] = [
    (7,  "offensive"),
    (6,  "defensive"),
    (8,  "web"),
    (10, "malware"),
    (4,  "general"),
    (9,  "programming"),
    (5,  "spotlight-discussion"),
]

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")

_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}

# Maximum post IDs per chunk when fetching remaining posts
_CHUNK_SIZE = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_html(cooked: str) -> str:
    """Strip HTML tags from a Discourse ``cooked`` field and collapse whitespace."""
    text = _TAG_RE.sub(" ", cooked)
    return re.sub(r"\s+", " ", text).strip()


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    return s


def _get_json(
    session: requests.Session,
    url: str,
    delay: float = 1.0,
    *,
    params: dict | None = None,
) -> dict | None:
    """Fetch *url* as JSON with rate-limit awareness.

    Returns the parsed dict, or ``None`` on any unrecoverable error.
    Retries once on 429 (waits 30 s), skips on 403/5xx.
    """
    try:
        r = session.get(url, params=params, timeout=30, allow_redirects=True)
    except requests.RequestException as exc:
        log.warning("0x00sec: request error %s: %s", url, exc)
        return None
    finally:
        time.sleep(delay)

    if r.status_code == 200:
        try:
            return r.json()
        except ValueError as exc:
            log.warning("0x00sec: JSON decode error %s: %s", url, exc)
            return None

    if r.status_code == 429:
        log.warning("0x00sec: 429 rate-limited on %s — sleeping 30 s then retrying", url)
        time.sleep(30)
        try:
            r2 = session.get(url, params=params, timeout=30, allow_redirects=True)
            time.sleep(delay)
            if r2.status_code == 200:
                try:
                    return r2.json()
                except ValueError:
                    return None
            log.warning("0x00sec: retry still failed HTTP %d for %s", r2.status_code, url)
        except requests.RequestException as exc:
            log.warning("0x00sec: retry request error %s: %s", url, exc)
        return None

    log.warning("0x00sec: HTTP %d for %s — skipping", r.status_code, url)
    return None


# ---------------------------------------------------------------------------
# Category → topic iteration
# ---------------------------------------------------------------------------


def _iter_category_topics(
    session: requests.Session,
    cat_id: int,
    cat_slug: str,
    max_pages: int = 200,
    delay: float = 1.0,
):
    """Yield topic dicts for every page of *cat_id*/*cat_slug*."""
    for page in range(0, max_pages):
        url = f"{BASE}/c/{cat_slug}/{cat_id}/l/latest.json"
        data = _get_json(session, url, delay=delay, params={"page": page})
        if data is None:
            log.warning(
                "0x00sec: failed to fetch page %d for category %s/%d — stopping",
                page, cat_slug, cat_id,
            )
            break

        topic_list = data.get("topic_list", {})
        topics = topic_list.get("topics", [])
        if not topics:
            log.debug("0x00sec: category %s page %d returned no topics", cat_slug, page)
            break

        for topic in topics:
            yield topic

        # Discourse signals end of pages with a null more_topics_url
        if not topic_list.get("more_topics_url"):
            log.debug(
                "0x00sec: category %s/%d exhausted at page %d",
                cat_slug, cat_id, page,
            )
            break


# ---------------------------------------------------------------------------
# Topic → post iteration
# ---------------------------------------------------------------------------


def _iter_topic_posts(
    session: requests.Session,
    topic_id: int,
    delay: float = 1.0,
):
    """Yield every post dict in *topic_id*, loading additional pages as needed."""
    url = f"{BASE}/t/{topic_id}.json"
    data = _get_json(session, url, delay=delay)
    if data is None:
        return

    post_stream = data.get("post_stream", {})
    initial_posts: list[dict] = post_stream.get("posts", [])
    full_stream: list[int] = post_stream.get("stream", [])

    # Emit the initial batch
    for p in initial_posts:
        yield p

    # Determine which IDs still need to be fetched
    fetched_ids: set[int] = {p["id"] for p in initial_posts}
    remaining_ids = [pid for pid in full_stream if pid not in fetched_ids]

    if not remaining_ids:
        return

    log.debug(
        "0x00sec: topic %d — fetching %d remaining posts in chunks of %d",
        topic_id, len(remaining_ids), _CHUNK_SIZE,
    )

    for chunk_start in range(0, len(remaining_ids), _CHUNK_SIZE):
        chunk = remaining_ids[chunk_start: chunk_start + _CHUNK_SIZE]
        # Build query string manually so repeated keys are accepted
        qs = "&".join(f"post_ids[]={pid}" for pid in chunk)
        chunk_url = f"{BASE}/t/{topic_id}/posts.json?{qs}"
        chunk_data = _get_json(session, chunk_url, delay=delay)
        if chunk_data is None:
            log.warning(
                "0x00sec: failed to fetch post chunk for topic %d (ids %s…) — skipping chunk",
                topic_id, chunk[:3],
            )
            continue
        for p in chunk_data.get("post_stream", {}).get("posts", []):
            yield p


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect(
    output: Path = Path("data/0x00sec_posts.jsonl"),
    cve_index_output: Path = Path("data/0x00sec_cve_index.jsonl"),
    request_delay: float = 1.0,
    max_pages: int = 200,
) -> tuple[int, int]:
    """Scrape forum.0x00sec.org and write posts + CVE index JSONL files.

    Parameters
    ----------
    output:
        Destination path for the posts JSONL file.
    cve_index_output:
        Destination path for the CVE-index JSONL file.
    request_delay:
        Seconds to sleep between API requests (default 1.0 respects the
        Discourse unauthenticated limit of 60 req/min).
    max_pages:
        Maximum topic-list pages to fetch per category.

    Returns
    -------
    ``(post_count, cve_index_count)``
    """
    output = Path(output)
    cve_index_output = Path(cve_index_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cve_index_output.parent.mkdir(parents=True, exist_ok=True)

    session = _make_session()
    seen_topic_ids: set[int] = set()
    seen_post_pids: set[str] = set()
    post_count = 0
    cve_index_count = 0

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        for cat_id, cat_slug in _CATEGORIES:
            log.info("0x00sec: scraping category %s (id=%d)", cat_slug, cat_id)

            for topic in _iter_category_topics(
                session, cat_id, cat_slug,
                max_pages=max_pages,
                delay=request_delay,
            ):
                topic_id: int = topic["id"]
                if topic_id in seen_topic_ids:
                    continue
                seen_topic_ids.add(topic_id)

                topic_title: str = topic.get("title", "")
                log.debug("0x00sec: topic %d — %s", topic_id, topic_title)

                first_post = True
                for post_data in _iter_topic_posts(
                    session, topic_id, delay=request_delay,
                ):
                    raw_id: int = post_data.get("id", 0)
                    cooked: str = post_data.get("cooked", "")
                    username: str = post_data.get("username", "anonymous")
                    created_at: str = post_data.get("created_at", "")

                    body = _strip_html(cooked)
                    if not body:
                        first_post = False
                        continue

                    # First post in topic prefixed with title
                    if first_post and topic_title:
                        text = f"{topic_title}\n\n{body}"
                        first_post = False
                    else:
                        text = body
                        first_post = False

                    ts = parse_date(created_at)
                    pid = post_id(FORUM_ID, str(raw_id))
                    ahash = author_hash(FORUM_ID, username)

                    if pid in seen_post_pids:
                        continue
                    seen_post_pids.add(pid)

                    post_dict = {
                        "id": pid,
                        "text": text,
                        "timestamp": ts.isoformat(),
                        "forum_id": FORUM_ID,
                        "author_hash": ahash,
                    }
                    post_fh.write(json.dumps(post_dict) + "\n")
                    post_fh.flush()
                    post_count += 1

                    cves = list({m.upper() for m in _CVE_RE.findall(text)})
                    for cve in cves:
                        entry = {
                            "exploit_id": pid,
                            "cve_id": cve,
                            "exploit_text": text[:500],
                            "title": topic_title,
                            "published": ts.isoformat(),
                            "source": FORUM_ID,
                        }
                        cve_fh.write(json.dumps(entry) + "\n")
                        cve_fh.flush()
                        cve_index_count += 1

                    if post_count % 200 == 0:
                        log.info(
                            "0x00sec: %d posts, %d CVE refs so far",
                            post_count, cve_index_count,
                        )

    log.info(
        "0x00sec: done — %d posts, %d CVE index entries → %s",
        post_count, cve_index_count, output,
    )
    return post_count, cve_index_count


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    collect()
