"""Collectors for public Discourse-based security communities.

The ETG corpus uses these communities as contemporary public practitioner
discussion sources.  Discourse exposes stable JSON endpoints, so this module
keeps collection API-first instead of scraping rendered HTML.
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests

from etg.collectors._common import author_hash, parse_date, post_id

log = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
_CHUNK_SIZE = 20


@dataclass(frozen=True)
class DiscourseCommunity:
    forum_id: str
    base_url: str
    label: str
    include_category_slugs: tuple[str, ...] = ()
    exclude_category_slugs: tuple[str, ...] = ()


COMMUNITIES: dict[str, DiscourseCommunity] = {
    "hackersploit": DiscourseCommunity(
        forum_id="hackersploit",
        base_url="https://forum.hackersploit.org",
        label="HackerSploit",
        include_category_slugs=(
            "pentesting",
            "malware-analysis",
            "reverse-engineering",
            "forensics",
            "networking",
            "python",
            "android",
            "tools",
            "advice",
            "general-discussion",
        ),
    ),
    "parrotsec": DiscourseCommunity(
        forum_id="parrotsec",
        base_url="https://community.parrotsec.org",
        label="ParrotSec Community",
        include_category_slugs=("hacking", "support", "project"),
    ),
    "hackthebox": DiscourseCommunity(
        forum_id="hackthebox",
        base_url="https://forum.hackthebox.com",
        label="Hack The Box Forum",
        include_category_slugs=("ctfs", "tutorials", "content", "uncategorized"),
        exclude_category_slugs=("off-topic", "site-feedback", "links"),
    ),
}


def _strip_html(cooked: str) -> str:
    text = _TAG_RE.sub(" ", cooked or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 ETG academic collector",
        "Accept": "application/json",
    })
    return session


def _get_json(
    session: requests.Session,
    url: str,
    *,
    request_delay: float,
    params: dict | None = None,
) -> dict | None:
    try:
        resp = session.get(url, params=params, timeout=30, allow_redirects=True)
    except requests.RequestException as exc:
        log.warning("discourse: request error %s: %s", url, exc)
        return None
    finally:
        time.sleep(request_delay)

    if resp.status_code == 200:
        try:
            return resp.json()
        except ValueError as exc:
            log.warning("discourse: JSON decode error %s: %s", url, exc)
            return None

    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", "30"))
        log.warning("discourse: rate-limited on %s; sleeping %ds", url, retry_after)
        time.sleep(retry_after)
        try:
            retry = session.get(url, params=params, timeout=30, allow_redirects=True)
            time.sleep(request_delay)
            if retry.status_code == 200:
                return retry.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("discourse: retry failed %s: %s", url, exc)
        return None

    log.warning("discourse: HTTP %d for %s", resp.status_code, url)
    return None


def _category_path(category: dict, prefix: str = "") -> str:
    slug = str(category.get("slug") or category.get("name") or category.get("id"))
    return f"{prefix}/{slug}" if prefix else slug


def _flatten_categories(data: dict) -> list[dict]:
    categories = data.get("category_list", {}).get("categories", [])
    flattened: list[dict] = []

    def visit(category: dict, prefix: str = "") -> None:
        path = _category_path(category, prefix)
        item = dict(category)
        item["path_slug"] = path
        flattened.append(item)
        for child in category.get("subcategory_list") or []:
            if isinstance(child, dict):
                visit(child, path)

    for category in categories:
        if isinstance(category, dict):
            visit(category)
    return flattened


def _selected_categories(
    community: DiscourseCommunity,
    categories: list[dict],
) -> list[dict]:
    include = {slug.lower() for slug in community.include_category_slugs}
    include_order = {
        slug.lower(): idx
        for idx, slug in enumerate(community.include_category_slugs)
    }
    exclude = {slug.lower() for slug in community.exclude_category_slugs}
    selected: list[dict] = []
    for category in categories:
        slug = str(category.get("slug") or "").lower()
        path_slug = str(category.get("path_slug") or slug).lower()
        if exclude and (slug in exclude or path_slug in exclude):
            continue
        if include and slug not in include and path_slug not in include:
            continue
        selected.append(category)
    return sorted(
        selected,
        key=lambda category: include_order.get(
            str(category.get("slug") or category.get("path_slug") or "").lower(),
            len(include_order),
        ),
    )


def _iter_category_topics(
    session: requests.Session,
    community: DiscourseCommunity,
    category: dict,
    *,
    max_pages: int,
    request_delay: float,
):
    cid = category.get("id")
    path_slug = category.get("path_slug") or category.get("slug")
    if not cid or not path_slug:
        return
    for page in range(max_pages):
        url = urljoin(community.base_url + "/", f"c/{path_slug}/{cid}/l/latest.json")
        data = _get_json(
            session,
            url,
            params={"page": page},
            request_delay=request_delay,
        )
        if not data:
            break
        topic_list = data.get("topic_list", {})
        topics = topic_list.get("topics", [])
        if not topics:
            break
        for topic in topics:
            yield topic, category
        if not topic_list.get("more_topics_url"):
            break


def _iter_topic_posts(
    session: requests.Session,
    community: DiscourseCommunity,
    topic_id: int,
    *,
    request_delay: float,
):
    topic_url = urljoin(community.base_url + "/", f"t/{topic_id}.json")
    data = _get_json(session, topic_url, request_delay=request_delay)
    if not data:
        return

    post_stream = data.get("post_stream", {})
    initial_posts: list[dict] = post_stream.get("posts", [])
    full_stream: list[int] = post_stream.get("stream", [])
    for post in initial_posts:
        yield post, data

    fetched = {post.get("id") for post in initial_posts}
    remaining = [pid for pid in full_stream if pid not in fetched]
    for start in range(0, len(remaining), _CHUNK_SIZE):
        chunk = remaining[start:start + _CHUNK_SIZE]
        qs = "&".join(f"post_ids[]={pid}" for pid in chunk)
        chunk_url = urljoin(community.base_url + "/", f"t/{topic_id}/posts.json?{qs}")
        chunk_data = _get_json(session, chunk_url, request_delay=request_delay)
        if not chunk_data:
            continue
        for post in chunk_data.get("post_stream", {}).get("posts", []):
            yield post, data


def collect_discourse_community(
    community: DiscourseCommunity,
    *,
    output: Path,
    cve_index_output: Path,
    request_delay: float = 1.0,
    max_pages: int = 200,
    max_topics: int | None = None,
) -> tuple[int, int]:
    output = Path(output)
    cve_index_output = Path(cve_index_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cve_index_output.parent.mkdir(parents=True, exist_ok=True)

    session = _make_session()
    category_data = _get_json(
        session,
        urljoin(community.base_url + "/", "categories.json"),
        request_delay=request_delay,
    )
    if not category_data:
        raise RuntimeError(f"Could not load categories for {community.label}")

    categories = _selected_categories(community, _flatten_categories(category_data))
    if not categories:
        raise RuntimeError(f"No selected categories for {community.label}")

    seen_topics: set[int] = set()
    seen_posts: set[str] = set()
    post_count = 0
    cve_count = 0

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        for category in categories:
            log.info(
                "%s: collecting category %s",
                community.label,
                category.get("path_slug") or category.get("slug"),
            )
            for topic, topic_category in _iter_category_topics(
                session,
                community,
                category,
                max_pages=max_pages,
                request_delay=request_delay,
            ):
                topic_id = topic.get("id")
                if not isinstance(topic_id, int) or topic_id in seen_topics:
                    continue
                topic_slug = str(topic.get("slug") or "")
                if topic_slug.startswith("about-the-") and topic_slug.endswith("-category"):
                    continue
                seen_topics.add(topic_id)
                if max_topics is not None and len(seen_topics) > max_topics:
                    break

                topic_title = topic.get("title") or ""
                topic_slug = topic_slug or str(topic_id)
                thread_url = urljoin(community.base_url + "/", f"t/{topic_slug}/{topic_id}")
                category_slug = topic_category.get("path_slug") or topic_category.get("slug")
                category_name = topic_category.get("name") or category_slug
                first_post = True

                for post_data, topic_data in _iter_topic_posts(
                    session,
                    community,
                    topic_id,
                    request_delay=request_delay,
                ):
                    raw_post_id = post_data.get("id")
                    if raw_post_id is None:
                        continue
                    pid = post_id(community.forum_id, str(raw_post_id))
                    if pid in seen_posts:
                        continue
                    seen_posts.add(pid)

                    body = _strip_html(post_data.get("cooked", ""))
                    if not body:
                        first_post = False
                        continue
                    text = f"{topic_title}\n\n{body}" if first_post and topic_title else body
                    first_post = False

                    created_at = post_data.get("created_at") or topic.get("created_at")
                    ts = parse_date(created_at) if created_at else datetime.now(timezone.utc)
                    username = post_data.get("username") or "unknown"
                    tags = topic_data.get("tags") or topic.get("tags") or []

                    record = {
                        "id": pid,
                        "text": text,
                        "timestamp": ts.isoformat(),
                        "forum_id": community.forum_id,
                        "author_hash": author_hash(community.forum_id, username),
                        "thread_url": thread_url,
                        "thread_title": topic_title,
                        "section": str(category_name),
                        "post_index": post_data.get("post_number"),
                        "post_id": str(raw_post_id),
                        "thread_id": str(topic_id),
                        "reply_count": max(int(topic_data.get("posts_count") or 1) - 1, 0),
                        "view_count": int(topic_data.get("views") or topic.get("views") or 0),
                        "tags": tags,
                        "scraped_at": datetime.now(timezone.utc).isoformat(),
                    }
                    post_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    post_count += 1

                    cves = sorted({match.upper() for match in _CVE_RE.findall(text)})
                    for cve in cves:
                        cve_fh.write(json.dumps({
                            "exploit_id": pid,
                            "cve_id": cve,
                            "exploit_text": text[:500],
                            "title": topic_title,
                            "published": ts.isoformat(),
                            "source": community.forum_id,
                        }, ensure_ascii=False) + "\n")
                        cve_count += 1

                if max_topics is not None and len(seen_topics) >= max_topics:
                    break
            if max_topics is not None and len(seen_topics) >= max_topics:
                break

    log.info(
        "%s: done - %d posts, %d CVE refs from %d topics",
        community.label,
        post_count,
        cve_count,
        len(seen_topics),
    )
    return post_count, cve_count


def collect_hackersploit(
    output: Path = Path("data/hackersploit_posts.jsonl"),
    cve_index_output: Path = Path("data/hackersploit_cve_index.jsonl"),
    request_delay: float = 1.0,
    max_pages: int = 200,
    max_topics: int | None = None,
) -> tuple[int, int]:
    return collect_discourse_community(
        COMMUNITIES["hackersploit"],
        output=output,
        cve_index_output=cve_index_output,
        request_delay=request_delay,
        max_pages=max_pages,
        max_topics=max_topics,
    )


def collect_parrotsec(
    output: Path = Path("data/parrotsec_posts.jsonl"),
    cve_index_output: Path = Path("data/parrotsec_cve_index.jsonl"),
    request_delay: float = 1.0,
    max_pages: int = 200,
    max_topics: int | None = None,
) -> tuple[int, int]:
    return collect_discourse_community(
        COMMUNITIES["parrotsec"],
        output=output,
        cve_index_output=cve_index_output,
        request_delay=request_delay,
        max_pages=max_pages,
        max_topics=max_topics,
    )


def collect_hackthebox(
    output: Path = Path("data/hackthebox_posts.jsonl"),
    cve_index_output: Path = Path("data/hackthebox_cve_index.jsonl"),
    request_delay: float = 1.0,
    max_pages: int = 200,
    max_topics: int | None = None,
) -> tuple[int, int]:
    return collect_discourse_community(
        COMMUNITIES["hackthebox"],
        output=output,
        cve_index_output=cve_index_output,
        request_delay=request_delay,
        max_pages=max_pages,
        max_topics=max_topics,
    )
