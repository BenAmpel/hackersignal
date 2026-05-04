"""Extended data models for raw scraped posts.

`RawPost` is a superset of `ForumPost` that carries all harvested metadata.
Use `RawPost.to_forum_post()` to produce the minimal schema expected by the
rest of the ETG pipeline (id / text / timestamp / forum_id / author_hash).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

from etg.data.schemas import ForumPost

# ---------------------------------------------------------------------------
# Forum index entry (one row from forum.md)
# ---------------------------------------------------------------------------

_ONION_RE = re.compile(r"\.onion(/|$)")
_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


@dataclass
class ForumEntry:
    """One row from the deepdarkCTI forum.md index."""

    name: str
    url: str
    status: str  # "ONLINE", "OFFLINE", "OFFLINE (seized)", …

    @property
    def is_online(self) -> bool:
        return self.status.strip().upper().startswith("ONLINE")

    @property
    def is_onion(self) -> bool:
        return bool(_ONION_RE.search(self.url))

    @property
    def forum_id(self) -> str:
        """Normalised slug used as ForumPost.forum_id."""
        return re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")


# ---------------------------------------------------------------------------
# Raw post (full metadata, produced by scrapers)
# ---------------------------------------------------------------------------


@dataclass
class RawPost:
    """All metadata extracted for a single forum post / thread opening."""

    # ---- required fields ---------------------------------------------------
    forum_id: str           # from ForumEntry.forum_id
    thread_url: str         # canonical URL of the thread
    post_index: int         # 0 = opening post, 1+ = replies
    raw_author: str         # username as-scraped (NOT stored in output)
    body: str               # post body text (BBCode / HTML stripped)
    scraped_at: datetime    # UTC when we fetched the page

    # ---- optional fields (best-effort) -------------------------------------
    thread_title: str = ""
    section: str = ""           # subforum / board name
    post_id: str = ""           # per-forum post ID (numeric string)
    thread_id: str = ""         # per-forum thread ID (numeric string)
    reply_count: int = 0
    view_count: int = 0
    tags: list[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None   # UTC; falls back to scraped_at

    # ---- computed properties -----------------------------------------------

    @property
    def effective_timestamp(self) -> datetime:
        ts = self.timestamp or self.scraped_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts

    @property
    def cve_refs(self) -> list[str]:
        """CVE IDs mentioned anywhere in title + body."""
        return list({
            m.upper()
            for m in _CVE_RE.findall(self.thread_title + " " + self.body)
        })

    @property
    def unique_id(self) -> str:
        """Deterministic ID that survives re-crawls."""
        key = f"{self.forum_id}::{self.thread_url}::{self.post_index}"
        return hashlib.sha256(key.encode()).hexdigest()[:20]

    @property
    def author_hash(self) -> str:
        """SHA-256 of (forum_id + raw_author) — anonymised, forum-scoped."""
        key = f"{self.forum_id}::{self.raw_author}"
        return hashlib.sha256(key.encode()).hexdigest()

    def full_text(self) -> str:
        """Title prepended to body for the opening post; body only otherwise."""
        if self.post_index == 0 and self.thread_title:
            return f"{self.thread_title}\n\n{self.body}"
        return self.body

    # ---- schema conversion -------------------------------------------------

    def to_forum_post(self) -> ForumPost:
        """Return the minimal ForumPost consumed by the ETG pipeline."""
        return ForumPost(
            id=self.unique_id,
            text=self.full_text(),
            timestamp=self.effective_timestamp,
            forum_id=self.forum_id,
            author_hash=self.author_hash,
        )

    def to_dict(self) -> dict:
        """Full metadata dict for JSONL archiving."""
        d = {
            "id": self.unique_id,
            "forum_id": self.forum_id,
            "thread_url": self.thread_url,
            "thread_title": self.thread_title,
            "section": self.section,
            "post_index": self.post_index,
            "post_id": self.post_id,
            "thread_id": self.thread_id,
            "text": self.full_text(),
            "body": self.body,
            "timestamp": self.effective_timestamp.isoformat(),
            "scraped_at": self.scraped_at.isoformat(),
            "author_hash": self.author_hash,
            "reply_count": self.reply_count,
            "view_count": self.view_count,
            "tags": self.tags,
            "cve_refs": self.cve_refs,
        }
        return d
