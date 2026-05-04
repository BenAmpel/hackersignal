"""Parse the deepdarkCTI forum.md index into ForumEntry objects.

Sources:
  - Remote: GitHub raw content (default)
  - Local: a path to a downloaded forum.md file
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import urlopen

from .models import ForumEntry

# Official raw URL for the deepdarkCTI forum index
_FORUM_MD_URL = (
    "https://raw.githubusercontent.com/fastfire/deepdarkCTI/main/forum.md"
)

# Extra publicly-accessible CTI forums not listed in the deepdarkCTI index.
# All are clearnet and currently ONLINE.
_EXTRA_CLEARNET_FORUMS = [
    ForumEntry(name="0x00sec", url="https://0x00sec.org", status="ONLINE"),
    ForumEntry(name="HackForums", url="https://hackforums.net", status="ONLINE"),
    ForumEntry(name="Sinister.ly", url="https://sinister.ly", status="ONLINE"),
]

# Matches rows like:
#   |[NAME](URL)| STATUS | optional description |
#   |[NAME](URL)|STATUS||
_ROW_RE = re.compile(
    r"^\|\s*\[(?P<name>[^\]]+)\]\((?P<url>[^)]+)\)\s*\|"
    r"\s*(?P<status>[^|]+?)\s*\|",
    re.MULTILINE,
)


def _parse_markdown(text: str) -> list[ForumEntry]:
    entries: list[ForumEntry] = []
    for m in _ROW_RE.finditer(text):
        name = m.group("name").strip()
        url = m.group("url").strip()
        status = m.group("status").strip()
        if name and url:
            entries.append(ForumEntry(name=name, url=url, status=status))
    return entries


def _normalised_host(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host.removeprefix("www.")


def _dedupe_entries(entries: Iterable[ForumEntry]) -> list[ForumEntry]:
    """Prefer the first occurrence of each forum_id/host pair.

    deepdarkCTI sometimes carries multiple aliases for the same community, and
    we also append a few curated extras.  Keeping one representative entry per
    normalized host prevents duplicate crawls and inflated coverage counts.
    """
    seen_ids: set[str] = set()
    seen_hosts: set[str] = set()
    deduped: list[ForumEntry] = []
    for entry in entries:
        host = _normalised_host(entry.url)
        if entry.forum_id in seen_ids or host in seen_hosts:
            continue
        deduped.append(entry)
        seen_ids.add(entry.forum_id)
        seen_hosts.add(host)
    return deduped


def load_index(source: str | Path | None = None) -> list[ForumEntry]:
    """Return all ForumEntry objects from forum.md.

    Parameters
    ----------
    source:
        - ``None`` (default) — fetch the live GitHub raw file
        - a ``str`` URL  — fetch from that URL
        - a ``Path``     — read from disk
    """
    if source is None or isinstance(source, str):
        url = source or _FORUM_MD_URL
        with urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    else:
        text = Path(source).read_text(encoding="utf-8", errors="replace")

    return _parse_markdown(text)


def extra_clearnet_entries() -> list[ForumEntry]:
    """Return the extra clearnet CTI forums not in the deepdarkCTI index."""
    return list(_EXTRA_CLEARNET_FORUMS)


def online_entries(source: str | Path | None = None, extra: bool = True) -> list[ForumEntry]:
    """Return only ONLINE clearnet or onion entries.

    Parameters
    ----------
    source:
        Passed through to :func:`load_index`.
    extra:
        When ``True`` (default), append :data:`_EXTRA_CLEARNET_FORUMS` to the
        returned list.  All extra entries are clearnet and ONLINE so no
        additional filtering is needed.
    """
    entries = [e for e in load_index(source) if e.is_online]
    if extra:
        entries = entries + list(_EXTRA_CLEARNET_FORUMS)
    return _dedupe_entries(entries)


def clearnet_entries(source: str | Path | None = None) -> list[ForumEntry]:
    """Return only ONLINE clearnet (non-.onion) entries."""
    return [e for e in online_entries(source) if not e.is_onion]


def onion_entries(source: str | Path | None = None) -> list[ForumEntry]:
    """Return only ONLINE .onion entries (requires Tor)."""
    return [e for e in online_entries(source) if e.is_onion]
