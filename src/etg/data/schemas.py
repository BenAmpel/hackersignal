"""Canonical data schemas. Synthetic and real loaders both yield these."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Protocol


@dataclass(frozen=True)
class ForumPost:
    id: str
    text: str
    timestamp: datetime  # timezone-aware UTC
    forum_id: str
    author_hash: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ForumPost":
        ts = datetime.fromisoformat(d["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        post_id = d.get("id") or d.get("unified_id")
        if post_id is None:
            raise KeyError("ForumPost requires 'id' or 'unified_id'")
        return cls(
            id=post_id,
            text=d["text"],
            timestamp=ts,
            forum_id=d["forum_id"],
            author_hash=d["author_hash"],
        )


@dataclass(frozen=True)
class EVPair:
    """(exploit, vulnerability) training sample."""

    exploit_text: str
    vulnerability_text: str
    cve_id: str | None
    timestamp: datetime
    label: int  # 1 positive, 0 negative/distractor

    def to_dict(self) -> dict:
        return {
            "exploit_text": self.exploit_text,
            "vulnerability_text": self.vulnerability_text,
            "cve_id": self.cve_id,
            "timestamp": self.timestamp.isoformat(),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EVPair":
        ts = datetime.fromisoformat(d["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return cls(
            exploit_text=d["exploit_text"],
            vulnerability_text=d["vulnerability_text"],
            cve_id=d.get("cve_id"),
            timestamp=ts,
            label=int(d["label"]),
        )


class ForumLoader(Protocol):
    def __iter__(self) -> Iterator[ForumPost]: ...


class EVLoader(Protocol):
    def __iter__(self) -> Iterator[EVPair]: ...


# ---- JSONL round-trip helpers ----


def dump_jsonl(items: Iterable, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict()) + "\n")


def load_posts_jsonl(path: str | Path) -> list[ForumPost]:
    out: list[ForumPost] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            out.append(ForumPost.from_dict(json.loads(line)))
    return out


def load_ev_jsonl(path: str | Path) -> list[EVPair]:
    out: list[EVPair] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            out.append(EVPair.from_dict(json.loads(line)))
    return out
