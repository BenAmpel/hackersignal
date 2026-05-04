"""Shared helpers for ETG data collectors."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)


def post_id(source: str, uid: str) -> str:
    """Return the first 20 hex characters of SHA-256('{source}::{uid}')."""
    raw = f"{source}::{uid}".encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def author_hash(source: str, author: str) -> str:
    """Return the full SHA-256 hex digest of '{source}::{author}'."""
    raw = f"{source}::{author}".encode()
    return hashlib.sha256(raw).hexdigest()


def parse_date(s: str) -> datetime:
    """Parse *s* into a timezone-aware UTC datetime.

    Tries, in order:
    1. ``datetime.fromisoformat``
    2. ``dateutil.parser.parse``
    3. Falls back to ``datetime.now(tz=timezone.utc)`` and logs a warning.

    Always returns a timezone-aware datetime in UTC.
    """
    if not s:
        return datetime.now(tz=timezone.utc)

    # --- attempt 1: stdlib ISO 8601 ---
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        pass

    # --- attempt 2: dateutil ---
    try:
        from dateutil import parser as du_parser  # type: ignore[import]

        dt = du_parser.parse(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    log.warning("parse_date: could not parse %r, using now(UTC)", s)
    return datetime.now(tz=timezone.utc)
