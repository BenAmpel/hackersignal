"""Collectors for the DTL-EL (Deep Transfer Learning – Exploit Labeling) HDF5 datasets.

These files come from prior research by Samtani et al. and contain:

- ``hackerExploits.h5``    — 2.5 GB; real hacker forum posts from multiple underground
                             forums.  The ``forum_title`` column drives the ``forum_id``.
- ``exploits.h5``           — 356 MB; ~96 k public exploit entries (ExploitDB + others)
                             with ``sourceData`` text and ``attackType`` labels.
- ``PublicExploits.h5``     — 293 MB; 24 843 ExploitDB entries with CVE IDs, CVSS scores,
                             ``attackType``, ``platform``, and full ``sourceData`` text.
- ``KaeliHackerExploits.h5``— 384 MB; three hacker forums in three HDF5 keys:
                               ``ao`` = AntiOnline  (452 775 rows)
                               ``c``  = CipherSpace ( 67 527 rows)
                               ``g4`` = Go4Expert   ( 60 694 rows)
- ``CVEHackerForum.h5``     — 1.7 MB; 81 annotated CVE-discussion posts from AntiChat.

Design principles
-----------------
* **Each forum is its own ``forum_id``** — no collector-name prefixes.
* **Within-file deduplication** — posts are fingerprinted on the first 400 chars of
  normalised text.  Exact/near-exact duplicate rows in the same file are skipped.
* **ExploitDB deduplication** — ``exploits.h5`` and ``PublicExploits.h5`` both contain
  ExploitDB entries identified by ``EDB-ID:NNNNN``.  If the caller supplies an
  ``existing_ids`` set (pre-loaded from ``exploitdb_posts.jsonl``) those entries are
  skipped, keeping the richer git-repo version.

All five functions follow the standard ETG collector contract:
    collect_*(output, ...) -> tuple[int, int]   (post_count, cve_index_count)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from etg.collectors._common import author_hash, parse_date, post_id
from etg.data.schemas import ForumPost

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_ONEDRIVE = Path.home() / "Library/CloudStorage/OneDrive-Personal/Academic Resources/Code"
_DTL_DATA = _ONEDRIVE / "DTL-EL/data"
_ATTACK_DATA = _ONEDRIVE / "ATTACK-Link/data"

DEFAULT_HACKER_EXPLOITS    = _DTL_DATA / "hackerExploits.h5"
DEFAULT_EXPLOITS           = _DTL_DATA / "exploits.h5"
DEFAULT_PUBLIC_EXPLOITS    = _DTL_DATA / "PublicExploits.h5"
DEFAULT_KAELI              = _DTL_DATA / "KaeliHackerExploits.h5"
DEFAULT_CVE_HACKER_FORUM   = _DTL_DATA / "CVEHackerForum.h5"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_str(val) -> str:
    if val is None:
        return ""
    try:
        import numpy as np
        if isinstance(val, float) and (val != val):
            return ""
        if isinstance(val, (bytes, bytearray)):
            return val.decode("utf-8", errors="replace")
    except ImportError:
        pass
    return str(val).strip()


def _parse_ts(val) -> datetime:
    s = _safe_str(val)
    if not s:
        return datetime.now(tz=timezone.utc)
    try:
        return parse_date(s)
    except Exception:
        return datetime.now(tz=timezone.utc)


def _extract_cves(text: str) -> list[str]:
    return list(dict.fromkeys(m.upper() for m in re.findall(r"CVE-\d{4}-\d+", text, re.I)))


def _text_fingerprint(text: str) -> str:
    """MD5 of the first 400 chars of lowercased, whitespace-collapsed text."""
    t = re.sub(r"\s+", " ", text[:400].lower()).strip()
    return hashlib.md5(t.encode()).hexdigest()


def _forum_id(raw: str) -> str:
    """Normalise a raw forum name into a clean ``forum_id``."""
    return re.sub(r"[^a-z0-9]+", "_", raw.lower().strip()).strip("_") or "unknown"


# ---------------------------------------------------------------------------
# 1. hackerExploits.h5  — the hacker community corpus
# ---------------------------------------------------------------------------

def collect_hacker_exploits(
    h5_path: Path = DEFAULT_HACKER_EXPLOITS,
    output: Path = Path("data/hacker_exploits_posts.jsonl"),
    cve_index_output: Path = Path("data/hacker_exploits_cve_index.jsonl"),
    exclude_forums: tuple[str, ...] = ("kernelmode",),
    min_text_len: int = 40,
    log_every: int = 10000,
) -> tuple[int, int]:
    """Convert ``hackerExploits.h5`` to ForumPost + CVE-index JSONL.

    ``forum_id`` is taken directly from the ``forum_title`` column — no prefix.
    Duplicate posts (same text fingerprint) within this file are skipped.

    Returns ``(post_count, cve_index_count)``.
    """
    h5_path = Path(h5_path)
    output = Path(output)
    cve_index_output = Path(cve_index_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cve_index_output.parent.mkdir(parents=True, exist_ok=True)

    if not h5_path.exists():
        log.error("hackerExploits: file not found: %s", h5_path)
        return 0, 0

    log.info("hackerExploits: loading %s …", h5_path)
    df = pd.read_hdf(str(h5_path), key="df")
    log.info("hackerExploits: loaded %d rows, cols=%s", len(df), list(df.columns))
    df.columns = [c.lower() for c in df.columns]

    seen_fps: set[str] = set()
    post_count = 0
    cve_index_count = 0
    skipped_dupe = 0

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        for idx, row in df.iterrows():
            try:
                raw_forum   = _safe_str(row.get("forum_title", ""))
                # Strip researcher-applied prefixes baked into forum_title values
                # e.g. "hacker_antichat" → "antichat", "hacker_hackforums" → "hackforums"
                raw_forum   = re.sub(r"^hacker[_\-\s]+", "", raw_forum, flags=re.I)
                fid         = _forum_id(raw_forum)
                if fid in exclude_forums:
                    continue

                username    = _safe_str(row.get("username", ""))
                post_date   = _safe_str(row.get("postdate", ""))
                title       = _safe_str(row.get("threadtitle", ""))
                content     = _safe_str(row.get("postcontent", ""))
                source_code = _safe_str(row.get("sourcecode", ""))
                exploit_type = _safe_str(row.get("dtlel", row.get("exploitlabel", "")))

                parts = []
                if title:
                    parts.append(title)
                if content:
                    parts.append(content)
                if source_code and source_code != content:
                    parts.append(source_code[:3000])
                text = "\n\n".join(parts).strip()

                if len(text) < min_text_len:
                    continue

                fp = _text_fingerprint(text)
                if fp in seen_fps:
                    skipped_dupe += 1
                    continue
                seen_fps.add(fp)

                ts  = _parse_ts(post_date)
                uid = hashlib.sha1(f"{fid}|{username}|{post_date}|{text[:100]}".encode()).hexdigest()[:20]

                post = ForumPost(
                    id=post_id(fid, uid),
                    text=text,
                    timestamp=ts,
                    forum_id=fid,
                    author_hash=author_hash(fid, username or "unknown"),
                )
                post_fh.write(json.dumps(post.to_dict()) + "\n")
                post_count += 1

                for cve in _extract_cves(text):
                    cve_fh.write(json.dumps({
                        "exploit_id": uid,
                        "cve_id": cve,
                        "exploit_text": text[:500],
                        "title": title,
                        "published": ts.isoformat(),
                        "source": fid,
                        "exploit_type": exploit_type,
                    }) + "\n")
                    cve_index_count += 1

                if post_count % log_every == 0:
                    post_fh.flush()
                    cve_fh.flush()
                    log.info(
                        "hackerExploits: %d posts / %d CVE entries (skipped_dupe=%d)",
                        post_count, cve_index_count, skipped_dupe,
                    )

            except Exception as exc:
                log.warning("hackerExploits: skipping row %s: %s", idx, exc)
                continue

    log.info(
        "hackerExploits: done. posts=%d, cve_index=%d, duplicates_skipped=%d",
        post_count, cve_index_count, skipped_dupe,
    )
    return post_count, cve_index_count


# ---------------------------------------------------------------------------
# 2. exploits.h5  — source-domain public exploits with attack-type labels
# ---------------------------------------------------------------------------

def collect_exploits(
    h5_path: Path = DEFAULT_EXPLOITS,
    output: Path = Path("data/dtl_exploits_posts.jsonl"),
    cve_index_output: Path = Path("data/dtl_exploits_cve_index.jsonl"),
    existing_edb_ids: set[str] | None = None,
    min_text_len: int = 40,
    log_every: int = 5000,
) -> tuple[int, int]:
    """Convert ``exploits.h5`` (~96 k rows) to ForumPost JSONL.

    ``forum_id`` is the ``type`` column value (e.g., ``exploitdb``).
    Rows whose ``id`` field matches an EDB-ID already in ``existing_edb_ids``
    are skipped to avoid duplicating data from the git-repo ExploitDB collector.
    Within-file deduplication is performed on text fingerprints.

    Returns ``(post_count, cve_index_count)``.
    """
    h5_path = Path(h5_path)
    output = Path(output)
    cve_index_output = Path(cve_index_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cve_index_output.parent.mkdir(parents=True, exist_ok=True)

    if not h5_path.exists():
        log.error("dtl_exploits: file not found: %s", h5_path)
        return 0, 0

    log.info("dtl_exploits: loading %s …", h5_path)
    df = pd.read_hdf(str(h5_path), key="df")
    log.info("dtl_exploits: loaded %d rows", len(df))
    df.columns = [c.lower() for c in df.columns]

    seen_fps: set[str] = set()
    post_count = 0
    cve_index_count = 0
    skipped_existing = 0
    skipped_dupe = 0

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        for idx, row in df.iterrows():
            try:
                exploit_id  = _safe_str(row.get("id", ""))
                source_data = _safe_str(row.get("sourcedata", ""))
                description = _safe_str(row.get("description", ""))
                title       = _safe_str(row.get("title", ""))
                cve         = _safe_str(row.get("cve", "")).upper()
                published   = _safe_str(row.get("published", ""))
                reporter    = _safe_str(row.get("reporter", ""))
                attack_type = _safe_str(row.get("attacktype", ""))
                platform    = _safe_str(row.get("platform", ""))
                cvss        = _safe_str(row.get("cvss", ""))
                src_type    = _forum_id(_safe_str(row.get("type", "exploitdb")))

                # Skip if the git-repo ExploitDB collector already has this entry
                if existing_edb_ids and exploit_id in existing_edb_ids:
                    skipped_existing += 1
                    continue

                parts = []
                if title:
                    parts.append(title)
                if description and description != title:
                    parts.append(description)
                if source_data and source_data not in (description, title):
                    parts.append(source_data)
                text = "\n\n".join(parts).strip()

                if len(text) < min_text_len:
                    continue

                fp = _text_fingerprint(text)
                if fp in seen_fps:
                    skipped_dupe += 1
                    continue
                seen_fps.add(fp)

                ts  = _parse_ts(published) if published else datetime.now(tz=timezone.utc)
                uid = hashlib.sha1((exploit_id or text[:200]).encode()).hexdigest()[:20]

                post = ForumPost(
                    id=post_id(src_type, uid),
                    text=text,
                    timestamp=ts,
                    forum_id=src_type,
                    author_hash=author_hash(src_type, reporter or "public"),
                )
                post_fh.write(json.dumps(post.to_dict()) + "\n")
                post_count += 1

                cves = ([cve] if cve and re.match(r"CVE-\d{4}-\d+", cve) else []) or _extract_cves(text)
                for c in cves:
                    cve_fh.write(json.dumps({
                        "exploit_id": exploit_id or uid,
                        "cve_id": c,
                        "exploit_text": text[:500],
                        "title": title,
                        "published": ts.isoformat(),
                        "source": src_type,
                        "exploit_type": attack_type,
                        "cvss": cvss,
                        "platform": platform,
                    }) + "\n")
                    cve_index_count += 1

                if post_count % log_every == 0:
                    post_fh.flush()
                    cve_fh.flush()
                    log.info("dtl_exploits: %d posts / %d CVE entries", post_count, cve_index_count)

            except Exception as exc:
                log.warning("dtl_exploits: skipping row %s: %s", idx, exc)
                continue

    log.info(
        "dtl_exploits: done. posts=%d, cve_index=%d, skipped_existing=%d, skipped_dupe=%d",
        post_count, cve_index_count, skipped_existing, skipped_dupe,
    )
    return post_count, cve_index_count


# ---------------------------------------------------------------------------
# 3. PublicExploits.h5
# ---------------------------------------------------------------------------

def collect_public_exploits(
    h5_path: Path = DEFAULT_PUBLIC_EXPLOITS,
    output: Path = Path("data/public_exploits_posts.jsonl"),
    cve_index_output: Path = Path("data/public_exploits_cve_index.jsonl"),
    existing_edb_ids: set[str] | None = None,
    min_text_len: int = 40,
) -> tuple[int, int]:
    """Convert ``PublicExploits.h5`` (24 843 ExploitDB entries) to JSONL.

    Same schema as ``exploits.h5``.  Rows already present in ``existing_edb_ids``
    are skipped.  ``forum_id`` is the ``type`` column (e.g., ``exploitdb``).

    Returns ``(post_count, cve_index_count)``.
    """
    h5_path = Path(h5_path)
    output = Path(output)
    cve_index_output = Path(cve_index_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cve_index_output.parent.mkdir(parents=True, exist_ok=True)

    if not h5_path.exists():
        log.error("public_exploits: file not found: %s", h5_path)
        return 0, 0

    log.info("public_exploits: loading %s …", h5_path)
    df = pd.read_hdf(str(h5_path), key="df")
    log.info("public_exploits: loaded %d rows", len(df))
    df.columns = [c.lower() for c in df.columns]

    seen_fps: set[str] = set()
    post_count = 0
    cve_index_count = 0
    skipped_existing = 0
    skipped_dupe = 0

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        for idx, row in df.iterrows():
            try:
                exploit_id  = _safe_str(row.get("id", ""))
                source_data = _safe_str(row.get("sourcedata", ""))
                description = _safe_str(row.get("description", ""))
                title       = _safe_str(row.get("title", ""))
                cve         = _safe_str(row.get("cve", "")).upper()
                published   = _safe_str(row.get("published", ""))
                reporter    = _safe_str(row.get("reporter", ""))
                attack_type = _safe_str(row.get("attacktype", ""))
                platform    = _safe_str(row.get("platform", ""))
                cvss        = _safe_str(row.get("cvss", ""))
                src_type    = _forum_id(_safe_str(row.get("type", "exploitdb")))

                if existing_edb_ids and exploit_id in existing_edb_ids:
                    skipped_existing += 1
                    continue

                parts = []
                if title:
                    parts.append(title)
                if description and description != title:
                    parts.append(description)
                if source_data and source_data not in (description, title):
                    parts.append(source_data)
                text = "\n\n".join(parts).strip()

                if len(text) < min_text_len:
                    continue

                fp = _text_fingerprint(text)
                if fp in seen_fps:
                    skipped_dupe += 1
                    continue
                seen_fps.add(fp)

                ts  = _parse_ts(published) if published else datetime.now(tz=timezone.utc)
                uid = hashlib.sha1((exploit_id or text[:200]).encode()).hexdigest()[:20]

                post = ForumPost(
                    id=post_id(src_type, uid),
                    text=text,
                    timestamp=ts,
                    forum_id=src_type,
                    author_hash=author_hash(src_type, reporter or "unknown"),
                )
                post_fh.write(json.dumps(post.to_dict()) + "\n")
                post_count += 1

                if cve and re.match(r"CVE-\d{4}-\d+", cve):
                    cve_fh.write(json.dumps({
                        "exploit_id": exploit_id or uid,
                        "cve_id": cve,
                        "exploit_text": text[:500],
                        "title": title,
                        "published": ts.isoformat(),
                        "source": src_type,
                        "exploit_type": attack_type,
                        "cvss": cvss,
                        "platform": platform,
                    }) + "\n")
                    cve_index_count += 1

            except Exception as exc:
                log.warning("public_exploits: skipping row %s: %s", idx, exc)
                continue

    log.info(
        "public_exploits: done. posts=%d, cve_index=%d, skipped_existing=%d, skipped_dupe=%d",
        post_count, cve_index_count, skipped_existing, skipped_dupe,
    )
    return post_count, cve_index_count


# ---------------------------------------------------------------------------
# 4. KaeliHackerExploits.h5  — three forums, three HDF5 keys
# ---------------------------------------------------------------------------

# Maps HDF5 key → canonical forum_id
_KAELI_KEY_FORUM = {
    "ao": "antionline",
    "c":  "cipherspace",
    "g4": "go4expert",
}


def collect_kaeli(
    h5_path: Path = DEFAULT_KAELI,
    output: Path = Path("data/kaeli_hacker_posts.jsonl"),
    cve_index_output: Path = Path("data/kaeli_hacker_cve_index.jsonl"),
    min_text_len: int = 40,
    log_every: int = 10000,
) -> tuple[int, int]:
    """Convert ``KaeliHackerExploits.h5`` (three forum keys) to ForumPost JSONL.

    HDF5 keys and their forums:
      ``ao`` → ``antionline``   (452 775 rows)
      ``c``  → ``cipherspace``  ( 67 527 rows)
      ``g4`` → ``go4expert``    ( 60 694 rows)

    Each forum gets its own ``forum_id`` directly (no ``kaeli_`` prefix).
    Within-file deduplication is applied per-forum.

    Returns ``(post_count, cve_index_count)``.
    """
    import h5py as _h5

    h5_path = Path(h5_path)
    output = Path(output)
    cve_index_output = Path(cve_index_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cve_index_output.parent.mkdir(parents=True, exist_ok=True)

    if not h5_path.exists():
        log.error("kaeli: file not found: %s", h5_path)
        return 0, 0

    with _h5.File(str(h5_path), "r") as f:
        keys = list(f.keys())
    log.info("kaeli: found keys %s", keys)

    post_count = 0
    cve_index_count = 0

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        for key in keys:
            fid = _KAELI_KEY_FORUM.get(key, _forum_id(key))
            log.info("kaeli: loading key=%r → forum_id=%r …", key, fid)

            df = pd.read_hdf(str(h5_path), key=key)
            df.columns = [c.lower() for c in df.columns]
            log.info("kaeli[%s]: %d rows, cols=%s", key, len(df), list(df.columns))

            text_cols = [c for c in ("threadtitle", "postcontent", "sourcecode") if c in df.columns]
            date_col  = next((c for c in ("postdate", "published") if c in df.columns), None)
            user_col  = next((c for c in ("username", "reporter") if c in df.columns), None)

            seen_fps: set[str] = set()
            skipped_dupe = 0
            key_count = 0

            for idx, row in df.iterrows():
                try:
                    parts: list[str] = []
                    for c in text_cols:
                        v = _safe_str(row.get(c, ""))
                        if v and v not in parts:
                            parts.append(v)
                    text = "\n\n".join(parts).strip()

                    if len(text) < min_text_len:
                        continue

                    fp = _text_fingerprint(text)
                    if fp in seen_fps:
                        skipped_dupe += 1
                        continue
                    seen_fps.add(fp)

                    ts       = _parse_ts(row.get(date_col) if date_col else None)
                    username = _safe_str(row.get(user_col, "")) if user_col else ""
                    title    = _safe_str(row.get("threadtitle", ""))
                    uid      = hashlib.sha1(f"{fid}|{username}|{text[:100]}".encode()).hexdigest()[:20]

                    post = ForumPost(
                        id=post_id(fid, uid),
                        text=text,
                        timestamp=ts,
                        forum_id=fid,
                        author_hash=author_hash(fid, username or "unknown"),
                    )
                    post_fh.write(json.dumps(post.to_dict()) + "\n")
                    post_count += 1
                    key_count  += 1

                    for cve in _extract_cves(text):
                        cve_fh.write(json.dumps({
                            "exploit_id": uid,
                            "cve_id": cve,
                            "exploit_text": text[:500],
                            "title": title,
                            "published": ts.isoformat(),
                            "source": fid,
                        }) + "\n")
                        cve_index_count += 1

                    if post_count % log_every == 0:
                        post_fh.flush()
                        cve_fh.flush()
                        log.info("kaeli: %d posts total / %d CVE entries", post_count, cve_index_count)

                except Exception as exc:
                    log.warning("kaeli[%s]: skipping row %s: %s", key, idx, exc)
                    continue

            log.info("kaeli[%s] (%s): %d posts written, %d dupes skipped",
                     key, fid, key_count, skipped_dupe)

    log.info("kaeli: all keys done. posts=%d, cve_index=%d", post_count, cve_index_count)
    return post_count, cve_index_count


# ---------------------------------------------------------------------------
# 5. CVEHackerForum.h5  — 81 annotated AntiChat CVE posts
# ---------------------------------------------------------------------------

def collect_cve_hacker_forum(
    h5_path: Path = DEFAULT_CVE_HACKER_FORUM,
    output: Path = Path("data/cve_hacker_forum_posts.jsonl"),
    cve_index_output: Path = Path("data/cve_hacker_forum_cve_index.jsonl"),
) -> tuple[int, int]:
    """Convert ``CVEHackerForum.h5`` (81 annotated CVE posts) to JSONL.

    ``forum_id`` is taken from the ``forum_title`` column (e.g., ``antichat``).

    Returns ``(post_count, cve_index_count)``.
    """
    h5_path = Path(h5_path)
    output = Path(output)
    cve_index_output = Path(cve_index_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cve_index_output.parent.mkdir(parents=True, exist_ok=True)

    if not h5_path.exists():
        log.error("cve_hacker_forum: file not found: %s", h5_path)
        return 0, 0

    log.info("cve_hacker_forum: loading %s …", h5_path)
    df = pd.read_hdf(str(h5_path), key="df")
    log.info("cve_hacker_forum: loaded %d rows", len(df))
    df.columns = [c.lower() for c in df.columns]

    seen_fps: set[str] = set()
    post_count = 0
    cve_index_count = 0

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        for idx, row in df.iterrows():
            try:
                raw_f    = re.sub(r"^hacker[_\-\s]+", "", _safe_str(row.get("forum_title", "antichat")), flags=re.I)
                fid      = _forum_id(raw_f)
                username = _safe_str(row.get("username", ""))
                post_date = _safe_str(row.get("postdate", ""))
                title    = _safe_str(row.get("threadtitle", ""))
                content  = _safe_str(row.get("postcontent", ""))
                src      = _safe_str(row.get("sourcecode", ""))

                parts = [p for p in (title, content, src) if p]
                text = "\n\n".join(parts).strip()
                if len(text) < 20:
                    continue

                fp = _text_fingerprint(text)
                if fp in seen_fps:
                    continue
                seen_fps.add(fp)

                ts  = _parse_ts(post_date)
                uid = hashlib.sha1(f"{fid}|{username}|{post_date}|{text[:80]}".encode()).hexdigest()[:20]

                post = ForumPost(
                    id=post_id(fid, uid),
                    text=text,
                    timestamp=ts,
                    forum_id=fid,
                    author_hash=author_hash(fid, username or "unknown"),
                )
                post_fh.write(json.dumps(post.to_dict()) + "\n")
                post_count += 1

                for cve in _extract_cves(text):
                    cve_fh.write(json.dumps({
                        "exploit_id": uid,
                        "cve_id": cve,
                        "exploit_text": text[:500],
                        "title": title,
                        "published": ts.isoformat(),
                        "source": fid,
                    }) + "\n")
                    cve_index_count += 1

            except Exception as exc:
                log.warning("cve_hacker_forum: skipping row %s: %s", idx, exc)
                continue

    log.info("cve_hacker_forum: done. posts=%d, cve_index=%d", post_count, cve_index_count)
    return post_count, cve_index_count


# ---------------------------------------------------------------------------
# Convenience: run all five collectors
# ---------------------------------------------------------------------------

def collect_all(
    output_dir: Path = Path("data"),
    hacker_exploits_h5: Path = DEFAULT_HACKER_EXPLOITS,
    exploits_h5: Path = DEFAULT_EXPLOITS,
    public_exploits_h5: Path = DEFAULT_PUBLIC_EXPLOITS,
    kaeli_h5: Path = DEFAULT_KAELI,
    cve_hacker_forum_h5: Path = DEFAULT_CVE_HACKER_FORUM,
    load_existing_edb_ids: bool = True,
) -> dict[str, tuple[int, int]]:
    """Run all five DTL-EL collectors and return a results dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing ExploitDB IDs for deduplication
    existing_edb_ids: set[str] | None = None
    if load_existing_edb_ids:
        edb_path = output_dir / "exploitdb_posts.jsonl"
        if edb_path.exists():
            existing_edb_ids = set()
            with edb_path.open() as f:
                for line in f:
                    d = json.loads(line)
                    existing_edb_ids.add(d.get("id", ""))
            log.info("Loaded %d existing ExploitDB IDs for dedup", len(existing_edb_ids))

    results: dict[str, tuple[int, int]] = {}

    results["hacker_exploits"] = collect_hacker_exploits(
        h5_path=hacker_exploits_h5,
        output=output_dir / "hacker_exploits_posts.jsonl",
        cve_index_output=output_dir / "hacker_exploits_cve_index.jsonl",
    )
    results["exploits"] = collect_exploits(
        h5_path=exploits_h5,
        output=output_dir / "dtl_exploits_posts.jsonl",
        cve_index_output=output_dir / "dtl_exploits_cve_index.jsonl",
        existing_edb_ids=existing_edb_ids,
    )
    results["public_exploits"] = collect_public_exploits(
        h5_path=public_exploits_h5,
        output=output_dir / "public_exploits_posts.jsonl",
        cve_index_output=output_dir / "public_exploits_cve_index.jsonl",
        existing_edb_ids=existing_edb_ids,
    )
    results["kaeli"] = collect_kaeli(
        h5_path=kaeli_h5,
        output=output_dir / "kaeli_hacker_posts.jsonl",
        cve_index_output=output_dir / "kaeli_hacker_cve_index.jsonl",
    )
    results["cve_hacker_forum"] = collect_cve_hacker_forum(
        h5_path=cve_hacker_forum_h5,
        output=output_dir / "cve_hacker_forum_posts.jsonl",
        cve_index_output=output_dir / "cve_hacker_forum_cve_index.jsonl",
    )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = collect_all()
    for name, (posts, cves) in results.items():
        print(f"{name:<25} posts={posts:>8,}  cve_index={cves:>6,}")
