"""Collect exploit records from a local 0day.today HDF5 archive.

Reads the ``0DayData.h5`` file (key ``zdt``) — a 19k-record export from
0day.today — and emits:
- A ForumPost JSONL file (title + description + source text).
- A CVE index JSONL file mapping exploit IDs to CVE identifiers.

The H5 schema has columns::

    sourceData, id, CVE, title, published, reporter, score,
    description, href, attackType, platform, riskScore

``sourceData`` holds the full exploit text / PoC source code.
``description`` holds a short one-liner description (always present).
``CVE`` is a CVE ID or None for ~75 % of records.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from etg.collectors._common import author_hash, parse_date, post_id
from etg.data.schemas import ForumPost

log = logging.getLogger(__name__)

FORUM_ID = "zeroday_today"

# CVE regex — handles comma- / space-separated lists in the CVE column
_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


def _extract_cves(raw: object) -> list[str]:
    """Return a deduplicated list of CVE IDs found in *raw*."""
    if not raw or (isinstance(raw, float) and raw != raw):  # NaN
        return []
    return list(dict.fromkeys(_CVE_RE.findall(str(raw))))


def collect(
    h5_path: Path = (
        Path.home()
        / "Library/CloudStorage/OneDrive-Personal/Academic Resources/Code/"
          "Exploit Source Code/0DayData.h5"
    ),
    output: Path = Path("data/zeroday_posts.jsonl"),
    cve_index_output: Path = Path("data/zeroday_cve_index.jsonl"),
    log_every: int = 2000,
) -> tuple[int, int]:
    """Read *h5_path* and write ForumPost + CVE-index JSONL files.

    Parameters
    ----------
    h5_path:
        Local path to the ``0DayData.h5`` file.
    output:
        Destination JSONL for ForumPost records.
    cve_index_output:
        Destination JSONL mapping exploit IDs → CVE IDs.
    log_every:
        Emit a progress log every *log_every* records written.

    Returns
    -------
    tuple[int, int]
        ``(post_count, cve_index_count)`` — where *cve_index_count* counts
        rows written to the CVE index (one row per CVE reference).
    """
    try:
        import pandas as pd  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("pandas is required: pip install pandas tables") from exc

    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"0DayData.h5 not found: {h5_path}")

    output = Path(output)
    cve_index_output = Path(cve_index_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cve_index_output.parent.mkdir(parents=True, exist_ok=True)

    log.info("0day.today: loading %s …", h5_path)
    df = pd.read_hdf(str(h5_path), key="zdt")
    log.info("0day.today: %d records loaded", len(df))

    post_count = 0
    cve_index_count = 0

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        for _, row in df.iterrows():
            try:
                exploit_id: str = str(row.get("id", "")).strip() or f"zdt-{_}id"
                title: str = str(row.get("title") or "").strip()
                description: str = str(row.get("description") or "").strip()
                source_data: str = str(row.get("sourceData") or "").strip()
                published_raw = row.get("published")
                reporter: str = str(row.get("reporter") or "unknown").strip()
                cve_raw = row.get("CVE")
                attack_type: str = str(row.get("attackType") or "").strip()
                platform: str = str(row.get("platform") or "").strip()

                # Build the ForumPost text
                parts = []
                if title:
                    parts.append(title)
                if description and description != title:
                    parts.append(description)
                if attack_type or platform:
                    meta = " | ".join(filter(None, [attack_type, platform]))
                    parts.append(f"[{meta}]")
                if source_data and source_data not in (title, description):
                    parts.append(source_data)
                text = "\n\n".join(parts)[:4000]

                # Parse date
                if hasattr(published_raw, "to_pydatetime"):
                    published_dt = published_raw.to_pydatetime()
                    if published_dt.tzinfo is None:
                        published_dt = published_dt.replace(tzinfo=timezone.utc)
                else:
                    published_dt = parse_date(str(published_raw) if published_raw else "")

                post = ForumPost(
                    id=post_id("zeroday", exploit_id),
                    text=text,
                    timestamp=published_dt,
                    forum_id=FORUM_ID,
                    author_hash=author_hash("zeroday", reporter),
                )
                post_fh.write(json.dumps(post.to_dict()) + "\n")
                post_count += 1

                # CVE index entries
                cves = _extract_cves(cve_raw)
                for cve in cves:
                    entry = {
                        "exploit_id": exploit_id,
                        "cve_id": cve.upper(),
                        "title": title,
                        "published": published_dt.isoformat(),
                        "source": "zeroday_today",
                    }
                    cve_fh.write(json.dumps(entry) + "\n")
                    cve_index_count += 1

                if post_count % log_every == 0:
                    log.info(
                        "0day.today: written %d posts / %d CVE index entries",
                        post_count,
                        cve_index_count,
                    )

            except Exception as exc:
                log.warning("0day.today: skipping malformed row: %s", exc)
                continue

    log.info(
        "0day.today: done. posts=%d, cve_index=%d", post_count, cve_index_count
    )
    return post_count, cve_index_count
