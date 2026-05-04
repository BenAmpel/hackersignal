"""Importer for the CVEfixes SQLite database.

Data source: SQLite database discovered via glob at ``data/external/zenodo/**/*.db``
(canonical location: ``data/external/zenodo/CVEfixes_v1.0.7/CVEfixes.db``)

CVEfixes links CVEs to the open-source repository commits that fixed them.
Each row in the join combines:

- ``fixes``: mapping from CVE ID to commit hash + repository URL
- ``commits``: the commit message associated with a hash
- ``cve``: the NVD CVE record (description, published date, CVSS score)

The combined text for each post is the CVE description together with the fix
commit message, providing both the vulnerability context and the remediation.

Because every row contains a known CVE ID, every post also generates a CVE
index entry.

Outputs
-------
- Forum posts JSONL: one record per (CVE, commit) fix pair.
- CVE index JSONL: one record per (CVE, commit) pair — all rows included.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from etg.collectors._common import author_hash, parse_date, post_id

log = logging.getLogger(__name__)

FORUM_ID = "cvefixes"

_QUERY = """
    SELECT f.cve_id,
           f.hash        AS commit_hash,
           f.repo_url,
           c.msg         AS commit_message,
           c.author      AS commit_author,
           cv.description,
           cv.published_date
    FROM   fixes f
    JOIN   commits c  ON f.hash     = c.hash
                     AND f.repo_url = c.repo_url
    JOIN   cve    cv  ON f.cve_id   = cv.cve_id
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect(
    data_dir: Path = Path("data/external/zenodo"),
    output: Path = Path("data/cvefixes_posts.jsonl"),
    cve_index_output: Path = Path("data/cvefixes_cve_index.jsonl"),
    log_every: int = 500,
) -> tuple[int, int]:
    """Import CVEfixes database and write ForumPost + CVE-index JSONL.

    Parameters
    ----------
    data_dir:
        Root directory to search for the CVEfixes ``.db`` file using
        ``rglob("*.db")``.  The canonical path is
        ``data/external/zenodo/CVEfixes_v1.0.7/CVEfixes.db``.
    output:
        Destination JSONL path for forum post records.
    cve_index_output:
        Destination JSONL path for the CVE reference index.
    log_every:
        Log a progress message every *log_every* rows processed.

    Returns
    -------
    tuple[int, int]
        ``(post_count, cve_index_count)``
    """
    data_dir = Path(data_dir)
    output = Path(output)
    cve_index_output = Path(cve_index_output)

    # --- Locate the database file ---
    db_files = list(data_dir.rglob("*.db"))
    if not db_files:
        log.error("cvefixes: no .db file found under %s", data_dir)
        return 0, 0

    db_path = db_files[0]
    log.info("cvefixes: using database at %s", db_path)
    if len(db_files) > 1:
        log.warning(
            "cvefixes: multiple .db files found; using first: %s",
            db_path,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    cve_index_output.parent.mkdir(parents=True, exist_ok=True)

    post_count = 0
    cve_index_count = 0

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute(_QUERY)

        with (
            output.open("w", encoding="utf-8") as post_fh,
            cve_index_output.open("w", encoding="utf-8") as cve_fh,
        ):
            for row in cursor:
                cve_id = str(row["cve_id"] or "").strip()
                commit_hash = str(row["commit_hash"] or "").strip()
                repo_url = str(row["repo_url"] or "").strip()
                commit_message = str(row["commit_message"] or "").strip()
                commit_author = str(row["commit_author"] or "").strip()
                description = str(row["description"] or "").strip()
                published_date = str(row["published_date"] or "").strip()

                # Skip rows with no meaningful text content
                if not description and not commit_message:
                    continue

                # --- Composed text ---
                text = f"CVE: {cve_id}\n\n{description}\n\nFix commit: {commit_message}"

                # --- IDs ---
                pid = post_id(FORUM_ID, f"{cve_id}::{commit_hash}")
                # Use git author as author identity, fall back to repo_url
                author_id = commit_author or repo_url or "unknown"
                ahash = author_hash(FORUM_ID, author_id)

                # --- Timestamp ---
                ts = parse_date(published_date)

                # --- Write post record ---
                post_dict = {
                    "id": pid,
                    "text": text,
                    "timestamp": ts.isoformat(),
                    "forum_id": FORUM_ID,
                    "author_hash": ahash,
                }
                post_fh.write(json.dumps(post_dict) + "\n")
                post_count += 1

                # --- CVE index: always emit since cve_id is always known ---
                entry = {
                    "exploit_id": pid,
                    "cve_id": cve_id.upper(),
                    "exploit_text": text[:500],
                    "title": cve_id,
                    "published": ts.isoformat(),
                    "source": FORUM_ID,
                }
                cve_fh.write(json.dumps(entry) + "\n")
                cve_index_count += 1

                if post_count % log_every == 0:
                    log.info(
                        "cvefixes: wrote %d posts / %d CVE entries so far",
                        post_count,
                        cve_index_count,
                    )
                    post_fh.flush()
                    cve_fh.flush()

    finally:
        conn.close()

    log.info(
        "cvefixes: done — %d posts, %d CVE index entries → %s",
        post_count,
        cve_index_count,
        output,
    )
    return post_count, cve_index_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collect()
