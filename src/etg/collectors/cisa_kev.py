"""Collect the CISA Known Exploited Vulnerabilities (KEV) catalog.

Downloads the single-file JSON catalog from CISA and emits:
- A ForumPost JSONL (one post per KEV entry, rich text with description + action).
- A CVE index JSONL mapping exploit IDs to CVE identifiers.

The KEV catalog currently contains ~1,500+ CVEs confirmed to have been
actively exploited in the wild — the most high-value positive examples for
the exploit–vulnerability linker.
"""

from __future__ import annotations

import json
import logging
from datetime import timezone
from pathlib import Path

import requests

from etg.collectors._common import author_hash, parse_date, post_id
from etg.data.schemas import ForumPost

log = logging.getLogger(__name__)

FORUM_ID = "cisa_kev"
KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


def collect(
    output: Path = Path("data/cisa_kev_posts.jsonl"),
    cve_index_output: Path = Path("data/cisa_kev_cve_index.jsonl"),
) -> tuple[int, int]:
    """Download and parse the CISA KEV catalog.

    Returns ``(post_count, cve_index_count)``.
    """
    output = Path(output)
    cve_index_output = Path(cve_index_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cve_index_output.parent.mkdir(parents=True, exist_ok=True)

    log.info("CISA KEV: downloading catalog from %s", KEV_URL)
    resp = requests.get(KEV_URL, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    vulnerabilities = data.get("vulnerabilities", [])
    log.info("CISA KEV: loaded %d entries", len(vulnerabilities))

    post_count = 0
    cve_index_count = 0

    with (
        output.open("w", encoding="utf-8") as post_fh,
        cve_index_output.open("w", encoding="utf-8") as cve_fh,
    ):
        for vuln in vulnerabilities:
            try:
                cve_id: str = vuln.get("cveID", "").upper()
                if not cve_id:
                    continue

                vendor: str = vuln.get("vendorProject", "")
                product: str = vuln.get("product", "")
                name: str = vuln.get("vulnerabilityName", "")
                short_desc: str = vuln.get("shortDescription", "")
                required_action: str = vuln.get("requiredAction", "")
                ransomware: str = vuln.get("knownRansomwareCampaignUse", "")
                cwes: list = vuln.get("cwes", [])
                date_added: str = vuln.get("dateAdded", "")

                # Build rich text
                parts: list[str] = []
                if name:
                    parts.append(name)
                if vendor or product:
                    parts.append(f"Affected: {vendor} {product}".strip())
                if short_desc:
                    parts.append(short_desc)
                if required_action:
                    parts.append(f"Required Action: {required_action}")
                if ransomware and ransomware.lower() not in ("unknown", "no"):
                    parts.append(f"Ransomware: {ransomware}")
                if cwes:
                    parts.append(f"CWE(s): {', '.join(cwes)}")
                parts.append("[Source: CISA Known Exploited Vulnerabilities Catalog]")

                text = "\n\n".join(parts)

                ts = parse_date(date_added) if date_added else __import__("datetime").datetime.now(tz=timezone.utc)

                post = ForumPost(
                    id=post_id("cisa_kev", cve_id),
                    text=text,
                    timestamp=ts,
                    forum_id=FORUM_ID,
                    author_hash=author_hash("cisa_kev", "cisa"),
                )
                post_fh.write(json.dumps(post.to_dict()) + "\n")
                post_count += 1

                cve_entry = {
                    "exploit_id": cve_id,
                    "cve_id": cve_id,
                    "exploit_text": text[:500],
                    "title": name,
                    "published": ts.isoformat(),
                    "source": "cisa_kev",
                    "ransomware": ransomware,
                }
                cve_fh.write(json.dumps(cve_entry) + "\n")
                cve_index_count += 1

            except Exception as exc:
                log.warning("CISA KEV: skipping entry %s: %s", vuln.get("cveID"), exc)
                continue

    log.info(
        "CISA KEV: done. posts=%d, cve_index=%d", post_count, cve_index_count
    )
    return post_count, cve_index_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collect()
