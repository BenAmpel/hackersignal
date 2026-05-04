"""Stub for loading real exploit–vulnerability pairs.

Expected format: JSONL with one `EVPair` per line (keys: exploit_text,
vulnerability_text, cve_id, timestamp, label). Pull exploits from
ExploitDB and vulnerabilities from NVD, construct positives from explicit
CVE references in exploit metadata, sample negatives as needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .schemas import EVPair, load_ev_jsonl


class RealEV:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Real EV data not found: {self.path}. "
                "Point `path` to a JSONL file with one EVPair per line."
            )

    def __iter__(self) -> Iterator[EVPair]:
        yield from load_ev_jsonl(self.path)
