"""Shared helpers for the camera-ready audit runs: paths, provenance, result rows."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_release" / "benchmark_v2"
OUT = ROOT / "results" / "camera_ready"
RAW = OUT / "raw"
PERQ = OUT / "per_query"
for _d in (OUT, RAW, PERQ):
    _d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_T0 = time.time()


def log(msg: str) -> None:
    el = int(time.time() - _T0)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [+{el // 3600}h{el % 3600 // 60:02d}m] {msg}", flush=True)


def git_sha() -> str:
    if os.environ.get("AUDIT_GIT_SHA"):  # cluster copy: the worktree's .git link does not resolve there
        return os.environ["AUDIT_GIT_SHA"]
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src", "scripts", "audit"],
                           capture_output=True, text=True).stdout.strip()
    return sha + ("+audit-uncommitted" if dirty else "")


def data_hashes() -> dict:
    p = OUT / "data_integrity.json"
    files = json.loads(p.read_text())["files"]
    return {k: v["sha256"] for k, v in files.items()}


def model_revision(path: str) -> str | None:
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(path).sha
    except Exception as e:  # offline or gated
        return f"unknown ({type(e).__name__})"


def machine() -> str:
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip())
        return f"{platform.node()} {chip} {mem // 2**30} GB macOS {platform.mac_ver()[0]}"
    except Exception:
        return platform.platform()


def write_row(stream: str, **row) -> None:
    """Append one result row (with provenance) to raw/<stream>.jsonl immediately."""
    row.setdefault("git_sha", git_sha())
    row.setdefault("machine", machine())
    row["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(RAW / f"{stream}.jsonl", "a") as f:
        f.write(json.dumps(row, default=float) + "\n")
        f.flush()
        os.fsync(f.fileno())


def done_keys(stream: str, fields: tuple[str, ...]) -> set:
    """Keys of rows already written, so a restarted run skips finished work."""
    p = RAW / f"{stream}.jsonl"
    if not p.exists():
        return set()
    out = set()
    for line in open(p):
        r = json.loads(line)
        out.add(tuple(r.get(k) for k in fields))
    return out


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()
