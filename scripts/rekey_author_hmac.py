"""Re-key author pseudonyms in the public HackerSignal release with keyed HMAC-SHA-256 (rebuttal commitment C1).

Each non-empty author_hash h (an unkeyed hex digest of the source-namespaced author string) becomes
HMAC-SHA-256(key, "author_hash::" + h). Without the key, hashing candidate usernames no longer
reproduces a released value. unified_id took the old author_hash as an input, so it is recomputed
with the same formula over the re-keyed value; otherwise it would leave an unkeyed path back to the
author string. No other field changes. The key is read from a file outside the repository and is
never written to the output or the manifest.

Usage:
  python -I scripts/rekey_author_hmac.py --in <public.jsonl.gz> --out <new.jsonl.gz> \
      --key <key file, 32 bytes, mode 600> --key-id v1 --manifest <manifest.json>
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import hmac
import json
import os
import stat
import sys
import time


def unified_id(record: dict) -> str:
    # Same formula as etg.data.unify_hacker_communities._unified_id.
    parts = [record.get("source_dataset", ""), record.get("source_file", ""), record.get("source_record_id", ""),
             record.get("forum_id", ""), record.get("timestamp", ""), record.get("author_hash", "")]
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:24]


def load_key(path: str) -> bytes:
    st = os.stat(path)
    if stat.S_IMODE(st.st_mode) & 0o077:
        sys.exit(f"STOP: key file {path} is readable by group or others; chmod 600 it")
    key = open(path, "rb").read()
    if len(key) != 32:
        sys.exit(f"STOP: key must be 32 bytes, got {len(key)}")
    return key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--key-id", required=True)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists; refusing to overwrite")
    key = load_key(a.key)

    rows = rekeyed = empty = uid_formula_match = 0
    old_uids, new_uids = set(), set()
    sha_in, sha_out = hashlib.sha256(), hashlib.sha256()
    t0 = time.time()
    tmp = a.out + ".partial"
    with gzip.open(a.inp, "rb") as fin, gzip.open(tmp, "wb", compresslevel=6) as fout:
        for raw in fin:
            sha_in.update(raw)
            if not raw.strip():
                continue
            r = json.loads(raw)
            rows += 1
            old_uid = r.get("unified_id", "")
            uid_formula_match += old_uid == unified_id(r)
            old_uids.add(old_uid)
            h = r.get("author_hash") or ""
            if h:
                r["author_hash"] = hmac.new(key, ("author_hash::" + h).encode("utf-8"), hashlib.sha256).hexdigest()
                rekeyed += 1
            else:
                empty += 1
            r["unified_id"] = unified_id(r)
            new_uids.add(r["unified_id"])
            line = (json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8")
            sha_out.update(line)
            fout.write(line)
            if rows % 500000 == 0:
                print(f"{rows:,} rows  {time.time() - t0:.0f}s", flush=True)
    os.rename(tmp, a.out)
    man = {
        "commitment": "C1: keyed HMAC-SHA-256 author pseudonyms",
        "input": os.path.basename(a.inp), "output": os.path.basename(a.out), "key_id": a.key_id,
        "construction": 'author_hash = HMAC-SHA-256(key, "author_hash::" + previous unkeyed hex digest of the '
                        'source-namespaced author string, 64 or 56 hex characters depending on the collector); '
                        "unified_id recomputed with the release formula over the re-keyed author_hash",
        "rows": rows, "author_hash_rekeyed": rekeyed, "author_hash_empty_kept": empty,
        "old_unified_id_matched_formula": uid_formula_match,
        "unique_unified_id_old": len(old_uids), "unique_unified_id_new": len(new_uids),
        "input_jsonl_sha256": sha_in.hexdigest(), "output_jsonl_sha256": sha_out.hexdigest(),
        "seconds": round(time.time() - t0),
    }
    json.dump(man, open(a.manifest, "w"), indent=2)
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
