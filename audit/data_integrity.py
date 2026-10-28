"""Step 2 of the camera-ready audit: row counts, hashes and split-integrity checks.

Writes results/camera_ready/data_integrity.json and prints a readable summary.
Reads only; never modifies data.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "data_release" / "benchmark_v2"
TRACKED = ROOT / "data" / "benchmark_v2"
TARBALL = ROOT / "reference" / "july_rerun" / "hackersignal_data.tgz"
OUT = ROOT / "results" / "camera_ready" / "data_integrity.json"
HF = "https://huggingface.co/api/datasets/DatasetSubmission/HackerSignal"

# Table 3 of the paper (current tex) and task meta.json
TABLE3 = {
    "task1_cve_linkage": {"train": 56692, "val": 2584, "test": 1990, "corpus": 340536},
    "task2_exploit_type": {"train": 64413, "val": 4735, "test": 1735},
    "task3_temporal_generalization": {"train": 56833, "val": 2535, "test": 1898, "corpus": 340536},
}
CVE_YEAR = re.compile(r"CVE-(\d{4})-", re.I)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(l) for l in f]


def norm(t: str) -> str:
    return " ".join(t.lower().split())


def hf_lfs() -> dict:
    out = {}
    for sub in ["task1_cve_linkage", "task2_exploit_type", "task3_temporal_generalization"]:
        try:
            with urllib.request.urlopen(f"{HF}/tree/main/benchmark_v2/{sub}", timeout=30) as r:
                for e in json.load(r):
                    lfs = e.get("lfs") or {}
                    out[e["path"].replace("benchmark_v2/", "")] = {
                        "sha256": lfs.get("oid"), "size": lfs.get("size", e.get("size"))}
        except Exception as e:  # recorded, not fatal
            out[f"{sub}/ERROR"] = repr(e)
    try:
        with urllib.request.urlopen(HF, timeout=30) as r:
            out["_repo_sha"] = json.load(r).get("sha")
    except Exception as e:
        out["_repo_sha"] = repr(e)
    return out


def tar_hashes() -> dict:
    out = {}
    with tarfile.open(TARBALL) as t:
        for m in t.getmembers():
            if m.isfile() and m.name.endswith(".jsonl") and "cve_index" not in m.name:
                h = hashlib.sha256(t.extractfile(m).read()).hexdigest()
                out[m.name.replace("hackersignal_data/", "")] = h
    return out


def main() -> int:
    rep: dict = {"files": {}, "checks": {}}
    hf = hf_lfs()
    july = tar_hashes()
    rep["hf_repo_sha"] = hf.get("_repo_sha")

    for p in sorted(REL.rglob("*.jsonl")):
        rel = str(p.relative_to(REL))
        n = sum(1 for _ in open(p, "rb"))
        h = sha256(p)
        tracked = TRACKED / rel
        rep["files"][rel] = {
            "rows": n, "sha256": h, "bytes": p.stat().st_size,
            "hf_sha256": (hf.get(rel) or {}).get("sha256"),
            "matches_hf": (hf.get(rel) or {}).get("sha256") == h if rel in hf else None,
            "july_tarball_sha256": july.get(rel),
            "matches_july_tarball": (july[rel] == h) if rel in july else None,
            "git_tracked_sha256": sha256(tracked) if tracked.exists() else None,
        }
        task, split = rel.split("/")[0], p.stem
        want = TABLE3.get(task, {}).get(split)
        rep["files"][rel]["table3_rows"] = want
        rep["files"][rel]["matches_table3"] = (want == n) if want is not None else None
        meta = REL / task / "meta.json"
        if meta.exists() and split in json.load(open(meta)).get("splits", {}):
            rep["files"][rel]["meta_rows"] = json.load(open(meta))["splits"][split]
    rep["hf_files_not_local"] = sorted(k for k in hf if not k.startswith("_") and k.endswith(".jsonl")
                                       and k not in rep["files"])

    # ---- retrieval tasks ----
    c1 = load(REL / "task1_cve_linkage/corpus.jsonl")
    ids = [d["id"] for d in c1]
    rep["checks"]["corpus_rows"] = len(c1)
    rep["checks"]["corpus_unique_ids"] = len(set(ids))
    rep["checks"]["task1_task3_corpus_identical"] = (
        rep["files"]["task1_cve_linkage/corpus.jsonl"]["sha256"]
        == rep["files"]["task3_temporal_generalization/corpus.jsonl"]["sha256"])
    if not rep["checks"]["task1_task3_corpus_identical"]:
        c3 = load(REL / "task3_temporal_generalization/corpus.jsonl")
        d1 = {d["id"]: d["text"] for d in c1}
        d3 = {d.get("id", d.get("cve_id")): d["text"] for d in c3}
        rep["checks"]["corpus_diff"] = {
            "ids_only_task1": len(set(d1) - set(d3)), "ids_only_task3": len(set(d3) - set(d1)),
            "same_id_text_differs": sum(1 for k in set(d1) & set(d3) if d1[k] != d3[k]),
            "order_differs": list(d1) != list(d3), "task3_keys": sorted(c3[0]),
            "examples": [k for k in sorted(set(d1) & set(d3)) if d1[k] != d3[k]][:5],
        }
    idset = set(ids)
    corpus_text = {norm(d["text"]): d["id"] for d in c1}

    R = {}
    for task in ["task1_cve_linkage", "task3_temporal_generalization"]:
        sp = {s: load(REL / task / f"{s}.jsonl") for s in ["train", "val", "test"]}
        R[task] = sp
        g = {s: {r["cve_id"] for r in rows} for s, rows in sp.items()}
        tr_text = Counter(norm(r["text"]) for r in sp["train"])
        test_dup = [r for r in sp["test"] if norm(r["text"]) in tr_text]
        test_dup_same_gold = [r for r in test_dup
                              if r["cve_id"] in {x["cve_id"] for x in sp["train"] if norm(x["text"]) == norm(r["text"])}]
        years = {s: Counter(int(m.group(1)) if (m := CVE_YEAR.match(r["cve_id"])) else None for r in rows)
                 for s, rows in sp.items()}
        ts = {s: Counter((r.get("timestamp") or "")[:4] or "EMPTY" for r in rows) for s, rows in sp.items()}
        rep["checks"][task] = {
            "rows": {s: len(v) for s, v in sp.items()},
            "unique_gold_cves": {s: len(v) for s, v in g.items()},
            "gold_not_in_corpus": {s: sum(1 for r in rows if r["cve_id"] not in idset) for s, rows in sp.items()},
            "cve_overlap_train_test": len(g["train"] & g["test"]),
            "cve_overlap_train_val": len(g["train"] & g["val"]),
            "cve_overlap_val_test": len(g["val"] & g["test"]),
            "test_exact_dup_of_train_text": len(test_dup),
            "test_exact_dup_of_train_text_same_gold": len(test_dup_same_gold),
            "test_query_identical_to_corpus_doc": sum(1 for r in sp["test"] if norm(r["text"]) in corpus_text),
            "test_internal_exact_dups": len(sp["test"]) - len({norm(r["text"]) for r in sp["test"]}),
            "duplicate_ids_in_test": len(sp["test"]) - len({r["id"] for r in sp["test"]}),
            "gold_cve_year_range": {s: [min(k for k in y if k), max(k for k in y if k)] for s, y in years.items()},
            "gold_cve_year_hist": {s: dict(sorted((str(k), v) for k, v in y.items())) for s, y in years.items()},
            "timestamp_year_hist": {s: dict(sorted(t.items())) for s, t in ts.items()},
            "test_sources": dict(Counter(r["source"] for r in sp["test"]).most_common()),
            "train_ids_equal_gold_share": sum(1 for r in sp["train"] if r["id"] == r["cve_id"]) / len(sp["train"]),
            "train_ids_equal_gold_rows": sum(1 for r in sp["train"] if r["id"] == r["cve_id"]),
        }
    t1, t3 = R["task1_cve_linkage"]["test"], R["task3_temporal_generalization"]["test"]
    k1 = {(r["id"], r["cve_id"], r["text"]) for r in t1}
    rep["checks"]["task3_test_subset_of_task1_test"] = {
        "by_id_gold_text": sum(1 for r in t3 if (r["id"], r["cve_id"], r["text"]) in k1),
        "by_id": sum(1 for r in t3 if r["id"] in {x["id"] for x in t1}),
        "task3_test_rows": len(t3),
    }
    rep["checks"]["task3_test_gold_in_task1_train"] = len(
        {r["cve_id"] for r in t3} & {r["cve_id"] for r in R["task1_cve_linkage"]["train"]})

    # ---- task 2 ----
    t2 = {s: load(REL / "task2_exploit_type" / f"{s}.jsonl") for s in ["train", "val", "test"]}
    classes = json.load(open(REL / "task2_exploit_type/meta.json"))["classes"]
    tr_text = {norm(r["text"]) for r in t2["train"]}
    va_text = {norm(r["text"]) for r in t2["val"]}
    rep["checks"]["task2_exploit_type"] = {
        "rows": {s: len(v) for s, v in t2.items()},
        "total": sum(len(v) for v in t2.values()),
        "labels_outside_8_classes": {s: sorted({r["label"] for r in v} - set(classes)) for s, v in t2.items()},
        "label_counts": {s: dict(Counter(r["label"] for r in v)) for s, v in t2.items()},
        "label_counts_match_meta": all(
            dict(Counter(r["label"] for r in t2[s])) == json.load(open(REL / "task2_exploit_type/meta.json"))["label_counts"][s]
            for s in t2),
        "timestamp_range": {s: [min(r["timestamp"] for r in v if r.get("timestamp")),
                                max(r["timestamp"] for r in v if r.get("timestamp"))] for s, v in t2.items()},
        "missing_timestamp": {s: sum(1 for r in v if not r.get("timestamp")) for s, v in t2.items()},
        "test_exact_dup_of_train_text": sum(1 for r in t2["test"] if norm(r["text"]) in tr_text),
        "test_exact_dup_of_val_text": sum(1 for r in t2["test"] if norm(r["text"]) in va_text),
        "val_exact_dup_of_train_text": sum(1 for r in t2["val"] if norm(r["text"]) in tr_text),
        "duplicate_ids_across_splits": len({r["id"] for r in t2["train"]} & {r["id"] for r in t2["test"]}),
        "sources": {s: dict(Counter(r["source"] for r in v)) for s, v in t2.items()},
        "text_len_words_median": {s: sorted(len(r["text"].split()) for r in v)[len(v) // 2] for s, v in t2.items()},
        "share_over_300_words": {s: sum(1 for r in v if len(r["text"].split()) > 300) / len(v) for s, v in t2.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
