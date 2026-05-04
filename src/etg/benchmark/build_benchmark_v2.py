"""Build the 3-task HackerSignal benchmark (v2).

Tasks:
  1. CVE Linkage Retrieval — cross-source temporally OOD entity grounding
  2. Hacker Signal Detection — two-stage detection + CVE grounding
  3. Temporal Generalization — CVE-disjoint prospective evaluation

Usage:
    python -m etg.benchmark.build_benchmark_v2 \
        --data-dir data/ \
        --output data/benchmark_v2/
"""

from __future__ import annotations

import json
import hashlib
import random
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

TRAIN_END = date(2022, 1, 1)
VAL_END = date(2024, 1, 1)

# Task 1: CVE Linkage sources
T1_QUALITY_SOURCES = {"cisa_kev", "cvefixes", "dtl_exploits", "hackerone"}
T1_ALL_SOURCES = [
    "exploitdb", "dtl_exploits", "github_advisory",
    "hackerone", "zeroscience", "vulnlab", "cisa_kev",
    "zeroday", "cvefixes",
]
T1_MIN_QUERY_TOKENS = 8

# Task 2: Signal detection sources (hacker community)
T2_HACKER_SOURCES = [
    "hacker_exploits", "hackforums", "hackerone",
    "dtl_exploits", "exploitdb_hf", "gayanku", "evolution",
    "hackersploit", "hackthebox", "0x00sec", "kaeli_hacker",
    "cve_hacker_forum",
]

# Structural patterns for actionability
ACTIONABLE_PATTERNS = [
    re.compile(r'\\x[0-9a-fA-F]{2}', re.IGNORECASE),
    re.compile(r'0x[0-9a-fA-F]{4,}'),
    re.compile(r'\b(exploit|payload|shellcode|PoC|proof.of.concept)\b', re.IGNORECASE),
    re.compile(r'\b(metasploit|msfconsole|msfvenom)\b', re.IGNORECASE),
    re.compile(r'\b(curl|wget|nc|netcat|nmap)\s+', re.IGNORECASE),
    re.compile(r'(GET|POST|PUT|DELETE)\s+/\S+\s+HTTP', re.IGNORECASE),
    re.compile(r"(SELECT|INSERT|UPDATE|DELETE|DROP|UNION)\s+.*(FROM|INTO|TABLE|ALL)", re.IGNORECASE),
    re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
    re.compile(r'(def |import |require\(|#include|public class|void main)', re.IGNORECASE),
    re.compile(r'/bin/(sh|bash|zsh|cmd)', re.IGNORECASE),
    re.compile(r'CVE-\d{4}-\d{4,}'),
]

ACTIONABLE_THRESHOLD = 2  # Need >=2 structural patterns to be actionable


def parse_date(ts: str) -> date | None:
    if not ts:
        return None
    try:
        return date.fromisoformat(ts[:10])
    except (ValueError, TypeError):
        return None


def get_split(d: date) -> str:
    if d < TRAIN_END:
        return "train"
    elif d < VAL_END:
        return "val"
    else:
        return "test"


def extract_cve_ids(text: str) -> list[str]:
    return re.findall(r'CVE-\d{4}-\d{4,}', text)


def cve_year(cve_id: str) -> int | None:
    m = re.match(r'CVE-(\d{4})-', cve_id)
    return int(m.group(1)) if m else None


def count_actionable_patterns(text: str) -> int:
    return sum(1 for p in ACTIONABLE_PATTERNS if p.search(text))


def build_task1_cve_linkage(data_dir: Path, output_dir: Path):
    """Task 1: CVE Linkage Retrieval.

    Query = exploit/advisory text. Target = correct NVD CVE description.
    Temporal split on publication date. Quality-controlled to remove
    unretrievable rows (min token count).
    """
    print("\n=== Task 1: CVE Linkage Retrieval ===")
    task_dir = output_dir / "task1_cve_linkage"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Load NVD corpus
    nvd_path = data_dir / "cve_index" / "nvd_cve_descriptions.jsonl"
    if not nvd_path.exists():
        # Try alternate location
        nvd_path = data_dir / "nvd_cve_descriptions.jsonl"

    corpus = {}
    if nvd_path.exists():
        with open(nvd_path) as f:
            for line in f:
                rec = json.loads(line)
                cve_id = rec.get("cve_id", rec.get("id", ""))
                text = rec.get("description", rec.get("text", ""))
                if cve_id and text:
                    corpus[cve_id] = {"id": cve_id, "text": text}

    # If no NVD file found, use existing corpus
    if not corpus:
        existing_corpus = data_dir / "benchmark" / "task2_cve_linkage" / "corpus.jsonl"
        if existing_corpus.exists():
            with open(existing_corpus) as f:
                for line in f:
                    rec = json.loads(line)
                    cve_id = rec.get("cve_id", rec.get("id", ""))
                    if cve_id:
                        corpus[cve_id] = {"id": cve_id, "text": rec.get("text", "")}

    print(f"  Corpus size: {len(corpus)}")

    # Build queries from multiple CVE index sources
    queries = {"train": [], "val": [], "test": []}

    cve_index_dir = data_dir / "cve_index"
    if cve_index_dir.exists():
        for idx_file in cve_index_dir.glob("*.jsonl"):
            if "nvd" in idx_file.name:
                continue
            source_name = idx_file.stem.replace("_cve_index", "").replace("_index", "")
            with open(idx_file) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    cve_id = rec.get("cve_id", "")
                    text = rec.get("text", rec.get("description", ""))
                    ts = rec.get("timestamp", rec.get("date", ""))

                    if not cve_id or not text or cve_id not in corpus:
                        continue
                    if len(text.split()) < T1_MIN_QUERY_TOKENS:
                        continue

                    d = parse_date(ts)
                    if d is None:
                        # Use CVE year as fallback
                        yr = cve_year(cve_id)
                        if yr and yr < 2022:
                            split = "train"
                        elif yr and yr < 2024:
                            split = "val"
                        else:
                            split = "test"
                    else:
                        split = get_split(d)

                    query_id = rec.get("id", hashlib.sha256(
                        f"{source_name}:{cve_id}:{text[:50]}".encode()
                    ).hexdigest()[:20])

                    queries[split].append({
                        "id": query_id,
                        "text": text[:4000],
                        "cve_id": cve_id,
                        "title": rec.get("title", ""),
                        "source": source_name,
                        "timestamp": ts,
                        "split": split,
                    })

    # Also pull from existing task2 data if cve_index is sparse
    if sum(len(v) for v in queries.values()) < 1000:
        existing_t2 = data_dir / "benchmark" / "task2_cve_linkage"
        if existing_t2.exists():
            for split_name in ["train", "val", "test"]:
                split_file = existing_t2 / f"{split_name}.jsonl"
                if split_file.exists():
                    with open(split_file) as f:
                        for line in f:
                            rec = json.loads(line)
                            rec["split"] = split_name
                            queries[split_name].append(rec)

    for split_name, split_data in queries.items():
        print(f"  {split_name}: {len(split_data)} queries")
        out_path = task_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for rec in split_data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write corpus
    corpus_path = task_dir / "corpus.jsonl"
    with open(corpus_path, "w") as f:
        for rec in corpus.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Corpus: {len(corpus)} CVE descriptions")

    meta = {
        "task": "cve_linkage_retrieval",
        "description": "Cross-source temporally OOD entity grounding: given exploit evidence text, retrieve the correct NVD CVE entry.",
        "metrics": ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"],
        "split_boundaries": {"train_end": "2022-01-01", "val_end": "2024-01-01"},
        "corpus_size": len(corpus),
        "splits": {s: len(v) for s, v in queries.items()},
    }
    with open(task_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def build_task2_signal_detection(data_dir: Path, output_dir: Path):
    """Task 2: Hacker Signal Detection.

    Two-stage: (1) Detect actionable exploit signals from hacker community noise.
    (2) For positives, ground to NVD CVE.
    """
    print("\n=== Task 2: Hacker Signal Detection ===")
    task_dir = output_dir / "task2_signal_detection"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Use existing task5 data which has proper structure
    existing_t5 = data_dir / "benchmark" / "task5_hacker_signal_detection"

    records = {"train": [], "val": [], "test": []}

    if existing_t5.exists():
        for split_name in ["train", "val", "test"]:
            split_file = existing_t5 / f"{split_name}.jsonl"
            if split_file.exists():
                with open(split_file) as f:
                    for line in f:
                        rec = json.loads(line)
                        # Rename actionable_label to label for clarity
                        rec["label"] = rec.pop("actionable_label", rec.get("label", 0))
                        rec["split"] = split_name
                        records[split_name].append(rec)

    for split_name, split_data in records.items():
        pos = sum(1 for r in split_data if r["label"] > 0)
        neg = len(split_data) - pos
        print(f"  {split_name}: {len(split_data)} rows (pos={pos}, neg={neg})")
        out_path = task_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for rec in split_data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Copy corpus
    existing_corpus = existing_t5 / "corpus.jsonl"
    if existing_corpus.exists():
        import shutil
        shutil.copy2(existing_corpus, task_dir / "corpus.jsonl")
        corpus_size = sum(1 for _ in open(existing_corpus))
    else:
        corpus_size = 0

    total_pos = sum(1 for s in records.values() for r in s if r["label"] > 0)
    total_neg = sum(1 for s in records.values() for r in s if r["label"] == 0)

    meta = {
        "task": "hacker_signal_detection",
        "description": "Two-stage: detect actionable exploit signals from hacker community noise, then ground positives to NVD CVE.",
        "label_map": {"0": "not_actionable", "1": "actionable_exploit_signal"},
        "metrics": ["f1_actionable", "precision", "recall", "joint_recall@5", "mrr"],
        "split_boundaries": {"train_end": "2022-01-01", "val_end": "2024-01-01"},
        "negative_ratio": f"{total_neg/max(total_pos,1):.0f}:1",
        "corpus_size": corpus_size,
        "splits": {s: len(v) for s, v in records.items()},
        "label_distribution": {
            s: {"pos": sum(1 for r in v if r["label"] > 0), "neg": sum(1 for r in v if r["label"] == 0)}
            for s, v in records.items()
        },
    }
    with open(task_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def build_task3_temporal_generalization(data_dir: Path, output_dir: Path):
    """Task 3: Temporal Generalization (Prospective CVE-disjoint evaluation).

    Same retrieval setup as Task 1 but with strict constraint:
    C_train ∩ C_test = ∅ (no CVE appears in both train and test).
    Split by CVE publication year, not query timestamp.
    """
    print("\n=== Task 3: Temporal Generalization ===")
    task_dir = output_dir / "task3_temporal_generalization"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Load Task 1 data and re-split by CVE year
    t1_dir = output_dir / "task1_cve_linkage"

    all_queries = []
    for split_name in ["train", "val", "test"]:
        split_file = t1_dir / f"{split_name}.jsonl"
        if split_file.exists():
            with open(split_file) as f:
                for line in f:
                    all_queries.append(json.loads(line))

    print(f"  Total queries from Task 1: {len(all_queries)}")

    # Re-split by CVE year for strict disjointness
    # Train: CVE-20xx where xx < 2022
    # Val: CVE-2022 and CVE-2023
    # Test: CVE-2024+
    train, val, test = [], [], []

    for q in all_queries:
        cve_id = q.get("cve_id", "")
        yr = cve_year(cve_id)
        if yr is None:
            continue

        if yr < 2022:
            q["split"] = "train"
            train.append(q)
        elif yr < 2024:
            q["split"] = "val"
            val.append(q)
        else:
            q["split"] = "test"
            test.append(q)

    # Verify disjointness
    train_cves = {q["cve_id"] for q in train}
    val_cves = {q["cve_id"] for q in val}
    test_cves = {q["cve_id"] for q in test}

    overlap_train_test = train_cves & test_cves
    overlap_train_val = train_cves & val_cves
    overlap_val_test = val_cves & test_cves

    print(f"  Train CVEs: {len(train_cves)}, Val CVEs: {len(val_cves)}, Test CVEs: {len(test_cves)}")
    print(f"  Train∩Test overlap: {len(overlap_train_test)} (must be 0)")
    print(f"  Train∩Val overlap: {len(overlap_train_val)}")
    print(f"  Val∩Test overlap: {len(overlap_val_test)}")

    # Remove any overlapping CVEs (shouldn't happen with year-based split, but safety check)
    if overlap_train_test:
        test = [q for q in test if q["cve_id"] not in train_cves]
        test_cves = {q["cve_id"] for q in test}

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        print(f"  {split_name}: {len(split_data)} queries")
        out_path = task_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for rec in split_data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Symlink corpus from task1
    corpus_src = t1_dir / "corpus.jsonl"
    corpus_dst = task_dir / "corpus.jsonl"
    if corpus_src.exists() and not corpus_dst.exists():
        import shutil
        shutil.copy2(corpus_src, corpus_dst)

    corpus_size = sum(1 for _ in open(corpus_dst)) if corpus_dst.exists() else 0

    meta = {
        "task": "temporal_generalization",
        "description": "Prospective CVE-disjoint retrieval: C_train ∩ C_test = ∅. Tests whether models generalize to wholly unseen vulnerabilities.",
        "metrics": ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"],
        "cve_year_boundaries": {"train": "<2022", "val": "2022-2023", "test": "2024+"},
        "cve_disjointness": {
            "train_test_overlap": len(overlap_train_test),
            "verified_disjoint": len(overlap_train_test) == 0,
        },
        "unique_cves": {"train": len(train_cves), "val": len(val_cves), "test": len(test_cves)},
        "corpus_size": corpus_size,
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
    }
    with open(task_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/benchmark_v2"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    m1 = build_task1_cve_linkage(args.data_dir, args.output)
    m2 = build_task2_signal_detection(args.data_dir, args.output)
    m3 = build_task3_temporal_generalization(args.data_dir, args.output)

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK V2 SUMMARY")
    print("=" * 60)
    print(f"\nTask 1 (CVE Linkage): {m1['splits']}")
    print(f"  Corpus: {m1['corpus_size']}")
    print(f"\nTask 2 (Signal Detection): {m2['splits']}")
    print(f"  Neg ratio: {m2['negative_ratio']}")
    print(f"\nTask 3 (Temporal Generalization): {m3['splits']}")
    print(f"  CVE disjoint: {m3['cve_disjointness']['verified_disjoint']}")


if __name__ == "__main__":
    main()
