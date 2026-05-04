"""Temporal train / validation / test split generation for HackerSignal benchmarks.

Split design
------------
All HackerSignal benchmark tasks use a **temporal holdout** strategy that mirrors
real-world threat-intelligence scenarios: models are trained on historical data
and evaluated on future, unseen posts.

    ┌─────────────────┬────────────────┬──────────────────┐
    │  TRAIN          │  VAL           │  TEST            │
    │  pre-2022-01-01 │  2022 – 2023   │  2024-01-01+     │
    └─────────────────┴────────────────┴──────────────────┘

Rationale
---------
- The 2022 boundary aligns with the Log4Shell / Log4j era, which is well-studied
  and provides a clean break between well-labelled historical data and more
  contemporary posts.
- The 2024+ test window is entirely future relative to most existing published
  models (SecBERT, etc.), ensuring test contamination is minimal.

Tasks
-----
Five JSONL datasets are produced under ``{output_dir}/``:

1. ``task1_exploit_clf/``   — Exploit Relevance Classification
2. ``task2_cve_linkage/``   — CVE Linkage / Retrieval
3. ``task3_severity/``      — Severity Prediction
4. ``task4_hacker_exploit_labeling/`` — Hacker Exploit Labeling
5. ``task5_hacker_signal_detection/`` — Hacker Exploit Signal Detection

Each task directory contains:
    train.jsonl, val.jsonl, test.jsonl, corpus.jsonl (Tasks 2 and 5 only), meta.json

Usage
-----
    python -m etg.benchmark.splits --data-dir data/ --output data/benchmark/
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Split boundaries
# ---------------------------------------------------------------------------

TRAIN_END = date(2022, 1, 1)    # exclusive upper bound for train
VAL_END   = date(2024, 1, 1)    # exclusive upper bound for val (test starts here)

# ---------------------------------------------------------------------------
# Source configuration for each task
# ---------------------------------------------------------------------------

# Task 1 — CVE-linked exploit relevance classification
# Labels are built within each source: positive if a post is linked to at
# least one CVE in that source's CVE index, negative if it is from the same
# source/split and has no CVE linkage or explicit CVE token. This removes the
# source-membership shortcut in the earlier source-derived task.
_T1_WITHIN_SOURCE_CANDIDATES = [
    "exploitdb", "zeroscience", "hackerone", "vulnlab", "seebug",
    "0x00sec", "hackthebox", "hackersploit", "parrotsec",
    "gayanku", "evolution", "hackforums",
    "dtl_exploits", "github_advisory", "cvefixes", "zeroday",
    "hacker_exploits", "kaeli_hacker", "cisa_kev",
]
_T1_MIN_POS_PER_SOURCE_SPLIT = 5
_T1_MAX_POS_PER_SOURCE_SPLIT = 100_000

# Task 2 — CVE Linkage
# Uses the CVE index JSONL files (exploit_id → cve_id mapping with text)
# Corpus is the NVD CVE descriptions
_T2_SOURCES = [
    "exploitdb", "dtl_exploits", "github_advisory",
    "hackerone", "zeroscience", "vulnlab", "cisa_kev",
    "zeroday", "cvefixes",
]
_T2_QUALITY_SOURCES = {
    # Low/medium risk in the LLM-assisted CVE-link triage and generally
    # nonempty, source-supported linkage evidence.
    "cisa_kev",
    "cvefixes",
    "dtl_exploits",
    "hackerone",
}
_T2_MIN_QUERY_TOKENS = 8

# Task 3 — Severity Prediction
# Sources that carry a severity label: github_advisory (has 'severity' field
# embedded in text as "Severity: X"), zeroscience (sev in advisory index),
# cisa_kev (all "critical" — known exploited in the wild).
_T3_SOURCES = {
    "github_advisory": "github_advisory",
    "zeroscience": "zeroscience",
    "hackerone": "hackerone",
    "cisa_kev": "cisa_kev",
    "vulnlab": "vulnlab",
}
_T3_LABEL_KEYWORDS = {
    "critical": 3,
    "high": 2,
    "medium": 1,
    "moderate": 1,
    "low": 0,
}

# Task 4 — Hacker Exploit Labeling
# Weak ternary labels over hacker-community messages. The labels intentionally
# use observable metadata and lexical actionability cues so the benchmark can be
# regenerated without private hand labels.
_HACKER_COMMUNITY_SOURCES = [
    "hacker_exploits",
    "cve_hacker_forum",
    "hackforums",
    "hackerone",
    "dtl_exploits",
    "exploitdb_hf",
    "gayanku",
    "evolution",
    "hackersploit",
    "hackthebox",
    "0x00sec",
    "kaeli_hacker",
]
_T4_MIN_TOKENS = 12
_T4_MAX_PER_LABEL_SPLIT = 100_000
_ACTIONABLE_TERMS = [
    "exploit",
    "proof of concept",
    "poc",
    "metasploit",
    "payload",
    "shellcode",
    "reverse shell",
    "privilege escalation",
    "remote code execution",
    "rce",
    "sql injection",
    "xss",
    "csrf",
    "command injection",
    "buffer overflow",
    "use-after-free",
    "deserialization",
    "bypass",
    "scanner",
    "wpscan",
    "root shell",
]
_VULNERABILITY_TERMS = [
    "cve-",
    "vulnerability",
    "vulnerable",
    "patch",
    "advisory",
    "severity",
    "cvss",
    "zero-day",
    "0day",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(ts: str) -> date | None:
    """Parse an ISO timestamp string to a date, returning None on failure."""
    if not ts:
        return None
    try:
        return date.fromisoformat(ts[:10])
    except (ValueError, TypeError):
        return None


def _split_label(ts_date: date | None) -> str | None:
    """Return 'train' / 'val' / 'test' or None if date is missing."""
    if ts_date is None:
        return None
    if ts_date < TRAIN_END:
        return "train"
    if ts_date < VAL_END:
        return "val"
    return "test"


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def _write_jsonl(records: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def _write_meta(info: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def _label_counts(records: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        label = str(record.get(key, "unknown"))
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _word_count(text: str) -> int:
    return len((text or "").split())


def _contains_any(text: str, terms: list[str]) -> bool:
    text_l = (text or "").lower()
    return any(term in text_l for term in terms)


def _source_cve_lookup(data_dir: Path, sources: list[str]) -> dict[str, dict[str, list[str]]]:
    """Return source -> post/exploit id -> CVE IDs."""
    lookup: dict[str, dict[str, list[str]]] = {}
    for src in sources:
        idx_path = data_dir / f"{src}_cve_index.jsonl"
        if not idx_path.exists():
            continue
        src_lookup: dict[str, list[str]] = {}
        for row in _iter_jsonl(idx_path):
            exploit_id = str(row.get("exploit_id") or row.get("post_id") or "")
            if not exploit_id:
                continue
            raw_cves = row.get("cve_id") or row.get("cves") or []
            if isinstance(raw_cves, str):
                raw_cves = [raw_cves]
            cves: list[str] = []
            for raw in raw_cves:
                cve = str(raw).split(";")[0].upper().strip()
                if cve.startswith("CVE-"):
                    cves.append(cve)
            if cves:
                src_lookup.setdefault(exploit_id, [])
                src_lookup[exploit_id].extend(c for c in cves if c not in src_lookup[exploit_id])
        if src_lookup:
            lookup[src] = src_lookup
    return lookup


def _build_nvd_corpus(data_dir: Path) -> list[dict]:
    """Build NVD CVE description corpus used by retrieval tasks."""
    import re

    nvd_path = data_dir / "nvd_posts.jsonl"
    cve_dict_path = data_dir / "nvd_cve_dict.jsonl"
    corpus: list[dict] = []
    seen: set[str] = set()

    if cve_dict_path.exists():
        for d in _iter_jsonl(cve_dict_path):
            cve_id = (d.get("cve_id") or "").upper()
            if cve_id and cve_id not in seen:
                corpus.append({"cve_id": cve_id, "text": (d.get("description") or d.get("text") or "")[:2048]})
                seen.add(cve_id)
    elif nvd_path.exists():
        for d in _iter_jsonl(nvd_path):
            text = d.get("text") or ""
            m = re.match(r"CVE:\s*(CVE-\d{4}-\d+)", text)
            if m:
                cve_id = m.group(1).upper()
                if cve_id not in seen:
                    corpus.append({"cve_id": cve_id, "text": text[:2048]})
                    seen.add(cve_id)

    return corpus


# ---------------------------------------------------------------------------
# Task 1 — Exploit Relevance Classification
# ---------------------------------------------------------------------------


def build_task1(data_dir: Path, output_dir: Path, seed: int = 42) -> dict:
    """Build source-controlled CVE-linked exploit relevance splits.

    Each record:
        {"id": str, "text": str, "label": 0|1, "source": str,
         "timestamp": str, "split": "train"|"val"|"test"}
    """
    import re

    rng = random.Random(seed)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    source_split_counts: dict[str, dict[str, dict[str, int]]] = {}
    cve_re = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)

    for src in _T1_WITHIN_SOURCE_CANDIDATES:
        posts_path = data_dir / f"{src}_posts.jsonl"
        index_path = data_dir / f"{src}_cve_index.jsonl"
        if not posts_path.exists() or not index_path.exists():
            log.warning("task1: missing posts or CVE index for %s", src)
            continue

        linked_ids = {
            str(d.get("exploit_id") or "")
            for d in _iter_jsonl(index_path)
            if d.get("exploit_id")
        }
        if not linked_ids:
            continue

        by_split: dict[str, dict[str, list[dict]]] = {
            sp: {"pos": [], "neg": []}
            for sp in ("train", "val", "test")
        }

        for row in _iter_jsonl(posts_path):
            ts_date = _parse_date(row.get("timestamp", ""))
            split = _split_label(ts_date)
            if split is None:
                continue
            record_id = str(row.get("id", ""))
            text = row.get("text") or ""
            base = {
                "id": record_id,
                "text": text[:4096],
                "source": src,
                "timestamp": (row.get("timestamp") or "")[:10],
                "split": split,
                "label_source": "within_source_cve_index",
            }
            if record_id in linked_ids:
                by_split[split]["pos"].append({**base, "label": 1})
            elif not cve_re.search(text):
                by_split[split]["neg"].append({**base, "label": 0})

        source_split_counts[src] = {}
        for split, groups in by_split.items():
            pos = groups["pos"]
            neg = groups["neg"]
            if len(pos) < _T1_MIN_POS_PER_SOURCE_SPLIT or not neg:
                source_split_counts[src][split] = {
                    "positive_available": len(pos),
                    "negative_available": len(neg),
                    "positive_used": 0,
                    "negative_used": 0,
                }
                continue
            rng.shuffle(pos)
            rng.shuffle(neg)
            n = min(len(pos), len(neg), _T1_MAX_POS_PER_SOURCE_SPLIT)
            used = pos[:n] + neg[:n]
            splits[split].extend(used)
            source_split_counts[src][split] = {
                "positive_available": len(pos),
                "negative_available": len(neg),
                "positive_used": n,
                "negative_used": n,
            }

    # --- Shuffle and write ---
    task_dir = output_dir / "task1_exploit_clf"
    total = 0
    counts = {}
    for sp, records in splits.items():
        rng.shuffle(records)
        n = _write_jsonl(records, task_dir / f"{sp}.jsonl")
        counts[sp] = n
        total += n
        log.info("task1: wrote %d records → %s/%s.jsonl", n, task_dir.name, sp)

    meta = {
        "task": "exploit_relevance_classification",
        "description": (
            "Source-controlled binary classification: given a post or advisory text, "
            "predict whether it is CVE-linked exploit-relevant (1) or not (0). "
            "Each source/split contributes balanced positives and negatives from "
            "the same source to remove source-membership confounding."
        ),
        "label_map": {"0": "not_exploit_relevant", "1": "exploit_relevant"},
        "candidate_sources": sorted(_T1_WITHIN_SOURCE_CANDIDATES),
        "labeling_rule": (
            "positive if post ID appears in the same source CVE index; negative if "
            "from the same source/split, absent from the CVE index, and containing no CVE token"
        ),
        "source_split_sampling": source_split_counts,
        "split_boundaries": {
            "train_end": str(TRAIN_END),
            "val_end": str(VAL_END),
        },
        "metrics": ["f1_macro", "f1_positive", "auc_roc", "precision", "recall"],
        "splits": counts,
        "label_distribution_by_split": {
            sp: _label_counts(records, "label")
            for sp, records in splits.items()
        },
        "total": total,
    }
    _write_meta(meta, task_dir / "meta.json")
    return meta


# ---------------------------------------------------------------------------
# Task 2 — CVE Linkage / Retrieval
# ---------------------------------------------------------------------------


def build_task2(data_dir: Path, output_dir: Path, seed: int = 42, quality_controlled: bool = True) -> dict:
    """Build CVE linkage retrieval splits.

    Query records (train/val/test):
        {"id": str, "text": str, "cve_id": str, "source": str,
         "timestamp": str, "split": str}

    Corpus (corpus.jsonl — all NVD CVE descriptions):
        {"cve_id": str, "text": str}

    By default this builds the quality-controlled Task 2 split used in the
    paper: rows from high-risk source families are excluded, query text must be
    nonempty and minimally informative, and the source title is used as
    fallback text when the CVE index has no body snippet. This removes
    unretrievable empty GitHub Advisory rows and malformed high-risk source
    rows without changing the temporal split or NVD retrieval corpus.
    """
    rng = random.Random(seed)
    task_dir = output_dir / "task2_cve_linkage"

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    skipped: dict[str, int] = {
        "quality_source_filter": 0,
        "empty_or_short_query": 0,
        "duplicate": 0,
        "missing_id_or_cve": 0,
    }

    # --- Build query set from CVE index files ---
    seen: set[tuple[str, str]] = set()  # (exploit_id, cve_id) dedup

    for src in _T2_SOURCES:
        if quality_controlled and src not in _T2_QUALITY_SOURCES:
            idx_path = data_dir / f"{src}_cve_index.jsonl"
            if idx_path.exists():
                skipped["quality_source_filter"] += sum(1 for _ in _iter_jsonl(idx_path))
            continue
        idx_path = data_dir / f"{src}_cve_index.jsonl"
        if not idx_path.exists():
            log.warning("task2: missing %s", idx_path)
            continue
        posts_path = data_dir / f"{src}_posts.jsonl"

        # Build a timestamp lookup from posts if available
        ts_lookup: dict[str, str] = {}
        if posts_path.exists():
            for d in _iter_jsonl(posts_path):
                ts_lookup[d.get("id", "")] = (d.get("timestamp") or "")[:10]

        for d in _iter_jsonl(idx_path):
            exploit_id = d.get("exploit_id", "")
            cve_id = (d.get("cve_id") or "").upper()
            if not exploit_id or not cve_id:
                skipped["missing_id_or_cve"] += 1
                continue
            key = (exploit_id, cve_id)
            if key in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(key)

            query_text = (d.get("exploit_text") or d.get("title") or "").strip()
            if _word_count(query_text) < _T2_MIN_QUERY_TOKENS:
                skipped["empty_or_short_query"] += 1
                continue

            ts = ts_lookup.get(exploit_id) or (d.get("published") or "")[:10]
            ts_date = _parse_date(ts)
            sp = _split_label(ts_date)
            if sp is None:
                # No date — assign to train (safest assumption)
                sp = "train"

            splits[sp].append({
                "id": exploit_id,
                "text": query_text[:2048],
                "cve_id": cve_id,
                "title": d.get("title", ""),
                "source": src,
                "timestamp": ts,
                "split": sp,
            })

    # Shuffle and write query splits
    counts = {}
    total = 0
    for sp, records in splits.items():
        rng.shuffle(records)
        n = _write_jsonl(records, task_dir / f"{sp}.jsonl")
        counts[sp] = n
        total += n
        log.info("task2: wrote %d query records → %s/%s.jsonl", n, task_dir.name, sp)

    # --- Build NVD corpus ---
    corpus = _build_nvd_corpus(data_dir)

    if corpus:
        n = _write_jsonl(corpus, task_dir / "corpus.jsonl")
        log.info("task2: wrote %d corpus records → corpus.jsonl", n)
    else:
        log.warning("task2: no NVD corpus found — corpus.jsonl will be empty")
        n = 0

    # Collect all CVE IDs that appear in queries (for evaluation lookup)
    all_query_cves = set(
        r["cve_id"] for sp_records in splits.values() for r in sp_records
    )

    meta = {
        "task": "cve_linkage_retrieval",
        "description": (
            "Retrieval task: given a post or advisory text, retrieve the "
            "correct NVD CVE record from a corpus of ~340K CVE descriptions. "
            "Evaluated as a ranking problem."
        ),
        "query_sources": _T2_SOURCES,
        "quality_controlled": quality_controlled,
        "quality_sources": sorted(_T2_QUALITY_SOURCES) if quality_controlled else _T2_SOURCES,
        "min_query_tokens": _T2_MIN_QUERY_TOKENS if quality_controlled else 0,
        "query_text_rule": "exploit_text, falling back to title when body text is absent",
        "skipped": skipped,
        "corpus_source": "nvd",
        "corpus_size": n,
        "unique_query_cves": len(all_query_cves),
        "split_boundaries": {
            "train_end": str(TRAIN_END),
            "val_end": str(VAL_END),
        },
        "metrics": ["recall_at_1", "recall_at_5", "recall_at_10", "mrr"],
        "splits": counts,
        "query_source_distribution_by_split": {
            sp: _label_counts(records, "source")
            for sp, records in splits.items()
        },
        "total_queries": total,
    }
    _write_meta(meta, task_dir / "meta.json")
    return meta


# ---------------------------------------------------------------------------
# Task 3 — Severity Prediction
# ---------------------------------------------------------------------------


def build_task3(data_dir: Path, output_dir: Path, seed: int = 42) -> dict:
    """Build severity prediction splits.

    Each record:
        {"id": str, "text": str, "severity": str, "severity_label": int,
         "source": str, "timestamp": str, "split": str}

    Labels: 0=low, 1=medium, 2=high, 3=critical
    """
    import re

    rng = random.Random(seed)
    task_dir = output_dir / "task3_severity"

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    _SEV_RE = re.compile(
        r"(?:^|\n)\s*\[?\s*(?:severity|sev|cvss[^:]*)\s*:\s*(\w+)\s*\]?",
        re.IGNORECASE,
    )
    _SEV_LINE_RE = re.compile(
        r"(?im)^\s*\[?\s*(?:severity|sev|cvss[^:]*)\s*:\s*(?:critical|high|medium|moderate|low)\s*\]?\s*$"
    )
    _SEV_INLINE_RE = re.compile(
        r"\[?\s*(?:severity|sev|cvss[^:]*)\s*:\s*(?:critical|high|medium|moderate|low)\s*\]?",
        re.IGNORECASE,
    )
    _CVSS_SCORE_RE = re.compile(
        r"(?im)^\s*cvss(?:\s+(?:score|base score))?\s*:\s*[0-9.]+(?:\s*/\s*10)?\s*$"
    )
    _CVSS_VECTOR_RE = re.compile(r"CVSS:\d\.\d/[A-Z:0-9./-]+", re.IGNORECASE)
    _SEVERITY_WORD_RE = re.compile(r"\b(?:critical|high|medium|moderate|low)\s+severity\b|\bseverity\s+(?:critical|high|medium|moderate|low)\b", re.IGNORECASE)

    def _extract_severity(text: str, source: str) -> tuple[str, int] | None:
        """Extract severity label from post text. Returns (label_str, int) or None."""
        # CISA KEV — all known exploited = critical
        if source == "cisa_kev":
            return ("critical", 3)

        m = _SEV_RE.search(text)
        if m:
            word = m.group(1).lower()
            for kw, lbl in _T3_LABEL_KEYWORDS.items():
                if kw in word:
                    label_name = "medium" if kw == "moderate" else kw
                    return (label_name, lbl)
        return None

    def _strip_severity_leakage(text: str) -> str:
        """Remove explicit severity labels from model input after label extraction."""
        cleaned = _SEV_LINE_RE.sub("", text)
        cleaned = _CVSS_SCORE_RE.sub("", cleaned)
        cleaned = _CVSS_VECTOR_RE.sub("", cleaned)
        cleaned = _SEV_INLINE_RE.sub("", cleaned)
        paragraphs = re.split(r"\n\s*\n", cleaned)
        kept: list[str] = []
        for para in paragraphs:
            para_l = para.lower()
            has_severity_label = "severity" in para_l and (
                _SEVERITY_WORD_RE.search(para) or any(label in para_l for label in _T3_LABEL_KEYWORDS)
            )
            has_cvss_score = "cvss" in para_l and re.search(r"\b[0-9](?:\.[0-9])?\s*(?:/10)?\b", para)
            if has_severity_label or has_cvss_score:
                continue
            kept.append(para.strip())
        cleaned = "\n\n".join(p for p in kept if p)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    for src in sorted(_T3_SOURCES.keys()):
        posts_path = data_dir / f"{src}_posts.jsonl"
        if not posts_path.exists():
            log.warning("task3: missing %s", posts_path)
            continue

        included = 0
        skipped = 0
        for d in _iter_jsonl(posts_path):
            text = d.get("text") or ""
            sev = _extract_severity(text, src)
            if sev is None:
                skipped += 1
                continue

            ts_date = _parse_date(d.get("timestamp", ""))
            sp = _split_label(ts_date)
            if sp is None:
                sp = "train"

            sev_str, sev_int = sev
            clean_text = _strip_severity_leakage(text)
            splits[sp].append({
                "id": d.get("id", ""),
                "text": clean_text[:4096],
                "text_raw": text[:4096],
                "severity": sev_str,
                "severity_label": sev_int,
                "source": src,
                "timestamp": (d.get("timestamp") or "")[:10],
                "split": sp,
            })
            included += 1

        log.info("task3: %s — included=%d skipped=%d", src, included, skipped)

    # Shuffle and write
    counts = {}
    total = 0
    for sp, records in splits.items():
        rng.shuffle(records)
        n = _write_jsonl(records, task_dir / f"{sp}.jsonl")
        counts[sp] = n
        total += n
        log.info("task3: wrote %d records → %s/%s.jsonl", n, task_dir.name, sp)

    # Class distribution
    all_records = [r for sp_recs in splits.values() for r in sp_recs]
    from collections import Counter
    dist = Counter(r["severity"] for r in all_records)

    meta = {
        "task": "severity_prediction",
        "description": (
            "4-class classification: given an advisory or post text, predict "
            "the vulnerability severity as critical / high / medium / low."
        ),
        "label_map": {"0": "low", "1": "medium", "2": "high", "3": "critical"},
        "sources": sorted(_T3_SOURCES.keys()),
        "split_boundaries": {
            "train_end": str(TRAIN_END),
            "val_end": str(VAL_END),
        },
        "metrics": ["f1_macro", "accuracy", "f1_per_class"],
        "text_preprocessing": "explicit severity and CVSS label strings stripped from model input after label extraction",
        "class_distribution": dict(dist),
        "class_distribution_by_split": {
            sp: _label_counts(records, "severity")
            for sp, records in splits.items()
        },
        "splits": counts,
        "total": total,
    }
    _write_meta(meta, task_dir / "meta.json")
    return meta


# ---------------------------------------------------------------------------
# Task 4 — Hacker Exploit Labeling
# ---------------------------------------------------------------------------


def _task4_label(text: str, cves: list[str]) -> tuple[int, str]:
    """Weak ternary exploit-intelligence label for hacker-community messages."""
    has_cve = bool(cves)
    has_actionable = _contains_any(text, _ACTIONABLE_TERMS)
    has_vuln = has_cve or _contains_any(text, _VULNERABILITY_TERMS)
    if has_cve and has_actionable:
        return 2, "actionable_exploit"
    if has_vuln:
        return 1, "vulnerability_discussion"
    return 0, "non_exploit_noise"


def build_task4(data_dir: Path, output_dir: Path, seed: int = 42) -> dict:
    """Build ternary hacker exploit-labeling splits.

    Each record:
        {"id": str, "text": str, "label": 0|1|2, "label_name": str,
         "cve_ids": list[str], "source": str, "timestamp": str, "split": str}
    """
    rng = random.Random(seed)
    task_dir = output_dir / "task4_hacker_exploit_labeling"
    cve_lookup = _source_cve_lookup(data_dir, _HACKER_COMMUNITY_SOURCES)
    splits_by_label: dict[str, dict[int, list[dict]]] = {
        sp: {0: [], 1: [], 2: []}
        for sp in ("train", "val", "test")
    }
    skipped = {"missing_file": 0, "missing_date": 0, "short_text": 0}

    for src in _HACKER_COMMUNITY_SOURCES:
        posts_path = data_dir / f"{src}_posts.jsonl"
        if not posts_path.exists():
            skipped["missing_file"] += 1
            continue
        src_cves = cve_lookup.get(src, {})
        for row in _iter_jsonl(posts_path):
            text = (row.get("text") or "").strip()
            if _word_count(text) < _T4_MIN_TOKENS:
                skipped["short_text"] += 1
                continue
            ts = (row.get("timestamp") or "")[:10]
            sp = _split_label(_parse_date(ts))
            if sp is None:
                skipped["missing_date"] += 1
                continue
            record_id = str(row.get("id") or "")
            cves = src_cves.get(record_id, [])
            label, label_name = _task4_label(text, cves)
            splits_by_label[sp][label].append({
                "id": record_id,
                "text": text[:4096],
                "label": label,
                "label_name": label_name,
                "cve_ids": cves,
                "source": src,
                "forum_id": row.get("forum_id", src),
                "timestamp": ts,
                "split": sp,
                "label_source": "weak_cve_metadata_plus_actionability_terms",
            })

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    sampling: dict[str, dict[str, int]] = {}
    for sp, label_groups in splits_by_label.items():
        sampling[sp] = {}
        for label, records in label_groups.items():
            rng.shuffle(records)
            used = records[:_T4_MAX_PER_LABEL_SPLIT]
            splits[sp].extend(used)
            sampling[sp][str(label)] = len(used)
        rng.shuffle(splits[sp])

    counts = {}
    total = 0
    for sp, records in splits.items():
        n = _write_jsonl(records, task_dir / f"{sp}.jsonl")
        counts[sp] = n
        total += n
        log.info("task4: wrote %d records → %s/%s.jsonl", n, task_dir.name, sp)

    meta = {
        "task": "hacker_exploit_labeling",
        "description": (
            "Ternary weak-supervision task over hacker-community messages: "
            "non-exploit/noise, vulnerability discussion, or actionable exploit intelligence."
        ),
        "label_map": {
            "0": "non_exploit_noise",
            "1": "vulnerability_discussion",
            "2": "actionable_exploit",
        },
        "sources": _HACKER_COMMUNITY_SOURCES,
        "labeling_rule": (
            "actionable_exploit if a source CVE linkage is present and the message contains an "
            "actionability cue; vulnerability_discussion if a CVE/vulnerability cue is present "
            "without actionability; non_exploit_noise otherwise"
        ),
        "actionability_terms": _ACTIONABLE_TERMS,
        "vulnerability_terms": _VULNERABILITY_TERMS,
        "min_tokens": _T4_MIN_TOKENS,
        "max_per_label_split": _T4_MAX_PER_LABEL_SPLIT,
        "skipped": skipped,
        "split_boundaries": {
            "train_end": str(TRAIN_END),
            "val_end": str(VAL_END),
        },
        "metrics": ["f1_macro", "accuracy", "f1_per_class"],
        "splits": counts,
        "label_distribution_by_split": {
            sp: _label_counts(records, "label")
            for sp, records in splits.items()
        },
        "source_distribution_by_split": {
            sp: _label_counts(records, "source")
            for sp, records in splits.items()
        },
        "total": total,
    }
    _write_meta(meta, task_dir / "meta.json")
    return meta


# ---------------------------------------------------------------------------
# Task 5 — Hacker Exploit Signal Detection
# ---------------------------------------------------------------------------


def build_task5(data_dir: Path, output_dir: Path, seed: int = 42) -> dict:
    """Build actionable hacker-signal detection plus CVE-context retrieval splits."""
    rng = random.Random(seed)
    task4_dir = output_dir / "task4_hacker_exploit_labeling"
    if not (task4_dir / "train.jsonl").exists():
        build_task4(data_dir, output_dir, seed=seed)

    task_dir = output_dir / "task5_hacker_signal_detection"
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for sp in ("train", "val", "test"):
        positives: list[dict] = []
        negatives: list[dict] = []
        for row in _iter_jsonl(task4_dir / f"{sp}.jsonl"):
            cves = row.get("cve_ids") or []
            actionable = int(row.get("label") == 2 and bool(cves))
            record = {
                "id": row.get("id", ""),
                "text": row.get("text", ""),
                "actionable_label": actionable,
                "label_name": "actionable_exploit_signal" if actionable else "not_actionable_exploit_signal",
                "cve_id": cves[0] if actionable else "",
                "cve_ids": cves if actionable else [],
                "source": row.get("source", ""),
                "forum_id": row.get("forum_id", ""),
                "timestamp": row.get("timestamp", ""),
                "split": sp,
                "label_source": "derived_from_task4_actionable_cve_subset",
            }
            if actionable:
                positives.append(record)
            else:
                negatives.append(record)
        rng.shuffle(positives)
        rng.shuffle(negatives)
        n_neg = min(len(negatives), max(len(positives) * 2, len(positives)))
        splits[sp] = positives + negatives[:n_neg]
        rng.shuffle(splits[sp])

    counts = {}
    total = 0
    for sp, records in splits.items():
        n = _write_jsonl(records, task_dir / f"{sp}.jsonl")
        counts[sp] = n
        total += n
        log.info("task5: wrote %d records → %s/%s.jsonl", n, task_dir.name, sp)

    corpus = _build_nvd_corpus(data_dir)
    corpus_size = _write_jsonl(corpus, task_dir / "corpus.jsonl") if corpus else 0
    actionable_cves = {
        r["cve_id"] for records in splits.values() for r in records
        if r.get("actionable_label") == 1 and r.get("cve_id")
    }

    meta = {
        "task": "hacker_exploit_signal_detection",
        "description": (
            "Joint task: given a hacker-community message, detect whether it contains "
            "actionable exploit intelligence and, when actionable, recover the associated "
            "NVD CVE context."
        ),
        "label_map": {"0": "not_actionable_exploit_signal", "1": "actionable_exploit_signal"},
        "sources": _HACKER_COMMUNITY_SOURCES,
        "labeling_rule": "positive examples are Task 4 actionable_exploit records with CVE metadata",
        "negative_sampling": "up to two non-actionable messages per actionable message within each temporal split",
        "corpus_source": "nvd",
        "corpus_size": corpus_size,
        "unique_actionable_cves": len(actionable_cves),
        "split_boundaries": {
            "train_end": str(TRAIN_END),
            "val_end": str(VAL_END),
        },
        "metrics": ["f1_actionable", "recall_at_1", "recall_at_5", "recall_at_10", "mrr", "joint_recall_at_5"],
        "splits": counts,
        "label_distribution_by_split": {
            sp: _label_counts(records, "actionable_label")
            for sp, records in splits.items()
        },
        "source_distribution_by_split": {
            sp: _label_counts(records, "source")
            for sp, records in splits.items()
        },
        "total": total,
    }
    _write_meta(meta, task_dir / "meta.json")
    return meta


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build HackerSignal benchmark splits")
    parser.add_argument("--data-dir", default="data", help="Directory with *_posts.jsonl files")
    parser.add_argument("--output", default="data/benchmark", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", choices=["all", "1", "2", "3", "4", "5"], default="all")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)

    summary = {}
    if args.task in ("all", "1"):
        log.info("Building Task 1: Exploit Relevance Classification")
        summary["task1"] = build_task1(data_dir, output_dir, seed=args.seed)

    if args.task in ("all", "2"):
        log.info("Building Task 2: CVE Linkage Retrieval")
        summary["task2"] = build_task2(data_dir, output_dir, seed=args.seed)

    if args.task in ("all", "3"):
        log.info("Building Task 3: Severity Prediction")
        summary["task3"] = build_task3(data_dir, output_dir, seed=args.seed)

    if args.task in ("all", "4"):
        log.info("Building Task 4: Hacker Exploit Labeling")
        summary["task4"] = build_task4(data_dir, output_dir, seed=args.seed)

    if args.task in ("all", "5"):
        log.info("Building Task 5: Hacker Exploit Signal Detection")
        summary["task5"] = build_task5(data_dir, output_dir, seed=args.seed)

    print("\n=== Benchmark Split Summary ===")
    for task_id, meta in summary.items():
        print(f"\n[{task_id.upper()}: {meta['task']}]")
        for sp, n in meta.get("splits", {}).items():
            print(f"  {sp:<8} {n:>8,} records")

    (output_dir / "benchmark_meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
