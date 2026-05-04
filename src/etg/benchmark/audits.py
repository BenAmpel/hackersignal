"""Reviewer-facing audit utilities for ETG benchmark hardening."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import ssl
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SEVERITY_STRICT_PATTERNS: dict[str, re.Pattern[str]] = {
    "severity_field": re.compile(
        r"(?:^|\n)\s*\[?\s*(?:severity|sev|cvss[^:]*)\s*:\s*(?:critical|high|medium|moderate|low)\s*\]?",
        re.IGNORECASE,
    ),
    "cvss_score_field": re.compile(
        r"(?im)^\s*cvss(?:\s+(?:score|base score))?\s*:\s*[0-9.]+(?:\s*/\s*10)?\s*$"
    ),
    "cvss_vector": re.compile(r"CVSS:\d\.\d/[A-Z:0-9./-]+", re.IGNORECASE),
}

SEVERITY_WARNING_PATTERNS: dict[str, re.Pattern[str]] = {
    "severity_word_phrase": re.compile(
        r"\b(?:critical|high|medium|moderate|low)\s+severity\b|\bseverity\s+(?:critical|high|medium|moderate|low)\b",
        re.IGNORECASE,
    ),
    "cvss_near_number": re.compile(r"\bcvss\b.{0,40}\b[0-9](?:\.[0-9])?\s*(?:/10)?\b", re.IGNORECASE),
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _fingerprint(text: str) -> str:
    return hashlib.sha256(_norm_text(text).encode("utf-8", errors="ignore")).hexdigest()


def _sample_by_strata(
    records: Iterable[dict[str, Any]],
    strata_keys: list[str],
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(str(record.get(k, "")) for k in strata_keys)
        groups[key].append(record)
    if not groups:
        return []
    for group in groups.values():
        rng.shuffle(group)
    ordered = sorted(groups)
    chosen: list[dict[str, Any]] = []
    idx = 0
    while len(chosen) < n and any(groups.values()):
        key = ordered[idx % len(ordered)]
        if groups[key]:
            chosen.append(groups[key].pop())
        idx += 1
    rng.shuffle(chosen)
    return chosen


def sample_manual(
    benchmark_dir: Path,
    output_dir: Path,
    n_task1: int = 240,
    n_task2: int = 240,
    seed: int = 42,
) -> dict[str, Any]:
    """Create annotation packets; does not adjudicate labels."""
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    t1_records: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        for row in _load_jsonl(benchmark_dir / "task1_exploit_clf" / f"{split}.jsonl"):
            row = dict(row)
            row["split"] = split
            t1_records.append(row)
    t1_sample = _sample_by_strata(t1_records, ["split", "source", "label"], n_task1, rng)
    t1_rows = [
        {
            "audit_id": f"T1-{i:04d}",
            "task": "task1_exploit_relevance",
            "split": r.get("split", ""),
            "source": r.get("source", ""),
            "record_id": r.get("id", ""),
            "benchmark_label": r.get("label", ""),
            "label_meaning": "exploit_relevant" if int(r.get("label", 0)) == 1 else "not_exploit_relevant",
            "text": r.get("text", ""),
            "annotator_label": "",
            "annotator_confidence": "",
            "needs_second_review": "",
            "notes": "",
        }
        for i, r in enumerate(t1_sample, 1)
    ]

    t2_records: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        for row in _load_jsonl(benchmark_dir / "task2_cve_linkage" / f"{split}.jsonl"):
            row = dict(row)
            row["split"] = split
            t2_records.append(row)
    t2_sample = _sample_by_strata(t2_records, ["split", "source"], n_task2, rng)
    t2_rows = [
        {
            "audit_id": f"CVE-{i:04d}",
            "task": "task2_cve_linkage",
            "split": r.get("split", ""),
            "source": r.get("source", ""),
            "record_id": r.get("id", ""),
            "cve_id": r.get("cve_id", ""),
            "title": r.get("title", ""),
            "text": r.get("text", ""),
            "cve_link_correct": "",
            "annotator_confidence": "",
            "needs_second_review": "",
            "notes": "",
        }
        for i, r in enumerate(t2_sample, 1)
    ]

    t1_path = output_dir / "task1_manual_label_audit_packet.csv"
    t2_path = output_dir / "task2_cve_linkage_manual_audit_packet.csv"
    _write_csv(t1_rows, t1_path, list(t1_rows[0].keys()) if t1_rows else [])
    _write_csv(t2_rows, t2_path, list(t2_rows[0].keys()) if t2_rows else [])

    summary = {
        "seed": seed,
        "mode": "annotation_packet_only",
        "task1_packet": str(t1_path),
        "task1_rows": len(t1_rows),
        "task1_strata": ["split", "source", "label"],
        "task2_packet": str(t2_path),
        "task2_rows": len(t2_rows),
        "task2_strata": ["split", "source"],
        "human_labels_present": False,
    }
    _write_json(summary, output_dir / "manual_audit_packet_summary.json")
    return summary


def summarize_manual(packet: Path, output: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(packet.open(encoding="utf-8")))
    completed = 0
    agreements = 0
    auditable = 0
    for row in rows:
        if row.get("annotator_label") not in ("", None):
            completed += 1
            if "benchmark_label" in row:
                auditable += 1
                expected = "1" if row.get("annotator_label") in {"1", "yes", "true", "exploit_relevant"} else "0"
                agreements += int(expected == str(row.get("benchmark_label")))
            elif row.get("cve_link_correct") not in ("", None):
                auditable += 1
                agreements += int(str(row.get("cve_link_correct")).lower() in {"1", "yes", "true", "correct"})
    summary = {
        "packet": str(packet),
        "rows": len(rows),
        "completed_rows": completed,
        "auditable_rows": auditable,
        "agreement_or_precision": round(agreements / auditable, 4) if auditable else None,
        "status": "complete" if completed == len(rows) and rows else "annotation_pending",
    }
    _write_json(summary, output)
    return summary


LLM_JUDGES: dict[str, dict[str, str]] = {
    "cti_triage": {
        "name": "CTI triage analyst",
        "style": (
            "You are an automated cybersecurity triage judge. Prioritize practical CTI relevance, "
            "exploitability, vulnerability evidence, and conservative uncertainty handling."
        ),
    },
    "vuln_research": {
        "name": "vulnerability research analyst",
        "style": (
            "You are an automated vulnerability-research judge. Prioritize technical specificity, "
            "CVE evidence, proof-of-concept language, affected component details, and false-positive control."
        ),
    },
}


def _truncate_text(text: str, limit: int = 4500) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + " ... [truncated]"


def _json_from_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in model response: {text[:300]}")
    return json.loads(match.group(0))


def _openai_chat_json(
    *,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for live LLM judge audits")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    context = None
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(f"{base_url}/chat/completions", data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
                doc = json.loads(resp.read().decode("utf-8"))
            content = doc["choices"][0]["message"]["content"]
            return _json_from_response(content)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    last_error = exc.read().decode("utf-8", errors="replace")[:1000]
                except Exception:
                    pass
            if attempt == max_retries:
                break
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"OpenAI judge request failed after {max_retries} attempts: {last_error}")


def _task1_prompt(row: dict[str, str], judge: dict[str, str]) -> list[dict[str, str]]:
    system = (
        f"{judge['style']} This is an LLM-assisted audit, not a human annotation. "
        "Return only JSON. Do not include chain-of-thought."
    )
    user = {
        "audit_task": "task1_exploit_relevance_label_audit",
        "definition": (
            "Judge whether the text is exploit-relevant. Positive means the text contains actionable exploit, "
            "vulnerability, proof-of-concept, bug-bounty vulnerability, exploit-code, attack technique, or concrete "
            "security flaw evidence. Negative means generic chat, marketplace/payment talk, non-exploit admin text, "
            "or insufficient security content."
        ),
        "allowed_json_schema": {
            "llm_label": "0 or 1",
            "agrees_with_benchmark": "true or false",
            "confidence": "low, medium, or high",
            "needs_human_review": "true or false",
            "error_type": "none, weak_negative, weak_positive, ambiguous, off_topic, insufficient_context, other",
            "rationale": "one concise sentence",
        },
        "record": {
            "audit_id": row.get("audit_id"),
            "source": row.get("source"),
            "split": row.get("split"),
            "benchmark_label": row.get("benchmark_label"),
            "label_meaning": row.get("label_meaning"),
            "text": _truncate_text(row.get("text", "")),
        },
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def _task2_prompt(row: dict[str, str], judge: dict[str, str]) -> list[dict[str, str]]:
    system = (
        f"{judge['style']} This is an LLM-assisted audit, not a human annotation. "
        "Return only JSON. Do not include chain-of-thought."
    )
    user = {
        "audit_task": "task2_cve_linkage_precision_audit",
        "definition": (
            "Judge whether the provided text is correctly linked to the given CVE. Mark correct if the text explicitly "
            "mentions the CVE, describes the same affected product/vulnerability, is an official advisory/fix record for "
            "that CVE, or otherwise provides clear evidence of the linkage. Mark incorrect when the text points to a "
            "different CVE, is too vague, or lacks evidence."
        ),
        "allowed_json_schema": {
            "cve_link_correct": "true or false",
            "confidence": "low, medium, or high",
            "needs_human_review": "true or false",
            "error_type": "none, wrong_cve, too_vague, multiple_cves, metadata_parse_error, insufficient_context, other",
            "rationale": "one concise sentence",
        },
        "record": {
            "audit_id": row.get("audit_id"),
            "source": row.get("source"),
            "split": row.get("split"),
            "cve_id": row.get("cve_id"),
            "title": row.get("title"),
            "text": _truncate_text(row.get("text", "")),
        },
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def _normalise_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "correct", "agree"}:
            return True
        if value in {"0", "false", "no", "incorrect", "disagree"}:
            return False
    return None


def _normalise_task1_label(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "positive", "exploit_relevant"}:
        return "1"
    return "0"


def _load_jsonl_cache(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            doc = json.loads(line)
            cache[(str(doc.get("audit_id")), str(doc.get("judge_id")))] = doc
    return cache


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_llm_csv(task: str, source_rows: list[dict[str, str]], judgments: dict[tuple[str, str], dict[str, Any]], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    judge_ids = list(LLM_JUDGES)
    for row in source_rows:
        out = dict(row)
        decisions: list[Any] = []
        for judge_id in judge_ids:
            judgment = judgments.get((row["audit_id"], judge_id), {})
            prefix = f"{judge_id}_"
            out[prefix + "status"] = judgment.get("status", "")
            out[prefix + "confidence"] = judgment.get("confidence", "")
            out[prefix + "needs_human_review"] = judgment.get("needs_human_review", "")
            out[prefix + "error_type"] = judgment.get("error_type", "")
            out[prefix + "rationale"] = judgment.get("rationale", "")
            if task == "task1":
                label = judgment.get("llm_label", "")
                out[prefix + "llm_label"] = label
                out[prefix + "agrees_with_benchmark"] = judgment.get("agrees_with_benchmark", "")
                if label != "":
                    decisions.append(str(label))
            else:
                correct = judgment.get("cve_link_correct", "")
                out[prefix + "cve_link_correct"] = correct
                if correct != "":
                    decisions.append(bool(correct))
        if len(decisions) == 2 and decisions[0] == decisions[1]:
            out["llm_consensus"] = decisions[0]
            out["llm_disagreement"] = False
        else:
            out["llm_consensus"] = ""
            out["llm_disagreement"] = True
        out["audit_mode"] = "llm_assisted_not_human"
        rows.append(out)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    _write_csv(rows, path, fieldnames)


def summarize_llm_judge(task1_csv: Path | None, task2_csv: Path | None, output: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "audit_mode": "llm_assisted_not_human",
        "human_labels_present": False,
        "tasks": {},
        "disclosure": "These labels are automated LLM judgments and must not be reported as qualified human-professional labels.",
    }
    if task1_csv and task1_csv.exists():
        rows = list(csv.DictReader(task1_csv.open(encoding="utf-8")))
        completed = [r for r in rows if r.get("cti_triage_status") == "ok" and r.get("vuln_research_status") == "ok"]
        consensus = [r for r in completed if str(r.get("llm_disagreement", "")).lower() == "false"]
        agreements = [r for r in consensus if str(r.get("llm_consensus")) == str(r.get("benchmark_label"))]
        summary["tasks"]["task1"] = {
            "rows": len(rows),
            "completed_rows": len(completed),
            "consensus_rows": len(consensus),
            "disagreement_rows": len(completed) - len(consensus),
            "benchmark_agreement_on_consensus": round(len(agreements) / max(len(consensus), 1), 4) if consensus else None,
            "needs_human_review_rows": sum(
                str(r.get("cti_triage_needs_human_review")).lower() == "true"
                or str(r.get("vuln_research_needs_human_review")).lower() == "true"
                or str(r.get("llm_disagreement")).lower() == "true"
                for r in completed
            ),
        }
    if task2_csv and task2_csv.exists():
        rows = list(csv.DictReader(task2_csv.open(encoding="utf-8")))
        completed = [r for r in rows if r.get("cti_triage_status") == "ok" and r.get("vuln_research_status") == "ok"]
        consensus = [r for r in completed if str(r.get("llm_disagreement", "")).lower() == "false"]
        correct = [r for r in consensus if str(r.get("llm_consensus")).lower() == "true"]
        summary["tasks"]["task2"] = {
            "rows": len(rows),
            "completed_rows": len(completed),
            "consensus_rows": len(consensus),
            "disagreement_rows": len(completed) - len(consensus),
            "consensus_precision_estimate": round(len(correct) / max(len(consensus), 1), 4) if consensus else None,
            "needs_human_review_rows": sum(
                str(r.get("cti_triage_needs_human_review")).lower() == "true"
                or str(r.get("vuln_research_needs_human_review")).lower() == "true"
                or str(r.get("llm_disagreement")).lower() == "true"
                for r in completed
            ),
        }
    _write_json(summary, output)
    return summary


def llm_judge(
    input_dir: Path,
    output_dir: Path,
    task: str = "all",
    model: str = "gpt-4o-mini",
    limit: int | None = None,
    temperature: float = 0.0,
    timeout: int = 60,
    max_retries: int = 4,
) -> dict[str, Any]:
    """Run two independent LLM-assisted judges over audit packets.

    Outputs are explicitly marked as automated LLM judgments and must not be
    merged into the manual annotator columns.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "llm_judge_raw.jsonl"
    cache = _load_jsonl_cache(raw_path)
    generated = 0
    failures = 0
    outputs: dict[str, str] = {}

    task_specs: list[tuple[str, Path, Path, Any]] = []
    if task in {"all", "task1"}:
        task_specs.append(
            (
                "task1",
                input_dir / "task1_manual_label_audit_packet.csv",
                output_dir / "task1_llm_judge_audit.csv",
                _task1_prompt,
            )
        )
    if task in {"all", "task2"}:
        task_specs.append(
            (
                "task2",
                input_dir / "task2_cve_linkage_manual_audit_packet.csv",
                output_dir / "task2_llm_judge_audit.csv",
                _task2_prompt,
            )
        )
    if not task_specs:
        raise ValueError(f"Unsupported task: {task}")

    for task_name, packet, csv_output, prompt_fn in task_specs:
        rows = list(csv.DictReader(packet.open(encoding="utf-8")))
        if limit is not None:
            rows = rows[:limit]
        for row in rows:
            for judge_id, judge in LLM_JUDGES.items():
                key = (row["audit_id"], judge_id)
                if key in cache and cache[key].get("status") == "ok":
                    continue
                try:
                    doc = _openai_chat_json(
                        messages=prompt_fn(row, judge),
                        model=model,
                        temperature=temperature,
                        timeout=timeout,
                        max_retries=max_retries,
                    )
                    normalized: dict[str, Any] = {
                        "audit_id": row["audit_id"],
                        "task": task_name,
                        "judge_id": judge_id,
                        "judge_name": judge["name"],
                        "model": model,
                        "status": "ok",
                        "confidence": str(doc.get("confidence", "")).lower(),
                        "needs_human_review": _normalise_bool(doc.get("needs_human_review")),
                        "error_type": str(doc.get("error_type", "other")).lower(),
                        "rationale": str(doc.get("rationale", ""))[:500],
                    }
                    if task_name == "task1":
                        label = _normalise_task1_label(doc.get("llm_label"))
                        normalized["llm_label"] = label
                        normalized["agrees_with_benchmark"] = label == str(row.get("benchmark_label"))
                    else:
                        normalized["cve_link_correct"] = _normalise_bool(doc.get("cve_link_correct"))
                    cache[key] = normalized
                    _append_jsonl(raw_path, normalized)
                    generated += 1
                except Exception as exc:
                    failures += 1
                    failure = {
                        "audit_id": row["audit_id"],
                        "task": task_name,
                        "judge_id": judge_id,
                        "judge_name": judge["name"],
                        "model": model,
                        "status": "failed",
                        "error": str(exc)[:1000],
                    }
                    cache[key] = failure
                    _append_jsonl(raw_path, failure)
        task_cache = {k: v for k, v in cache.items() if v.get("task") == task_name}
        _write_llm_csv(task_name, rows, task_cache, csv_output)
        outputs[task_name] = str(csv_output)

    summary = summarize_llm_judge(
        Path(outputs["task1"]) if "task1" in outputs else None,
        Path(outputs["task2"]) if "task2" in outputs else None,
        output_dir / "llm_judge_summary.json",
    )
    summary["model"] = model
    summary["generated_judgments_this_run"] = generated
    summary["failed_judgments_this_run"] = failures
    summary["raw_judgments"] = str(raw_path)
    summary["outputs"] = outputs
    _write_json(summary, output_dir / "llm_judge_summary.json")
    return summary


def leakage_audit(benchmark_dir: Path, output: Path, fail_on_strict: bool = False) -> dict[str, Any]:
    task_dir = benchmark_dir / "task3_severity"
    by_split: dict[str, Any] = {}
    strict_total = 0
    warning_total = 0
    raw_strict_total = 0
    examples: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        rows = _load_jsonl(task_dir / f"{split}.jsonl")
        split_counts = {
            "rows": len(rows),
            "strict_sanitized_hits": 0,
            "warning_sanitized_hits": 0,
            "strict_raw_hits": 0,
            "pattern_counts": Counter(),
        }
        for row in rows:
            text = row.get("text", "") or ""
            raw = row.get("text_raw", "") or ""
            strict_hits = [name for name, pat in SEVERITY_STRICT_PATTERNS.items() if pat.search(text)]
            warning_hits = [name for name, pat in SEVERITY_WARNING_PATTERNS.items() if pat.search(text)]
            raw_hits = [name for name, pat in SEVERITY_STRICT_PATTERNS.items() if pat.search(raw)]
            if strict_hits:
                split_counts["strict_sanitized_hits"] += 1
                strict_total += 1
                if len(examples) < 20:
                    examples.append({"split": split, "id": row.get("id"), "patterns": strict_hits, "text": text[:500]})
            if warning_hits:
                split_counts["warning_sanitized_hits"] += 1
                warning_total += 1
            if raw_hits:
                split_counts["strict_raw_hits"] += 1
                raw_strict_total += 1
            for name in strict_hits + warning_hits:
                split_counts["pattern_counts"][name] += 1
        split_counts["pattern_counts"] = dict(split_counts["pattern_counts"])
        by_split[split] = split_counts
    report = {
        "task": "task3_severity",
        "blocking_policy": "strict sanitized hits must be zero",
        "strict_sanitized_hits": strict_total,
        "warning_sanitized_hits": warning_total,
        "strict_raw_hits": raw_strict_total,
        "by_split": by_split,
        "examples": examples,
        "passed": strict_total == 0,
    }
    _write_json(report, output)
    if fail_on_strict and strict_total:
        raise SystemExit(f"Task 3 strict leakage audit failed with {strict_total} sanitized hits")
    return report


def overlap_audit(benchmark_dir: Path, output: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"tasks": {}, "cross_task_text_overlap": {}}
    task_specs = {
        "task1": ("task1_exploit_clf", ("train", "val", "test")),
        "task2": ("task2_cve_linkage", ("train", "val", "test")),
        "task3": ("task3_severity", ("train", "val", "test")),
        "task4": ("task4_hacker_exploit_labeling", ("train", "val", "test")),
        "task5": ("task5_hacker_signal_detection", ("train", "val", "test")),
    }
    task_hashes: dict[str, dict[str, set[str]]] = {}
    for task, (dirname, splits) in task_specs.items():
        if not (benchmark_dir / dirname).exists():
            continue
        split_hashes: dict[str, set[str]] = {}
        for split in splits:
            split_hashes[split] = {
                _fingerprint(r.get("text", "") or "")
                for r in _load_jsonl(benchmark_dir / dirname / f"{split}.jsonl")
                if r.get("text")
            }
        task_hashes[task] = split_hashes
        report["tasks"][task] = {
            "train_val_overlap": len(split_hashes["train"] & split_hashes["val"]),
            "train_test_overlap": len(split_hashes["train"] & split_hashes["test"]),
            "val_test_overlap": len(split_hashes["val"] & split_hashes["test"]),
        }
    all_by_task = {task: set().union(*splits.values()) for task, splits in task_hashes.items()}
    for left in task_hashes:
        for right in task_hashes:
            if left >= right:
                continue
            report["cross_task_text_overlap"][f"{left}_{right}"] = len(all_by_task[left] & all_by_task[right])
    _write_json(report, output)
    return report


def quality_by_task(benchmark_dir: Path, output: Path) -> dict[str, Any]:
    task_specs = {
        "task1": ("task1_exploit_clf", "label"),
        "task2": ("task2_cve_linkage", "cve_id"),
        "task3": ("task3_severity", "severity_label"),
        "task4": ("task4_hacker_exploit_labeling", "label"),
        "task5": ("task5_hacker_signal_detection", "actionable_label"),
    }
    report: dict[str, Any] = {"tasks": {}}
    for task, (dirname, label_key) in task_specs.items():
        if not (benchmark_dir / dirname).exists():
            continue
        task_report: dict[str, Any] = {}
        for split in ("train", "val", "test"):
            rows = _load_jsonl(benchmark_dir / dirname / f"{split}.jsonl")
            hashes = [_fingerprint(r.get("text", "") or "") for r in rows if r.get("text")]
            counter = Counter(hashes)
            short = sum(1 for r in rows if len((r.get("text", "") or "").split()) < 8)
            non_ascii = sum(1 for r in rows if sum(ord(ch) > 127 for ch in (r.get("text", "") or "")) > 10)
            label_counts = Counter(str(r.get(label_key, "")) for r in rows)
            task_report[split] = {
                "rows": len(rows),
                "unique_labels": len(label_counts),
                "label_counts": dict(sorted(label_counts.items())) if len(label_counts) <= 20 else {},
                "top_labels": dict(label_counts.most_common(20)) if len(label_counts) > 20 else {},
                "short_under_8_tokens": short,
                "short_under_8_tokens_pct": round(short / max(len(rows), 1), 4),
                "non_ascii_signal_rows": non_ascii,
                "non_ascii_signal_rows_pct": round(non_ascii / max(len(rows), 1), 4),
                "exact_duplicate_text_rows": sum(v - 1 for v in counter.values() if v > 1),
            }
        report["tasks"][task] = task_report
    _write_json(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="ETG benchmark audit utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sample-manual")
    p.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmark"))
    p.add_argument("--output-dir", type=Path, default=Path("data/audits"))
    p.add_argument("--n-task1", type=int, default=240)
    p.add_argument("--n-task2", type=int, default=240)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("summarize-manual")
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("leakage")
    p.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmark"))
    p.add_argument("--output", type=Path, default=Path("data/audits/task3_leakage_audit.json"))
    p.add_argument("--fail-on-strict", action="store_true")

    p = sub.add_parser("overlap")
    p.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmark"))
    p.add_argument("--output", type=Path, default=Path("data/audits/split_overlap_audit.json"))

    p = sub.add_parser("quality-by-task")
    p.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmark"))
    p.add_argument("--output", type=Path, default=Path("data/audits/quality_by_task.json"))

    p = sub.add_parser("llm-judge")
    p.add_argument("--input-dir", type=Path, default=Path("data/audits"))
    p.add_argument("--output-dir", type=Path, default=Path("data/audits/llm_judge"))
    p.add_argument("--task", choices=["all", "task1", "task2"], default="all")
    p.add_argument("--model", default=os.environ.get("ETG_LLM_JUDGE_MODEL", "gpt-4o-mini"))
    p.add_argument("--limit", type=int)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--max-retries", type=int, default=4)

    args = parser.parse_args()
    if args.command == "sample-manual":
        out = sample_manual(args.benchmark_dir, args.output_dir, args.n_task1, args.n_task2, args.seed)
    elif args.command == "summarize-manual":
        out = summarize_manual(args.packet, args.output)
    elif args.command == "leakage":
        out = leakage_audit(args.benchmark_dir, args.output, args.fail_on_strict)
    elif args.command == "overlap":
        out = overlap_audit(args.benchmark_dir, args.output)
    elif args.command == "quality-by-task":
        out = quality_by_task(args.benchmark_dir, args.output)
    elif args.command == "llm-judge":
        out = llm_judge(
            args.input_dir,
            args.output_dir,
            args.task,
            args.model,
            args.limit,
            args.temperature,
            args.timeout,
            args.max_retries,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
