"""Corpus audit and filtering helpers for raw forum crawls."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .preprocessing import (
    PreprocessConfig,
    _ascii_ratio,
    _text_fingerprint,
    cti_score,
    has_real_timestamp,
    parse_timestamp,
    profile_raw_file,
    run_pipeline,
)
from .schemas import dump_jsonl


def default_corpus_config() -> PreprocessConfig:
    """Return a broad retention config suitable for multilingual corpus growth."""
    return PreprocessConfig(
        require_real_timestamp=False,
        min_ascii_ratio=0.0,
        use_langdetect=False,
        min_tokens=8,
        max_tokens=3_000,
        apply_cti_filter=False,
        exact_dedup=True,
        near_dedup=True,
    )


def _pct(n: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * n / total, 2)


def audit_raw_corpus(raw_path: str | Path) -> dict[str, Any]:
    """Return a streaming audit summary for a raw crawl JSONL file."""
    path = Path(raw_path)
    total_rows = 0
    rows_with_body = 0
    rows_with_timestamp = 0
    rows_with_real_timestamp = 0
    rows_with_cve_refs = 0
    rows_cti_nonzero = 0
    exact_duplicate_rows = 0
    duplicate_id_rows = 0
    ascii_like_rows = 0
    cyrillic_like_rows = 0

    seen_text_fingerprints: set[str] = set()
    seen_ids: set[str] = set()
    forums: Counter[str] = Counter()
    sections: Counter[str] = Counter()
    years: Counter[int] = Counter()
    thread_urls: set[str] = set()
    token_lengths: list[int] = []

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_rows += 1
            record_id = record.get("id")
            if record_id in seen_ids:
                duplicate_id_rows += 1
            elif record_id:
                seen_ids.add(record_id)

            forum_id = record.get("forum_id") or record.get("forum_name") or "unknown"
            forums[forum_id] += 1

            section = (record.get("section") or "").strip()
            if section:
                sections[section] += 1

            thread_url = record.get("thread_url")
            if thread_url:
                thread_urls.add(thread_url)

            body = (record.get("body") or record.get("text") or "").strip()
            if body:
                rows_with_body += 1

            timestamp = record.get("timestamp")
            if timestamp:
                rows_with_timestamp += 1
                ts = parse_timestamp(timestamp)
                if ts is not None:
                    years[ts.year] += 1

            if has_real_timestamp(record):
                rows_with_real_timestamp += 1

            if record.get("cve_refs"):
                rows_with_cve_refs += 1

            if not body:
                continue

            cleaned = body.strip()
            token_lengths.append(len(cleaned.split()))
            ratio = _ascii_ratio(cleaned)
            if ratio >= 0.70:
                ascii_like_rows += 1
            if any("\u0400" <= c <= "\u04ff" for c in cleaned):
                cyrillic_like_rows += 1

            if cti_score(cleaned) > 0:
                rows_cti_nonzero += 1

            fingerprint = _text_fingerprint(cleaned)
            if fingerprint in seen_text_fingerprints:
                exact_duplicate_rows += 1
            else:
                seen_text_fingerprints.add(fingerprint)

    profile = profile_raw_file(path, n_sample=min(10_000, total_rows or 10_000))
    median_tokens = sorted(token_lengths)[len(token_lengths) // 2] if token_lengths else 0

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(path),
        "total_rows": total_rows,
        "unique_forums": len(forums),
        "unique_threads": len(thread_urls),
        "rows_with_body": rows_with_body,
        "rows_with_timestamp": rows_with_timestamp,
        "rows_with_real_timestamp": rows_with_real_timestamp,
        "rows_with_cve_refs": rows_with_cve_refs,
        "rows_cti_nonzero": rows_cti_nonzero,
        "unique_ids": len(seen_ids),
        "duplicate_id_rows": duplicate_id_rows,
        "exact_duplicate_rows": exact_duplicate_rows,
        "exact_duplicate_rate_pct": _pct(exact_duplicate_rows, total_rows),
        "real_timestamp_rate_pct": _pct(rows_with_real_timestamp, total_rows),
        "cti_rate_pct": _pct(rows_cti_nonzero, total_rows),
        "ascii_like_rate_pct": _pct(ascii_like_rows, total_rows),
        "cyrillic_like_rate_pct": _pct(cyrillic_like_rows, total_rows),
        "median_tokens": median_tokens,
        "top_forums": forums.most_common(15),
        "top_sections": sections.most_common(15),
        "year_counts": sorted(years.items()),
        "sample_profile": profile,
    }


def render_audit_markdown(report: dict[str, Any]) -> str:
    """Render a compact markdown audit report."""
    top_forums = report.get("top_forums", [])
    top_sections = report.get("top_sections", [])
    year_counts = report.get("year_counts", [])
    pipeline_stats = report.get("pipeline_stats")
    config = report.get("filter_config")

    lines = [
        "# Corpus Audit",
        "",
        f"- Generated: `{report.get('generated_at', '')}`",
        f"- Raw path: `{report.get('raw_path', '')}`",
        f"- Total rows: `{report.get('total_rows', 0)}`",
        f"- Unique forums: `{report.get('unique_forums', 0)}`",
        f"- Unique threads: `{report.get('unique_threads', 0)}`",
        f"- Exact duplicate rows: `{report.get('exact_duplicate_rows', 0)}` ({report.get('exact_duplicate_rate_pct', 0)}%)",
        f"- Real timestamps: `{report.get('rows_with_real_timestamp', 0)}` ({report.get('real_timestamp_rate_pct', 0)}%)",
        f"- CTI-nonzero rows: `{report.get('rows_cti_nonzero', 0)}` ({report.get('cti_rate_pct', 0)}%)",
        f"- ASCII-like rows: `{report.get('ascii_like_rate_pct', 0)}%`",
        f"- Cyrillic-like rows: `{report.get('cyrillic_like_rate_pct', 0)}%`",
        "",
        "## Top Forums",
        "",
    ]
    lines.extend(f"- `{forum}`: `{count}`" for forum, count in top_forums[:10])
    lines.extend(["", "## Top Sections", ""])
    lines.extend(f"- `{section}`: `{count}`" for section, count in top_sections[:10])
    lines.extend(["", "## Year Counts", ""])
    lines.extend(f"- `{year}`: `{count}`" for year, count in year_counts[:10])

    if pipeline_stats:
        lines.extend(
            [
                "",
                "## Filtered Corpus",
                "",
                f"- Filtered posts written: `{report.get('filtered_posts', 0)}`",
                f"- Filtered output: `{report.get('filtered_output', '')}`",
                f"- After forum filter: `{pipeline_stats.get('after_allowlist', 0)}`",
                f"- After timestamp filter: `{pipeline_stats.get('after_timestamp_filter', 0)}`",
                f"- After language filter: `{pipeline_stats.get('after_language_filter', 0)}`",
                f"- After length filter: `{pipeline_stats.get('after_length_filter', 0)}`",
                f"- After CTI filter: `{pipeline_stats.get('after_cti_filter', 0)}`",
                f"- After exact dedup: `{pipeline_stats.get('after_exact_dedup', 0)}`",
                f"- After near dedup: `{pipeline_stats.get('after_near_dedup', 0)}`",
                f"- Final after sample: `{pipeline_stats.get('after_sample', 0)}`",
            ]
        )
    if config:
        lines.extend(["", "## Filter Config", "", "```json", json.dumps(config, indent=2), "```"])

    return "\n".join(lines) + "\n"


def build_filtered_corpus(
    raw_path: str | Path,
    filtered_output: str | Path,
    *,
    config: PreprocessConfig | None = None,
    audit_json: str | Path | None = None,
    audit_markdown: str | Path | None = None,
) -> dict[str, Any]:
    """Audit a raw corpus, write a filtered corpus, and persist audit artifacts."""
    if config is None:
        config = default_corpus_config()

    posts, stats = run_pipeline(raw_path, config=config, verbose=False)
    filtered_output = Path(filtered_output)
    dump_jsonl(posts, filtered_output)

    report = audit_raw_corpus(raw_path)
    report["filtered_output"] = str(filtered_output)
    report["filtered_posts"] = len(posts)
    report["pipeline_stats"] = asdict(stats)
    report["filter_config"] = asdict(config)

    if audit_json:
        audit_json = Path(audit_json)
        audit_json.parent.mkdir(parents=True, exist_ok=True)
        audit_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if audit_markdown:
        audit_markdown = Path(audit_markdown)
        audit_markdown.parent.mkdir(parents=True, exist_ok=True)
        audit_markdown.write_text(render_audit_markdown(report), encoding="utf-8")

    return report
