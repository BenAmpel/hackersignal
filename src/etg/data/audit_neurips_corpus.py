"""Streaming audit for the expanded NeurIPS ETG corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from etg.data.preprocessing import cti_score, parse_timestamp

_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")
_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 3) if total else 0.0


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _stable_int(value: str, *, nbytes: int = 8) -> int:
    return int.from_bytes(
        hashlib.blake2b(value.encode("utf-8", errors="ignore"), digest_size=nbytes).digest(),
        "big",
    )


def _simhash(text: str, *, bits: int = 64, shingle_k: int = 5) -> int:
    toks = _tokens(text)
    if len(toks) < shingle_k:
        shingles = toks
    else:
        shingles = [" ".join(toks[i:i + shingle_k]) for i in range(len(toks) - shingle_k + 1)]
    if not shingles:
        return 0
    vector = [0] * bits
    for shingle in shingles[:500]:
        h = _stable_int(shingle)
        for bit in range(bits):
            vector[bit] += 1 if h & (1 << bit) else -1
    out = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            out |= 1 << bit
    return out


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _sample_keep(key: str, sample_rate: float) -> bool:
    if sample_rate >= 1:
        return True
    threshold = int(sample_rate * (2**64 - 1))
    return _stable_int(key) <= threshold


def _write_counter_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_neurips_corpus(
    input_path: str | Path,
    *,
    output_prefix: str | Path = "data/unified_hacker_communities_neurips_audit",
    sample_rate: float = 0.03,
    near_hamming_threshold: int = 3,
    max_bucket_candidates: int = 40,
) -> dict[str, Any]:
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be in the interval (0, 1]")

    input_path = Path(input_path)
    output_prefix = Path(output_prefix)

    total = 0
    with_timestamp = 0
    cti_nonzero = 0
    cve_rows = 0
    short_rows = 0
    long_rows = 0
    ascii_like = 0
    cyrillic_like = 0

    layer_counts: Counter[str] = Counter()
    forum_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    forum_category_counts: Counter[tuple[str, str, str]] = Counter()
    forum_layer_counts: Counter[tuple[str, str]] = Counter()
    year_counts: Counter[int] = Counter()
    token_hist: Counter[str] = Counter()

    sampled = 0
    near_duplicate_sample_rows = 0
    band_index: dict[tuple[int, int], list[tuple[int, str, str, str]]] = {}
    near_examples: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1

            text = (row.get("text") or row.get("body") or "").strip()
            layer = row.get("source_layer") or "unknown"
            forum = row.get("forum_id") or "unknown"
            source = row.get("source_dataset") or "unknown"
            section = (row.get("section") or "uncategorized").strip() or "uncategorized"

            layer_counts[layer] += 1
            forum_counts[forum] += 1
            source_counts[source] += 1
            forum_layer_counts[(layer, forum)] += 1
            forum_category_counts[(forum, layer, section)] += 1

            ts = parse_timestamp(row.get("timestamp") or "")
            if ts is not None:
                with_timestamp += 1
                year_counts[ts.year] += 1

            toks = _tokens(text)
            ntok = len(toks)
            if ntok < 8:
                short_rows += 1
            if ntok > 3000:
                long_rows += 1
            if ntok < 8:
                token_hist["lt_8"] += 1
            elif ntok < 32:
                token_hist["8_31"] += 1
            elif ntok < 128:
                token_hist["32_127"] += 1
            elif ntok < 512:
                token_hist["128_511"] += 1
            elif ntok < 3000:
                token_hist["512_2999"] += 1
            else:
                token_hist["gte_3000"] += 1

            if text:
                ascii_chars = sum(1 for c in text if ord(c) < 128)
                if ascii_chars / max(len(text), 1) >= 0.70:
                    ascii_like += 1
                if any("\u0400" <= c <= "\u04ff" for c in text):
                    cyrillic_like += 1
                if cti_score(text) > 0:
                    cti_nonzero += 1
                if row.get("cve_refs") or _CVE_RE.search(text):
                    cve_rows += 1

            sample_key = row.get("unified_id") or row.get("id") or text[:200]
            if text and _sample_keep(str(sample_key), sample_rate):
                sampled += 1
                sim = _simhash(text)
                matched = None
                for band in range(4):
                    value = (sim >> (band * 16)) & 0xFFFF
                    candidates = band_index.get((band, value), [])
                    for prev_sim, prev_forum, prev_section, prev_preview in candidates[:max_bucket_candidates]:
                        dist = _hamming(sim, prev_sim)
                        if dist <= near_hamming_threshold:
                            matched = (dist, prev_forum, prev_section, prev_preview)
                            break
                    if matched:
                        break
                if matched:
                    near_duplicate_sample_rows += 1
                    if len(near_examples) < 20:
                        dist, prev_forum, prev_section, prev_preview = matched
                        near_examples.append({
                            "forum_id": forum,
                            "section": section,
                            "matched_forum_id": prev_forum,
                            "matched_section": prev_section,
                            "hamming_distance": dist,
                            "text_preview": text[:180],
                            "matched_preview": prev_preview,
                        })
                else:
                    preview = text[:180]
                    for band in range(4):
                        value = (sim >> (band * 16)) & 0xFFFF
                        bucket = band_index.setdefault((band, value), [])
                        if len(bucket) < max_bucket_candidates:
                            bucket.append((sim, forum, section, preview))

    forum_rows = [
        {
            "forum_id": forum,
            "rows": count,
            "pct": _pct(count, total),
        }
        for forum, count in forum_counts.most_common()
    ]
    category_rows = [
        {
            "forum_id": forum,
            "source_layer": layer,
            "section": section,
            "rows": count,
            "pct": _pct(count, total),
        }
        for (forum, layer, section), count in forum_category_counts.most_common()
    ]
    layer_rows = [
        {"source_layer": layer, "rows": count, "pct": _pct(count, total)}
        for layer, count in layer_counts.most_common()
    ]
    forum_layer_rows = [
        {"source_layer": layer, "forum_id": forum, "rows": count, "pct": _pct(count, total)}
        for (layer, forum), count in forum_layer_counts.most_common()
    ]

    _write_counter_csv(
        output_prefix.with_name(output_prefix.name + "_forum_counts.csv"),
        ["forum_id", "rows", "pct"],
        forum_rows,
    )
    _write_counter_csv(
        output_prefix.with_name(output_prefix.name + "_forum_category_counts.csv"),
        ["forum_id", "source_layer", "section", "rows", "pct"],
        category_rows,
    )
    _write_counter_csv(
        output_prefix.with_name(output_prefix.name + "_layer_counts.csv"),
        ["source_layer", "rows", "pct"],
        layer_rows,
    )
    _write_counter_csv(
        output_prefix.with_name(output_prefix.name + "_forum_layer_counts.csv"),
        ["source_layer", "forum_id", "rows", "pct"],
        forum_layer_rows,
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "total_rows": total,
        "rows_with_parseable_timestamp": with_timestamp,
        "rows_with_parseable_timestamp_pct": _pct(with_timestamp, total),
        "cti_nonzero_rows": cti_nonzero,
        "cti_nonzero_pct": _pct(cti_nonzero, total),
        "cve_rows": cve_rows,
        "cve_rows_pct": _pct(cve_rows, total),
        "ascii_like_rows": ascii_like,
        "ascii_like_pct": _pct(ascii_like, total),
        "cyrillic_like_rows": cyrillic_like,
        "cyrillic_like_pct": _pct(cyrillic_like, total),
        "short_rows_lt_8_tokens": short_rows,
        "short_rows_pct": _pct(short_rows, total),
        "long_rows_gt_3000_tokens": long_rows,
        "long_rows_pct": _pct(long_rows, total),
        "sample_rate": sample_rate,
        "near_duplicate_sample_rows": near_duplicate_sample_rows,
        "sampled_rows": sampled,
        "near_duplicate_sample_pct": _pct(near_duplicate_sample_rows, sampled),
        "near_duplicate_estimated_rows": round(total * near_duplicate_sample_rows / sampled) if sampled else 0,
        "near_duplicate_method": (
            "deterministic sample; 64-bit token-shingle simhash; duplicate when "
            f"Hamming distance <= {near_hamming_threshold} in a shared 16-bit band"
        ),
        "token_hist": dict(token_hist),
        "year_counts": dict(sorted(year_counts.items())),
        "source_layer_counts": layer_rows,
        "top_forums": forum_rows[:50],
        "top_forum_categories": category_rows[:100],
        "new_source_rows": {
            forum: forum_counts.get(forum, 0)
            for forum in ("hackersploit", "parrotsec", "hackthebox")
        },
        "near_duplicate_examples": near_examples,
        "artifacts": {
            "forum_counts_csv": str(output_prefix.with_name(output_prefix.name + "_forum_counts.csv")),
            "forum_category_counts_csv": str(output_prefix.with_name(output_prefix.name + "_forum_category_counts.csv")),
            "layer_counts_csv": str(output_prefix.with_name(output_prefix.name + "_layer_counts.csv")),
            "forum_layer_counts_csv": str(output_prefix.with_name(output_prefix.name + "_forum_layer_counts.csv")),
        },
    }

    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# NeurIPS Corpus Audit",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Input: `{report['input_path']}`",
        f"- Total rows: `{total:,}`",
        f"- Parseable timestamps: `{with_timestamp:,}` ({report['rows_with_parseable_timestamp_pct']}%)",
        f"- CTI-nonzero rows: `{cti_nonzero:,}` ({report['cti_nonzero_pct']}%)",
        f"- CVE rows: `{cve_rows:,}` ({report['cve_rows_pct']}%)",
        f"- Short rows (<8 tokens): `{short_rows:,}` ({report['short_rows_pct']}%)",
        f"- Long rows (>3000 tokens): `{long_rows:,}` ({report['long_rows_pct']}%)",
        f"- Near-duplicate sample: `{near_duplicate_sample_rows:,}` / `{sampled:,}` ({report['near_duplicate_sample_pct']}%)",
        f"- Near-duplicate estimated rows: `{report['near_duplicate_estimated_rows']:,}`",
        "",
        "## Source Layers",
        "",
    ]
    lines.extend(
        f"- `{row['source_layer']}`: `{row['rows']:,}` ({row['pct']}%)"
        for row in layer_rows
    )
    lines.extend(["", "## Top Forums", ""])
    lines.extend(
        f"- `{row['forum_id']}`: `{row['rows']:,}` ({row['pct']}%)"
        for row in forum_rows[:25]
    )
    lines.extend(["", "## Top Forum Categories", ""])
    lines.extend(
        f"- `{row['forum_id']}` / `{row['section']}`: `{row['rows']:,}` ({row['pct']}%)"
        for row in category_rows[:50]
    )
    lines.extend(["", "## Artifacts", ""])
    lines.extend(f"- `{path}`" for path in report["artifacts"].values())
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit expanded ETG NeurIPS corpus.")
    parser.add_argument("input_path")
    parser.add_argument("--output-prefix", default="data/unified_hacker_communities_neurips_audit")
    parser.add_argument("--sample-rate", type=float, default=0.03)
    parser.add_argument("--near-hamming-threshold", type=int, default=3)
    args = parser.parse_args()
    report = audit_neurips_corpus(
        args.input_path,
        output_prefix=args.output_prefix,
        sample_rate=args.sample_rate,
        near_hamming_threshold=args.near_hamming_threshold,
    )
    print(json.dumps({
        "total_rows": report["total_rows"],
        "sampled_rows": report["sampled_rows"],
        "near_duplicate_sample_pct": report["near_duplicate_sample_pct"],
        "near_duplicate_estimated_rows": report["near_duplicate_estimated_rows"],
        "artifacts": report["artifacts"],
    }, indent=2))


if __name__ == "__main__":
    main()
