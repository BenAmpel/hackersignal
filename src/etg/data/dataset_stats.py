"""Compute and report dataset statistics for the ETG corpus.

Produces per-source and aggregate statistics suitable for a
NeurIPS Datasets & Benchmarks paper table.

Usage
-----
    python -m etg.data.dataset_stats --data-dir data/ --output data/stats/

Outputs
-------
- ``stats.json``    — machine-readable per-source stats
- ``table.tex``     — LaTeX booktabs table for the paper
- ``summary.txt``   — human-readable summary
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source metadata — human-readable names, categories, and access model
# ---------------------------------------------------------------------------

SOURCE_META: dict[str, dict] = {
    # ---- Hacker / Security Forums ----
    "hackforums": {
        "display": "HackForums",
        "category": "Hacker Forum",
        "access": "Public (crawled)",
        "language": "en",
    },
    "kaeli_hacker": {
        "display": "Kaeli Hacker Forums",
        "category": "Hacker Forum",
        "access": "Public (Kaggle)",
        "language": "en",
    },
    "deepdarkcti_public_broad_2026-04-18": {
        "display": "DeepDarkCTI (Broad)",
        "category": "Hacker Forum",
        "access": "Public (GitHub)",
        "language": "multi",
    },
    "0x00sec": {
        "display": "0x00sec",
        "category": "Hacker Forum",
        "access": "Public (scraped)",
        "language": "en",
    },
    "cve_hacker_forum": {
        "display": "CVE Hacker Forum",
        "category": "Hacker Forum",
        "access": "Public (GitHub)",
        "language": "en",
    },
    "crackingarena": {
        "display": "CrackingArena",
        "category": "Hacker Forum",
        "access": "Public (Kaggle)",
        "language": "en",
    },
    "go4expert": {
        "display": "Go4Expert",
        "category": "Hacker Forum",
        "access": "Public (scraped)",
        "language": "en",
    },
    "seebug": {
        "display": "Seebug",
        "category": "Hacker Forum",
        "access": "Public (scraped)",
        "language": "zh",
    },
    "antionline": {
        "display": "AntiOnline",
        "category": "Hacker Forum",
        "access": "Public (scraped)",
        "language": "en",
    },
    # ---- Exploit Databases ----
    "exploitdb": {
        "display": "Exploit-DB",
        "category": "Exploit Database",
        "access": "Public (CSV)",
        "language": "en",
    },
    "exploitdb_hf": {
        "display": "Exploit-DB (HuggingFace)",
        "category": "Exploit Database",
        "access": "Public (HuggingFace)",
        "language": "en",
    },
    "hacker_exploits": {
        "display": "Hacker Exploits (Kaggle)",
        "category": "Exploit Database",
        "access": "Public (Kaggle)",
        "language": "en",
    },
    "dtl_exploits": {
        "display": "DTL Exploits (AZSecure)",
        "category": "Exploit Database",
        "access": "Public (GitHub)",
        "language": "en",
    },
    "zeroday": {
        "display": "0day.today",
        "category": "Exploit Database",
        "access": "Public (GitHub)",
        "language": "en",
    },
    # ---- Vulnerability Advisories ----
    "nvd": {
        "display": "NVD CVE",
        "category": "Vulnerability Advisory",
        "access": "Public (API)",
        "language": "en",
    },
    "github_advisory": {
        "display": "GitHub Advisory DB",
        "category": "Vulnerability Advisory",
        "access": "Public (API/GraphQL)",
        "language": "en",
    },
    "cisa_kev": {
        "display": "CISA KEV",
        "category": "Vulnerability Advisory",
        "access": "Public (API)",
        "language": "en",
    },
    "hackerone": {
        "display": "HackerOne Disclosed",
        "category": "Vulnerability Advisory",
        "access": "Public (HuggingFace)",
        "language": "en",
    },
    "zeroscience": {
        "display": "ZeroScience Lab",
        "category": "Vulnerability Advisory",
        "access": "Public (scraped)",
        "language": "en",
    },
    "vulnlab": {
        "display": "Vulnerability-Lab",
        "category": "Vulnerability Advisory",
        "access": "Public (scraped)",
        "language": "en",
    },
    "fulldisclosure": {
        "display": "Full Disclosure ML",
        "category": "Vulnerability Advisory",
        "access": "Public (scraped)",
        "language": "en",
    },
    # ---- Fix Commits ----
    "cvefixes": {
        "display": "CVEfixes",
        "category": "Fix Commit",
        "access": "Public (Zenodo)",
        "language": "en",
    },
    # ---- Darknet Markets (Gayanku CSV — individual forums) ----
    # NOTE: Evolution Forum was a darknet marketplace, not a hacker forum.
    # "gayanku" is the legacy combined source file; once the importer is re-run,
    # it is replaced by the per-forum entries below.
    "gayanku": {
        "display": "Darknet Markets (Gayanku)",
        "category": "Darknet Market",
        "access": "Public (Kaggle)",
        "language": "en",
    },
    "evolution": {
        "display": "Evolution Forum",
        "category": "Darknet Market",
        "access": "Public (Zenodo)",
        "language": "en",
    },
    "gayanku_silk_road_1": {
        "display": "Silk Road 1",
        "category": "Darknet Market",
        "access": "Public (Kaggle)",
        "language": "en",
    },
    "gayanku_silk_road_2": {
        "display": "Silk Road 2",
        "category": "Darknet Market",
        "access": "Public (Kaggle)",
        "language": "en",
    },
    "gayanku_agora": {
        "display": "Agora",
        "category": "Darknet Market",
        "access": "Public (Kaggle)",
        "language": "en",
    },
    "gayanku_evolution": {
        "display": "Evolution Forum (Gayanku)",
        "category": "Darknet Market",
        "access": "Public (Kaggle)",
        "language": "en",
    },
    "gayanku_reddit_darknet_markets": {
        "display": "Reddit (Darknet Markets)",
        "category": "Darknet Market",
        "access": "Public (Kaggle)",
        "language": "en",
    },
    "gayanku_reddit_silk_road": {
        "display": "Reddit (Silk Road)",
        "category": "Darknet Market",
        "access": "Public (Kaggle)",
        "language": "en",
    },
    "gayanku_black_market_reloaded": {
        "display": "Black Market Reloaded",
        "category": "Darknet Market",
        "access": "Public (Kaggle)",
        "language": "en",
    },
}

CATEGORY_ORDER = [
    "Hacker Forum",
    "Darknet Market",
    "Exploit Database",
    "Vulnerability Advisory",
    "Fix Commit",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SourceStats:
    source_id: str
    display_name: str
    category: str
    access_model: str
    language: str
    post_count: int = 0
    cve_entry_count: int = 0
    posts_with_cve: int = 0          # unique exploit_ids in CVE index
    min_date: str = "—"
    max_date: str = "—"
    avg_tokens: float = 0.0
    unique_cve_ids: int = 0


@dataclass
class CorpusStats:
    sources: dict[str, SourceStats] = field(default_factory=dict)
    total_posts: int = 0
    total_cve_entries: int = 0
    unique_cve_ids: int = 0
    date_range: tuple[str, str] = ("—", "—")

    def to_dict(self) -> dict:
        return {
            "total_posts": self.total_posts,
            "total_cve_entries": self.total_cve_entries,
            "unique_cve_ids": self.unique_cve_ids,
            "date_range": list(self.date_range),
            "sources": {k: asdict(v) for k, v in self.sources.items()},
        }


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

_TOKEN_SAMPLE = 5_000   # rows to sample for avg token estimate


def _estimate_tokens(text: str) -> int:
    """Rough GPT-style token count: chars / 4."""
    return max(1, len(text) // 4)


def compute_stats(data_dir: Path, sample_rows: int = _TOKEN_SAMPLE) -> CorpusStats:
    """Scan all *_posts.jsonl and *_cve_index.jsonl files and return CorpusStats."""
    corpus = CorpusStats()

    # --- Phase 1: posts ---
    for posts_file in sorted(data_dir.glob("*_posts.jsonl")):
        src_id = posts_file.stem.replace("_posts", "")

        meta = SOURCE_META.get(src_id, {})
        ss = SourceStats(
            source_id=src_id,
            display_name=meta.get("display", src_id),
            category=meta.get("category", "Other"),
            access_model=meta.get("access", "Unknown"),
            language=meta.get("language", "en"),
        )

        post_count = 0
        token_total = 0
        token_sample_n = 0
        min_date = "9999-99-99"
        max_date = "0000-00-00"

        with posts_file.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                post_count += 1

                ts = (d.get("timestamp") or "")[:10]
                if ts and len(ts) == 10 and ts > "1990":
                    if ts < min_date:
                        min_date = ts
                    if ts > max_date:
                        max_date = ts

                if token_sample_n < sample_rows:
                    token_total += _estimate_tokens(d.get("text") or "")
                    token_sample_n += 1

        ss.post_count = post_count
        ss.min_date = min_date if min_date != "9999-99-99" else "—"
        ss.max_date = max_date if max_date != "0000-00-00" else "—"
        ss.avg_tokens = round(token_total / max(token_sample_n, 1), 1)

        corpus.sources[src_id] = ss
        corpus.total_posts += post_count
        log.info("stats: %s — %d posts", src_id, post_count)

    # --- Phase 2: CVE index ---
    all_cve_ids: set[str] = set()
    for cve_file in sorted(data_dir.glob("*_cve_index.jsonl")):
        src_id = cve_file.stem.replace("_cve_index", "")
        ss = corpus.sources.get(src_id)
        if ss is None:
            # source has CVE index but no posts file — create stub
            meta = SOURCE_META.get(src_id, {})
            ss = SourceStats(
                source_id=src_id,
                display_name=meta.get("display", src_id),
                category=meta.get("category", "Other"),
                access_model=meta.get("access", "Unknown"),
                language=meta.get("language", "en"),
            )
            corpus.sources[src_id] = ss

        linked_post_ids: set[str] = set()
        source_cve_ids: set[str] = set()

        with cve_file.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ss.cve_entry_count += 1
                corpus.total_cve_entries += 1

                cve_id = (d.get("cve_id") or "").upper()
                if cve_id:
                    all_cve_ids.add(cve_id)
                    source_cve_ids.add(cve_id)

                pid = d.get("exploit_id") or ""
                if pid:
                    linked_post_ids.add(pid)

        ss.posts_with_cve = len(linked_post_ids)
        ss.unique_cve_ids = len(source_cve_ids)

    # --- Aggregate ---
    corpus.unique_cve_ids = len(all_cve_ids)

    all_dates = [
        ss.min_date for ss in corpus.sources.values() if ss.min_date != "—"
    ] + [
        ss.max_date for ss in corpus.sources.values() if ss.max_date != "—"
    ]
    if all_dates:
        corpus.date_range = (min(all_dates), max(all_dates))

    return corpus


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def render_summary(corpus: CorpusStats) -> str:
    lines = [
        "=" * 80,
        "ETG (Exploit Text Graph) Dataset — Summary Statistics",
        "=" * 80,
        f"  Total posts:        {_fmt_int(corpus.total_posts)}",
        f"  Total CVE entries:  {_fmt_int(corpus.total_cve_entries)}",
        f"  Unique CVEs:        {_fmt_int(corpus.unique_cve_ids)}",
        f"  Date range:         {corpus.date_range[0]} → {corpus.date_range[1]}",
        f"  Sources:            {len(corpus.sources)}",
        "",
    ]

    for cat in CATEGORY_ORDER:
        sources = [s for s in corpus.sources.values() if s.category == cat]
        if not sources:
            continue
        lines.append(f"  [{cat}]")
        lines.append(
            f"    {'Source':<32} {'Posts':>10} {'CVE Refs':>10} {'CVE%':>7} "
            f"{'AvgTok':>8} {'Dates'}"
        )
        lines.append("    " + "-" * 88)
        cat_posts = 0
        cat_cves = 0
        for ss in sorted(sources, key=lambda s: -s.post_count):
            pct = (
                f"{100 * ss.posts_with_cve / ss.post_count:.1f}%"
                if ss.post_count > 0
                else "—"
            )
            dates = (
                f"{ss.min_date[:7]} – {ss.max_date[:7]}"
                if ss.min_date != "—"
                else "—"
            )
            lines.append(
                f"    {ss.display_name:<32} {_fmt_int(ss.post_count):>10} "
                f"{_fmt_int(ss.cve_entry_count):>10} {pct:>7} "
                f"{ss.avg_tokens:>8.0f} {dates}"
            )
            cat_posts += ss.post_count
            cat_cves += ss.cve_entry_count
        lines.append(
            f"    {'  Subtotal':<32} {_fmt_int(cat_posts):>10} {_fmt_int(cat_cves):>10}"
        )
        lines.append("")

    return "\n".join(lines)


def render_latex(corpus: CorpusStats) -> str:
    """Render a booktabs LaTeX table for the NeurIPS paper."""
    rows = []
    rows.append(r"\begin{table}[t]")
    rows.append(r"\centering")
    rows.append(r"\small")
    rows.append(r"\caption{ETG corpus statistics by source. \emph{CVE\%} is the")
    rows.append(r"  fraction of posts linked to at least one CVE record.}")
    rows.append(r"\label{tab:corpus-stats}")
    rows.append(r"\begin{tabular}{llrrrrr}")
    rows.append(r"\toprule")
    rows.append(
        r"\textbf{Source} & \textbf{Category} & \textbf{Posts} & "
        r"\textbf{CVE Refs} & \textbf{CVE\%} & \textbf{Avg Tok.} & \textbf{Years} \\"
    )
    rows.append(r"\midrule")

    for cat in CATEGORY_ORDER:
        sources = [s for s in corpus.sources.values() if s.category == cat]
        if not sources:
            continue

        cat_label = cat.replace(" ", r"\ ")
        rows.append(r"\multicolumn{7}{l}{\textit{" + cat + r"}} \\")

        for ss in sorted(sources, key=lambda s: -s.post_count):
            if ss.post_count == 0 and ss.cve_entry_count == 0:
                continue
            pct = (
                f"{100 * ss.posts_with_cve / ss.post_count:.0f}\\%"
                if ss.post_count > 0
                else "—"
            )
            yr_min = ss.min_date[:4] if ss.min_date != "—" else "—"
            yr_max = ss.max_date[:4] if ss.max_date != "—" else "—"
            years = f"{yr_min}--{yr_max}" if yr_min != "—" else "—"
            rows.append(
                f"  {ss.display_name} & {ss.category} & "
                f"{_fmt_int(ss.post_count)} & "
                f"{_fmt_int(ss.cve_entry_count)} & "
                f"{pct} & "
                f"{ss.avg_tokens:.0f} & "
                f"{years} \\\\"
            )

        rows.append(r"\midrule")

    # Totals row
    rows.append(
        f"  \\textbf{{Total}} & & "
        f"\\textbf{{{_fmt_int(corpus.total_posts)}}} & "
        f"\\textbf{{{_fmt_int(corpus.total_cve_entries)}}} & "
        f"— & — & "
        f"{corpus.date_range[0][:4]}--{corpus.date_range[1][:4]} \\\\"
    )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"\end{table}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Compute ETG dataset statistics")
    parser.add_argument("--data-dir", default="data", help="Directory containing *_posts.jsonl files")
    parser.add_argument("--output", default="data/stats", help="Output directory for stats files")
    parser.add_argument("--sample-rows", type=int, default=5000, help="Rows to sample per source for avg token estimate")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = compute_stats(data_dir, sample_rows=args.sample_rows)

    # Write JSON
    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(corpus.to_dict(), indent=2), encoding="utf-8")
    log.info("Wrote %s", stats_path)

    # Write LaTeX
    tex_path = output_dir / "table.tex"
    tex_path.write_text(render_latex(corpus), encoding="utf-8")
    log.info("Wrote %s", tex_path)

    # Write summary
    summary_path = output_dir / "summary.txt"
    summary = render_summary(corpus)
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)
    log.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
