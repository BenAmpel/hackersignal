#!/usr/bin/env python3
"""
Generate paper/figures/temporal_heatmap.{pdf,png}

Temporal coverage by source layer (log10 scale) for the HackerSignal dataset.
Colored bands mark the benchmark train/val/test boundaries.

Data sources (in priority order):
  1. Per-source JSONL files in data/ (most accurate year-level counts)
  2. data/stats/stats.json (min/max dates + total count — used as fallback)

Usage
-----
  cd /path/to/exploit-text-graph
  PYTHONPATH=src python3 scripts/plot_temporal_heatmap.py [--data-dir data] [--out paper/figures]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# NeurIPS benchmark split boundaries (year boundaries)
SPLIT_TRAIN_END = 2022  # train: < 2022
SPLIT_VAL_END   = 2024  # val:   2022–2023;  test: >= 2024

YEAR_RANGE = (1990, 2027)  # inclusive start, exclusive end → columns 1990…2026

# Layer → colour for bars
LAYER_COLORS: dict[str, str] = {
    "Hacker Forum":         "#4e79a7",
    "Darknet Market":       "#b07aa1",
    "Exploit Database":     "#f28e2b",
    "Vulnerability Advisory": "#59a14f",
    "Fix Commit":           "#76b7b2",
    "Other":                "#9e9e9e",
}

# Sources to show (display name, category, source_id prefix for JSONL lookup)
# Only sources with ≥ 100 posts and non-trivial date ranges are included.
SOURCES_ORDERED = [
    # --- Hacker Forum ---
    ("HackForums",          "Hacker Forum",         "hackforums"),
    ("Kaeli Forums",        "Hacker Forum",         "kaeli_hacker"),
    ("DeepDarkCTI",         "Hacker Forum",         "deepdarkcti"),
    ("CrackingArena",       "Hacker Forum",         "crackingarena"),
    ("Go4Expert",           "Hacker Forum",         "go4expert"),
    ("0x00sec",             "Hacker Forum",         "0x00sec"),
    # --- Darknet Market ---
    ("Evolution Forum",     "Darknet Market",       "evolution"),
    # --- Exploit Database ---
    ("Hacker Exploits",     "Exploit Database",     "hacker_exploits"),
    ("DTL Exploits",        "Exploit Database",     "dtl_exploits"),
    ("Exploit-DB",          "Exploit Database",     "exploitdb"),
    ("0day.today",          "Exploit Database",     "zeroday"),
    # --- Vulnerability Advisory ---
    ("NVD CVE",             "Vulnerability Advisory", "nvd"),
    ("GitHub Advisory",     "Vulnerability Advisory", "github_advisory"),
    ("HackerOne",           "Vulnerability Advisory", "hackerone"),
    ("Full Disclosure",     "Vulnerability Advisory", "fulldisclosure"),
    ("CISA KEV",            "Vulnerability Advisory", "cisa_kev"),
    # --- Fix Commit ---
    ("CVEfixes",            "Fix Commit",           "cvefixes"),
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_stats_json(data_dir: Path) -> dict:
    stats_path = data_dir / "stats" / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"stats.json not found at {stats_path}")
    with open(stats_path) as f:
        return json.load(f)


def _year_counts_from_jsonl(jsonl_path: Path) -> dict[int, int]:
    """Stream a JSONL file and count posts per year (fast path)."""
    counts: dict[int, int] = defaultdict(int)
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                ts = rec.get("timestamp", "")
                if ts and len(ts) >= 4:
                    year = int(ts[:4])
                    counts[year] += 1
            except (json.JSONDecodeError, ValueError):
                pass
    return dict(counts)


def _uniform_year_counts(min_year: int, max_year: int, total: int) -> dict[int, int]:
    """Distribute `total` posts uniformly across years min_year..max_year (fallback)."""
    span = max(1, max_year - min_year + 1)
    base = total // span
    remainder = total % span
    counts = {}
    for y in range(min_year, max_year + 1):
        counts[y] = base + (1 if y - min_year < remainder else 0)
    return counts


def build_year_matrix(
    data_dir: Path,
    sources_ordered: list[tuple],
    stats: dict,
) -> tuple[np.ndarray, list[tuple]]:
    """
    Returns
    -------
    matrix : ndarray of shape (n_sources, n_years)  — log10(count+1)
    included : list of (display_name, layer) for rows that have any data
    """
    years = list(range(YEAR_RANGE[0], YEAR_RANGE[1]))
    n_years = len(years)
    year_idx = {y: i for i, y in enumerate(years)}

    matrix_rows = []
    included = []

    for display_name, layer, src_id in sources_ordered:
        # --- Try to load from JSONL ---
        year_counts: dict[int, int] | None = None
        for candidate in [
            data_dir / f"{src_id}_posts.jsonl",
            data_dir / f"{src_id}_raw.jsonl",
        ]:
            if candidate.exists():
                print(f"  Reading {candidate.name} …", file=sys.stderr)
                year_counts = _year_counts_from_jsonl(candidate)
                break

        if year_counts is None:
            # Fallback: use stats.json
            src_stats = stats.get("sources", {}).get(src_id)
            if src_stats is None:
                # Try fuzzy match
                for k, v in stats["sources"].items():
                    if src_id in k or k in src_id:
                        src_stats = v
                        break
            if src_stats is None or src_stats.get("post_count", 0) == 0:
                continue
            min_d = src_stats.get("min_date", "")
            max_d = src_stats.get("max_date", "")
            if not min_d or min_d == "—":
                continue
            min_year = int(min_d[:4])
            max_year = int(max_d[:4])
            total = src_stats["post_count"]
            year_counts = _uniform_year_counts(min_year, max_year, total)
            print(
                f"  {display_name}: no JSONL found — using uniform fallback "
                f"({total:,} posts, {min_year}–{max_year})",
                file=sys.stderr,
            )

        row = np.zeros(n_years)
        for year, cnt in year_counts.items():
            if year in year_idx:
                row[year_idx[year]] += cnt

        if row.sum() == 0:
            continue

        matrix_rows.append(np.log10(row + 1))
        included.append((display_name, layer))

    if not matrix_rows:
        raise RuntimeError("No data found for any source.")

    return np.array(matrix_rows), included, years


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_heatmap(
    matrix: np.ndarray,
    included: list[tuple],
    years: list[int],
    out_dir: Path,
) -> None:
    n_sources, n_years = matrix.shape
    fig_height = max(4.5, n_sources * 0.38 + 1.2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Heatmap cells
    cmap = plt.cm.YlOrRd
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=np.percentile(matrix[matrix > 0], 98) if matrix.max() > 0 else 1,
        interpolation="nearest",
    )

    # X ticks: every 5 years
    tick_years = [y for y in years if y % 5 == 0]
    tick_pos   = [years.index(y) for y in tick_years]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(y) for y in tick_years], fontsize=8)

    # Y ticks
    ax.set_yticks(range(n_sources))
    ax.set_yticklabels(
        [name for name, _ in included],
        fontsize=8.5,
    )

    # Colour left-side row labels by layer
    for i, (name, layer) in enumerate(included):
        ax.get_yticklabels()[i].set_color(LAYER_COLORS.get(layer, "#333333"))

    # Benchmark split vertical bands
    def _year_to_x(y: int) -> float:
        return years.index(y) - 0.5 if y in years else -0.5

    split_alpha = 0.18
    # val band: 2022–2023
    x0 = _year_to_x(SPLIT_TRAIN_END)
    x1 = _year_to_x(SPLIT_VAL_END)
    ax.axvspan(x0, x1, color="#f59e0b", alpha=split_alpha, zorder=0, label="val (2022–23)")
    # test band: 2024+
    x2 = n_years - 0.5
    ax.axvspan(x1, x2, color="#ef4444", alpha=split_alpha, zorder=0, label="test (2024+)")
    # train band: shading on the left (subtle)
    ax.axvspan(-0.5, x0, color="#3b82f6", alpha=0.07, zorder=0, label="train (<2022)")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
    cbar.set_label("log₁₀(posts + 1)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Layer legend
    layer_patches = [
        mpatches.Patch(color=c, label=lyr)
        for lyr, c in LAYER_COLORS.items()
        if any(l == lyr for _, l in included)
    ]
    split_patches = [
        mpatches.Patch(color="#3b82f6", alpha=0.3, label="train (<2022)"),
        mpatches.Patch(color="#f59e0b", alpha=0.4, label="val (2022–23)"),
        mpatches.Patch(color="#ef4444", alpha=0.4, label="test (2024+)"),
    ]
    ax.legend(
        handles=layer_patches + split_patches,
        loc="upper left",
        fontsize=7.5,
        ncol=3,
        framealpha=0.85,
        handlelength=1.0,
        columnspacing=0.8,
    )

    ax.set_xlabel("Year", fontsize=9)
    ax.set_title(
        "HackerSignal — temporal coverage by source (log₁₀ scale)",
        fontsize=10,
    )

    fig.tight_layout()

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        dest = out_dir / f"temporal_heatmap.{ext}"
        fig.savefig(dest, dpi=200, bbox_inches="tight")
        print(f"Saved {dest}", file=sys.stderr)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data"),
        help="Path to the data/ directory (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "paper" / "figures"),
        help="Output directory for figures (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out)

    print("Loading stats.json …", file=sys.stderr)
    stats = _load_stats_json(data_dir)

    print("Building year × source matrix …", file=sys.stderr)
    matrix, included, years = build_year_matrix(data_dir, SOURCES_ORDERED, stats)

    print(f"Matrix shape: {matrix.shape}  ({len(included)} sources × {len(years)} years)", file=sys.stderr)
    plot_heatmap(matrix, included, years, out_dir)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
