#!/usr/bin/env python3
"""
Generate supplemental corpus statistics figures:
  - paper/figures/corpus_temporal_distribution.{pdf,png}
  - paper/figures/source_layer_composition.{pdf,png}

Data source: data/stats/stats.json (always available in the repo)

Usage
-----
  cd /path/to/exploit-text-graph
  PYTHONPATH=src python3 scripts/plot_corpus_stats.py [--data-dir data] [--out paper/figures]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Layer → color and display name
# ---------------------------------------------------------------------------

LAYER_META: dict[str, dict] = {
    "Hacker Forum":           {"color": "#4e79a7", "display": "Hacker Forum"},
    "Darknet Market":         {"color": "#b07aa1", "display": "Darknet Market"},
    "Exploit Database":       {"color": "#f28e2b", "display": "Exploit Archive"},
    "Vulnerability Advisory": {"color": "#59a14f", "display": "Advisory"},
    "Fix Commit":             {"color": "#76b7b2", "display": "Fix Commit"},
    "Other":                  {"color": "#9e9e9e", "display": "Other"},
}

# Canonical ordering for stacked area chart
LAYER_ORDER = [
    "Hacker Forum",
    "Darknet Market",
    "Exploit Database",
    "Vulnerability Advisory",
    "Fix Commit",
]

# NeurIPS benchmark split boundaries
SPLIT_TRAIN_END = 2022
SPLIT_VAL_END   = 2024


# ---------------------------------------------------------------------------
# Load stats
# ---------------------------------------------------------------------------

def load_stats(data_dir: Path) -> dict:
    p = data_dir / "stats" / "stats.json"
    if not p.exists():
        raise FileNotFoundError(f"stats.json not found at {p}")
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: corpus_temporal_distribution
# Stacked bar chart of post counts per year, coloured by source layer.
# Because we only have min/max per source (not year-level counts),
# we distribute posts uniformly within each source's active period.
# ---------------------------------------------------------------------------

def _uniform_year_counts(min_year: int, max_year: int, total: int) -> dict[int, int]:
    span = max(1, max_year - min_year + 1)
    base, rem = divmod(total, span)
    return {y: base + (1 if y - min_year < rem else 0)
            for y in range(min_year, max_year + 1)}


def build_temporal_matrix(
    stats: dict,
    year_range: tuple[int, int] = (1990, 2027),
) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Returns
    -------
    matrix : (n_layers, n_years) float64 — post counts (uniformly distributed)
    layer_names : list of layer display names (same row order)
    years : list of int year labels (same column order)
    """
    years = list(range(year_range[0], year_range[1]))
    year_idx = {y: i for i, y in enumerate(years)}
    n_years = len(years)

    layer_rows: dict[str, np.ndarray] = {
        l: np.zeros(n_years) for l in LAYER_ORDER
    }

    for src_id, src in stats["sources"].items():
        total = src.get("post_count", 0)
        if total == 0:
            continue
        min_d = src.get("min_date", "")
        max_d = src.get("max_date", "")
        if not min_d or min_d == "—":
            continue
        layer = src.get("category", "Other")
        if layer not in layer_rows:
            layer = "Other"
            if "Other" not in layer_rows:
                layer_rows["Other"] = np.zeros(n_years)
        min_year = int(min_d[:4])
        max_year = int(max_d[:4])
        for y, cnt in _uniform_year_counts(min_year, max_year, total).items():
            if y in year_idx:
                layer_rows[layer][year_idx[y]] += cnt

    matrix = np.array([layer_rows.get(l, np.zeros(n_years)) for l in LAYER_ORDER])
    return matrix, LAYER_ORDER, years


def plot_temporal_distribution(
    matrix: np.ndarray,
    layer_names: list[str],
    years: list[int],
    out_dir: Path,
) -> None:
    # Filter years with any posts
    totals = matrix.sum(axis=0)
    active = totals > 0
    y_arr = np.array(years)[active]
    m_arr = matrix[:, active]

    fig, ax = plt.subplots(figsize=(12, 4.5))

    x = np.arange(len(y_arr))
    bottom = np.zeros(len(y_arr))

    for i, layer in enumerate(layer_names):
        row = m_arr[i]
        log_row = np.log10(row + 1)  # log10 scale
        color = LAYER_META.get(layer, LAYER_META["Other"])["color"]
        disp  = LAYER_META.get(layer, LAYER_META["Other"])["display"]
        ax.bar(x, log_row, bottom=np.log10(bottom + 1), color=color,
               label=disp, width=0.9, align="center")
        bottom += row  # sum in linear space, but display in log

    # Benchmark split regions
    def _x_for_year(yr: int) -> float:
        hits = [i for i, y in enumerate(y_arr) if y == yr]
        return hits[0] - 0.5 if hits else -0.5

    x_train_end = _x_for_year(SPLIT_TRAIN_END)
    x_val_end   = _x_for_year(SPLIT_VAL_END)
    ax.axvspan(x_train_end, x_val_end,       color="#f59e0b", alpha=0.18, label="val (2022–23)")
    ax.axvspan(x_val_end,   len(y_arr) - 0.5, color="#ef4444", alpha=0.18, label="test (2024+)")
    ax.axvspan(-0.5,        x_train_end,       color="#3b82f6", alpha=0.07, label="train (<2022)")

    # X axis
    tick_step = 5
    tick_x     = [i for i, y in enumerate(y_arr) if y % tick_step == 0]
    tick_labels = [str(y_arr[i]) for i in tick_x]
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_labels, fontsize=8)

    ax.set_ylabel("log₁₀(posts + 1)", fontsize=9)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_title("HackerSignal — temporal distribution of posts by source layer", fontsize=10)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Put source-layer patches first
    layer_h = handles[:len(layer_names)]
    layer_l = labels[:len(layer_names)]
    split_h = handles[len(layer_names):]
    split_l = labels[len(layer_names):]
    ax.legend(
        layer_h + split_h,
        layer_l + split_l,
        fontsize=8,
        ncol=4,
        loc="upper left",
        framealpha=0.85,
    )

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        dest = out_dir / f"corpus_temporal_distribution.{ext}"
        fig.savefig(dest, dpi=200, bbox_inches="tight")
        print(f"Saved {dest}", file=sys.stderr)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: source_layer_composition
# Horizontal stacked bar chart showing the post count per source
# coloured by layer, sorted by total descending.
# ---------------------------------------------------------------------------

def plot_layer_composition(stats: dict, out_dir: Path) -> None:
    # Collect (display, layer, count) for sources with > 0 posts
    rows = []
    for src_id, src in stats["sources"].items():
        total = src.get("post_count", 0)
        if total == 0:
            continue
        layer = src.get("category", "Other")
        disp  = src.get("display_name", src_id)
        rows.append((disp, layer, total))
    rows.sort(key=lambda x: x[2], reverse=True)

    # Limit to top 25 for readability
    rows = rows[:25]

    labels = [r[0] for r in rows]
    counts = np.array([r[2] for r in rows], dtype=float)
    colors = [LAYER_META.get(r[1], LAYER_META["Other"])["color"] for r in rows]
    layers = [r[1] for r in rows]

    n = len(rows)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.42 + 1)))

    y_pos = np.arange(n)
    bars = ax.barh(y_pos, np.log10(counts + 1), color=colors, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)

    ax.set_xlabel("log₁₀(posts)", fontsize=9)
    ax.set_title("HackerSignal — source composition (top 25 sources by post count)", fontsize=10)

    # Add raw count annotations
    for i, (bar, cnt) in enumerate(zip(bars, counts)):
        ax.text(
            bar.get_width() + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{int(cnt):,}",
            va="center",
            fontsize=7,
        )

    # Layer legend
    seen: set[str] = set()
    handles = []
    for r in rows:
        layer = r[1]
        if layer not in seen:
            seen.add(layer)
            handles.append(
                mpatches.Patch(
                    color=LAYER_META.get(layer, LAYER_META["Other"])["color"],
                    label=LAYER_META.get(layer, LAYER_META["Other"])["display"],
                )
            )
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.85)
    ax.invert_yaxis()

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        dest = out_dir / f"source_layer_composition.{ext}"
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

    print("Loading data/stats/stats.json …", file=sys.stderr)
    stats = load_stats(data_dir)

    print("Plotting corpus temporal distribution …", file=sys.stderr)
    matrix, layer_names, years = build_temporal_matrix(stats)
    plot_temporal_distribution(matrix, layer_names, years, out_dir)

    print("Plotting source layer composition …", file=sys.stderr)
    plot_layer_composition(stats, out_dir)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
