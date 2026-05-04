#!/usr/bin/env python3
"""
Generate paper/figures/domain_overlap.{pdf,png}

Pairwise vocabulary overlap (Jaccard similarity) across hacker community
forums, displayed as a hierarchically-clustered heatmap.

Data sources (in priority order):
  1. Per-source JSONL files in data/ (computes 5% token sample → Jaccard)
  2. data/domain_overlap_cache.json (precomputed matrix, committed if available)

If neither is available the script exits with a clear error message.

Usage
-----
  cd /path/to/exploit-text-graph
  PYTHONPATH=src python3 scripts/plot_domain_overlap.py [--data-dir data] [--out paper/figures]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

REPO_ROOT = Path(__file__).resolve().parent.parent

# Sampling seed for the 5% deterministic token sample
SAMPLE_SEED = 42
SAMPLE_RATE = 0.05  # 5% of tokens per source

# Source IDs to include (forum_id as stored in individual JSONL files)
FORUM_SOURCES = [
    "0x00sec",
    "antionline",
    "crackingarena",
    "cve_hacker_forum",
    "deepdarkcti_public_broad_2026-04-18",
    "evolution",
    "go4expert",
    "hackforums",
    "kaeli_hacker",
    "seebug",
]

# Minimum documents per source to include
MIN_DOCS = 50


# ---------------------------------------------------------------------------
# Token-set extraction
# ---------------------------------------------------------------------------

import re

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_\-]{3,}")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _build_token_set_from_jsonl(path: Path, sample_rate: float = SAMPLE_RATE) -> set[str]:
    """Stream JSONL and build a sampled vocabulary set."""
    rng = random.Random(SAMPLE_SEED)
    tokens: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("text", "")
            if not text:
                continue
            for tok in _tokenize(text):
                if rng.random() < sample_rate:
                    tokens.add(tok)
    return tokens


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Load or compute
# ---------------------------------------------------------------------------

def load_or_compute(data_dir: Path) -> tuple[np.ndarray, list[str]]:
    cache_path = data_dir / "domain_overlap_cache.json"

    # --- Try cache ---
    if cache_path.exists():
        print(f"Loading precomputed overlap from {cache_path}", file=sys.stderr)
        with open(cache_path) as f:
            cache = json.load(f)
        labels = cache["labels"]
        matrix = np.array(cache["matrix"])
        return matrix, labels

    # --- Try JSONL files ---
    token_sets: dict[str, set[str]] = {}
    labels: list[str] = []

    all_candidates = []
    for path in sorted(data_dir.glob("*_posts.jsonl")):
        # Derive forum_id from filename
        forum_id = path.stem.replace("_posts", "")
        all_candidates.append((forum_id, path))
    for path in sorted(data_dir.glob("*_raw.jsonl")):
        forum_id = path.stem.replace("_raw", "")
        if forum_id not in {f for f, _ in all_candidates}:
            all_candidates.append((forum_id, path))

    if not all_candidates:
        print(
            "ERROR: No JSONL files found and no precomputed cache at:\n"
            f"  {cache_path}\n\n"
            "To regenerate domain_overlap.pdf you need either:\n"
            "  (a) The per-source JSONL files in data/ (gitignored — download from HuggingFace), or\n"
            "  (b) A precomputed data/domain_overlap_cache.json (run this script once with data,\n"
            "      then commit the cache).\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(all_candidates)} JSONL files — building token sets …", file=sys.stderr)
    for forum_id, path in all_candidates:
        print(f"  {forum_id} …", file=sys.stderr)
        ts = _build_token_set_from_jsonl(path)
        if len(ts) >= MIN_DOCS:
            token_sets[forum_id] = ts
            labels.append(forum_id)

    if len(labels) < 2:
        print("ERROR: Not enough sources with sufficient data for overlap computation.", file=sys.stderr)
        sys.exit(1)

    print(f"Computing {len(labels)}×{len(labels)} Jaccard matrix …", file=sys.stderr)
    n = len(labels)
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            j_val = _jaccard(token_sets[labels[i]], token_sets[labels[j]])
            matrix[i, j] = matrix[j, i] = j_val

    # Save cache for future runs
    cache = {"labels": labels, "matrix": matrix.tolist()}
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    print(f"Saved cache to {cache_path}", file=sys.stderr)

    return matrix, labels


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_overlap(matrix: np.ndarray, labels: list[str], out_dir: Path) -> None:
    n = len(labels)
    # Hierarchical clustering on distance = 1 - Jaccard
    dist = 1.0 - matrix
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    order = dendrogram(Z, no_plot=True)["leaves"]

    reordered = matrix[np.ix_(order, order)]
    reordered_labels = [labels[i] for i in order]

    fig_size = max(7.0, n * 0.38 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(
        reordered,
        cmap="Blues",
        vmin=0,
        vmax=min(0.5, reordered.max()),
        aspect="auto",
    )

    ax.set_xticks(range(n))
    ax.set_xticklabels(reordered_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(reordered_labels, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Jaccard similarity (5% token sample)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(
        f"Pairwise vocabulary overlap across {n} hacker community forums\n"
        "(5% deterministic token sample, hierarchically clustered)",
        fontsize=9,
    )

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        dest = out_dir / f"domain_overlap.{ext}"
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

    matrix, labels = load_or_compute(data_dir)
    plot_overlap(matrix, labels, out_dir)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
