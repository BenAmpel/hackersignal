#!/usr/bin/env python3
"""Publication-quality 2x3 case-study figure for MISQ.

Generates a clean, annotated panel figure showing how six well-known
cybersecurity terms shifted meaning in hacker forums from 2016-2026,
anchored to real-world events a general audience would recognise.

Usage (from repo root):
    python scripts/generate_case_study_figure.py
    python scripts/generate_case_study_figure.py --output path/to/fig.png --dpi 600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# -- Repo-relative defaults --------------------------------------------------
REPO      = Path(__file__).resolve().parent.parent
EMB_PATH  = REPO / "ETG_MISQ/output/dgt/dgt_embeddings.npz"
OUT_PATH  = REPO / "ETG_MISQ/figures/misq_case_study_shifts.png"

# -- Term definitions ---------------------------------------------------------
# Each dict: term (vocab key), display label, Okabe-Ito colour,
#            peak_t (1-indexed transition; 1 = S01->S02),
#            event text (shown in callout box),
#            insight (shown as italic panel subtitle),
#            text_side ("left" or "right" of the peak for the callout box).
TERMS = [
    {
        "term":      "iot",
        "label":     "IoT",
        "color":     "#0072B2",   # Okabe-Ito blue
        "peak_t":    1,
        "event":     "Mirai Botnet (Oct 2016)\nLargest-ever DDoS attack",
        "insight":   "IoT devices reframed as attack infrastructure",
        "text_side": "right",
    },
    {
        "term":      "rce",
        "label":     "Remote Code Execution (RCE)",
        "color":     "#D55E00",   # Okabe-Ito vermilion
        "peak_t":    2,
        "event":     "WannaCry / EternalBlue (May 2017)\nNSA exploit leaked and weaponised",
        "insight":   "RCE becomes the defining exploit shorthand of the era",
        "text_side": "right",
    },
    {
        "term":      "ransomware",
        "label":     "Ransomware",
        "color":     "#009E73",   # Okabe-Ito green
        "peak_t":    4,
        "event":     "GandCrab / REvil (2018-19)\nRansomware-as-a-Service launched",
        "insight":   "Malware campaign evolves into criminal franchise",
        "text_side": "right",
    },
    {
        "term":      "stealer",
        "label":     "Stealer",
        "color":     "#CC79A7",   # Okabe-Ito pink
        "peak_t":    6,
        "event":     "RedLine & Raccoon (2020-21)\nSubscription credential-theft market",
        "insight":   "A new underground product category is born",
        "text_side": "right",
    },
    {
        "term":      "ivanti",
        "label":     "Ivanti",
        "color":     "#E69F00",   # Okabe-Ito orange
        "peak_t":    9,
        "event":     "Ivanti Connect Secure (2023-24)\nNation-state zero-day exploitation",
        "insight":   "Vendor name becomes shorthand for VPN zero-days",
        "text_side": "left",
    },
    {
        "term":      "jailbreak",
        "label":     "Jailbreak",
        "color":     "#404040",   # dark grey
        "peak_t":    10,
        "event":     "ChatGPT / GPT-4 (2023-24)\nLLM safety-bypass techniques surge",
        "insight":   "iOS firmware bypass repurposed for AI safety filters",
        "text_side": "left",
    },
]

# Approximate calendar year for the END of each spell (= transition x-position)
# Spell ends: Nov-16, Sep-17, Jul-18, Jun-19, Apr-20, Feb-21,
#             Jan-22, Nov-22, Sep-23, Aug-24, Jun-25
_SPELL_END_YEARS = [
    2016.87,  # S01->S02
    2017.70,  # S02->S03
    2018.53,  # S03->S04
    2019.46,  # S04->S05
    2020.29,  # S05->S06
    2021.12,  # S06->S07
    2022.04,  # S07->S08
    2022.87,  # S08->S09
    2023.70,  # S09->S10
    2024.62,  # S10->S11
    2025.46,  # S11->S12
]


# -- Embedding helpers --------------------------------------------------------

def _cosine_shifts(emb_dict: dict, term: str, n_spells: int = 12) -> list[float]:
    """Per-transition cosine distances for *term* (NaN if zero vector)."""
    word_list = emb_dict["words"].tolist()
    if term not in word_list:
        return [float("nan")] * (n_spells - 1)
    idx = word_list.index(term)
    out = []
    for t in range(1, n_spells):
        a = emb_dict[f"G_{t:02d}"][idx].astype(np.float64)
        b = emb_dict[f"G_{t+1:02d}"][idx].astype(np.float64)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            out.append(float("nan"))
        else:
            out.append(float(np.clip(1.0 - np.dot(a / na, b / nb), 0.0, 2.0)))
    return out


# -- Figure -------------------------------------------------------------------

def make_figure(emb_dict: dict, output_path: Path, dpi: int = 300) -> None:
    """Generate and save the 2x3 MISQ case-study figure."""
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "axes.grid.axis":     "y",
        "grid.color":         "#E8E8E8",
        "grid.linewidth":     0.55,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
    })

    xs = np.array(_SPELL_END_YEARS)   # (11,) calendar years as floats
    x_span = xs[-1] - xs[0]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(14, 8.2),
        gridspec_kw={"hspace": 0.72, "wspace": 0.36},
    )

    for ax, meta in zip(axes.flatten(), TERMS):
        term      = meta["term"]
        color     = meta["color"]
        label     = meta["label"]
        peak_i    = meta["peak_t"] - 1   # 0-indexed into xs
        event     = meta["event"]
        insight   = meta["insight"]
        text_side = meta.get("text_side", "right")

        # Compute shifts; replace NaN with 0 for plotting but keep NaN array
        raw   = _cosine_shifts(emb_dict, term)
        ys    = np.array([v if not np.isnan(v) else np.nan for v in raw])
        ys0   = np.where(np.isnan(ys), 0.0, ys)   # fill for area plot
        y_max = float(np.nanmax(ys))

        # -- Area + line ------------------------------------------------------
        ax.fill_between(xs, ys0, alpha=0.13, color=color, linewidth=0)
        ax.plot(xs, ys0, color=color, linewidth=2.1,
                solid_capstyle="round", zorder=3)
        ax.plot(xs, ys0, "o", color=color, markersize=4.0,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)

        # -- Set axes limits before annotating --------------------------------
        y_ceil = y_max * 1.80     # generous headroom for the callout box
        ax.set_ylim(0, y_ceil)
        ax.set_xlim(xs[0] - 0.4, xs[-1] + 0.4)

        # -- Peak marker ------------------------------------------------------
        peak_x = xs[peak_i]
        peak_y = ys0[peak_i]
        ax.axvline(peak_x, color=color, linewidth=0.8,
                   linestyle="--", alpha=0.40, zorder=2)
        ax.plot(peak_x, peak_y, "o", color=color, markersize=9, zorder=5,
                markeredgecolor="white", markeredgewidth=1.6)

        # -- Event callout box ------------------------------------------------
        # Place the box well above the peak value, left or right as specified
        text_y  = y_max * 1.30
        offset  = x_span * 0.22

        if text_side == "right":
            text_x = min(peak_x + offset, xs[-1] - x_span * 0.04)
            ha     = "left"
            rad    = 0.20
        else:
            text_x = max(peak_x - offset, xs[0] + x_span * 0.04)
            ha     = "right"
            rad    = -0.20

        ax.annotate(
            event,
            xy=(peak_x, peak_y),
            xytext=(text_x, text_y),
            fontsize=8.4,
            ha=ha, va="center",
            color="#111111",
            bbox=dict(
                boxstyle="round,pad=0.42",
                facecolor="#F9F9F9",
                edgecolor=color,
                linewidth=1.5,
                alpha=0.97,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=1.2,
                connectionstyle=f"arc3,rad={rad}",
            ),
            zorder=7,
        )

        # -- Axes formatting --------------------------------------------------
        ax.set_xticks([2017, 2019, 2021, 2023, 2025])
        ax.set_xticklabels(["2017", "2019", "2021", "2023", "2025"],
                           fontsize=9.2)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(3, prune="upper"))
        ax.tick_params(axis="y", labelsize=9.2)
        ax.set_ylabel("Semantic drift", fontsize=9.5, color="#444444", labelpad=3)

        # -- Panel heading: bold term name + italic insight on next line ------
        ax.set_title(label, fontsize=13, fontweight="bold",
                     loc="left", pad=16, color="#111111")
        ax.text(
            0.0, 1.042, insight,
            transform=ax.transAxes,
            fontsize=8.4, style="italic", color="#666666",
            va="bottom", ha="left",
        )

    # -- x-axis label on bottom row only -------------------------------------
    for ax in axes[1]:
        ax.set_xlabel("Year", fontsize=10, color="#333333")

    # -- Figure-level title and subtitle -------------------------------------
    fig.text(
        0.5, 1.002,
        "How Hacker-Forum Vocabulary Shifted Around Six Key Cybersecurity Events (2016-2026)",
        ha="center", va="bottom",
        fontsize=14, fontweight="bold", color="#111111",
    )
    fig.text(
        0.5, 0.994,
        "Semantic drift score = cosine distance between consecutive 9-month embedding windows. "
        "Near zero = stable usage; higher values = term entered a new context.",
        ha="center", va="top",
        fontsize=9.5, color="#555555", style="italic",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Figure saved -> {output_path}  ({dpi} dpi)")


# -- CLI ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emb",    default=str(EMB_PATH),
                   help="Path to dgt_embeddings.npz")
    p.add_argument("--output", default=str(OUT_PATH),
                   help="Output PNG/PDF path")
    p.add_argument("--dpi",    default=300, type=int,
                   help="Output resolution (default 300)")
    args = p.parse_args(argv)

    emb_path = Path(args.emb)
    if not emb_path.exists():
        sys.exit(
            f"Embeddings not found: {emb_path}\n"
            "Run the DGT pipeline first or point --emb at the correct file."
        )

    print(f"Loading embeddings from {emb_path} ...")
    emb_dict = dict(np.load(emb_path, allow_pickle=True))
    print(f"  vocab size: {len(emb_dict['words'])}")

    for meta in TERMS:
        t     = meta["term"]
        found = t in emb_dict["words"].tolist()
        print(f"  '{t}': {'found' if found else 'NOT IN VOCAB'}")

    make_figure(emb_dict, Path(args.output), dpi=args.dpi)


if __name__ == "__main__":
    main()
