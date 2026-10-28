"""Generate the baseline radar figure (baseline_results_summary.{pdf,png}).

Every plotted value is read from results/camera_ready/results.csv, the single
source of truth written by audit/finalize.py. Nothing is estimated, copied from
the paper, or recomputed here: if a model or metric is missing from results.csv
the script stops with an error instead of drawing a guess.

Tasks 1 and 3: retrieval rows, condition as_released, variant primary.
Task 2: classification rows (mean over seeds), one variant per model family
(--t2-variants; defaults match audit/finalize.py T2_PRIMARY).

Usage
-----
    python3 scripts/plot_baseline_radar.py [--results results/camera_ready/results.csv]
                                           [--out-dir results/camera_ready/figures]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths and plotted models
# ---------------------------------------------------------------------------

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results/camera_ready/results.csv"
OUT_DIR = ROOT / "results/camera_ready/figures"

RETRIEVAL_MODELS = ["BM25", "all-MiniLM-L6-v2", "all-mpnet-base-v2", "BGE-base-en-v1.5", "E5-base-v2",
                    "Hybrid BM25+mpnet"]
T2_MODELS = ["Decision Tree", "TF-IDF + LR", "SVM", "RNN", "GRU", "LSTM", "BiLSTM", "SecBERT", "SecureBERT", "CySecBERT"]
T2_FAMILY = {"Decision Tree": "bow", "TF-IDF + LR": "bow", "SVM": "bow", "RNN": "rnn", "GRU": "rnn",
             "LSTM": "rnn", "BiLSTM": "rnn", "SecBERT": "tf", "SecureBERT": "tf", "CySecBERT": "tf"}
T2_VARIANTS = {"bow": "repo_trunc2000", "rnn": "packed_last_token", "tf": "subsample15000"}
PAPER_NAME = {"all-MiniLM-L6-v2": "MiniLM-L6-v2", "all-mpnet-base-v2": "mpnet-base-v2",
              "BGE-base-en-v1.5": "BGE-base-v1.5"}

# ---------------------------------------------------------------------------
# Visual design — colours, widths, fills, styles
# ---------------------------------------------------------------------------

# Primary colour per model (consistent across all panels)
MODEL_COLORS: dict[str, str] = {
    # ── Retrieval models (Tasks 1 & 3) ──────────────────────────────────────
    "BM25":              "#595959",   # dark charcoal  – lexical baseline
    "MiniLM-L6-v2":      "#5ba4cf",   # sky blue       – small bi-encoder
    "mpnet-base-v2":     "#f28e2b",   # amber orange   – base bi-encoder
    "BGE-base-v1.5":     "#59a14f",   # leaf green     – base bi-encoder
    "E5-base-v2":        "#1d479e",   # deep navy      – base bi-encoder
    "Hybrid BM25+mpnet": "#76b7b2",   # teal cyan      – hybrid system
    # ── Classification models (Task 2) ──────────────────────────────────────
    "Decision Tree":     "#c8a227",   # harvest gold   – BoW tier
    "TF-IDF + LR":       "#499894",   # sea teal       – BoW tier
    "SVM":               "#9477ab",   # muted violet   – BoW tier
    "RNN":               "#b07aa1",   # mauve          – RNN family
    "GRU":               "#e05b5b",   # coral red      – RNN family
    "LSTM":              "#e8958e",   # salmon pink    – RNN family
    "BiLSTM":            "#982b1c",   # deep crimson   – RNN family
    "SecBERT":           "#c0602b",   # burnt sienna   – transformer
    "SecureBERT":        "#8c564b",   # umber          – transformer
    "CySecBERT":         "#d4a373",   # tan            – transformer
}

# Line style: dashed for hybrid/combined, solid for all others
MODEL_LS: dict[str, str] = {
    "Hybrid BM25+mpnet": (0, (5, 3)),   # custom dash
}

# Line width and fill: heavier for the top model of each panel (set in main from results.csv)
MODEL_LW: dict[str, float] = {}
MODEL_FA: dict[str, float] = {}

_DEFAULT_LW = 1.5
_TOP_LW     = 2.4
_DEFAULT_FA = 0.07
_TOP_FA     = 0.16
_DEFAULT_LS = "-"

# ---------------------------------------------------------------------------
# Results loading (fails loudly; never estimates)
# ---------------------------------------------------------------------------

class MissingResult(RuntimeError):
    pass


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise MissingResult(f"{path} not found; run audit/finalize.py first")
    with open(path) as f:
        return list(csv.DictReader(f))


def _value(row: dict, key: str, where: str) -> float:
    v = row.get(key, "")
    if v in ("", None):
        raise MissingResult(f"{where}: metric {key!r} is empty in results.csv")
    return float(v)


def _retrieval(rows: list[dict], task: str) -> dict[str, list[float]]:
    out = {}
    for m in RETRIEVAL_MODELS:
        hit = [r for r in rows if r["result_type"] == "retrieval" and r["task"] == task and r["model"] == m
               and r["condition"] == "as_released" and r["variant"] == "primary"]
        if len(hit) != 1:
            raise MissingResult(f"{task}/{m}: expected 1 as_released primary row, found {len(hit)}")
        out[PAPER_NAME.get(m, m)] = [_value(hit[0], k, f"{task}/{m}") for k in ("r_at_1", "r_at_5", "r_at_10", "mrr")]
    return out


def _task2(rows: list[dict], models: list[str], variants: dict[str, str]) -> dict[str, list[float]]:
    out = {}
    for m in models:
        var = variants[T2_FAMILY[m]]
        hit = [r for r in rows if r["result_type"] == "classification" and r["task"] == "task2"
               and r["model"] == m and r["variant"] == var]
        if len(hit) != 1:
            raise MissingResult(f"task2/{m}/{var}: expected 1 row, found {len(hit)}")
        out[m] = [_value(hit[0], k, f"task2/{m}") for k in ("macro_f1", "weighted_f1", "accuracy")]
    return out


# ---------------------------------------------------------------------------
# Radar chart helpers
# ---------------------------------------------------------------------------

def _plot_radar(
    ax,
    categories: list[str],
    models: dict[str, list[float]],
    r_min: float = 0.0,
    r_max: float = 1.0,
    n_rings: int = 5,
    panel_label: str = "",
) -> None:
    """Draw an elegant radar chart on a polar axes `ax`."""
    N      = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # ── Configure polar projection ─────────────────────────────────────────
    ax.set_theta_offset(np.pi / 2)   # first spoke points straight up
    ax.set_theta_direction(-1)        # clockwise (natural reading order)
    ax.set_facecolor("#fafafa")

    # ── Ring (radial) ticks ────────────────────────────────────────────────
    ring_vals = np.linspace(r_min, r_max, n_rings + 1)[1:]
    ax.set_ylim(r_min, r_max)
    ax.set_yticks(ring_vals)
    ax.set_yticklabels([])            # suppress default labels; add below

    # ── Subtle background fill for the outermost ring ─────────────────────
    theta_full = np.linspace(0, 2 * np.pi, 300)
    ax.fill(theta_full, np.full(300, r_max), color="#ececec", zorder=0)
    ax.fill(theta_full, np.full(300, r_min), color="#fafafa", zorder=0)

    # ── Grid lines ─────────────────────────────────────────────────────────
    ax.yaxis.grid(True,  color="#d0d0d0", linewidth=0.6, linestyle="--",
                  zorder=1)
    ax.xaxis.grid(True,  color="#d0d0d0", linewidth=0.7, zorder=1)
    ax.spines["polar"].set_visible(False)

    # ── Ring labels: small text with white background ──────────────────────
    label_angle_rad = np.deg2rad(360 / N * 1.1 + 90)  # just past first spoke
    for rv in ring_vals:
        ax.text(
            label_angle_rad, rv,
            f"{rv:.2f}",
            ha="center", va="center",
            fontsize=5.5, color="#777777",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.82),
            zorder=5,
        )

    # ── Category (spoke) labels ────────────────────────────────────────────
    ax.set_thetagrids(
        [a * 180 / np.pi for a in angles],
        labels=categories,
        fontsize=8.5,
        fontweight="bold",
        color="#333333",
    )
    # nudge labels outward slightly
    ax.tick_params(axis="x", pad=4)

    # ── Plot each model ────────────────────────────────────────────────────
    for model_name, values in models.items():
        vals_closed = values + [values[0]]
        angs_closed = angles + [angles[0]]
        color = MODEL_COLORS.get(model_name, "#aaaaaa")
        lw    = MODEL_LW.get(model_name, _DEFAULT_LW)
        ls    = MODEL_LS.get(model_name, _DEFAULT_LS)
        fa    = MODEL_FA.get(model_name, _DEFAULT_FA)
        is_top = lw > _DEFAULT_LW

        ax.plot(
            angs_closed, vals_closed,
            color=color, linewidth=lw, linestyle=ls,
            zorder=4 if is_top else 3,
            label=model_name,
            solid_capstyle="round",
        )
        ax.fill(
            angs_closed, vals_closed,
            color=color, alpha=fa,
            zorder=2,
        )

    # ── Panel label (A / B / C) ────────────────────────────────────────────
    if panel_label:
        ax.annotate(
            panel_label,
            xy=(0, 1.13), xycoords="axes fraction",
            fontsize=11, fontweight="bold", color="#222222",
            ha="left", va="top",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(results: Path, out_dir: Path, t2_names: list[str], t2_variants: dict[str, str],
         legend_fontsize: int = 8) -> None:
    rows = _load_rows(results)
    t1_models = _retrieval(rows, "task1")
    t3_models = _retrieval(rows, "task3")
    t2_models = _task2(rows, t2_names, t2_variants)
    metrics_t13 = ["R@1", "R@5", "R@10", "MRR"]
    metrics_t2  = ["Macro-F1", "Weighted-F1", "Accuracy"]

    # Heavier line for the top model of each panel by its headline metric (MRR / macro-F1)
    for d, k in ((t1_models, 3), (t3_models, 3), (t2_models, 0)):
        top = max(d, key=lambda m: d[m][k])
        MODEL_LW[top] = _TOP_LW
        MODEL_FA[top] = _TOP_FA

    print(f"Read {results}")
    for name, d in (("Task 1", t1_models), ("Task 2", t2_models), ("Task 3", t3_models)):
        for m, vals in d.items():
            print(f"  {name}  {m:20s} " + "  ".join(f"{v:.4f}" for v in vals))

    # ── Axis bounds ─────────────────────────────────────────────────────────
    def _nice_bounds(vals, margin=0.05):
        lo = max(0.0, math.floor((min(vals) - margin) * 20) / 20)
        hi = min(1.0, math.ceil( (max(vals) + margin) * 20) / 20)
        return lo, hi

    all_t13_vals = [v for d in (t1_models, t3_models)
                    for vs in d.values() for v in vs]
    all_t2_vals  = [v for vs in t2_models.values() for v in vs]

    t13_lo, t13_hi = _nice_bounds(all_t13_vals)
    t2_lo,  t2_hi  = _nice_bounds(all_t2_vals)

    # ── Build figure ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12.5, 4.4))
    fig.patch.set_facecolor("white")

    ax1 = fig.add_subplot(131, polar=True)
    ax2 = fig.add_subplot(132, polar=True)
    ax3 = fig.add_subplot(133, polar=True)

    panels = [
        (ax1, t1_models, metrics_t13, t13_lo, t13_hi,
         "Task 1 — CVE Linkage Retrieval", "A"),
        (ax2, t2_models, metrics_t2,  t2_lo,  t2_hi,
         "Task 2 — Exploit Type Classification", "B"),
        (ax3, t3_models, metrics_t13, t13_lo, t13_hi,
         "Task 3 — Temporal Generalization", "C"),
    ]
    for ax, models, cats, lo, hi, title, plabel in panels:
        _plot_radar(ax, cats, models, r_min=lo, r_max=hi,
                    n_rings=5, panel_label=plabel)
        ax.set_title(title, fontsize=9, fontweight="bold",
                     pad=14, color="#1a1a1a")

    # ── Shared legend ────────────────────────────────────────────────────────
    # Ordered by model family for readability
    legend_order = [
        # retrieval
        "BM25", "MiniLM-L6-v2", "mpnet-base-v2",
        "BGE-base-v1.5", "E5-base-v2", "Hybrid BM25+mpnet",
        # classification
        "Decision Tree", "TF-IDF + LR", "SVM",
        "RNN", "GRU", "LSTM", "BiLSTM", "SecBERT", "SecureBERT", "CySecBERT",
    ]
    all_present = list(dict.fromkeys(
        list(t1_models) + list(t2_models) + list(t3_models)
    ))
    ordered = [n for n in legend_order if n in all_present]
    ordered += [n for n in all_present if n not in ordered]

    legend_handles = [
        mpatches.Patch(
            facecolor=MODEL_COLORS.get(n, "#aaaaaa"),
            edgecolor=MODEL_COLORS.get(n, "#aaaaaa"),
            linewidth=0,
            label=n,
        )
        for n in ordered
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=7,
        fontsize=legend_fontsize,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        facecolor="white",
        bbox_to_anchor=(0.5, 0.01),
        handlelength=1.0,
        handleheight=0.85,
        columnspacing=0.9,
        handletextpad=0.5,
        borderpad=0.6,
    )

    plt.subplots_adjust(
        left=0.01, right=0.99,
        bottom=0.22, top=0.96,
        wspace=0.22,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf, png = out_dir / "baseline_results_summary.pdf", out_dir / "baseline_results_summary.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved → {pdf}")
    print(f"Saved → {png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate baseline radar chart from results.csv")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--t2-models", default=",".join(T2_MODELS),
                        help="comma list of Task 2 models to plot (each must exist in results.csv)")
    parser.add_argument("--t2-variants", default=",".join(f"{k}={v}" for k, v in T2_VARIANTS.items()),
                        help="family=variant pairs, e.g. bow=repo_trunc2000,rnn=packed_last_token,tf=subsample15000")
    parser.add_argument("--legend-fontsize", type=int, default=8,
                        help="Font size for legend entries (default: 8).")
    args = parser.parse_args()
    variants = dict(kv.split("=", 1) for kv in args.t2_variants.split(","))
    try:
        main(args.results, args.out_dir, [m for m in args.t2_models.split(",") if m], variants,
             legend_fontsize=args.legend_fontsize)
    except MissingResult as e:
        sys.exit(f"ERROR: {e}")
