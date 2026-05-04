"""Publication-quality matplotlib style and MIS Quarterly table formatter."""

from __future__ import annotations

import math
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from etg.eval.significance import GROUP_ORDER, MODEL_GROUPS

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

PUB_RC: dict = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
}

GROUP_COLORS: dict[str, str] = {
    "Trivial Baseline": "#9e9e9e",
    "Lexical / Count": "#4e79a7",
    "Classical ML": "#f28e2b",
    "Word Embedding": "#59a14f",
    "Static GNN": "#76b7b2",
    "Temporal GNN": "#edc948",
    "Recurrent / CNN": "#b07aa1",
    "Generative": "#ff9da7",
    "CTE Ablation": "#bab0ac",
    "Proposed (DGT / CTE)": "#e15759",
    "Unknown": "#cccccc",
}


def set_pub_style() -> None:
    """Apply PUB_RC publication style to matplotlib global rcParams."""
    plt.rcParams.update(PUB_RC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_group(name: str) -> str:
    return MODEL_GROUPS.get(name, "Unknown")


def _group_sort_key(name: str) -> int:
    grp = _get_group(name)
    try:
        return GROUP_ORDER.index(grp)
    except ValueError:
        return len(GROUP_ORDER)


def _fmt_mae(val: float) -> str:
    if math.isnan(val):
        return "NaN"
    if abs(val) < 0.001:
        return f"{val:.3e}"
    return f"{val:.5f}"


def _fmt_mape(val: float) -> str:
    if math.isnan(val):
        return "NaN"
    return f"{val:.2f}"


def _fmt_ir(val: float) -> str:
    if math.isnan(val):
        return "NaN"
    return f"{val:.4f}"


# ---------------------------------------------------------------------------
# Table formatters
# ---------------------------------------------------------------------------


def format_rt1_table(
    rt1_results: dict,
    best_baseline_mae: Optional[float] = None,
) -> pd.DataFrame:
    """Build a publication-quality RT1 results DataFrame.

    Parameters
    ----------
    rt1_results :
        Dict mapping model_name → {MAE, RMSE, MAPE, …}.
    best_baseline_mae :
        MAE of the best non-trivial baseline, used to compute Δ for DGT.

    Returns
    -------
    pd.DataFrame with columns:
        Model, Group, MAE, RMSE, MAPE, Δ vs Best Baseline (%), Rank
    """
    rows = []
    for name, m in rt1_results.items():
        rows.append(
            {
                "Model": name,
                "Group": _get_group(name),
                "_group_order": _group_sort_key(name),
                "_mae_raw": float(m.get("MAE", float("nan"))),
                "_rmse_raw": float(m.get("RMSE", float("nan"))),
                "_mape_raw": float(m.get("MAPE", float("nan"))),
            }
        )

    rows.sort(key=lambda r: (r["_group_order"], r["_mae_raw"]))

    # Compute rank (by MAE ascending, exclude CUSUM-DGT)
    rank_rows = [r for r in rows if r["Model"] != "CUSUM-DGT"]
    rank_rows_sorted = sorted(rank_rows, key=lambda r: r["_mae_raw"])
    rank_map = {r["Model"]: i + 1 for i, r in enumerate(rank_rows_sorted)}

    out_rows = []
    for r in rows:
        name = r["Model"]
        mae_raw = r["_mae_raw"]
        delta = float("nan")
        if name == "DGT" and best_baseline_mae is not None and not math.isnan(best_baseline_mae):
            delta = (best_baseline_mae - mae_raw) / abs(best_baseline_mae) * 100.0

        out_rows.append(
            {
                "Model": name,
                "Group": r["Group"],
                "MAE": _fmt_mae(mae_raw),
                "RMSE": _fmt_mae(r["_rmse_raw"]),
                "MAPE": _fmt_mape(r["_mape_raw"]),
                "Δ vs Best Baseline (%)": f"{delta:+.2f}%" if not math.isnan(delta) else "",
                "Rank": rank_map.get(name, "—"),
            }
        )

    df = pd.DataFrame(out_rows)
    df = df.set_index("Model")
    return df


def format_rt2_table(
    rt2_results: dict,
    sig_stars: Optional[dict[str, str]] = None,
    best_baseline: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """Build a publication-quality RT2 results DataFrame.

    Parameters
    ----------
    rt2_results :
        Dict mapping model_name → {HR@10, MRR@10, NDCG@10, MAP, …}.
    sig_stars :
        Optional dict model_name → significance stars string.
    best_baseline :
        Optional dict {metric: value} for the best non-trivial baseline,
        used to compute Δ HR@10 for CTE.

    Returns
    -------
    pd.DataFrame with columns:
        Model, Group, HR@10, MRR@10, NDCG@10, MAP, Δ HR@10 (%), Stars
    """
    rows = []
    for name, m in rt2_results.items():
        rows.append(
            {
                "Model": name,
                "Group": _get_group(name),
                "_group_order": _group_sort_key(name),
                "_hr_raw": float(m.get("HR@10", float("nan"))),
                "_mrr_raw": float(m.get("MRR@10", float("nan"))),
                "_ndcg_raw": float(m.get("NDCG@10", float("nan"))),
                "_map_raw": float(m.get("MAP", float("nan"))),
            }
        )

    rows.sort(key=lambda r: (r["_group_order"], -r["_hr_raw"]))

    out_rows = []
    for r in rows:
        name = r["Model"]
        hr_raw = r["_hr_raw"]
        delta = float("nan")
        if name == "CTE" and best_baseline is not None:
            base_hr = float(best_baseline.get("HR@10", float("nan")))
            if not math.isnan(base_hr) and base_hr != 0:
                delta = (hr_raw - base_hr) / abs(base_hr) * 100.0

        stars_str = (sig_stars or {}).get(name, "")
        display_name = f"{name}{stars_str}" if stars_str else name

        out_rows.append(
            {
                "Model": display_name,
                "Group": r["Group"],
                "HR@10": _fmt_ir(hr_raw),
                "MRR@10": _fmt_ir(r["_mrr_raw"]),
                "NDCG@10": _fmt_ir(r["_ndcg_raw"]),
                "MAP": _fmt_ir(r["_map_raw"]),
                "Δ HR@10 (%)": f"{delta:+.2f}%" if not math.isnan(delta) else "",
            }
        )

    df = pd.DataFrame(out_rows)
    df = df.set_index("Model")
    return df


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _group_legend_handles(groups_present: list[str]) -> list[mpatches.Patch]:
    seen: list[mpatches.Patch] = []
    seen_labels: set[str] = set()
    for grp in GROUP_ORDER:
        if grp in groups_present and grp not in seen_labels:
            seen.append(
                mpatches.Patch(color=GROUP_COLORS.get(grp, "#cccccc"), label=grp)
            )
            seen_labels.add(grp)
    return seen


# ---------------------------------------------------------------------------
# RT1 publication figure
# ---------------------------------------------------------------------------


def plot_rt1_pub(
    rt1_results: dict,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Horizontal bar chart of RT1 MAE, publication quality.

    - Sorted by MAE ascending (best at top).
    - Colored by model group.
    - Log x-axis.
    - Red dashed reference line at DGT's MAE.
    - CUSUM-DGT excluded (trivially zero / degenerate).
    """
    set_pub_style()

    # Build sorted data, exclude CUSUM-DGT
    data = [
        (name, float(m.get("MAE", float("nan"))))
        for name, m in rt1_results.items()
        if name != "CUSUM-DGT" and not math.isnan(float(m.get("MAE", float("nan"))))
    ]
    data.sort(key=lambda x: x[1], reverse=True)  # worst at top → best at bottom

    names = [d[0] for d in data]
    maes = [d[1] for d in data]
    colors = [GROUP_COLORS.get(_get_group(n), "#cccccc") for n in names]
    groups = [_get_group(n) for n in names]

    dgt_mae = next((m for n, m in data if n == "DGT"), None)

    fig, ax = plt.subplots(figsize=(8, 9))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, maes, color=colors, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Mean Absolute Error (log scale)")
    ax.set_title("RT1 Forecast MAE — Lower is Better", fontweight="bold")

    if dgt_mae is not None:
        ax.axvline(dgt_mae, color="#e15759", linestyle="--", linewidth=1.2, label="DGT")
        dgt_idx = names.index("DGT")
        ax.annotate(
            "[Best]",
            xy=(dgt_mae, dgt_idx),
            xytext=(dgt_mae * 1.15, dgt_idx),
            fontsize=7,
            color="#e15759",
            va="center",
            fontweight="bold",
        )

    handles = _group_legend_handles(list(set(groups)))
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=7,
        framealpha=0.9,
        title="Model Group",
        title_fontsize=7,
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# RT2 publication figure
# ---------------------------------------------------------------------------


def plot_rt2_pub(
    rt2_results: dict,
    sig_stars: Optional[dict[str, str]] = None,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Horizontal bar chart of RT2 HR@10, publication quality.

    - Sorted by HR@10 ascending (best at top).
    - Colored by model group.
    - Reference line at CTE's HR@10.
    - Excludes models with HR@10 == 0 (trivially zero, e.g. KNRM).
    - Dual x-axes: primary HR@10, secondary MAP as hollow circle markers.
    """
    set_pub_style()

    data = [
        (
            name,
            float(m.get("HR@10", float("nan"))),
            float(m.get("MAP", float("nan"))),
        )
        for name, m in rt2_results.items()
        if float(m.get("HR@10", 0.0)) > 0 and not math.isnan(float(m.get("HR@10", float("nan"))))
    ]
    data.sort(key=lambda x: x[1], reverse=True)  # worst first (top of horizontal bar)
    data = data[::-1]  # best at top

    names = [d[0] for d in data]
    hrs = [d[1] for d in data]
    maps = [d[2] for d in data]
    colors = [GROUP_COLORS.get(_get_group(n), "#cccccc") for n in names]
    groups = [_get_group(n) for n in names]

    cte_hr = next((h for n, h, _ in data if n == "CTE"), None)
    cte_idx = names.index("CTE") if "CTE" in names else None

    fig, ax = plt.subplots(figsize=(9, 11))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, hrs, color=colors, height=0.65)

    # Secondary: MAP as hollow circles
    ax2 = ax.twiny()
    ax2.scatter(maps, y_pos, s=20, facecolors="none", edgecolors="#333333", linewidths=0.8, zorder=5)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("MAP (hollow markers)", fontsize=8)
    ax2.tick_params(labelsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("HR@10")
    ax.set_title("RT2 Exploit–Vulnerability Linking (HR@10) — Higher is Better", fontweight="bold")

    if cte_hr is not None:
        ax.axvline(cte_hr, color="#e15759", linestyle="--", linewidth=1.2)
    if cte_idx is not None and cte_hr is not None:
        stars_str = (sig_stars or {}).get("CTE", "")
        label = f"CTE {stars_str}".strip()
        ax.annotate(
            label,
            xy=(cte_hr, cte_idx),
            xytext=(cte_hr + 0.02, cte_idx),
            fontsize=7,
            color="#e15759",
            va="center",
            fontweight="bold",
        )

    handles = _group_legend_handles(list(set(groups)))
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=7,
        framealpha=0.9,
        title="Model Group",
        title_fontsize=7,
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Combined two-panel summary figure
# ---------------------------------------------------------------------------


def plot_rt1_rt2_comparison(
    rt1_results: dict,
    rt2_results: dict,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Two-panel (1 row × 2 cols) summary figure for journal double-column.

    Left panel:  top-10 RT1 models by MAE (horizontal bars).
    Right panel: top-10 RT2 models by HR@10 (horizontal bars).
    Both panels share the same group color coding.
    """
    set_pub_style()

    # --- RT1 top-10 by MAE (exclude CUSUM-DGT) ---
    rt1_data = [
        (name, float(m.get("MAE", float("nan"))))
        for name, m in rt1_results.items()
        if name != "CUSUM-DGT" and not math.isnan(float(m.get("MAE", float("nan"))))
    ]
    rt1_data.sort(key=lambda x: x[1])
    rt1_top = rt1_data[:10]
    rt1_top.reverse()  # worst at top so best at bottom

    # --- RT2 top-10 by HR@10 ---
    rt2_data = [
        (name, float(m.get("HR@10", float("nan"))))
        for name, m in rt2_results.items()
        if not math.isnan(float(m.get("HR@10", float("nan")))) and float(m.get("HR@10", 0.0)) > 0
    ]
    rt2_data.sort(key=lambda x: x[1])
    rt2_top = rt2_data[-10:]
    rt2_top.reverse()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: RT1
    ax1 = axes[0]
    n1 = [d[0] for d in rt1_top]
    v1 = [d[1] for d in rt1_top]
    c1 = [GROUP_COLORS.get(_get_group(n), "#cccccc") for n in n1]
    y1 = np.arange(len(n1))
    ax1.barh(y1, v1, color=c1, height=0.65)
    ax1.set_yticks(y1)
    ax1.set_yticklabels(n1, fontsize=7)
    ax1.set_xscale("log")
    ax1.set_xlabel("MAE (log scale)", fontsize=9)
    ax1.set_title("Top-10 RT1 Models (MAE)", fontweight="bold", fontsize=10)

    # Right: RT2
    ax2 = axes[1]
    n2 = [d[0] for d in rt2_top]
    v2 = [d[1] for d in rt2_top]
    c2 = [GROUP_COLORS.get(_get_group(n), "#cccccc") for n in n2]
    y2 = np.arange(len(n2))
    ax2.barh(y2, v2, color=c2, height=0.65)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(n2, fontsize=7)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("HR@10", fontsize=9)
    ax2.set_title("Top-10 RT2 Models (HR@10)", fontweight="bold", fontsize=10)

    # Shared legend below both panels
    all_groups = list({_get_group(n) for n in n1 + n2})
    handles = _group_legend_handles(all_groups)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(5, len(handles)),
        fontsize=7,
        framealpha=0.9,
        title="Model Group",
        title_fontsize=7,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
