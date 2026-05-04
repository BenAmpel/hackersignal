"""Visualizations for RT1: ETG statistics, shift detection, forecasting, attention."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt


def plot_etg_degree_distributions(snapshots, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for snap in snapshots:
        V = snap["x"].shape[0]
        deg = np.zeros(V, dtype=np.int64)
        ei = snap["edge_index"].cpu().numpy()
        for s in ei[0]:
            deg[s] += 1
        for d in ei[1]:
            deg[d] += 1
        deg = deg[deg > 0]
        if len(deg) == 0:
            continue
        vals, counts = np.unique(deg, return_counts=True)
        ax.loglog(vals, counts, marker="o", linestyle="-", label=f"t={snap['t']}", alpha=0.7)
    ax.set_xlabel("degree")
    ax.set_ylabel("count")
    ax.set_title("ETG node-degree distribution per spell")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
    return fig


def plot_embedding_trajectories_umap(
    embeddings,
    vocab,
    watch_words: list[str],
    save_path: str | None = None,
):
    try:
        import umap

        reducer = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.3, random_state=0)
    except Exception:
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2)
    all_pts = np.stack([e.numpy() for e in embeddings], axis=0)  # [T, V, d]
    T, V, d = all_pts.shape
    flat = all_pts.reshape(T * V, d)
    coords = reducer.fit_transform(flat).reshape(T, V, 2)
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.scatter(coords[0, :, 0], coords[0, :, 1], s=2, alpha=0.1, color="gray")
    palette = plt.cm.tab10(np.linspace(0, 1, max(1, len(watch_words))))
    for i, w in enumerate(watch_words):
        if w not in vocab.tokens_to_id:
            continue
        v = vocab.id(w)
        path = coords[:, v, :]
        ax.plot(path[:, 0], path[:, 1], "-o", color=palette[i], alpha=0.85, label=w)
        for t in range(T):
            ax.annotate(str(t), (path[t, 0], path[t, 1]), fontsize=7)
    ax.set_title("Word trajectories across time-spells")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
    return fig


def plot_attention_heatmap(
    attn_matrix: np.ndarray,
    labels: list[str],
    save_path: str | None = None,
):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    im = ax.imshow(attn_matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("GT-MHMSA attention (sample)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
    return fig


def plot_shift_score_histograms(shift_results, save_path: str | None = None):
    n = len(shift_results)
    if n == 0:
        return None
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.0), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, shift_results):
        if len(r.scores) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        ax.hist(r.scores, bins=30, color="steelblue", alpha=0.8)
        ax.axvline(r.threshold, color="red", linestyle="--", label=f"thr={r.threshold:.3f}")
        ax.set_title(f"t={r.t_prev}→{r.t_curr}")
        ax.set_xlabel("cosine drift")
        ax.legend(fontsize=7)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
    return fig


def plot_forecast_band(
    series: np.ndarray,
    preds: np.ndarray,
    actuals: np.ndarray,
    save_path: str | None = None,
):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x_full = np.arange(len(series))
    ax.plot(x_full, series, "o-", color="black", label="observed")
    if len(preds) > 0:
        x_pred = np.arange(len(series) - len(preds), len(series))
        ax.plot(x_pred, preds, "s--", color="crimson", label="ARIMA pred")
    ax.set_xlabel("spell pair index")
    ax.set_ylabel("mean cosine drift")
    ax.set_title("Shift-score forecasting")
    ax.legend()
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
    return fig


def plot_rt1_summary_bar(results: dict[str, dict[str, float]], metric: str, save_path: str | None = None):
    methods = list(results.keys())
    vals = [results[m].get(metric, np.nan) for m in methods]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.bar(methods, vals, color="steelblue")
    ax.set_ylabel(metric)
    ax.set_title(f"RT1 — {metric} vs baselines")
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
    return fig
