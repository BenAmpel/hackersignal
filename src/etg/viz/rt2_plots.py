"""Visualizations for RT2: contrastive UMAP, score distributions, PR/ROC,
local-guided global context attention heatmap."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt


def plot_contrastive_view_umap(
    z_a: np.ndarray, z_b: np.ndarray, n_show: int = 150, save_path: str | None = None
):
    try:
        import umap

        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=0)
    except Exception:
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2)
    k = min(n_show, z_a.shape[0])
    combined = np.concatenate([z_a[:k], z_b[:k]], axis=0)
    coords = reducer.fit_transform(combined)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(coords[:k, 0], coords[:k, 1], s=14, color="steelblue", label="view A", alpha=0.8)
    ax.scatter(coords[k:, 0], coords[k:, 1], s=14, color="crimson", label="view B", alpha=0.8)
    for i in range(k):
        ax.plot(
            [coords[i, 0], coords[k + i, 0]],
            [coords[i, 1], coords[k + i, 1]],
            color="gray",
            alpha=0.3,
        )
    ax.set_title("CTE contrastive views (pairs connected)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
    return fig


def plot_score_histogram(
    scores: np.ndarray, positive_indices: np.ndarray, save_path: str | None = None
):
    pos_scores = np.array(
        [scores[q, int(positive_indices[q])] for q in range(scores.shape[0])]
    )
    neg_scores = []
    for q in range(scores.shape[0]):
        mask = np.ones(scores.shape[1], dtype=bool)
        mask[int(positive_indices[q])] = False
        neg_scores.extend(scores[q, mask].tolist())
    neg_scores = np.array(neg_scores[: 10 * len(pos_scores)])
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.hist(neg_scores, bins=40, alpha=0.5, color="gray", label="negatives")
    ax.hist(pos_scores, bins=40, alpha=0.7, color="crimson", label="positives")
    ax.set_xlabel("cosine similarity")
    ax.set_title("CTE score distribution — positives vs negatives")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
    return fig


def plot_pr_roc(
    scores: np.ndarray, positive_indices: np.ndarray, save_path: str | None = None
):
    # Stack positives and sampled negatives into a binary classification set.
    y_true = []
    y_score = []
    for q in range(scores.shape[0]):
        pos = int(positive_indices[q])
        y_true.append(1)
        y_score.append(float(scores[q, pos]))
        # sample up to 10 negatives
        neg_candidates = [i for i in range(scores.shape[1]) if i != pos][:10]
        for j in neg_candidates:
            y_true.append(0)
            y_score.append(float(scores[q, j]))
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].plot(recall, precision, color="crimson")
    axes[0].set_xlabel("recall")
    axes[0].set_ylabel("precision")
    axes[0].set_title("Precision–Recall")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(fpr, tpr, color="steelblue")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
    return fig


def plot_rt2_summary_bar(
    results: dict[str, dict[str, float]], metric: str, save_path: str | None = None
):
    methods = list(results.keys())
    vals = [results[m].get(metric, np.nan) for m in methods]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(methods, vals, color="crimson")
    ax.set_ylabel(metric)
    ax.set_title(f"RT2 — {metric} vs baselines")
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=20)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
    return fig


def plot_cte_attention_heatmap(
    attn: np.ndarray, tokens: list[str], save_path: str | None = None
):
    fig, ax = plt.subplots(figsize=(max(5.0, len(tokens) * 0.25), 2.5))
    im = ax.imshow(attn[None, :], aspect="auto", cmap="magma")
    ax.set_yticks([])
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=75, fontsize=8)
    ax.set_title("Local-guided global attention weights (sample)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
    return fig
