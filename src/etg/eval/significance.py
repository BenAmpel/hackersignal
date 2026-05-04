"""Statistical significance utilities for publication-quality reporting."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependencies (scipy / statsmodels) with graceful fallback
# ---------------------------------------------------------------------------
try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model group taxonomy
# ---------------------------------------------------------------------------

MODEL_GROUPS: dict[str, str] = {
    # Trivial Baseline
    "Random": "Trivial Baseline",
    "CUSUM-DGT": "Trivial Baseline",
    "naive_last": "Trivial Baseline",
    "ETS": "Trivial Baseline",
    # Lexical / Count
    "BoW-drift": "Lexical / Count",
    "FreqShift": "Lexical / Count",
    "PMI-ratio": "Lexical / Count",
    "PPMI-SVD": "Lexical / Count",
    "LDA-drift": "Lexical / Count",
    "TF-IDF": "Lexical / Count",
    "BM25": "Lexical / Count",
    "BM25+": "Lexical / Count",
    "BM25L": "Lexical / Count",
    "LSA": "Lexical / Count",
    "Jaccard": "Lexical / Count",
    "QueryLikelihood": "Lexical / Count",
    "CharNgram(3,4)": "Lexical / Count",
    "ROUGE-L": "Lexical / Count",
    "EditDist": "Lexical / Count",
    "WMD": "Lexical / Count",
    # Classical ML
    "LogisticReg": "Classical ML",
    "LinearSVM": "Classical ML",
    "RandomForest": "Classical ML",
    "XGBoost(GB)": "Classical ML",
    "HandFeature-GB": "Classical ML",
    # Word Embedding
    "word2vec": "Word Embedding",
    "fasttext": "Word Embedding",
    "GloVe": "Word Embedding",
    "W2V-Mean": "Word Embedding",
    "FastText-Mean": "Word Embedding",
    "GloVe-Mean": "Word Embedding",
    "Doc2Vec": "Word Embedding",
    "DeepWalk": "Word Embedding",
    "LINE": "Word Embedding",
    "LapPE-direct": "Word Embedding",
    # Static GNN
    "StaticGCN": "Static GNN",
    "GAT": "Static GNN",
    "GAE": "Static GNN",
    "GraphSAGE": "Static GNN",
    "GIN": "Static GNN",
    # Temporal GNN
    "TGAT": "Temporal GNN",
    "EvolveGCN": "Temporal GNN",
    # Recurrent / CNN
    "BiLSTM-CNN": "Recurrent / CNN",
    "SiameseBiLSTM": "Recurrent / CNN",
    "SiameseBiLSTM-HardNeg": "Recurrent / CNN",
    "CNNEncoder": "Recurrent / CNN",
    "DSSM": "Recurrent / CNN",
    "ESIM-lite": "Recurrent / CNN",
    "MatchPyramid": "Recurrent / CNN",
    "KNRM": "Recurrent / CNN",
    # Generative
    "Autoencoder": "Generative",
    "VAE": "Generative",
    # CTE Ablation
    "CTE-MLM-only": "CTE Ablation",
    "CTE-Contrastive-only": "CTE Ablation",
    "CTE-FT-only": "CTE Ablation",
    "CTE-no-temporal": "CTE Ablation",
    "CTE-dual-encoder": "CTE Ablation",
    "CTE-tau=0.01": "CTE Ablation",
    "CTE-tau=0.07": "CTE Ablation",
    "CTE-tau=0.2": "CTE Ablation",
    # Proposed
    "DGT": "Proposed (DGT / CTE)",
    "CTE": "Proposed (DGT / CTE)",
}

# Display order for table groups — proposed last (bottom of table = best)
GROUP_ORDER: list[str] = [
    "Trivial Baseline",
    "Lexical / Count",
    "Classical ML",
    "Word Embedding",
    "Static GNN",
    "Temporal GNN",
    "Recurrent / CNN",
    "Generative",
    "CTE Ablation",
    "Proposed (DGT / CTE)",
    "Unknown",
]


def _get_group(model_name: str) -> str:
    """Return the group string for a model, falling back to 'Unknown'."""
    return MODEL_GROUPS.get(model_name, "Unknown")


# ---------------------------------------------------------------------------
# Significance helpers
# ---------------------------------------------------------------------------


def significance_stars(p_value: float) -> str:
    """Return APA-style significance stars for a given p-value.

    Returns
    -------
    "***"  p < 0.001
    "**"   p < 0.01
    "*"    p < 0.05
    "†"    p < 0.10
    ""     otherwise
    """
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.10:
        return "\u2020"
    return ""


def relative_improvement(
    model_val: float,
    baseline_val: float,
    lower_is_better: bool = True,
) -> float:
    """Percentage improvement of *model_val* relative to *baseline_val*.

    For lower-is-better metrics (MAE, RMSE …) a reduction is positive gain.
    For higher-is-better metrics (HR, NDCG …) an increase is positive gain.

    Returns
    -------
    float  Signed percentage improvement (e.g. 12.3 means +12.3 %).
    """
    if baseline_val == 0:
        return float("nan")
    if lower_is_better:
        return (baseline_val - model_val) / abs(baseline_val) * 100.0
    else:
        return (model_val - baseline_val) / abs(baseline_val) * 100.0


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------


def per_query_ndcg(
    scores: np.ndarray,
    positive_indices: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Compute per-query NDCG@k for a binary-relevance single-positive setup.

    Parameters
    ----------
    scores:
        Shape (n_queries, n_candidates) — similarity score per candidate.
    positive_indices:
        Shape (n_queries,) — index of the positive candidate per query.
    k:
        Truncation depth.

    Returns
    -------
    ndcg_per_query : np.ndarray of shape (n_queries,)
    """
    scores = np.asarray(scores, dtype=np.float64)
    positive_indices = np.asarray(positive_indices, dtype=np.int64)
    n_queries = scores.shape[0]
    ndcg_vals = np.empty(n_queries, dtype=np.float64)
    idcg = 1.0 / np.log2(2.0)  # optimal DCG when positive is at rank 1

    order = np.argsort(-scores, axis=1)  # descending
    for q in range(n_queries):
        pos = int(positive_indices[q])
        rank = int(np.where(order[q] == pos)[0][0]) + 1  # 1-indexed
        dcg = 1.0 / np.log2(rank + 1) if rank <= k else 0.0
        ndcg_vals[q] = dcg / idcg
    return ndcg_vals


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test
# ---------------------------------------------------------------------------


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> float:
    """Wilcoxon signed-rank test (one-sided: a > b).

    Parameters
    ----------
    a, b : array-like of per-query scores.

    Returns
    -------
    p_value : float   (1.0 on degenerate input or missing scipy)
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    if not _SCIPY_AVAILABLE:
        return 1.0
    if np.all(diff == 0):
        return 1.0
    try:
        result = _scipy_wilcoxon(diff, alternative="greater")
        return float(result.pvalue)
    except Exception:  # noqa: BLE001
        return 1.0


# ---------------------------------------------------------------------------
# RT2 significance bundle
# ---------------------------------------------------------------------------


def compute_rt2_significance(
    cte_per_query: np.ndarray,
    baseline_per_query: np.ndarray,
) -> tuple[float, float, str]:
    """Compute Wilcoxon p-value and Cohen's d between CTE and a baseline.

    Parameters
    ----------
    cte_per_query :      Per-query NDCG@k for CTE.
    baseline_per_query : Per-query NDCG@k for the baseline.

    Returns
    -------
    (p_value, cohens_d, stars)
    """
    cte = np.asarray(cte_per_query, dtype=np.float64)
    base = np.asarray(baseline_per_query, dtype=np.float64)

    p_value = wilcoxon_test(cte, base)
    stars = significance_stars(p_value)

    mean_diff = float(np.mean(cte) - np.mean(base))
    pooled_std = float(
        np.sqrt((np.var(cte, ddof=1) + np.var(base, ddof=1)) / 2.0)
    )
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else float("nan")

    return p_value, cohens_d, stars


# ---------------------------------------------------------------------------
# Best-baseline helpers
# ---------------------------------------------------------------------------

_TRIVIAL_GROUPS = {"Trivial Baseline"}
_PROPOSED_NAMES = {"DGT", "CTE"}


def best_nontrivial_rt1(rt1_results: dict) -> tuple[str, float]:
    """Return (model_name, MAE) of the best non-trivial, non-proposed model.

    "Best" = lowest MAE.  Trivial baselines and DGT are excluded.
    """
    best_name = ""
    best_mae = float("inf")
    for name, metrics in rt1_results.items():
        group = _get_group(name)
        if group in _TRIVIAL_GROUPS or name in _PROPOSED_NAMES:
            continue
        mae = float(metrics.get("MAE", float("inf")))
        if mae < best_mae:
            best_mae = mae
            best_name = name
    return best_name, best_mae


def best_nontrivial_rt2(
    rt2_results: dict,
    metric: str = "HR@10",
) -> tuple[str, float]:
    """Return (model_name, value) of the best non-trivial, non-proposed model.

    "Best" = highest metric value.  Trivial baselines and CTE are excluded.
    """
    best_name = ""
    best_val = float("-inf")
    for name, metrics in rt2_results.items():
        group = _get_group(name)
        if group in _TRIVIAL_GROUPS or name in _PROPOSED_NAMES:
            continue
        val = float(metrics.get(metric, float("-inf")))
        if val > best_val:
            best_val = val
            best_name = name
    return best_name, best_val
