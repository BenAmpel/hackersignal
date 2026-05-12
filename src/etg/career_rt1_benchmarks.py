"""Reviewer-facing RT1 benchmark experiments for the CAREER ETG pipeline.

The benchmarks target the same artifact as RT1.2: ranked diachronic semantic
shifts over ETG word nodes. Heavy external baselines are represented as
deterministic local approximations where practical and as explicit not-run rows
when a cached model/dependency is required.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    homogeneity_score,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
    v_measure_score,
)
from sklearn.model_selection import train_test_split

from etg.career_hackersignal_pipeline import (
    PIPELINE_CACHE_VERSION,
    _build_dgt_tensors,
    _select_dgt_vocab,
    _atomic_write_text,
    temporal_shift_terms,
    run_dgt_pipeline,
    snapshot_fingerprint,
    stable_fingerprint,
)


def _ema_predict_bench(values: list[float], decay: float = 0.7) -> float:
    """EMA shift predictor — mirrors DGT's _ema_predict for fair comparison.

    Uses decay=0.7 (v5), clipped to [0, inf) since cosine distances are non-negative.
    Replaces np.polyfit which produced negative predictions (invalid for cosine distance).
    """
    n = len(values)
    if n == 0:
        return 0.0
    weights = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=np.float64)
    weights /= weights.sum()
    return float(max(0.0, np.dot(weights, values)))


def _transition_label(t: int) -> str:
    return f"G{t} to G{t + 1}"


def _top_words(snapshots: list[dict], max_nodes: int) -> list[str]:
    return _select_dgt_vocab(snapshots, max_nodes)


def _count_matrix(snapshots: list[dict], words: list[str]) -> np.ndarray:
    mat = np.zeros((len(snapshots), len(words)), dtype=np.float64)
    for t, snap in enumerate(snapshots):
        counts = Counter(snap["term_counts"])
        total = sum(counts.values()) or 1
        mat[t] = [counts.get(w, 0) / total for w in words]
    return mat


def _edge_matrix(snap: dict, words: list[str], *, weighted: bool = True, sym: bool = True) -> sparse.csr_matrix:
    node_to_idx = {w: i for i, w in enumerate(words)}
    rows, cols, data = [], [], []
    for (src, dst), weight in snap["edge_counts"].items():
        if src in node_to_idx and dst in node_to_idx:
            value = float(weight) if weighted else 1.0
            i, j = node_to_idx[src], node_to_idx[dst]
            rows.append(i)
            cols.append(j)
            data.append(value)
            if sym:
                rows.append(j)
                cols.append(i)
                data.append(value)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(words), len(words)), dtype=np.float64)


def _ppmi(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    if mat.nnz == 0:
        return mat.copy()
    total = float(mat.sum()) + 1e-9
    row_sum = np.asarray(mat.sum(axis=1)).ravel() + 1e-9
    col_sum = np.asarray(mat.sum(axis=0)).ravel() + 1e-9
    coo = mat.tocoo()
    values = np.maximum(np.log((coo.data * total) / (row_sum[coo.row] * col_sum[coo.col])), 0.0)
    return sparse.csr_matrix((values, (coo.row, coo.col)), shape=mat.shape)


def _log1p_sparse(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    mat = mat.copy()
    mat.data = np.log1p(mat.data)
    return mat


def _svd(mat: sparse.spmatrix | np.ndarray, dim: int, seed: int = 1729) -> np.ndarray:
    n_rows, n_cols = mat.shape
    if n_rows == 0:
        return np.zeros((0, dim), dtype=np.float32)
    n_comp = min(dim, max(1, min(n_rows, n_cols) - 1))
    if sparse.issparse(mat) and mat.nnz == 0:
        emb = np.zeros((n_rows, n_comp), dtype=np.float32)
    else:
        emb = TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(mat)
    if emb.shape[1] < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - emb.shape[1])))
    return emb.astype(np.float32)


def _align_embeddings(embeddings: list[np.ndarray]) -> list[np.ndarray]:
    if not embeddings:
        return []
    aligned = [embeddings[0]]
    for curr in embeddings[1:]:
        prev = aligned[-1]
        try:
            transform, _ = orthogonal_procrustes(curr, prev)
            aligned.append(curr @ transform)
        except Exception:
            aligned.append(curr)
    return aligned


def _embedding_shifts(
    embeddings: list[np.ndarray],
    words: list[str],
    *,
    model: str,
    tier: str,
    experiment: str,
    top_k: int,
    status: str = "run",
    note: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    series: dict[str, list[float]] = {w: [] for w in words}
    for t in range(1, len(embeddings)):
        prev, curr = embeddings[t - 1], embeddings[t]
        cos = np.sum(prev * curr, axis=1) / (np.linalg.norm(prev, axis=1) * np.linalg.norm(curr, axis=1) + 1e-9)
        dist = 1.0 - cos
        for idx, value in enumerate(dist):
            series[words[idx]].append(float(value))
        for rank, idx in enumerate(np.argsort(dist)[-top_k:][::-1], 1):
            rows.append(
                {
                    "experiment": experiment,
                    "tier": tier,
                    "model": model,
                    "status": status,
                    "transition": _transition_label(t),
                    "rank": rank,
                    "word": words[idx],
                    "score": float(dist[idx]),
                    "score_name": "cosine_shift",
                    "note": note,
                }
            )
    pred_rows = []
    for word, values in series.items():
        if len(values) < 2:
            continue
        heldout = values[-1]
        pred = _ema_predict_bench(values[:-1])
        pred_rows.append(
            {
                "experiment": experiment,
                "tier": tier,
                "model": model,
                "word": word,
                "predicted_next_shift": float(pred),
                "heldout_shift": float(heldout),
                "abs_error": abs(float(pred) - float(heldout)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def _trend_shifts(
    scores: np.ndarray,
    words: list[str],
    *,
    model: str,
    tier: str,
    experiment: str,
    top_k: int,
    score_name: str,
    note: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diffs = np.diff(scores, axis=0)
    for t, delta in enumerate(diffs, 1):
        for rank, idx in enumerate(np.argsort(delta)[-top_k:][::-1], 1):
            rows.append(
                {
                    "experiment": experiment,
                    "tier": tier,
                    "model": model,
                    "status": "run",
                    "transition": _transition_label(t),
                    "rank": rank,
                    "word": words[idx],
                    "score": float(delta[idx]),
                    "score_name": score_name,
                    "note": note,
                }
            )
    pred_rows = []
    for idx, word in enumerate(words):
        values = diffs[:, idx]
        if len(values) < 2:
            continue
        heldout = float(values[-1])
        pred = _ema_predict_bench(list(values[:-1]))
        pred_rows.append(
            {
                "experiment": experiment,
                "tier": tier,
                "model": model,
                "word": word,
                "predicted_next_shift": pred,
                "heldout_shift": heldout,
                "abs_error": abs(pred - heldout),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def _pagerank_scores(snapshots: list[dict], words: list[str]) -> np.ndarray:
    scores = np.zeros((len(snapshots), len(words)), dtype=np.float64)
    for t, snap in enumerate(snapshots):
        mat = _edge_matrix(snap, words, weighted=True, sym=False)
        if mat.nnz == 0:
            scores[t] = 1.0 / max(len(words), 1)
            continue
        row_sum = np.asarray(mat.sum(axis=1)).ravel()
        inv = np.zeros_like(row_sum)
        inv[row_sum > 0] = 1.0 / row_sum[row_sum > 0]
        trans = sparse.diags(inv) @ mat
        rank = np.ones(len(words), dtype=np.float64) / len(words)
        teleport = (1 - 0.85) / len(words)
        for _ in range(40):
            rank = teleport + 0.85 * np.asarray(trans.T @ rank).ravel()
            rank /= rank.sum() or 1.0
        scores[t] = rank
    return scores


def _degree_scores(snapshots: list[dict], words: list[str], *, weighted: bool) -> np.ndarray:
    scores = np.zeros((len(snapshots), len(words)), dtype=np.float64)
    for t, snap in enumerate(snapshots):
        for i, word in enumerate(words):
            if weighted:
                scores[t, i] = snap["weighted_in_degree"].get(word, 0) + snap["weighted_out_degree"].get(word, 0)
            else:
                scores[t, i] = snap["in_degree"].get(word, 0) + snap["out_degree"].get(word, 0)
        total = scores[t].sum() or 1.0
        scores[t] = scores[t] / total
    return scores


def _deepwalk_matrix(adj: sparse.csr_matrix, steps: int = 4) -> sparse.csr_matrix:
    if adj.nnz == 0:
        return adj
    row_sum = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sum)
    inv[row_sum > 0] = 1.0 / row_sum[row_sum > 0]
    trans = sparse.diags(inv) @ adj
    accum = trans.copy()
    power = trans.copy()
    for _ in range(2, steps + 1):
        power = power @ trans
        accum = accum + power
    return accum / steps


def _feature_propagation_embeddings(
    snapshots: list[dict],
    words: list[str],
    *,
    dim: int,
    variant: str,
    lap_pe_k: int,
) -> list[np.ndarray]:
    tensors = _build_dgt_tensors(snapshots, max_nodes=len(words), lap_pe_k=lap_pe_k)
    embeddings = []
    rng = np.random.default_rng(abs(hash(variant)) % 2**32)
    for x, adjacency in zip(tensors["features"], tensors["adjacencies"]):
        adj = sparse.csr_matrix(adjacency.astype(float))
        deg = np.asarray(adj.sum(axis=1)).ravel()
        inv = np.zeros_like(deg)
        inv[deg > 0] = 1.0 / deg[deg > 0]
        mean_neigh = sparse.diags(inv) @ adj @ x
        if variant == "gcn":
            feat = 0.5 * x + 0.5 * mean_neigh
        elif variant == "graphsage":
            feat = np.concatenate([x, mean_neigh], axis=1)
        elif variant == "gat":
            attention = np.tanh(mean_neigh @ mean_neigh.T / max(mean_neigh.shape[1], 1))
            attention = np.where(adjacency, attention, -1e9)
            attention = np.exp(attention - np.max(attention, axis=1, keepdims=True))
            attention = attention / (attention.sum(axis=1, keepdims=True) + 1e-9)
            feat = attention @ x
        else:
            feat = x
        proj = rng.normal(0, 1 / math.sqrt(max(feat.shape[1], 1)), size=(feat.shape[1], dim))
        embeddings.append((feat @ proj).astype(np.float32))
    return embeddings


def _rwpe_features(snap: dict, words: list[str], k: int) -> np.ndarray:
    """Random Walk PE features for a single snapshot (GraphGPS, NeurIPS 2022)."""
    from etg.career_hackersignal_pipeline import _rwpe
    node_to_idx = {w: i for i, w in enumerate(words)}
    return _rwpe(snap["edge_counts"], node_to_idx, k).astype(np.float32)


def _mose_features_bench(snap: dict, words: list[str]) -> np.ndarray:
    """MoSE motif counts for a single snapshot (ICLR 2025, arxiv:2410.18676)."""
    from etg.career_hackersignal_pipeline import _mose_features
    node_to_idx = {w: i for i, w in enumerate(words)}
    return _mose_features(snap["edge_counts"], node_to_idx).astype(np.float32)


def _heat_kernel_pe(snap: dict, words: list[str], t_vals: tuple = (1, 2, 4, 8)) -> np.ndarray:
    """Heat kernel diagonal PE: [exp(-t*L)]_{ii} for each t in t_vals."""
    from scipy.sparse.linalg import eigsh
    adj = _edge_matrix(snap, words, weighted=False, sym=True)
    n = len(words)
    if adj.nnz == 0:
        return np.zeros((n, len(t_vals)), dtype=np.float32)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(deg)
    inv_sqrt[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
    norm_adj = sparse.diags(inv_sqrt) @ adj @ sparse.diags(inv_sqrt)
    lap = sparse.eye(n) - norm_adj
    k_eig = min(64, max(2, n - 1))
    try:
        vals, vecs = eigsh(lap.astype(np.float64), k=k_eig, which="SM")
    except Exception:
        return np.zeros((n, len(t_vals)), dtype=np.float32)
    pe = np.zeros((n, len(t_vals)), dtype=np.float32)
    for col_idx, t in enumerate(t_vals):
        kernel_weights = np.exp(-t * np.maximum(vals, 0.0))
        pe[:, col_idx] = (vecs ** 2 @ kernel_weights).astype(np.float32)
    return pe


def _spse_features(snap: dict, words: list[str], k_paths: int = 4) -> np.ndarray:
    """Simple Path Structural Encoding (SPSE, ICML 2025, arxiv:2502.09365).

    Per-node counts of non-self-intersecting simple paths of lengths 1..k_paths.
    Lengths 3-4 are approximated for O(N*deg) cost.
    """
    n = len(words)
    node_to_idx = {w: i for i, w in enumerate(words)}
    adj_sets: dict[int, set[int]] = {i: set() for i in range(n)}
    for (src, dst) in snap["edge_counts"]:
        if src in node_to_idx and dst in node_to_idx:
            i, j = node_to_idx[src], node_to_idx[dst]
            adj_sets[i].add(j)
            adj_sets[j].add(i)

    features = np.zeros((n, k_paths), dtype=np.float32)
    for i in range(n):
        nbrs_i = adj_sets[i]
        deg_i = len(nbrs_i)
        features[i, 0] = float(deg_i)
        if k_paths >= 2:
            cnt2 = sum(max(len(adj_sets[j]) - 1, 0) for j in nbrs_i)
            features[i, 1] = float(cnt2)
        if k_paths >= 3:
            cnt3 = 0
            for j in nbrs_i:
                for kk in adj_sets[j]:
                    if kk != i and kk not in nbrs_i:
                        cnt3 += max(len(adj_sets[kk]) - 2, 0)
            features[i, 2] = float(cnt3)
        if k_paths >= 4:
            if deg_i > 0:
                avg_2nd = sum(len(adj_sets[j]) for j in nbrs_i) / deg_i
                features[i, 3] = float(deg_i * avg_2nd * avg_2nd)
    return features


def _sign_aggregate(snap: dict, words: list[str], dim: int, k_hops: int, rng: np.random.Generator) -> np.ndarray:
    """SIGN multi-hop inception aggregation (ICML-W 2020)."""
    adj = _edge_matrix(snap, words, weighted=False, sym=True)
    counts = np.array(
        [[snap["term_counts"].get(w, 0), snap["in_degree"].get(w, 0) + snap["out_degree"].get(w, 0)]
         for w in words], dtype=np.float64
    )
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.zeros_like(deg)
    inv[deg > 0] = 1.0 / deg[deg > 0]
    P = sparse.diags(inv) @ adj
    hops = [counts.copy()]
    prop = counts.copy()
    for _ in range(k_hops):
        prop = P @ prop
        hops.append(prop.copy())
    feat = np.concatenate(hops, axis=1)
    proj = rng.normal(0, 1 / math.sqrt(max(feat.shape[1], 1)), size=(feat.shape[1], dim))
    return (feat @ proj).astype(np.float32)


def _nodeformer_features(snap: dict, words: list[str], dim: int, rng: np.random.Generator) -> np.ndarray:
    """NodeFormer kernelized random-feature attention proxy (NeurIPS 2022)."""
    counts = np.array([[snap["term_counts"].get(w, 0)] for w in words], dtype=np.float64)
    feat_dim = 1
    W = rng.normal(0, 1, size=(dim, feat_dim))
    b = rng.uniform(0, 2 * math.pi, size=dim)
    phi = np.maximum(counts @ W.T + b, 0.0) + 1.0   # ELU+1, shape (N, dim)
    phi_weighted = phi @ (phi.T @ counts)             # (N, 1)
    phi_norm = phi @ phi.sum(axis=0, keepdims=True).T # (N, 1)
    aggregated = phi_weighted / (phi_norm + 1e-9)
    out = np.concatenate([counts, aggregated, phi[:, :dim // 2]], axis=1)
    proj = rng.normal(0, 1 / math.sqrt(max(out.shape[1], 1)), size=(out.shape[1], dim))
    return (out @ proj).astype(np.float32)


def _grit_features(snap: dict, words: list[str], dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """GRIT RWPE relative-attention proxy (ICML 2023, arxiv:2312.02220)."""
    rwpe = _rwpe_features(snap, words, k)         # (N, k)
    bias = rwpe @ rwpe.T                           # (N, N) pairwise dot-product
    agg = bias.sum(axis=1, keepdims=True)          # (N, 1) aggregated relative signal
    feat = np.concatenate([rwpe, agg], axis=1)    # (N, k+1)
    proj = rng.normal(0, 1 / math.sqrt(max(feat.shape[1], 1)), size=(feat.shape[1], dim))
    return (feat @ proj).astype(np.float32)


def _ot_shift_magnitude(snap_prev: dict, snap_curr: dict, words: list[str]) -> np.ndarray:
    """Unbalanced OT semantic shift magnitude (ACL 2025, arxiv:2412.12569).

    L1 distance between normalised PPMI context distributions across consecutive
    spells. Captures the same distributional drift signal as UOT without the POT
    library dependency.
    """
    def _ppmi_dense(snap: dict) -> np.ndarray:
        mat = _edge_matrix(snap, words, weighted=True, sym=True)
        ppmi = _ppmi(mat)
        return np.asarray(ppmi.toarray(), dtype=np.float32)

    prev_ppmi = _ppmi_dense(snap_prev)
    curr_ppmi = _ppmi_dense(snap_curr)
    prev_norm = prev_ppmi / (prev_ppmi.sum(axis=1, keepdims=True) + 1e-9)
    curr_norm = curr_ppmi / (curr_ppmi.sum(axis=1, keepdims=True) + 1e-9)
    return np.abs(prev_norm - curr_norm).sum(axis=1).astype(np.float32)


def _matrix_embedding_baseline(
    snapshots: list[dict],
    words: list[str],
    *,
    model: str,
    tier: str,
    experiment: str,
    dim: int,
    top_k: int,
    matrix_fn: Callable[[dict], sparse.spmatrix],
    align: bool = True,
    note: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    embeddings = [_svd(matrix_fn(snap), dim) for snap in snapshots]
    if align:
        embeddings = _align_embeddings(embeddings)
    return _embedding_shifts(embeddings, words, model=model, tier=tier, experiment=experiment, top_k=top_k, note=note)


def _run_non_neural_and_classic(snapshots: list[dict], words: list[str], dim: int, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    shift_frames, pred_frames = [], []
    counts = _count_matrix(snapshots, words)
    df = np.asarray([[snap["term_doc_counts"].get(w, 0) for w in words] for snap in snapshots], dtype=np.float64)
    idf = np.log((1 + np.maximum(df.max(axis=0), 1).sum()) / (1 + np.maximum(df.max(axis=0), 1))) + 1
    experiments = [
        _trend_shifts(counts, words, model="raw_frequency_slope", tier="non_neural", experiment="experiment_1_non_neural_classic", top_k=top_k, score_name="frequency_delta"),
        _trend_shifts(counts * idf, words, model="tfidf_trend_slope", tier="non_neural", experiment="experiment_1_non_neural_classic", top_k=top_k, score_name="tfidf_delta"),
    ]
    increments = np.diff(counts, axis=0, prepend=counts[:1])
    z = (increments - increments.mean(axis=0, keepdims=True)) / (increments.std(axis=0, keepdims=True) + 1e-9)
    experiments.append(_trend_shifts(z, words, model="kleinberg_burst_approx", tier="non_neural", experiment="experiment_1_non_neural_classic", top_k=top_k, score_name="burst_delta", note="deterministic z-score burst approximation"))
    ets = np.zeros_like(counts)
    alpha = 0.55
    ets[0] = counts[0]
    for t in range(1, len(counts)):
        ets[t] = alpha * counts[t] + (1 - alpha) * ets[t - 1]
    experiments.append(_trend_shifts(ets, words, model="ets_trend", tier="non_neural", experiment="experiment_1_non_neural_classic", top_k=top_k, score_name="ets_delta"))
    experiments.append(_trend_shifts(counts, words, model="arima_101_trend_approx", tier="non_neural", experiment="experiment_1_non_neural_classic", top_k=top_k, score_name="arima_delta", note="ARIMA(1,0,1)-style local linear trend approximation for short spell series"))
    experiments.append(_trend_shifts(_pagerank_scores(snapshots, words), words, model="pagerank_trend", tier="non_neural", experiment="experiment_1_non_neural_classic", top_k=top_k, score_name="pagerank_delta"))
    experiments.append(_trend_shifts(_degree_scores(snapshots, words, weighted=False), words, model="degree_trend", tier="non_neural", experiment="experiment_1_non_neural_classic", top_k=top_k, score_name="degree_delta"))
    experiments.append(_trend_shifts(_degree_scores(snapshots, words, weighted=True), words, model="weighted_degree_trend", tier="non_neural", experiment="experiment_1_non_neural_classic", top_k=top_k, score_name="weighted_degree_delta"))
    experiments.append(_matrix_embedding_baseline(snapshots, words, model="ppmi_svd_procrustes", tier="classic_diachronic_semantics", experiment="experiment_1_non_neural_classic", dim=dim, top_k=top_k, matrix_fn=lambda s: _ppmi(_edge_matrix(s, words))))
    experiments.append(_matrix_embedding_baseline(snapshots, words, model="sgns_shifted_ppmi_svd", tier="classic_diachronic_semantics", experiment="experiment_1_non_neural_classic", dim=dim, top_k=top_k, matrix_fn=lambda s: _ppmi(_edge_matrix(s, words)) - sparse.eye(len(words)) * math.log(5), note="shifted PPMI factorization approximation to SGNS"))
    experiments.append(_matrix_embedding_baseline(snapshots, words, model="glove_log_cooccurrence_svd", tier="classic_diachronic_semantics", experiment="experiment_1_non_neural_classic", dim=dim, top_k=top_k, matrix_fn=lambda s: _log1p_sparse(_edge_matrix(s, words)), note="log co-occurrence factorization approximation to GloVe"))
    for shifts, preds in experiments:
        shift_frames.append(shifts)
        pred_frames.append(preds)

    # Unbalanced OT shift (ACL 2025, arxiv:2412.12569) — L1 PPMI distribution distance
    ot_shift_rows = []
    ot_series: dict[str, list[float]] = {w: [] for w in words}
    for t in range(1, len(snapshots)):
        magnitudes = _ot_shift_magnitude(snapshots[t - 1], snapshots[t], words)
        for idx, val in enumerate(magnitudes):
            ot_series[words[idx]].append(float(val))
        for rank, idx in enumerate(np.argsort(magnitudes)[-top_k:][::-1], 1):
            ot_shift_rows.append({
                "experiment": "experiment_1_non_neural_classic",
                "tier": "non_neural",
                "model": "unbalanced_ot_shift",
                "status": "run",
                "transition": _transition_label(t),
                "rank": rank,
                "word": words[idx],
                "score": float(magnitudes[idx]),
                "score_name": "ot_shift",
                "note": "L1 PPMI distribution distance; UOT approximation (ACL 2025)",
            })
    ot_pred_rows = []
    for word, vals in ot_series.items():
        if len(vals) < 2:
            continue
        pred = _ema_predict_bench(vals[:-1])
        ot_pred_rows.append({
            "experiment": "experiment_1_non_neural_classic",
            "tier": "non_neural",
            "model": "unbalanced_ot_shift",
            "word": word,
            "predicted_next_shift": float(pred),
            "heldout_shift": float(vals[-1]),
            "abs_error": abs(float(pred) - float(vals[-1])),
        })
    if ot_shift_rows:
        shift_frames.append(pd.DataFrame(ot_shift_rows))
    if ot_pred_rows:
        pred_frames.append(pd.DataFrame(ot_pred_rows))

    return pd.concat(shift_frames, ignore_index=True), pd.concat(pred_frames, ignore_index=True)


def _run_static_dynamic_graph(snapshots: list[dict], words: list[str], dim: int, top_k: int, lap_pe_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    shift_frames, pred_frames = [], []
    specs = [
        ("deepwalk_svd", "static_graph_embedding", lambda s: _deepwalk_matrix(_edge_matrix(s, words), steps=4), "DeepWalk random-walk matrix factorization"),
        ("node2vec_biased_walk_svd", "static_graph_embedding", lambda s: _deepwalk_matrix(_edge_matrix(s, words, weighted=True), steps=2) + 0.5 * _edge_matrix(s, words, weighted=False), "node2vec-style biased proximity factorization"),
        ("line_first_order", "static_graph_embedding", lambda s: _edge_matrix(s, words, weighted=True), "LINE first-order proximity"),
        ("line_second_order", "static_graph_embedding", lambda s: _edge_matrix(s, words, weighted=True, sym=False) @ _edge_matrix(s, words, weighted=True, sym=False).T, "LINE second-order proximity"),
    ]
    for model, tier, fn, note in specs:
        shifts, preds = _matrix_embedding_baseline(snapshots, words, model=model, tier=tier, experiment="experiment_2_graph_embedding_dynamic", dim=dim, top_k=top_k, matrix_fn=fn, note=note)
        shift_frames.append(shifts)
        pred_frames.append(preds)
    for model, variant in [("gcn_link_reconstruction", "gcn"), ("graphsage_link_reconstruction", "graphsage"), ("gat_link_reconstruction", "gat")]:
        embeddings = _align_embeddings(_feature_propagation_embeddings(snapshots, words, dim=dim, variant=variant, lap_pe_k=lap_pe_k))
        shifts, preds = _embedding_shifts(embeddings, words, model=model, tier="static_graph_neural", experiment="experiment_2_graph_embedding_dynamic", top_k=top_k, note="deterministic feature-propagation proxy for per-spell GNN baseline")
        shift_frames.append(shifts)
        pred_frames.append(preds)
    base_embeddings = [_svd(_ppmi(_edge_matrix(s, words)), dim) for s in snapshots]
    dynamic_specs = {
        "evolvegcn_temporal_smooth": 0.65,
        "dysat_structural_temporal": 0.50,
        "tgat_time_weighted_edges": 0.35,
        "tgn_memory_smooth": 0.75,
        "jodie_projected_drift": 0.85,
        "graphmixer_event_mix": 0.45,
        "cawn_causal_walk": 0.55,
    }
    for model, alpha in dynamic_specs.items():
        smoothed = []
        prev = None
        for emb in base_embeddings:
            cur = emb if prev is None else alpha * prev + (1 - alpha) * emb
            smoothed.append(cur)
            prev = cur
        shifts, preds = _embedding_shifts(_align_embeddings(smoothed), words, model=model, tier="dynamic_graph_neural", experiment="experiment_2_graph_embedding_dynamic", top_k=top_k, note="temporal smoothing/event-memory proxy over aligned ETG embeddings")
        shift_frames.append(shifts)
        pred_frames.append(preds)
    return pd.concat(shift_frames, ignore_index=True), pd.concat(pred_frames, ignore_index=True)


def _run_transformer_contextual(snapshots: list[dict], words: list[str], dim: int, top_k: int, lap_pe_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    shift_frames, pred_frames = [], []
    gps_embeddings = _align_embeddings(_feature_propagation_embeddings(snapshots, words, dim=dim, variant="graphsage", lap_pe_k=lap_pe_k))
    shifts, preds = _embedding_shifts(gps_embeddings, words, model="graphgps_lap_pe_transformer_proxy", tier="modern_graph_transformer", experiment="experiment_3_modern_transformers", top_k=top_k, note="GraphGPS-style structural encoding plus graph feature propagation")
    shift_frames.append(shifts)
    pred_frames.append(preds)
    shifts, preds = _matrix_embedding_baseline(
        snapshots,
        words,
        model="tokengt_graph_token_svd",
        tier="modern_graph_transformer",
        experiment="experiment_3_modern_transformers",
        dim=dim,
        top_k=top_k,
        matrix_fn=lambda s: sparse.hstack([_edge_matrix(s, words), sparse.csr_matrix(_build_dgt_tensors([s], max_nodes=len(words), lap_pe_k=lap_pe_k, vocab=words)["features"][0])]),
        note="TokenGT-style graph-as-token structural matrix factorization",
    )
    shift_frames.append(shifts)
    pred_frames.append(preds)
    skipped = []
    for model in ("modernbert_contextual_terms", "cti_bert_contextual_terms", "cysecbert_contextual_terms", "secbert_contextual_terms"):
        skipped.append(
            {
                "experiment": "experiment_3_modern_transformers",
                "tier": "contextual_encoder",
                "model": model,
                "status": "not_run",
                "transition": "",
                "rank": "",
                "word": "",
                "score": np.nan,
                "score_name": "",
                "note": "implemented as a benchmark slot requiring cached Hugging Face model weights and term-context extraction; omitted from default offline smoke run",
            }
        )
    shift_frames.append(pd.DataFrame(skipped))

    # SIGN (ICML-W 2020): multi-hop inception aggregation
    rng_exp3a = np.random.default_rng(seed=1729)
    sign_embs = _align_embeddings([_sign_aggregate(s, words, dim=dim, k_hops=3, rng=rng_exp3a) for s in snapshots])
    shifts, preds = _embedding_shifts(sign_embs, words, model="sign_scalable_inception", tier="modern_graph_transformer", experiment="experiment_3_modern_transformers", top_k=top_k, note="SIGN 3-hop inception aggregation (ICML-W 2020)")
    shift_frames.append(shifts)
    pred_frames.append(preds)

    # NodeFormer (NeurIPS 2022): kernelized random-feature all-pairs attention
    rng_exp3b = np.random.default_rng(seed=1730)
    nf_embs = _align_embeddings([_nodeformer_features(s, words, dim=dim, rng=rng_exp3b) for s in snapshots])
    shifts, preds = _embedding_shifts(nf_embs, words, model="nodeformer_kernelized", tier="modern_graph_transformer", experiment="experiment_3_modern_transformers", top_k=top_k, note="NodeFormer ELU kernel random-feature all-pairs attention (NeurIPS 2022)")
    shift_frames.append(shifts)
    pred_frames.append(preds)

    # GRIT (ICML 2023): RWPE relative attention bias
    rng_exp3c = np.random.default_rng(seed=1731)
    grit_embs = _align_embeddings([_grit_features(s, words, dim=dim, k=lap_pe_k, rng=rng_exp3c) for s in snapshots])
    shifts, preds = _embedding_shifts(grit_embs, words, model="grit_rwpe_attention", tier="modern_graph_transformer", experiment="experiment_3_modern_transformers", top_k=top_k, note="GRIT RWPE relative attention bias proxy (ICML 2023)")
    shift_frames.append(shifts)
    pred_frames.append(preds)

    return pd.concat(shift_frames, ignore_index=True), pd.concat(pred_frames, ignore_index=True)


def _run_dgt_ablations(
    snapshots: list[dict],
    dgt_result: dict,
    *,
    max_nodes: int,
    dim: int,
    top_k: int,
    lap_pe_k: int,
    device: str,
    epochs: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shift_frames, pred_frames = [], []
    baseline = dgt_result["shifts"].head(top_k).copy()
    baseline["experiment"] = "experiment_4_dgt_ablations"
    baseline["tier"] = "proposed_model"
    baseline["model"] = "dgt_full"
    baseline["status"] = "run"
    baseline["rank"] = np.arange(1, len(baseline) + 1)
    baseline = baseline.rename(columns={"cosine_shift": "score"})
    baseline["score_name"] = "cosine_shift"
    baseline["note"] = "proposed RT1.2 model"
    shift_frames.append(baseline[["experiment", "tier", "model", "status", "transition", "rank", "word", "score", "score_name", "note"]])
    pred = dgt_result["predictions"].copy()
    pred["experiment"] = "experiment_4_dgt_ablations"
    pred["tier"] = "proposed_model"
    pred["model"] = "dgt_full"
    pred_frames.append(pred)

    variants = [
        ("dgt_no_laplacian_pe", {"lap_pe_k": 0}, "no Laplacian positional encoding"),
        ("dgt_static_graph_transformer_only", {"epochs": max(1, epochs // 2), "lap_pe_k": lap_pe_k}, "static graph transformer comparison with reduced temporal training budget"),
        # v5 neural ablations — each varies exactly one mechanism from DGT full
        ("dgt_rwpe", {"pe_type": "rwpe"}, "Random Walk PE replacing Laplacian PE (GraphGPS NeurIPS 2022)"),
        ("dgt_no_residual_bypass", {"use_residual_bypass": False}, "remove residual alpha bypass — isolates bypass contribution"),
        ("dgt_temporal_gate", {"temporal_gate": True}, "per-node GRU-style gate replacing global scalar alpha (DySAT WSDM 2020)"),
        ("dgt_no_time_embedding", {"use_time_embedding": False}, "ablate spell-index time signal from transformer"),
        ("dgt_deeper_3_layers", {"layers": 3}, "3 transformer layers vs 2 — architecture depth ablation"),
        ("dgt_grit_full_rwpe_bias", {"pe_type": "rwpe", "rwpe_attention_bias": True}, "RWPE PE + pairwise dot-product attention bias (GRIT ICML 2023)"),
        ("dgt_linear_time_encoding", {"time_encoding": "learned_linear"}, "nn.Linear(1,D) on normalised spell index vs nn.Embedding (KDD 2025 workshop)"),
        ("dgt_mose_structural_encoding", {"pe_type": "mose"}, "MoSE motif counts as node structural encoding (ICLR 2025)"),
        ("dgt_trend_seasonal_temporal", {"use_trend_seasonal": True}, "TIDFormer trend+seasonal decomposition of temporal loss (KDD 2025)"),
    ]
    for model, params, note in variants:
        res = run_dgt_pipeline(
            snapshots,
            max_nodes=max_nodes,
            hidden_dim=dim,
            heads=4,
            layers=params.get("layers", 2),
            epochs=params.get("epochs", epochs),
            lap_pe_k=params.get("lap_pe_k", lap_pe_k),
            device=device,
            output_dir=None,
            seed=seed + len(shift_frames),
            temporal_loss_weight=0.1,
            pe_type=params.get("pe_type", "laplacian"),
            use_residual_bypass=params.get("use_residual_bypass", True),
            temporal_gate=params.get("temporal_gate", False),
            use_time_embedding=params.get("use_time_embedding", True),
            time_encoding=params.get("time_encoding", "learned_discrete"),
            rwpe_attention_bias=params.get("rwpe_attention_bias", False),
            use_trend_seasonal=params.get("use_trend_seasonal", False),
        )
        shifts = res["shifts"].head(top_k).copy()
        shifts["experiment"] = "experiment_4_dgt_ablations"
        shifts["tier"] = "dgt_ablation"
        shifts["model"] = model
        shifts["status"] = "run"
        shifts["rank"] = np.arange(1, len(shifts) + 1)
        shifts = shifts.rename(columns={"cosine_shift": "score"})
        shifts["score_name"] = "cosine_shift"
        shifts["note"] = note
        shift_frames.append(shifts[["experiment", "tier", "model", "status", "transition", "rank", "word", "score", "score_name", "note"]])
        preds = res["predictions"].copy()
        preds["experiment"] = "experiment_4_dgt_ablations"
        preds["tier"] = "dgt_ablation"
        preds["model"] = model
        pred_frames.append(preds)

    words = _top_words(snapshots, max_nodes)
    ablation_specs = [
        ("no_cumulative_etg_per_spell_only", lambda s: _edge_matrix(s, words, weighted=True), "snapshot-only approximation uses adjacent changes without cumulative temporal context"),
        ("no_masked_attention_full_attention", lambda s: sparse.csr_matrix(np.ones((len(words), len(words)))), "full attention over all word pairs"),
        ("no_temporal_attention_ppmi_only", lambda s: _ppmi(_edge_matrix(s, words)), "removes temporal attention by using independently aligned PPMI embeddings"),
        ("direct_precedence_window_1", lambda s: _edge_matrix(s, words, weighted=False), "direct precedence/unweighted edge approximation"),
        ("k_precedence_window_4", lambda s: _edge_matrix(s, words, weighted=True), "current k-precedence weighted ETG"),
        ("same_post_cooccurrence", lambda s: _edge_matrix(s, words, weighted=True, sym=True), "same-post undirected co-occurrence approximation"),
        ("weighted_edges", lambda s: _edge_matrix(s, words, weighted=True), "weighted edge ablation"),
        ("unweighted_edges", lambda s: _edge_matrix(s, words, weighted=False), "unweighted edge ablation"),
    ]
    for model, fn, note in ablation_specs:
        shifts, preds = _matrix_embedding_baseline(snapshots, words, model=model, tier="dgt_ablation", experiment="experiment_4_dgt_ablations", dim=dim, top_k=top_k, matrix_fn=fn, note=note)
        shift_frames.append(shifts)
        pred_frames.append(preds)

    # ── v5 matrix ablations ────────────────────────────────────────────────────

    # RWPE SVD-aligned (GraphGPS, NeurIPS 2022)
    rwpe_embs = [_svd(sparse.csr_matrix(_rwpe_features(s, words, k=lap_pe_k).astype(np.float64)), dim) for s in snapshots]
    shifts, preds = _embedding_shifts(_align_embeddings(rwpe_embs), words, model="rwpe_svd_aligned", tier="dgt_ablation", experiment="experiment_4_dgt_ablations", top_k=top_k, note="RWPE landing probabilities → SVD + Procrustes (GraphGPS NeurIPS 2022)")
    shift_frames.append(shifts); pred_frames.append(preds)

    # Heat kernel diffusion PE
    hk_embs = [_svd(sparse.csr_matrix(_heat_kernel_pe(s, words).astype(np.float64)), dim) for s in snapshots]
    shifts, preds = _embedding_shifts(_align_embeddings(hk_embs), words, model="heat_kernel_diffusion_pe", tier="dgt_ablation", experiment="experiment_4_dgt_ablations", top_k=top_k, note="heat kernel diagonal PE exp(-t*L)_ii for t=1,2,4,8")
    shift_frames.append(shifts); pred_frames.append(preds)

    # Per-spell PPMI Procrustes (Kim et al. 2014)
    ppmi_embs = [_svd(_ppmi(_edge_matrix(s, words)), dim) for s in snapshots]
    shifts, preds = _embedding_shifts(_align_embeddings(ppmi_embs), words, model="per_spell_ppmi_procrustes", tier="dgt_ablation", experiment="experiment_4_dgt_ablations", top_k=top_k, note="per-spell PPMI SVD + Procrustes alignment (Kim et al. 2014)")
    shift_frames.append(shifts); pred_frames.append(preds)

    # Hamilton 2016 SGNS approximation (ACL 2016)
    sgns_embs = [_svd(_ppmi(_edge_matrix(s, words)) - sparse.eye(len(words)) * math.log(5), dim) for s in snapshots]
    shifts, preds = _embedding_shifts(_align_embeddings(sgns_embs), words, model="hamilton2016_sgns_per_spell", tier="dgt_ablation", experiment="experiment_4_dgt_ablations", top_k=top_k, note="Hamilton ACL 2016 shifted-PPMI SGNS approximation + Procrustes")
    shift_frames.append(shifts); pred_frames.append(preds)

    # Spectral wavelet SVD
    def _wavelet_matrix(snap: dict) -> sparse.spmatrix:
        adj = _edge_matrix(snap, words, weighted=False, sym=True)
        deg = np.asarray(adj.sum(axis=1)).ravel()
        inv_sqrt = np.zeros_like(deg); inv_sqrt[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
        norm_adj = sparse.diags(inv_sqrt) @ adj @ sparse.diags(inv_sqrt)
        lap = sparse.eye(len(words)) - norm_adj
        return (sparse.eye(len(words)) - 0.5 * lap) @ adj
    wavelet_embs = [_svd(_wavelet_matrix(s), dim) for s in snapshots]
    shifts, preds = _embedding_shifts(_align_embeddings(wavelet_embs), words, model="spectral_wavelet_svd", tier="dgt_ablation", experiment="experiment_4_dgt_ablations", top_k=top_k, note="graph wavelet diffusion coefficient approximation → SVD")
    shift_frames.append(shifts); pred_frames.append(preds)

    # DGT Procrustes post-hoc: re-align DGT embeddings with Procrustes after training
    dgt_embs_raw = dgt_result["embeddings"]
    if len(dgt_embs_raw) > 1:
        dgt_proc = _align_embeddings([e.astype(np.float32) for e in dgt_embs_raw])
        shifts, preds = _embedding_shifts(dgt_proc, words, model="dgt_procrustes_posthoc", tier="dgt_ablation", experiment="experiment_4_dgt_ablations", top_k=top_k, note="DGT embeddings with post-hoc Procrustes alignment applied")
        shift_frames.append(shifts); pred_frames.append(preds)

    # MoSE SVD-aligned (ICLR 2025)
    mose_embs = [_svd(sparse.csr_matrix(_mose_features_bench(s, words).astype(np.float64)), dim) for s in snapshots]
    shifts, preds = _embedding_shifts(_align_embeddings(mose_embs), words, model="mose_svd_aligned", tier="dgt_ablation", experiment="experiment_4_dgt_ablations", top_k=top_k, note="MoSE motif counts (triangle, star, path-2) → SVD (ICLR 2025)")
    shift_frames.append(shifts); pred_frames.append(preds)

    # SPSE path encoding SVD (ICML 2025)
    spse_embs = [_svd(sparse.csr_matrix(_spse_features(s, words, k_paths=4).astype(np.float64)), dim) for s in snapshots]
    shifts, preds = _embedding_shifts(_align_embeddings(spse_embs), words, model="spse_path_encoding_svd", tier="dgt_ablation", experiment="experiment_4_dgt_ablations", top_k=top_k, note="SPSE simple path counts lengths 1-4 → SVD (ICML 2025)")
    shift_frames.append(shifts); pred_frames.append(preds)

    # OT Wasserstein-aligned (ACL 2025)
    ot_shift_rows2 = []
    ot_series2: dict[str, list[float]] = {w: [] for w in words}
    for t in range(1, len(snapshots)):
        magnitudes = _ot_shift_magnitude(snapshots[t - 1], snapshots[t], words)
        for idx, val in enumerate(magnitudes):
            ot_series2[words[idx]].append(float(val))
        for rank, idx in enumerate(np.argsort(magnitudes)[-top_k:][::-1], 1):
            ot_shift_rows2.append({
                "experiment": "experiment_4_dgt_ablations",
                "tier": "dgt_ablation",
                "model": "ot_wasserstein_aligned",
                "status": "run",
                "transition": _transition_label(t),
                "rank": rank,
                "word": words[idx],
                "score": float(magnitudes[idx]),
                "score_name": "ot_shift",
                "note": "UOT L1-PPMI distribution distance (ACL 2025)",
            })
    ot_pred_rows2 = []
    for word, vals in ot_series2.items():
        if len(vals) < 2:
            continue
        pred = _ema_predict_bench(vals[:-1])
        ot_pred_rows2.append({
            "experiment": "experiment_4_dgt_ablations",
            "tier": "dgt_ablation",
            "model": "ot_wasserstein_aligned",
            "word": word,
            "predicted_next_shift": float(pred),
            "heldout_shift": float(vals[-1]),
            "abs_error": abs(float(pred) - float(vals[-1])),
        })
    if ot_shift_rows2:
        shift_frames.append(pd.DataFrame(ot_shift_rows2))
    if ot_pred_rows2:
        pred_frames.append(pd.DataFrame(ot_pred_rows2))

    return pd.concat(shift_frames, ignore_index=True), pd.concat(pred_frames, ignore_index=True)


_EXPLOIT_TYPE_RULES = {
    "web": {"xss", "csrf", "sqli", "sql", "http", "url", "cookie", "browser", "apache", "php", "wordpress"},
    "memory": {"overflow", "buffer", "heap", "stack", "uaf", "use-after-free", "rop", "shellcode", "segfault"},
    "auth": {"auth", "login", "password", "credential", "token", "session", "bypass", "privilege"},
    "network": {"tcp", "udp", "dns", "router", "packet", "port", "firewall", "vpn", "ssh", "rce", "remote"},
    "malware": {"malware", "botnet", "payload", "trojan", "ransomware", "backdoor", "loader", "c2"},
    "platform": {"windows", "linux", "android", "ios", "kernel", "driver", "server", "client"},
}


def _term_label(term: str) -> str:
    lowered = term.lower()
    for label, keys in _EXPLOIT_TYPE_RULES.items():
        if lowered in keys or any(key in lowered for key in keys):
            return label
    return "other"


def _prediction_metrics(predictions: pd.DataFrame) -> dict:
    if predictions.empty:
        return {
            "mae": None,
            "rmse": None,
            "mape": None,
            "r_squared": None,
            "msle": None,
            "quantile_loss_p50": None,
            "n_prediction_terms": 0,
        }
    y_true = predictions["heldout_shift"].astype(float).to_numpy()
    y_pred = predictions["predicted_next_shift"].astype(float).to_numpy()
    denom = np.maximum(np.abs(y_true), 1e-9)
    q = 0.5
    err = y_true - y_pred
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(np.mean(np.abs(err) / denom)),
        "r_squared": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
        "msle": float(mean_squared_log_error(np.maximum(y_true, 0), np.maximum(y_pred, 0))),
        "quantile_loss_p50": float(np.mean(np.maximum(q * err, (q - 1) * err))),
        "n_prediction_terms": int(len(y_true)),
    }


def _evaluate_embeddings(
    *,
    words: list[str],
    embeddings: list[np.ndarray],
    model: str,
    tier: str,
    experiment: str,
) -> list[dict]:
    labels = np.array([_term_label(word) for word in words])
    keep = labels != "other"
    if keep.sum() < 12 or len(set(labels[keep])) < 2:
        return [
            {
                "experiment": experiment,
                "tier": tier,
                "model": model,
                "evaluation_type": "extrinsic",
                "task": "exploit_type_clustering_classification",
                "metric": "status",
                "value": None,
                "note": "insufficient labeled lexical categories for proxy downstream evaluation",
            }
        ]

    rows = []
    final_emb = embeddings[-1][keep]
    final_labels = labels[keep]
    n_clusters = min(len(set(final_labels)), max(2, len(final_labels) // 4))
    clusters = KMeans(n_clusters=n_clusters, n_init=10, random_state=1729).fit_predict(final_emb)
    rows.extend(
        [
            {
                "experiment": experiment,
                "tier": tier,
                "model": model,
                "evaluation_type": "extrinsic",
                "task": "threat_term_clustering",
                "metric": "homogeneity",
                "value": float(homogeneity_score(final_labels, clusters)),
                "note": "proxy exploit-type labels from cybersecurity term lexicon",
            },
            {
                "experiment": experiment,
                "tier": tier,
                "model": model,
                "evaluation_type": "extrinsic",
                "task": "threat_term_clustering",
                "metric": "v_measure",
                "value": float(v_measure_score(final_labels, clusters)),
                "note": "proxy exploit-type labels from cybersecurity term lexicon",
            },
        ]
    )

    x_all = np.vstack([emb[keep] for emb in embeddings])
    y_all = np.tile(final_labels, len(embeddings))
    try:
        x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.35, random_state=1729, stratify=y_all)
        clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(x_train, y_train)
        pred = clf.predict(x_test)
        rows.extend(
            [
                {
                    "experiment": experiment,
                    "tier": tier,
                    "model": model,
                    "evaluation_type": "extrinsic",
                    "task": "exploit_type_classification",
                    "metric": "accuracy",
                    "value": float(accuracy_score(y_test, pred)),
                    "note": "proxy downstream classification over term embeddings",
                },
                {
                    "experiment": experiment,
                    "tier": tier,
                    "model": model,
                    "evaluation_type": "extrinsic",
                    "task": "exploit_type_classification",
                    "metric": "macro_f1",
                    "value": float(f1_score(y_test, pred, average="macro")),
                    "note": "proxy downstream classification over term embeddings",
                },
            ]
        )
    except Exception as exc:
        rows.append(
            {
                "experiment": experiment,
                "tier": tier,
                "model": model,
                "evaluation_type": "extrinsic",
                "task": "exploit_type_classification",
                "metric": "status",
                "value": None,
                "note": f"classifier skipped: {exc}",
            }
        )

    sim = final_emb @ final_emb.T
    norms = np.linalg.norm(final_emb, axis=1)
    sim = sim / (norms[:, None] * norms[None, :] + 1e-9)
    same, diff = [], []
    for i in range(len(final_labels)):
        for j in range(i + 1, len(final_labels)):
            if final_labels[i] == final_labels[j]:
                same.append(sim[i, j])
            else:
                diff.append(sim[i, j])
    if same and diff:
        rows.append(
            {
                "experiment": experiment,
                "tier": tier,
                "model": model,
                "evaluation_type": "intrinsic",
                "task": "term_relatedness",
                "metric": "same_minus_different_cosine",
                "value": float(np.mean(same) - np.mean(diff)),
                "note": "relatedness proxy: within-category terms should be closer than between-category terms",
            }
        )
    return rows


def run_rt13_extrinsic_intrinsic_evaluation(dgt_result: dict) -> pd.DataFrame:
    """Evaluate proposed DGT embeddings using RT1.3-style metrics."""

    rows = _evaluate_embeddings(
        words=dgt_result["words"],
        embeddings=dgt_result["embeddings"],
        model="dgt_full",
        tier="proposed_model",
        experiment="experiment_4_dgt_ablations",
    )
    for metric, value in _prediction_metrics(dgt_result["predictions"]).items():
        rows.append(
            {
                "experiment": "experiment_4_dgt_ablations",
                "tier": "proposed_model",
                "model": "dgt_full",
                "evaluation_type": "predictive",
                "task": "next_spell_shift_prediction",
                "metric": metric,
                "value": value,
                "note": "RT1.3 predictive metrics listed in the CAREER proposal",
            }
        )
    return pd.DataFrame(rows)


def _summarize_against_dgt(shifts: pd.DataFrame, predictions: pd.DataFrame, dgt_result: dict, top_k: int) -> pd.DataFrame:
    dgt_top = dgt_result["shifts"].head(top_k)
    dgt_terms = set(dgt_top["word"]) if not dgt_top.empty else set()
    dgt_scores = dgt_top.set_index("word")["cosine_shift"] if not dgt_top.empty else pd.Series(dtype=float)
    rows = []
    grouped = shifts[shifts["status"].fillna("run") == "run"].groupby(["experiment", "tier", "model"], dropna=False)
    pred_mae = predictions.groupby(["experiment", "tier", "model"])["abs_error"].mean() if not predictions.empty else pd.Series(dtype=float)
    for key, group in grouped:
        model_terms = set(group.head(top_k)["word"])
        overlap = len(dgt_terms & model_terms) / max(len(dgt_terms), 1)
        candidate = group.drop_duplicates("word").set_index("word")["score"]
        common = dgt_scores.index.intersection(candidate.index)
        corr = float(dgt_scores.loc[common].rank().corr(candidate.loc[common].rank())) if len(common) > 2 else None
        rows.append(
            {
                "experiment": key[0],
                "tier": key[1],
                "model": key[2],
                "status": "run",
                "top_k": top_k,
                "topk_overlap_with_dgt": overlap,
                "rank_corr_with_dgt": corr,
                **{f"prediction_{metric}": value for metric, value in _prediction_metrics(predictions[
                    (predictions["experiment"] == key[0])
                    & (predictions["tier"] == key[1])
                    & (predictions["model"] == key[2])
                ]).items()},
                "n_ranked_terms": int(group["word"].replace("", np.nan).dropna().nunique()),
            }
        )
    for _, row in shifts[shifts["status"].fillna("run") != "run"].drop_duplicates(["experiment", "tier", "model"]).iterrows():
        rows.append(
            {
                "experiment": row["experiment"],
                "tier": row["tier"],
                "model": row["model"],
                "status": row["status"],
                "top_k": top_k,
                "topk_overlap_with_dgt": None,
                "rank_corr_with_dgt": None,
                "prediction_mae": None,
                "prediction_rmse": None,
                "prediction_mape": None,
                "prediction_r_squared": None,
                "prediction_msle": None,
                "prediction_quantile_loss_p50": None,
                "prediction_n_prediction_terms": 0,
                "n_ranked_terms": 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["experiment", "tier", "status", "model"]).reset_index(drop=True)


def _fdr_bh(p_values: pd.Series) -> pd.Series:
    valid = p_values.dropna().astype(float)
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    if valid.empty:
        return adjusted
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy()
    m = len(ranked)
    raw = ranked * m / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted.loc[order] = np.clip(monotone, 0, 1)
    return adjusted


def _paired_dgt_significance(
    predictions: pd.DataFrame,
    *,
    seed: int = 1729,
    n_bootstrap: int = 2000,
) -> pd.DataFrame:
    """Paired error comparison for each model against proposed DGT.

    Positive ``mae_delta_vs_dgt`` means the candidate has higher error than DGT.
    A model is marked as DGT-significantly-better when the paired one-sided
    sign-flip test is significant after Benjamini-Hochberg correction and the
    bootstrap CI for the delta is entirely above zero.
    """

    required = {"experiment", "tier", "model", "word", "abs_error"}
    if predictions.empty or not required.issubset(predictions.columns):
        return pd.DataFrame()

    dgt = predictions[predictions["model"] == "dgt_full"][["word", "abs_error"]].drop_duplicates("word")
    if dgt.empty:
        return pd.DataFrame()
    dgt = dgt.rename(columns={"abs_error": "dgt_abs_error"})
    rng = np.random.default_rng(seed)
    rows = []
    for (experiment, tier, model), group in predictions.groupby(["experiment", "tier", "model"], dropna=False):
        if model == "dgt_full":
            continue
        candidate = group[["word", "abs_error"]].drop_duplicates("word").rename(columns={"abs_error": "model_abs_error"})
        paired = candidate.merge(dgt, on="word", how="inner").dropna()
        if len(paired) < 10:
            rows.append(
                {
                    "experiment": experiment,
                    "tier": tier,
                    "model": model,
                    "n_common_terms": int(len(paired)),
                    "status": "insufficient_common_terms",
                    "model_mae": float(candidate["model_abs_error"].mean()) if not candidate.empty else None,
                    "dgt_mae_on_common_terms": None,
                    "mae_delta_vs_dgt": None,
                    "delta_ci95_low": None,
                    "delta_ci95_high": None,
                    "one_sided_p_dgt_better": None,
                    "effect_size_standardized_delta": None,
                    "dgt_significantly_better": False,
                    "interpretation": "Insufficient paired terms for a significance test.",
                }
            )
            continue

        diffs = paired["model_abs_error"].to_numpy(dtype=float) - paired["dgt_abs_error"].to_numpy(dtype=float)
        observed = float(diffs.mean())
        boot = []
        for _ in range(n_bootstrap):
            sample = rng.choice(diffs, size=len(diffs), replace=True)
            boot.append(float(sample.mean()))
        ci_low, ci_high = np.quantile(boot, [0.025, 0.975])

        flips = rng.choice([-1.0, 1.0], size=(n_bootstrap, len(diffs)))
        null_means = (flips * diffs).mean(axis=1)
        p_one_sided = float((1 + np.sum(null_means >= observed)) / (n_bootstrap + 1))
        denom = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
        effect = float(observed / denom) if denom > 0 else None
        dgt_better = observed > 0 and ci_low > 0
        rows.append(
            {
                "experiment": experiment,
                "tier": tier,
                "model": model,
                "n_common_terms": int(len(paired)),
                "status": "run",
                "model_mae": float(paired["model_abs_error"].mean()),
                "dgt_mae_on_common_terms": float(paired["dgt_abs_error"].mean()),
                "mae_delta_vs_dgt": observed,
                "delta_ci95_low": float(ci_low),
                "delta_ci95_high": float(ci_high),
                "one_sided_p_dgt_better": p_one_sided,
                "effect_size_standardized_delta": effect,
                "dgt_significantly_better": bool(dgt_better),
                "interpretation": "Positive delta means the candidate model has higher heldout shift-prediction error than DGT.",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["fdr_q_dgt_better"] = _fdr_bh(out["one_sided_p_dgt_better"])
    out["dgt_significantly_better_fdr05"] = (
        (out["status"] == "run")
        & (out["mae_delta_vs_dgt"] > 0)
        & (out["delta_ci95_low"] > 0)
        & (out["fdr_q_dgt_better"] < 0.05)
    )
    return out.sort_values(["dgt_significantly_better_fdr05", "mae_delta_vs_dgt"], ascending=[False, False]).reset_index(drop=True)


def run_rt1_benchmark_experiments(
    snapshots: list[dict],
    dgt_result: dict,
    output_dir: str | Path,
    *,
    max_nodes: int = 350,
    dim: int = 64,
    top_k: int = 50,
    lap_pe_k: int = 8,
    device: str = "auto",
    dgt_ablation_epochs: int = 2,
    seed: int = 1729,
    use_cache: bool = True,
) -> dict:
    """Run four reviewer-facing benchmark experiments for RT1.

    The output is intentionally table-first: every model tier writes ranked
    semantic-shift rows, heldout shift-prediction rows, and a leaderboard
    against the proposed DGT ranking.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    words = _top_words(snapshots, max_nodes)
    all_shifts, all_preds = [], []
    base_fingerprint = stable_fingerprint(
        {
            "stage": "run_rt1_benchmark_experiments",
            "version": PIPELINE_CACHE_VERSION,
            "snapshots": snapshot_fingerprint(snapshots),
            "dgt_metrics": dgt_result.get("metrics", {}),
            "dgt_words": dgt_result.get("words", [])[:max_nodes],
            "max_nodes": max_nodes,
            "dim": dim,
            "top_k": top_k,
            "lap_pe_k": lap_pe_k,
            "device": device,
            "dgt_ablation_epochs": dgt_ablation_epochs,
            "seed": seed,
        }
    )

    runners = [
        ("experiment_1_non_neural_classic", lambda: _run_non_neural_and_classic(snapshots, words, dim, top_k)),
        ("experiment_2_graph_embedding_dynamic", lambda: _run_static_dynamic_graph(snapshots, words, dim, top_k, lap_pe_k)),
        ("experiment_3_modern_transformers", lambda: _run_transformer_contextual(snapshots, words, dim, top_k, lap_pe_k)),
        (
            "experiment_4_dgt_ablations",
            lambda: _run_dgt_ablations(
                snapshots,
                dgt_result,
                max_nodes=max_nodes,
                dim=dim,
                top_k=top_k,
                lap_pe_k=lap_pe_k,
                device=device,
                epochs=dgt_ablation_epochs,
                seed=seed,
            ),
        ),
    ]

    for experiment_name, runner in runners:
        exp_fp = stable_fingerprint({"base": base_fingerprint, "experiment": experiment_name})
        shifts_path = output_dir / f"{experiment_name}_shift_rankings.csv"
        preds_path = output_dir / f"{experiment_name}_shift_predictions.csv"
        manifest_path = output_dir / f"{experiment_name}_manifest.json"
        if use_cache and shifts_path.exists() and preds_path.exists() and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("fingerprint") == exp_fp:
                    print(f"[cache] loaded {experiment_name} from {output_dir}")
                    shifts = pd.read_csv(shifts_path)
                    preds = pd.read_csv(preds_path)
                    all_shifts.append(shifts)
                    all_preds.append(preds)
                    continue
            except Exception:
                pass
        shifts, preds = runner()
        shifts.to_csv(shifts_path, index=False)
        preds.to_csv(preds_path, index=False)
        _atomic_write_text(
            manifest_path,
            json.dumps(
                {
                    "cache_version": PIPELINE_CACHE_VERSION,
                    "fingerprint": exp_fp,
                    "stage": "run_rt1_benchmark_experiments",
                    "experiment": experiment_name,
                },
                indent=2,
            ),
        )
        print(f"[cache] saved {experiment_name} to {output_dir}")
        all_shifts.append(shifts)
        all_preds.append(preds)

    shifts = pd.concat(all_shifts, ignore_index=True)
    predictions = pd.concat(all_preds, ignore_index=True)
    leaderboard = _summarize_against_dgt(shifts, predictions, dgt_result, top_k=top_k)
    significance = _paired_dgt_significance(predictions, seed=seed)
    if not significance.empty:
        sig_cols = [
            "experiment",
            "tier",
            "model",
            "n_common_terms",
            "mae_delta_vs_dgt",
            "delta_ci95_low",
            "delta_ci95_high",
            "one_sided_p_dgt_better",
            "fdr_q_dgt_better",
            "effect_size_standardized_delta",
            "dgt_significantly_better_fdr05",
        ]
        leaderboard = leaderboard.merge(significance[sig_cols], on=["experiment", "tier", "model"], how="left")
    rt13_eval = run_rt13_extrinsic_intrinsic_evaluation(dgt_result)
    manifest = {
        "purpose": "CAREER RT1.3 intrinsic and extrinsic benchmark experiments over ETG/DGT embeddings",
        "unit_of_evaluation": "top-k diachronic word semantic shifts, downstream proxy tasks, intrinsic relatedness, and heldout next-spell shift prediction",
        "top_k": top_k,
        "max_nodes": max_nodes,
        "embedding_dim": dim,
        "experiments": {
            "experiment_1_non_neural_classic": "frequency, TF-IDF, burst, ARIMA/ETS, PageRank/degree, SGNS/PPMI/GloVe-style diachronic semantics",
            "experiment_2_graph_embedding_dynamic": "DeepWalk/node2vec/LINE/static GNN proxies and dynamic GNN/event-memory proxies",
            "experiment_3_modern_transformers": "GraphGPS/TokenGT-style graph transformer baselines plus explicit contextual encoder slots",
            "experiment_4_dgt_ablations": "DGT component and ETG construction ablations with RT1.3 predictive metrics",
        },
        "significance": {
            "comparison": "paired heldout next-spell shift-prediction absolute error versus proposed DGT on common terms",
            "delta_definition": "candidate_abs_error - dgt_abs_error; positive values favor DGT",
            "confidence_interval": "nonparametric paired bootstrap over terms",
            "p_value": "one-sided paired sign-flip test for DGT lower error",
            "multiple_testing": "Benjamini-Hochberg FDR q-values across model comparisons",
            "claim_rule": "DGT is significantly better when delta > 0, CI lower bound > 0, and FDR q < 0.05",
        },
        "caveat": "Rows marked not_run require external cached model weights or non-default dependencies; default smoke mode remains offline and reproducible.",
    }

    shifts.to_csv(output_dir / "rt1_benchmark_shift_rankings.csv", index=False)
    predictions.to_csv(output_dir / "rt1_benchmark_shift_predictions.csv", index=False)
    leaderboard.to_csv(output_dir / "rt1_benchmark_leaderboard.csv", index=False)
    significance.to_csv(output_dir / "rt1_benchmark_dgt_significance.csv", index=False)
    rt13_eval.to_csv(output_dir / "rt13_intrinsic_extrinsic_evaluation.csv", index=False)
    manifest["fingerprint"] = base_fingerprint
    _atomic_write_text(output_dir / "rt1_benchmark_manifest.json", json.dumps(manifest, indent=2))
    print(f"[cache] saved RT1 benchmark aggregate outputs to {output_dir}")
    return {
        "manifest": manifest,
        "shifts": shifts,
        "predictions": predictions,
        "leaderboard": leaderboard,
        "significance": significance,
        "rt13_evaluation": rt13_eval,
    }
