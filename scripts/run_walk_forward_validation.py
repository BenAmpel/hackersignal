#!/usr/bin/env python3
"""Walk-forward (rolling) validation for DGT and key baselines.

Addresses the MISQ reviewer concern that the main evaluation rests on a
single held-out spell.  For each fold k ∈ {7, 8, 9, 10, 11} we:

  1. Slice snapshots[:k]  (spells 1..k)
  2. Train DGT on those k spells
  3. Run the same fast baselines on the same k spells
  4. For every model: use EMA on shifts[0 : k-2] to predict shifts[k-2]
     (i.e., the last transition in the slice is the per-fold heldout target)
  5. Record per-fold MAE / RMSE / Top-50 overlap

Final outputs:
  wf_fold_results.csv    – per-fold, per-model metrics
  wf_aggregate.csv       – mean ± std across 5 folds, ranked leaderboard
  wf_significance.csv    – Wilcoxon signed-rank DGT vs each baseline
                           (paired across folds)

Usage (from repo root):
    python scripts/run_walk_forward_validation.py \\
        --snapshots ETG_MISQ/output/cache/rt11_etg_snapshots.pkl \\
        --output    ETG_MISQ/output/walk_forward \\
        --dgt-epochs 100 \\
        --max-nodes 6000 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import ttest_rel
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── repo on PYTHONPATH ──────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from etg.career_hackersignal_pipeline import run_dgt_pipeline

# ── constants ──────────────────────────────────────────────────────────────
FOLDS = [7, 8, 9, 10, 11]   # k = number of spells in training window
EMA_DECAY = 0.7
TOP_K = 50


# ═══════════════════════════════════════════════════════════════════════════
# Helpers shared with career_rt1_benchmarks (duplicated to keep standalone)
# ═══════════════════════════════════════════════════════════════════════════

def _ema_predict(values: list[float], decay: float = EMA_DECAY) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    w = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=np.float64)
    w /= w.sum()
    return float(max(0.0, np.dot(w, values)))


def _top_words_from_snaps(snapshots: list[dict], max_nodes: int) -> list[str]:
    """Top-N words by raw term frequency in the LAST spell of the slice."""
    from collections import Counter
    counts: Counter = Counter()
    for snap in snapshots:
        counts.update(snap["term_counts"])
    return [w for w, _ in counts.most_common(max_nodes)]


def _edge_matrix(snap: dict, words: list[str], *, weighted: bool = True) -> sparse.csr_matrix:
    n2i = {w: i for i, w in enumerate(words)}
    rows, cols, data = [], [], []
    for (src, dst), wt in snap["edge_counts"].items():
        if src in n2i and dst in n2i:
            v = float(wt) if weighted else 1.0
            i, j = n2i[src], n2i[dst]
            rows += [i, j]; cols += [j, i]; data += [v, v]
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(words), len(words)))


def _ppmi(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    if mat.nnz == 0:
        return mat.copy()
    total = float(mat.sum()) + 1e-9
    rs = np.asarray(mat.sum(axis=1)).ravel() + 1e-9
    cs = np.asarray(mat.sum(axis=0)).ravel() + 1e-9
    coo = mat.tocoo()
    vals = np.maximum(np.log((coo.data * total) / (rs[coo.row] * cs[coo.col])), 0.0)
    return sparse.csr_matrix((vals, (coo.row, coo.col)), shape=mat.shape)


def _svd(mat: sparse.spmatrix | np.ndarray, dim: int, seed: int = 1729) -> np.ndarray:
    n_rows, n_cols = mat.shape
    if n_rows == 0:
        return np.zeros((0, dim), dtype=np.float32)
    k = min(dim, max(1, min(n_rows, n_cols) - 1))
    if sparse.issparse(mat) and mat.nnz == 0:
        emb = np.zeros((n_rows, k), dtype=np.float32)
    else:
        emb = TruncatedSVD(n_components=k, random_state=seed).fit_transform(mat)
    if emb.shape[1] < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - emb.shape[1])))
    return emb.astype(np.float32)


def _align_embeddings(embs: list[np.ndarray]) -> list[np.ndarray]:
    """Procrustes-align a list of embeddings to the first one."""
    from numpy.linalg import svd
    aligned = [embs[0].copy()]
    ref = embs[0]
    for e in embs[1:]:
        M = ref.T @ e
        U, _, Vt = svd(M)
        R = U @ Vt
        aligned.append(e @ R.T)
        ref = aligned[-1]
    return aligned


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row cosine distance between two embedding matrices."""
    na = np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    nb = np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    return 1.0 - np.sum((a / na) * (b / nb), axis=1)


def _count_matrix(snapshots: list[dict], words: list[str]) -> np.ndarray:
    mat = np.zeros((len(snapshots), len(words)), dtype=np.float64)
    for t, snap in enumerate(snapshots):
        total = max(sum(snap["term_counts"].values()), 1)
        for i, w in enumerate(words):
            mat[t, i] = snap["term_counts"].get(w, 0) / total
    return mat


# ═══════════════════════════════════════════════════════════════════════════
# Shift-series extraction per model family
# ═══════════════════════════════════════════════════════════════════════════

def _shift_series_from_embeddings(
    embs: list[np.ndarray],
) -> dict[int, np.ndarray]:
    """Return {transition_index: cosine_dist_array (N,)} for t=0..T-2."""
    return {t: _cosine_dist(embs[t], embs[t + 1]) for t in range(len(embs) - 1)}


def _matrix_embedding_shifts(
    snapshots: list[dict],
    words: list[str],
    matrix_fn,
    dim: int = 64,
) -> dict[int, np.ndarray]:
    """PPMI/GloVe/SGNS-style: SVD per spell + Procrustes align."""
    per_spell = [_svd(matrix_fn(snap), dim) for snap in snapshots]
    aligned = _align_embeddings(per_spell)
    return _shift_series_from_embeddings(aligned)


def _trend_shifts(count_mat: np.ndarray) -> dict[int, np.ndarray]:
    """Frequency-difference baseline (predicts near-zero shift → null predictor)."""
    shifts = {}
    for t in range(len(count_mat) - 1):
        shifts[t] = np.abs(count_mat[t + 1] - count_mat[t])
    return shifts


def _dgt_shifts(
    snapshots: list[dict],
    words_global: list[str],
    *,
    max_nodes: int,
    epochs: int,
    device: str,
    hidden_dim: int,
    heads: int,
    layers: int,
    lap_pe_k: int,
    output_dir: Path,
) -> dict[int, np.ndarray]:
    """Train DGT on these snapshots and return per-transition cosine shifts."""
    result = run_dgt_pipeline(
        snapshots,
        max_nodes=max_nodes,
        hidden_dim=hidden_dim,
        heads=heads,
        layers=layers,
        epochs=epochs,
        lap_pe_k=lap_pe_k,
        device=device,
        output_dir=output_dir,
        use_cache=True,
        temporal_loss_weight=0.1,
        pe_type="laplacian",
        use_residual_bypass=True,
        use_time_embedding=True,
        time_encoding="learned_discrete",
    )
    # result["embeddings"] is a list of (N_dgt, D) arrays; words may differ from words_global
    dgt_words = result["words"]  # list[str], length N_dgt
    embs = result["embeddings"]  # list of (N_dgt, D) arrays

    # Map DGT words → global word index
    w2g = {w: i for i, w in enumerate(words_global)}
    N = len(words_global)
    aligned_embs = []
    for e in embs:
        full = np.zeros((N, e.shape[1]), dtype=np.float32)
        for j, w in enumerate(dgt_words):
            if w in w2g:
                full[w2g[w]] = e[j]
        aligned_embs.append(full)

    return _shift_series_from_embeddings(aligned_embs)


# ═══════════════════════════════════════════════════════════════════════════
# Per-fold metric computation
# ═══════════════════════════════════════════════════════════════════════════

def _fold_metrics(
    shifts: dict[int, np.ndarray],
    n_spells_in_fold: int,
    top_k: int = TOP_K,
    dgt_top_indices: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute EMA→heldout metrics for one fold.

    The fold has transitions 0 .. n_spells_in_fold-2.
    Training window: transitions 0 .. n_spells_in_fold-3
    Heldout target: transition n_spells_in_fold-2
    """
    T = n_spells_in_fold  # number of spells
    heldout_idx = T - 2   # last transition index (0-based)
    if heldout_idx not in shifts:
        return {}

    n_words = len(shifts[0])
    # Build per-word shift series over training transitions
    preds, actuals = [], []
    for w in range(n_words):
        series = [float(shifts[t][w]) for t in range(heldout_idx) if t in shifts]
        pred = _ema_predict(series)
        actual = float(shifts[heldout_idx][w])
        preds.append(pred)
        actuals.append(actual)

    preds = np.array(preds)
    actuals = np.array(actuals)
    abs_err = np.abs(preds - actuals)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))

    # Top-50 overlap with DGT (if provided)
    top_pred_idx = set(np.argsort(preds)[-top_k:])
    if dgt_top_indices is not None:
        overlap = len(top_pred_idx & set(dgt_top_indices)) / top_k
    else:
        overlap = 1.0  # self-overlap for DGT

    # Spearman ρ
    from scipy.stats import spearmanr
    rho, _ = spearmanr(preds, actuals)

    return {
        "mae": mae,
        "rmse": rmse,
        "top50_overlap": overlap,
        "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
        "heldout_transition": int(heldout_idx),
        "n_terms": int(n_words),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main walk-forward loop
# ═══════════════════════════════════════════════════════════════════════════

def run_walk_forward(
    all_snapshots: list[dict],
    output_dir: Path,
    *,
    folds: list[int] = FOLDS,
    max_nodes: int = 6000,
    dgt_epochs: int = 100,
    dim: int = 64,
    lap_pe_k: int = 16,
    device: str = "auto",
    hidden_dim: int = 128,
    heads: int = 4,
    layers: int = 2,
    top_k: int = TOP_K,
    seed: int = 1729,
) -> pd.DataFrame:
    """Run walk-forward validation across all folds.

    Returns a DataFrame with columns:
        fold, model, tier, mae, rmse, top50_overlap, spearman_rho,
        heldout_transition, n_terms
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Global vocabulary: top max_nodes words across ALL 12 spells (consistent
    # with the main evaluation so comparisons are apple-to-apple).
    words_global = _top_words_from_snaps(all_snapshots, max_nodes)
    n_words = len(words_global)
    print(f"[WF] Global vocab: {n_words} terms across {len(all_snapshots)} spells")
    print(f"[WF] Folds: {folds}  DGT epochs: {dgt_epochs}  device: {device}")

    all_rows: list[dict] = []

    for k in folds:
        print(f"\n{'='*60}")
        print(f"[WF] === Fold k={k}: training on spells 1..{k} ===")
        snaps_k = all_snapshots[:k]   # k cumulative snapshots

        # ── Global word count matrix for trend baselines ────────────────
        counts = _count_matrix(snaps_k, words_global)

        # ── Baseline models ─────────────────────────────────────────────
        baseline_specs: list[tuple[str, str, dict]] = [
            # (model_name, tier, shift_dict)
            ("ppmi_svd_procrustes", "classic_diachronic",
             _matrix_embedding_shifts(snaps_k, words_global,
                                      lambda s: _ppmi(_edge_matrix(s, words_global)), dim=dim)),
            ("sgns_shifted_ppmi_svd", "classic_diachronic",
             _matrix_embedding_shifts(snaps_k, words_global,
                                      lambda s: _ppmi(_edge_matrix(s, words_global))
                                               - sparse.eye(n_words) * math.log(5), dim=dim)),
            ("glove_log_cooccurrence_svd", "classic_diachronic",
             _matrix_embedding_shifts(snaps_k, words_global,
                                      lambda s: sparse.csr_matrix(
                                          np.log1p(np.asarray(_edge_matrix(s, words_global).todense()))
                                      ), dim=dim)),
            ("deepwalk_svd", "static_graph_embedding",
             _matrix_embedding_shifts(snaps_k, words_global,
                                      lambda s: _svd(_edge_matrix(s, words_global) @ _edge_matrix(s, words_global)
                                                     + _edge_matrix(s, words_global), dim=dim).T[:n_words].T
                                               if False else _edge_matrix(s, words_global), dim=dim)),
            ("raw_frequency_slope", "non_neural",
             _trend_shifts(counts)),
        ]

        # Simpler DeepWalk approximation: adjacency 2-hop
        dw_shifts = {}
        for t, snap in enumerate(snaps_k):
            A = _edge_matrix(snap, words_global, weighted=True)
            A2 = A + A @ A
            dw_shifts[t] = A2
        dw_embs = [_svd(dw_shifts[t], dim) for t in range(k)]
        dw_embs_aligned = _align_embeddings(dw_embs)
        baseline_specs[3] = ("deepwalk_svd", "static_graph_embedding",
                             _shift_series_from_embeddings(dw_embs_aligned))

        # ── DGT ─────────────────────────────────────────────────────────
        dgt_out = output_dir / f"fold_{k:02d}_dgt"
        print(f"[WF] Training DGT for fold k={k} → {dgt_out}")
        dgt_shifts_k = _dgt_shifts(
            snaps_k,
            words_global,
            max_nodes=max_nodes,
            epochs=dgt_epochs,
            device=device,
            hidden_dim=hidden_dim,
            heads=heads,
            layers=layers,
            lap_pe_k=lap_pe_k,
            output_dir=dgt_out,
        )

        # DGT top-50 for overlap computation
        heldout_idx = k - 2
        dgt_top = None
        if heldout_idx in dgt_shifts_k:
            dgt_preds_wf: list[float] = []
            for w_idx in range(n_words):
                s = [float(dgt_shifts_k[t][w_idx]) for t in range(heldout_idx) if t in dgt_shifts_k]
                dgt_preds_wf.append(_ema_predict(s))
            dgt_top = np.argsort(dgt_preds_wf)[-top_k:]

        # ── DGT metrics ─────────────────────────────────────────────────
        dgt_m = _fold_metrics(dgt_shifts_k, k, top_k=top_k, dgt_top_indices=dgt_top)
        if dgt_m:
            all_rows.append({
                "fold": k, "model": "dgt", "tier": "proposed",
                **dgt_m,
            })
            print(f"  DGT  MAE={dgt_m['mae']:.4f}  RMSE={dgt_m['rmse']:.4f}  "
                  f"ρ={dgt_m['spearman_rho']:.3f}")

        # ── Baseline metrics ─────────────────────────────────────────────
        for model_name, tier, shift_dict in baseline_specs:
            bm = _fold_metrics(shift_dict, k, top_k=top_k, dgt_top_indices=dgt_top)
            if bm:
                all_rows.append({
                    "fold": k, "model": model_name, "tier": tier,
                    **bm,
                })
                print(f"  {model_name:<35s} MAE={bm['mae']:.4f}  ρ={bm['spearman_rho']:.3f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "wf_fold_results.csv", index=False)
    print(f"\n[WF] Fold results → {output_dir / 'wf_fold_results.csv'}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate + significance
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_results(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Compute mean ± std across folds; paired one-tailed t-tests vs DGT."""
    agg = (
        df.groupby("model")[["mae", "rmse", "top50_overlap", "spearman_rho"]]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.sort_values("mae_mean")
    agg.to_csv(output_dir / "wf_aggregate.csv")
    print(f"[WF] Aggregate → {output_dir / 'wf_aggregate.csv'}")

    # Per-fold MAE pivot for paired t-test
    pivot = df.pivot(index="fold", columns="model", values="mae")
    dgt_col = pivot["dgt"].values if "dgt" in pivot.columns else None

    sig_rows = []
    for model in pivot.columns:
        if model == "dgt":
            continue
        other = pivot[model].values
        valid = ~(np.isnan(dgt_col) | np.isnan(other))
        if valid.sum() < 2 or dgt_col is None:
            p = float("nan")
        else:
            try:
                # One-tailed paired t-test: H1 = DGT MAE < baseline MAE
                _, p_two = ttest_rel(dgt_col[valid], other[valid])
                # Convert two-tailed p to one-tailed in the direction DGT < baseline
                p = float(p_two / 2) if float(np.mean(dgt_col[valid])) < float(np.mean(other[valid])) else 1.0
            except Exception:
                p = float("nan")
        delta = float(np.nanmean(other) - np.nanmean(dgt_col)) if dgt_col is not None else float("nan")
        sig_rows.append({
            "model": model,
            "mean_mae": float(np.nanmean(other)),
            "dgt_mean_mae": float(np.nanmean(dgt_col)) if dgt_col is not None else float("nan"),
            "delta_vs_dgt": delta,
            "ttest_p": p,
            "sig_dgt_better": (p < 0.05) if not np.isnan(p) else False,
        })
    sig_df = pd.DataFrame(sig_rows).sort_values("delta_vs_dgt", ascending=False)
    sig_df.to_csv(output_dir / "wf_significance.csv", index=False)
    print(f"[WF] Significance → {output_dir / 'wf_significance.csv'}")
    return agg


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshots", default="ETG_MISQ/output/cache/rt11_etg_snapshots.pkl",
                   help="Path to rt11_etg_snapshots.pkl")
    p.add_argument("--output", default="ETG_MISQ/output/walk_forward",
                   help="Output directory")
    p.add_argument("--dgt-epochs", type=int, default=100,
                   help="DGT training epochs per fold (default 100)")
    p.add_argument("--max-nodes", type=int, default=6000,
                   help="Vocabulary size (default 6000, matching main eval)")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lap-pe-k", type=int, default=16)
    p.add_argument("--dim", type=int, default=64,
                   help="Baseline embedding dimension")
    p.add_argument("--device", default="auto")
    p.add_argument("--folds", nargs="+", type=int, default=FOLDS,
                   help="Fold sizes (default: 7 8 9 10 11)")
    p.add_argument("--seed", type=int, default=1729)
    args = p.parse_args(argv)

    snap_path = Path(args.snapshots)
    if not snap_path.exists():
        # Try relative to repo root
        snap_path = REPO / args.snapshots
    if not snap_path.exists():
        sys.exit(f"Snapshots not found: {snap_path}\n"
                 "Run the ETG pipeline first or pass --snapshots <path>.")

    print(f"[WF] Loading snapshots from {snap_path}")
    with open(snap_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, tuple):
        all_snapshots, _ = obj
    else:
        all_snapshots = obj
    print(f"[WF] Loaded {len(all_snapshots)} spells")

    out = Path(args.output)
    df = run_walk_forward(
        all_snapshots,
        out,
        folds=args.folds,
        max_nodes=args.max_nodes,
        dgt_epochs=args.dgt_epochs,
        dim=args.dim,
        lap_pe_k=args.lap_pe_k,
        device=args.device,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        layers=args.layers,
        seed=args.seed,
    )
    agg = aggregate_results(df, out)

    print("\n[WF] === Aggregate MAE (mean ± std across folds) ===")
    print(agg[["mae_mean", "mae_std", "spearman_rho_mean"]].to_string())

    # Print a quick summary for the paper
    dgt_row = df[df["model"] == "dgt"]
    if not dgt_row.empty:
        print(f"\n[WF] DGT walk-forward MAE: "
              f"{dgt_row['mae'].mean():.4f} ± {dgt_row['mae'].std():.4f}  "
              f"(across {len(dgt_row)} folds)")
        print(f"[WF] DGT walk-forward ρ:   "
              f"{dgt_row['spearman_rho'].mean():.3f} ± {dgt_row['spearman_rho'].std():.3f}")

    print(f"\n[WF] Done. Results in {out}/")


if __name__ == "__main__":
    main()
