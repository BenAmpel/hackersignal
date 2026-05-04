"""Laplacian positional encoding per spell.

For each ETG snapshot we compute the top-K eigenvectors of the normalized
symmetric Laplacian. The Laplacian is re-computed per spell because the
graph changes. Sign ambiguity is handled with a canonical convention at
inference and random flipping during training.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from ..config import Config
from .etg_builder import ETGSnapshot


def _undirected_adjacency(edge_index: torch.Tensor, edge_weight: torch.Tensor, V: int) -> sp.csr_matrix:
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    w = edge_weight.cpu().numpy().astype(np.float32)
    # Symmetrize for Laplacian PE.
    data = np.concatenate([w, w])
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    A = sp.coo_matrix((data, (rows, cols)), shape=(V, V)).tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


def _normalized_laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.asarray(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    deg_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    I = sp.eye(A.shape[0])
    return I - D_inv_sqrt @ A @ D_inv_sqrt


def _canonical_sign(vecs: np.ndarray) -> np.ndarray:
    """Flip each eigenvector so its largest-magnitude entry is positive."""
    out = vecs.copy()
    for k in range(out.shape[1]):
        v = out[:, k]
        idx = int(np.argmax(np.abs(v)))
        if v[idx] < 0:
            out[:, k] = -v
    return out


def _pad_columns(vecs: np.ndarray, K: int, V: int) -> np.ndarray:
    if vecs.shape[1] >= K:
        return vecs[:, :K]
    pad = np.zeros((V, K - vecs.shape[1]), dtype=vecs.dtype)
    return np.concatenate([vecs, pad], axis=1)


def _dense_top_k(L: sp.csr_matrix, K: int) -> np.ndarray:
    """Deterministic K smallest eigenvectors for small graphs."""
    V = L.shape[0]
    if V == 0 or K <= 0:
        return np.zeros((V, max(K, 0)), dtype=np.float32)
    vals, vecs = np.linalg.eigh(L.toarray().astype(np.float64))
    order = np.argsort(vals, kind="stable")
    vecs = _pad_columns(vecs[:, order], K, V)
    return _canonical_sign(vecs.astype(np.float32))


def _eigsh_top_k(L: sp.csr_matrix, K: int) -> np.ndarray:
    """K smallest eigenvectors of a symmetric sparse matrix."""
    V = L.shape[0]
    if V <= 512:
        return _dense_top_k(L, K)
    K_eff = min(K, V - 2)
    if K_eff <= 0:
        # Tiny graph: return zeros of the right shape.
        return np.zeros((V, K), dtype=np.float32)
    try:
        # Fixed v0 removes avoidable run-to-run variation in ARPACK starts.
        v0 = np.linspace(1.0, 2.0, V, dtype=np.float64)
        vals, vecs = spla.eigsh(L, k=K_eff, which="SM", tol=1e-3, v0=v0)
    except Exception:
        # Fallback: dense eigendecomposition for small graphs.
        return _dense_top_k(L, K)
    order = np.argsort(vals)
    vecs = _pad_columns(vecs[:, order], K, V)
    return _canonical_sign(vecs.astype(np.float32))


def _nystrom_lap_pe(L: sp.csr_matrix, K: int, n_samples: int) -> np.ndarray:
    """Nyström approximation for very large V. Sample columns, SVD reduced block."""
    V = L.shape[0]
    n = min(n_samples, V)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(V, size=n, replace=False)
    # C = L[:, sample_idx]
    C = L[:, sample_idx].toarray().astype(np.float32)
    # W = L[sample_idx, sample_idx]
    W = L[sample_idx, :][:, sample_idx].toarray().astype(np.float32)
    # Stable pseudo-inverse via eigendecomposition of small W.
    w_vals, w_vecs = np.linalg.eigh(W)
    w_vals = np.clip(w_vals, 1e-6, None)
    W_inv_sqrt = w_vecs @ np.diag(1.0 / np.sqrt(w_vals)) @ w_vecs.T
    Z = C @ W_inv_sqrt  # [V, n]
    # Take the K leading columns of Z as positional encoding.
    return _canonical_sign(Z[:, :K].astype(np.float32))


def compute_lap_pe(snapshot: ETGSnapshot, config: Config) -> np.ndarray:
    V = snapshot["x"].shape[0]
    A = _undirected_adjacency(snapshot["edge_index"], snapshot["edge_weight"], V)
    L = _normalized_laplacian(A)
    method = config.lap_pe_method
    if method == "nystrom" and V > config.lap_pe_nystrom_samples * 2:
        return _nystrom_lap_pe(L, config.lap_pe_k, config.lap_pe_nystrom_samples)
    return _eigsh_top_k(L, config.lap_pe_k)


def attach_lap_pe(snapshots: list[ETGSnapshot], config: Config) -> list[ETGSnapshot]:
    """Compute per-spell Laplacian PE and write into the snapshots in place."""
    for snap in snapshots:
        vecs = compute_lap_pe(snap, config)
        snap["lap_pe"] = torch.from_numpy(vecs)
    return snapshots


def random_sign_flip(pe: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Train-time augmentation — random ±1 per column to be sign-invariant."""
    flips = torch.randint(
        0, 2, (pe.shape[1],), generator=generator, device=pe.device, dtype=pe.dtype
    ) * 2 - 1
    return pe * flips.unsqueeze(0)
