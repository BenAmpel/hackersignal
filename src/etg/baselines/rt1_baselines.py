"""Baselines for RT1 diachronic linguistics evaluation.

Classical / statistical:
  word2vec      : per-spell gensim Word2Vec, Procrustes-aligned (original)
  fasttext      : gensim FastText (subword skip-gram), Procrustes-aligned
  ppmi_svd      : PPMI co-occurrence matrix → truncated SVD embeddings
  bow_drift     : TF-IDF centroid cosine drift per spell (no graph)
  freq_shift    : L1 norm of normalized frequency-vector changes
  cusum         : cumulative-sum change-point severity on a shift series

Graph embedding (non-temporal):
  deepwalk      : uniform random walks on the ETG + Word2Vec
  node2vec      : biased random walks (p,q) + Word2Vec
  line          : LINE 1st-order proximity (direct edge modeling)
  static_gcn    : 2-layer GCN trained with BPR loss on last-spell ETG (original)
  gat           : Graph Attention Network (Veličković et al., 2018)
  gae           : Graph Autoencoder (GCN encoder + inner-product decoder)

All embedding-based baselines follow the same evaluation contract:
  train_X_per_spell(...) -> list[np.ndarray]   # one [V, d] per spell
  X_shift_series(embeddings)  -> np.ndarray    # one drift scalar per spell-pair
"""

from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn

from ..config import Config
from ..data.schemas import ForumPost
from ..rt1.etg_builder import ETGSnapshot
from ..rt1.vocab import Vocab, tokenize


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Orthogonal R minimising ||A R - B||_F."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt


def _align_sequence(embeddings: list[np.ndarray]) -> list[np.ndarray]:
    """Procrustes-align every embedding matrix to spell-0."""
    anchor = embeddings[0]
    aligned = [anchor.copy()]
    for t in range(1, len(embeddings)):
        nrm_a = np.linalg.norm(anchor, axis=1)
        nrm_t = np.linalg.norm(embeddings[t], axis=1)
        mask = (nrm_a > 0) & (nrm_t > 0)
        if mask.sum() < 4:
            aligned.append(embeddings[t].copy())
            continue
        R = _procrustes(embeddings[t][mask], anchor[mask])
        aligned.append((embeddings[t] @ R).astype(np.float32))
    return aligned


def embeddings_to_shift_series(embeddings: list[np.ndarray]) -> np.ndarray:
    """Mean cosine drift across adjacent spell pairs (∈ [0, 2])."""
    series = []
    for t in range(1, len(embeddings)):
        prev, curr = embeddings[t - 1], embeddings[t]
        nrm_p = np.linalg.norm(prev, axis=1)
        nrm_c = np.linalg.norm(curr, axis=1)
        mask = (nrm_p > 0) & (nrm_c > 0)
        if mask.sum() == 0:
            series.append(0.0)
            continue
        cos = (prev[mask] * curr[mask]).sum(axis=1) / (nrm_p[mask] * nrm_c[mask] + 1e-12)
        series.append(float(np.mean(1.0 - np.clip(cos, -1, 1))))
    return np.array(series, dtype=np.float32)


def make_gcn_norm_adj(edge_index: Tensor, V: int, device: torch.device) -> Tensor:
    """Symmetric D^{-1/2}(A+I)D^{-1/2} normalisation (dense)."""
    A = torch.zeros((V, V), device=device)
    if edge_index.numel() > 0:
        A[edge_index[0], edge_index[1]] = 1.0
        A[edge_index[1], edge_index[0]] = 1.0
    A.fill_diagonal_(1.0)
    deg = A.sum(dim=1)
    d_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    D = torch.diag(d_inv_sqrt)
    return D @ A @ D


# ---------------------------------------------------------------------------
# Word2Vec (original baseline)
# ---------------------------------------------------------------------------

def train_word2vec_per_spell(
    posts_by_spell: list[list[ForumPost]], vocab: Vocab, config: Config
) -> list[np.ndarray]:
    """One Word2Vec model per spell, Procrustes-aligned to spell 0."""
    from gensim.models import Word2Vec as GWord2Vec

    dim = config.dgt_hidden_dim
    V = vocab.size
    out: list[np.ndarray] = []
    for t, posts in enumerate(posts_by_spell):
        sentences = [tokenize(p.text, config.min_token_len) for p in posts]
        if not any(sentences):
            sentences = [["<UNK>"]]
        model = GWord2Vec(
            sentences=sentences, vector_size=dim, window=config.window_size,
            min_count=1, sg=1, epochs=3, seed=config.seed + t, workers=1,
        )
        emb = np.zeros((V, dim), dtype=np.float32)
        for i in range(V):
            tok = vocab.token(i)
            if tok in model.wv:
                emb[i] = model.wv[tok]
        out.append(emb)
    return _align_sequence(out)


def word2vec_shift_series(embeddings: list[np.ndarray]) -> np.ndarray:
    return embeddings_to_shift_series(embeddings)


# ---------------------------------------------------------------------------
# FastText
# ---------------------------------------------------------------------------

def train_fasttext_per_spell(
    posts_by_spell: list[list[ForumPost]], vocab: Vocab, config: Config
) -> list[np.ndarray]:
    """Gensim FastText (subword n-gram skip-gram), aligned across spells."""
    from gensim.models import FastText as GFastText

    dim = config.dgt_hidden_dim
    V = vocab.size
    out: list[np.ndarray] = []
    for t, posts in enumerate(posts_by_spell):
        sentences = [tokenize(p.text, config.min_token_len) for p in posts]
        if not any(sentences):
            sentences = [["<UNK>"]]
        model = GFastText(
            sentences=sentences, vector_size=dim, window=config.window_size,
            min_count=1, sg=1, epochs=3, seed=config.seed + t, workers=1,
        )
        emb = np.zeros((V, dim), dtype=np.float32)
        for i in range(V):
            try:
                emb[i] = model.wv[vocab.token(i)]  # handles OOV via subwords
            except KeyError:
                pass
        out.append(emb)
    return _align_sequence(out)


# ---------------------------------------------------------------------------
# PPMI + Truncated SVD
# ---------------------------------------------------------------------------

def ppmi_svd_per_spell(
    snapshots: list[ETGSnapshot], vocab: Vocab, config: Config
) -> list[np.ndarray]:
    """PPMI co-occurrence matrix per spell → truncated SVD word vectors."""
    from sklearn.utils.extmath import randomized_svd

    dim = config.dgt_hidden_dim
    V = vocab.size
    out: list[np.ndarray] = []
    for snap in snapshots:
        ei = snap["edge_index"]
        ew = snap["edge_weight"]
        # Sparse count matrix
        counts = np.zeros((V, V), dtype=np.float32)
        if ei.numel() > 0:
            rows = ei[0].cpu().numpy().astype(int)
            cols = ei[1].cpu().numpy().astype(int)
            vals = ew.cpu().numpy()
            np.add.at(counts, (rows, cols), vals)
            np.add.at(counts, (cols, rows), vals)
        row_s = counts.sum(axis=1, keepdims=True)
        col_s = counts.sum(axis=0, keepdims=True)
        total = counts.sum() + 1e-12
        # PMI = log P(i,j) / P(i)P(j), clipped to 0 → PPMI
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(counts * total / (row_s @ col_s + 1e-12) + 1e-12)
        ppmi = np.maximum(pmi, 0.0)
        nz = int((row_s > 0).sum())
        k = max(2, min(dim, nz - 1))
        try:
            U, s, _ = randomized_svd(ppmi, n_components=k, random_state=config.seed)
            emb = U * np.sqrt(np.maximum(s, 0))
        except Exception:
            emb = np.zeros((V, dim), dtype=np.float32)
        if emb.shape[1] < dim:
            emb = np.pad(emb, ((0, 0), (0, dim - emb.shape[1])))
        out.append(emb[:V].astype(np.float32))
    return _align_sequence(out)


# ---------------------------------------------------------------------------
# BoW cosine drift (no graph)
# ---------------------------------------------------------------------------

def bow_cosine_drift_series(
    posts_by_spell: list[list[ForumPost]], config: Config
) -> np.ndarray:
    """TF-IDF centroid cosine drift between adjacent spell centroids."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    all_texts = [p.text for spell in posts_by_spell for p in spell]
    vec = TfidfVectorizer(max_features=min(20_000, len(all_texts) * 10), min_df=1)
    vec.fit(all_texts)
    centroids = []
    for posts in posts_by_spell:
        if not posts:
            centroids.append(None)
            continue
        M = vec.transform([p.text for p in posts])
        c = np.asarray(M.mean(axis=0)).flatten().astype(np.float32)
        nrm = np.linalg.norm(c)
        centroids.append(c / (nrm + 1e-12))
    series = []
    for t in range(1, len(centroids)):
        if centroids[t - 1] is None or centroids[t] is None:
            series.append(0.0)
        else:
            series.append(float(1.0 - float(np.dot(centroids[t - 1], centroids[t]))))
    return np.array(series, dtype=np.float32)


# ---------------------------------------------------------------------------
# Term-frequency shift (non-embedding)
# ---------------------------------------------------------------------------

def freq_shift_series(snapshots: list[ETGSnapshot]) -> np.ndarray:
    """L1 norm of changes in the normalised per-node frequency vector."""
    series = []
    for t in range(1, len(snapshots)):
        f_p = snapshots[t - 1]["freq"].cpu().numpy().astype(np.float64)
        f_c = snapshots[t]["freq"].cpu().numpy().astype(np.float64)
        f_p /= f_p.sum() + 1e-12
        f_c /= f_c.sum() + 1e-12
        series.append(float(np.abs(f_c - f_p).sum()))
    return np.array(series, dtype=np.float32)


# ---------------------------------------------------------------------------
# CUSUM change-point severity
# ---------------------------------------------------------------------------

def cusum_series(shift_series: np.ndarray, k: float = 0.25) -> np.ndarray:
    """Two-sided CUSUM applied to a shift-score series.

    Returns the per-step max(S_high, S_low) — larger values indicate
    the signal has drifted beyond its rolling mean by more than k.
    """
    if len(shift_series) < 2:
        return shift_series.copy()
    mu = float(np.mean(shift_series))
    S_h, S_l = 0.0, 0.0
    out = []
    for x in shift_series:
        S_h = max(0.0, S_h + float(x) - mu - k)
        S_l = max(0.0, S_l + mu - float(x) - k)
        out.append(max(S_h, S_l))
    return np.array(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Random walk helpers (shared by DeepWalk and Node2Vec)
# ---------------------------------------------------------------------------

def _build_adj_lists(edge_index: Tensor, V: int) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(V)]
    if edge_index.numel() > 0:
        for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            adj[int(s)].append(int(d))
            adj[int(d)].append(int(s))
    return adj


def _deepwalk_walks(
    adj: list[list[int]], num_walks: int, walk_len: int, seed: int
) -> list[list[int]]:
    rng = random.Random(seed)
    V = len(adj)
    walks: list[list[int]] = []
    for _ in range(num_walks):
        order = list(range(V))
        rng.shuffle(order)
        for start in order:
            walk = [start]
            for _ in range(walk_len - 1):
                nbrs = adj[walk[-1]]
                if not nbrs:
                    break
                walk.append(rng.choice(nbrs))
            walks.append(walk)
    return walks


def _node2vec_precompute(
    adj: list[list[int]],
    p: float,
    q: float,
) -> dict[tuple[int, int], tuple[list[int], np.ndarray]]:
    """Pre-compute normalised transition probabilities for every (prev, curr) pair.

    This eliminates the O(degree²) per-step recomputation that makes the naive
    implementation hang on dense graphs.  Returns a dict mapping
    (prev, curr) → (neighbours, cumulative_prob_array).
    """
    V = len(adj)
    adj_set = [set(a) for a in adj]
    trans: dict[tuple[int, int], tuple[list[int], np.ndarray]] = {}

    for curr in range(V):
        nbrs = adj[curr]
        if not nbrs:
            continue
        # First-step: pretend prev == curr (uniform DeepWalk-like)
        w = np.ones(len(nbrs), dtype=np.float64)
        w /= w.sum()
        trans[(curr, curr)] = (nbrs, np.cumsum(w))
        # Biased steps
        for prev in adj[curr]:  # only precompute pairs that can actually occur
            weights = np.empty(len(nbrs), dtype=np.float64)
            prev_set = adj_set[prev]
            for k, nbr in enumerate(nbrs):
                if nbr == prev:
                    weights[k] = 1.0 / p
                elif nbr in prev_set:
                    weights[k] = 1.0
                else:
                    weights[k] = 1.0 / q
            weights /= weights.sum()
            trans[(prev, curr)] = (nbrs, np.cumsum(weights))
    return trans


def _node2vec_walks(
    adj: list[list[int]],
    num_walks: int,
    walk_len: int,
    p: float,
    q: float,
    seed: int,
    max_seconds: float = 30.0,
) -> list[list[int]]:
    """Node2Vec biased random walks with pre-computed transition tables.

    Pre-computing transition probabilities reduces per-step cost from
    O(degree²) to O(1) at the cost of O(E · degree) setup — acceptable for
    typical ETG densities and eliminates the freeze on dense real-data graphs.

    ``max_seconds`` is a wall-clock safety timeout: if the pre-computation
    exceeds this limit (pathologically dense graphs), we fall back to uniform
    DeepWalk-style walks so the pipeline never hangs.
    """
    import time

    t0 = time.time()
    try:
        trans = _node2vec_precompute(adj, p, q)
    except Exception:
        trans = {}

    if not trans or (time.time() - t0) > max_seconds:
        # Fallback: uniform DeepWalk walks
        return _deepwalk_walks(adj, num_walks, walk_len, seed)

    rng = np.random.default_rng(seed)
    V = len(adj)
    walks: list[list[int]] = []
    order = np.arange(V)

    for _ in range(num_walks):
        rng.shuffle(order)
        if (time.time() - t0) > max_seconds:
            break  # safety cut-off mid-walk-set
        for start in order.tolist():
            walk = [start]
            prev = start
            for _ in range(walk_len - 1):
                curr = walk[-1]
                key = (prev, curr)
                if key not in trans:
                    # Isolated node or unseen pair — stay put
                    break
                nbrs, cdf = trans[key]
                r = rng.random()
                idx = int(np.searchsorted(cdf, r))
                idx = min(idx, len(nbrs) - 1)
                nxt = nbrs[idx]
                walk.append(nxt)
                prev = curr
            walks.append(walk)
    return walks


def _walks_to_emb(walks: list[list[int]], V: int, dim: int, seed: int, window: int) -> np.ndarray:
    from gensim.models import Word2Vec as GW2V

    sentences = [[str(n) for n in w] for w in walks]
    model = GW2V(
        sentences=sentences, vector_size=dim, window=window,
        min_count=1, sg=1, epochs=1, seed=seed, workers=1,
    )
    emb = np.zeros((V, dim), dtype=np.float32)
    for v in range(V):
        key = str(v)
        if key in model.wv:
            emb[v] = model.wv[key]
    return emb


# ---------------------------------------------------------------------------
# DeepWalk
# ---------------------------------------------------------------------------

def train_deepwalk_per_spell(
    snapshots: list[ETGSnapshot],
    vocab: Vocab,
    config: Config,
    num_walks: int = 10,
    walk_len: int = 40,
) -> list[np.ndarray]:
    """DeepWalk: uniform random walks on the ETG → Word2Vec skip-gram."""
    V = vocab.size
    dim = config.dgt_hidden_dim
    out: list[np.ndarray] = []
    for t, snap in enumerate(snapshots):
        adj = _build_adj_lists(snap["edge_index"], V)
        walks = _deepwalk_walks(adj, num_walks, walk_len, config.seed + t)
        out.append(_walks_to_emb(walks, V, dim, config.seed + t, config.window_size))
    return _align_sequence(out)


# ---------------------------------------------------------------------------
# Node2Vec
# ---------------------------------------------------------------------------

def train_node2vec_per_spell(
    snapshots: list[ETGSnapshot],
    vocab: Vocab,
    config: Config,
    num_walks: int = 10,
    walk_len: int = 40,
    p: float = 1.0,
    q: float = 0.5,
) -> list[np.ndarray]:
    """Node2Vec: biased random walks (DFS/BFS interpolation) → Word2Vec."""
    V = vocab.size
    dim = config.dgt_hidden_dim
    out: list[np.ndarray] = []
    for t, snap in enumerate(snapshots):
        adj = _build_adj_lists(snap["edge_index"], V)
        walks = _node2vec_walks(adj, num_walks, walk_len, p, q, config.seed + t)
        out.append(_walks_to_emb(walks, V, dim, config.seed + t, config.window_size))
    return _align_sequence(out)


# ---------------------------------------------------------------------------
# LINE — Large-scale Information Network Embedding (1st-order)
# ---------------------------------------------------------------------------

class _LINEModel(nn.Module):
    def __init__(self, V: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(V, dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, src: Tensor, dst: Tensor) -> Tensor:
        return torch.sigmoid((self.emb(src) * self.emb(dst)).sum(-1))

    def get_emb(self) -> np.ndarray:
        return self.emb.weight.detach().cpu().numpy()


def train_line_per_spell(
    snapshots: list[ETGSnapshot],
    vocab: Vocab,
    config: Config,
    device: torch.device,
    epochs: int = 3,
) -> list[np.ndarray]:
    """LINE 1st-order: maximise P(edge) ∝ σ(u·v), noise-contrastive training."""
    V = vocab.size
    dim = config.dgt_hidden_dim
    out: list[np.ndarray] = []
    for snap in snapshots:
        model = _LINEModel(V, dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        ei = snap["edge_index"].to(device)
        ew = snap["edge_weight"].to(device)
        E = ei.shape[1]
        if E == 0:
            out.append(np.zeros((V, dim), dtype=np.float32))
            continue
        for _ in range(epochs):
            idx = torch.randint(0, E, (min(config.dgt_batch_size_edges, E),), device=device)
            src, dst = ei[0, idx], ei[1, idx]
            w = ew[idx] / (ew[idx].max() + 1e-8)
            pos = model(src, dst)
            neg_dst = torch.randint(0, V, (src.shape[0],), device=device)
            neg = model(src, neg_dst)
            loss = -(w * torch.log(pos + 1e-8) + torch.log(1 - neg + 1e-8)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        out.append(model.get_emb().astype(np.float32))
    return _align_sequence(out)


# ---------------------------------------------------------------------------
# Static GCN (original baseline)
# ---------------------------------------------------------------------------

class StaticGCNLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x: Tensor, A_norm: Tensor) -> Tensor:
        return torch.relu(A_norm @ self.lin(x))


class StaticGCN(nn.Module):
    def __init__(self, feature_dim: int, hidden: int):
        super().__init__()
        self.l1 = StaticGCNLayer(feature_dim, hidden)
        self.l2 = StaticGCNLayer(hidden, hidden)
        self.edge_score = nn.Bilinear(hidden, hidden, 1)

    def forward(self, x: Tensor, A_norm: Tensor) -> Tensor:
        return self.l2(self.l1(x, A_norm), A_norm)

    def score_edges(self, h: Tensor, ei: Tensor) -> Tensor:
        return self.edge_score(h[ei[0]], h[ei[1]]).squeeze(-1)


def train_gcn_per_spell(
    snapshots: list[ETGSnapshot],
    vocab: Vocab,
    config: Config,
    device: torch.device,
    epochs: int = 3,
) -> list[np.ndarray]:
    """Train one StaticGCN per spell with BPR edge-reconstruction loss."""
    feature_dim = snapshots[0]["x"].shape[1]
    hidden = config.dgt_hidden_dim
    V = vocab.size
    out: list[np.ndarray] = []
    for snap in snapshots:
        model = StaticGCN(feature_dim, hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = snap["x"].to(device)
        ei = snap["edge_index"].to(device)
        A_norm = make_gcn_norm_adj(ei, V, device)
        E = ei.shape[1]
        for _ in range(epochs):
            model.train()
            h = model(x, A_norm)
            if E == 0:
                break
            idx = torch.randint(0, E, (min(config.dgt_batch_size_edges, E),), device=device)
            pos_s = model.score_edges(h, ei[:, idx])
            neg_dst = torch.randint(0, V, (idx.shape[0],), device=device)
            neg_ei = torch.stack([ei[0, idx], neg_dst])
            neg_s = model.score_edges(h, neg_ei)
            loss = -torch.log(torch.sigmoid(pos_s - neg_s) + 1e-8).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            h = model(x, A_norm)
        out.append(h.cpu().numpy().astype(np.float32))
    return _align_sequence(out)


# ---------------------------------------------------------------------------
# GAT — Graph Attention Network
# ---------------------------------------------------------------------------

class _GATLayer(nn.Module):
    """Single multi-head additive GAT layer (Veličković et al., 2018).

    Uses sparse edge-wise computation: O(E·H) instead of O(V²·H).
    """

    def __init__(self, d_in: int, d_out: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.d_out = d_out
        self.W = nn.Linear(d_in, d_out * heads, bias=False)
        # Attention parameter: a^T [Wh_i || Wh_j], one per head.
        self.a_src = nn.Parameter(torch.empty(heads, d_out))
        self.a_dst = nn.Parameter(torch.empty(heads, d_out))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.leaky = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_out * heads)

    def forward(self, x: Tensor, edge_index: Tensor, V: int) -> Tensor:
        Wh = self.W(x).view(V, self.heads, self.d_out)  # [V, H, d_out]
        if edge_index.numel() == 0:
            return self.ln(torch.relu(Wh.view(V, -1)))
        src_idx, dst_idx = edge_index[0], edge_index[1]
        # e_ij = LeakyReLU(a_src · Wh_i + a_dst · Wh_j)
        e = self.leaky(
            (Wh[src_idx] * self.a_src).sum(-1)  # [E, H]
            + (Wh[dst_idx] * self.a_dst).sum(-1)
        )
        # Numerically-stable softmax over in-edges per destination node.
        # Step 1: per-dst max for stability.
        e_max = torch.full((V, self.heads), float("-inf"), device=x.device)
        e_max.scatter_reduce_(0, dst_idx.unsqueeze(1).expand(-1, self.heads), e, reduce="amax", include_self=True)
        exp_e = torch.exp(e - e_max[dst_idx])  # [E, H]
        agg_exp = torch.zeros(V, self.heads, device=x.device)
        agg_exp.scatter_add_(0, dst_idx.unsqueeze(1).expand(-1, self.heads), exp_e)
        alpha = self.drop(exp_e / (agg_exp[dst_idx] + 1e-12))  # [E, H]
        # Weighted message aggregation.
        msg = alpha.unsqueeze(-1) * Wh[src_idx]  # [E, H, d_out]
        out = torch.zeros(V, self.heads, self.d_out, device=x.device)
        out.scatter_add_(
            0,
            dst_idx.view(-1, 1, 1).expand(-1, self.heads, self.d_out),
            msg,
        )
        return self.ln(torch.relu(out.view(V, self.heads * self.d_out)))


class GAT(nn.Module):
    def __init__(self, feature_dim: int, hidden: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        d_per_head = max(4, hidden // heads)
        self.l1 = _GATLayer(feature_dim, d_per_head, heads, dropout)
        self.l2 = _GATLayer(d_per_head * heads, hidden, 1, dropout)
        self.edge_score = nn.Bilinear(hidden, hidden, 1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        V = x.shape[0]
        h = self.l1(x, edge_index, V)
        return self.l2(h, edge_index, V)

    def score_edges(self, h: Tensor, ei: Tensor) -> Tensor:
        return self.edge_score(h[ei[0]], h[ei[1]]).squeeze(-1)


def train_gat_per_spell(
    snapshots: list[ETGSnapshot],
    vocab: Vocab,
    config: Config,
    device: torch.device,
    epochs: int = 3,
) -> list[np.ndarray]:
    """Train one GAT per spell with BPR reconstruction loss."""
    feature_dim = snapshots[0]["x"].shape[1]
    hidden = config.dgt_hidden_dim
    V = vocab.size
    out: list[np.ndarray] = []
    for snap in snapshots:
        model = GAT(feature_dim, hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = snap["x"].to(device)
        ei = snap["edge_index"].to(device)
        E = ei.shape[1]
        for _ in range(epochs):
            model.train()
            h = model(x, ei)
            if E == 0:
                break
            idx = torch.randint(0, E, (min(config.dgt_batch_size_edges, E),), device=device)
            pos_s = model.score_edges(h, ei[:, idx])
            neg_dst = torch.randint(0, V, (idx.shape[0],), device=device)
            neg_ei = torch.stack([ei[0, idx], neg_dst])
            neg_s = model.score_edges(h, neg_ei)
            loss = -torch.log(torch.sigmoid(pos_s - neg_s) + 1e-8).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            h = model(x, ei)
        out.append(h.cpu().numpy().astype(np.float32))
    return _align_sequence(out)


# ---------------------------------------------------------------------------
# GAE — Graph Autoencoder
# ---------------------------------------------------------------------------

class _GAEEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden: int):
        super().__init__()
        self.l1 = StaticGCNLayer(feature_dim, hidden * 2)
        self.l2 = nn.Linear(hidden * 2, hidden)
        self.bn = nn.LayerNorm(hidden * 2)

    def forward(self, x: Tensor, A_norm: Tensor) -> Tensor:
        h = self.l1(x, A_norm)
        return self.l2(self.bn(h))


def train_gae_per_spell(
    snapshots: list[ETGSnapshot],
    vocab: Vocab,
    config: Config,
    device: torch.device,
    epochs: int = 3,
) -> list[np.ndarray]:
    """Graph Autoencoder: GCN encoder, inner-product decoder, BPR loss."""
    feature_dim = snapshots[0]["x"].shape[1]
    hidden = config.dgt_hidden_dim
    V = vocab.size
    out: list[np.ndarray] = []
    for snap in snapshots:
        enc = _GAEEncoder(feature_dim, hidden).to(device)
        opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
        x = snap["x"].to(device)
        ei = snap["edge_index"].to(device)
        A_norm = make_gcn_norm_adj(ei, V, device)
        E = ei.shape[1]
        for _ in range(epochs):
            enc.train()
            z = enc(x, A_norm)
            if E == 0:
                break
            idx = torch.randint(0, E, (min(config.dgt_batch_size_edges, E),), device=device)
            src, dst = ei[0, idx], ei[1, idx]
            pos_s = (z[src] * z[dst]).sum(-1)
            neg_dst = torch.randint(0, V, (src.shape[0],), device=device)
            neg_s = (z[src] * z[neg_dst]).sum(-1)
            loss = -torch.log(torch.sigmoid(pos_s - neg_s) + 1e-8).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        enc.eval()
        with torch.no_grad():
            z = enc(x, A_norm)
        out.append(z.cpu().numpy().astype(np.float32))
    return _align_sequence(out)


# ---------------------------------------------------------------------------
# Additional imports for new baselines (lazy, inside functions where needed)
# sklearn.decomposition.TruncatedSVD, LatentDirichletAllocation
# sklearn.feature_extraction.text.CountVectorizer
# statsmodels.tsa.holtwinters.ExponentialSmoothing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# GraphSAGE (mean aggregation)
# ---------------------------------------------------------------------------

class _SAGELayer(nn.Module):
    """Single GraphSAGE layer with mean aggregation."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # W applied to concat(h_self, h_mean_neigh)
        self.lin = nn.Linear(d_in * 2, d_out)

    def forward(self, x: Tensor, edge_index: Tensor, V: int) -> Tensor:
        # Mean-aggregate neighbour features
        neigh_mean = torch.zeros_like(x)
        count = torch.zeros(V, 1, device=x.device)
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            # both directions
            neigh_mean.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.shape[1]), x[src])
            count.scatter_add_(0, dst.unsqueeze(1), torch.ones(src.shape[0], 1, device=x.device))
            neigh_mean.scatter_add_(0, src.unsqueeze(1).expand(-1, x.shape[1]), x[dst])
            count.scatter_add_(0, src.unsqueeze(1), torch.ones(dst.shape[0], 1, device=x.device))
        neigh_mean = neigh_mean / (count + 1e-12)
        # Self-loop: if no neighbours keep self
        no_neigh = (count.squeeze(1) == 0)
        neigh_mean[no_neigh] = x[no_neigh]
        h = torch.cat([x, neigh_mean], dim=-1)
        return torch.relu(self.lin(h))


class _GraphSAGE(nn.Module):
    def __init__(self, feature_dim: int, hidden: int):
        super().__init__()
        self.l1 = _SAGELayer(feature_dim, hidden)
        self.l2 = _SAGELayer(hidden, hidden)
        self.edge_score = nn.Bilinear(hidden, hidden, 1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        V = x.shape[0]
        h = self.l1(x, edge_index, V)
        return self.l2(h, edge_index, V)

    def score_edges(self, h: Tensor, ei: Tensor) -> Tensor:
        return self.edge_score(h[ei[0]], h[ei[1]]).squeeze(-1)


def train_graphsage_per_spell(
    snapshots: list[ETGSnapshot],
    vocab: Vocab,
    config: Config,
    device: torch.device,
) -> list[np.ndarray]:
    """GraphSAGE with mean aggregation, BPR loss, Procrustes-aligned."""
    feature_dim = snapshots[0]["x"].shape[1]
    hidden = config.dgt_hidden_dim
    epochs = config.dgt_epochs
    lr = config.dgt_lr
    V = vocab.size
    out: list[np.ndarray] = []
    for snap in snapshots:
        model = _GraphSAGE(feature_dim, hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        x = snap["x"].to(device)
        ei = snap["edge_index"].to(device)
        E = ei.shape[1]
        for _ in range(epochs):
            model.train()
            h = model(x, ei)
            if E == 0:
                break
            idx = torch.randint(0, E, (min(config.dgt_batch_size_edges, E),), device=device)
            pos_s = model.score_edges(h, ei[:, idx])
            neg_dst = torch.randint(0, V, (idx.shape[0],), device=device)
            neg_ei = torch.stack([ei[0, idx], neg_dst])
            neg_s = model.score_edges(h, neg_ei)
            loss = -torch.log(torch.sigmoid(pos_s - neg_s) + 1e-8).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            h = model(x, ei)
        out.append(h.cpu().numpy().astype(np.float32))
    return _align_sequence(out)


# ---------------------------------------------------------------------------
# GIN — Graph Isomorphism Network
# ---------------------------------------------------------------------------

class _GINLayer(nn.Module):
    """Single GIN layer: h = MLP((1+ε)·h + Σ_{neigh} h_u), ε fixed at 0."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x: Tensor, edge_index: Tensor, V: int) -> Tensor:
        agg = x.clone()  # self term (ε=0 → (1+0)·h = h)
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.shape[1]), x[src])
            agg.scatter_add_(0, src.unsqueeze(1).expand(-1, x.shape[1]), x[dst])
        return self.mlp(agg)


class _GIN(nn.Module):
    def __init__(self, feature_dim: int, hidden: int):
        super().__init__()
        self.l1 = _GINLayer(feature_dim, hidden)
        self.l2 = _GINLayer(hidden, hidden)
        self.edge_score = nn.Bilinear(hidden, hidden, 1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        V = x.shape[0]
        h = self.l1(x, edge_index, V)
        return self.l2(h, edge_index, V)

    def score_edges(self, h: Tensor, ei: Tensor) -> Tensor:
        return self.edge_score(h[ei[0]], h[ei[1]]).squeeze(-1)


def train_gin_per_spell(
    snapshots: list[ETGSnapshot],
    vocab: Vocab,
    config: Config,
    device: torch.device,
) -> list[np.ndarray]:
    """GIN (ε=0) with BPR loss, Procrustes-aligned across spells."""
    feature_dim = snapshots[0]["x"].shape[1]
    hidden = config.dgt_hidden_dim
    epochs = config.dgt_epochs
    lr = config.dgt_lr
    V = vocab.size
    out: list[np.ndarray] = []
    for snap in snapshots:
        model = _GIN(feature_dim, hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        x = snap["x"].to(device)
        ei = snap["edge_index"].to(device)
        E = ei.shape[1]
        for _ in range(epochs):
            model.train()
            h = model(x, ei)
            if E == 0:
                break
            idx = torch.randint(0, E, (min(config.dgt_batch_size_edges, E),), device=device)
            pos_s = model.score_edges(h, ei[:, idx])
            neg_dst = torch.randint(0, V, (idx.shape[0],), device=device)
            neg_ei = torch.stack([ei[0, idx], neg_dst])
            neg_s = model.score_edges(h, neg_ei)
            loss = -torch.log(torch.sigmoid(pos_s - neg_s) + 1e-8).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            h = model(x, ei)
        out.append(h.cpu().numpy().astype(np.float32))
    return _align_sequence(out)


# ---------------------------------------------------------------------------
# TGAT — Temporal Graph Attention (simplified, spell-index time encoding)
# ---------------------------------------------------------------------------

def _sinusoidal_time_enc(t: int, d: int, device: torch.device) -> Tensor:
    """Sinusoidal encoding of scalar spell index t, shape [d]."""
    pos = float(t)
    enc = torch.zeros(d, device=device)
    for k in range(d // 2):
        denom = 10000 ** (2 * k / d)
        enc[2 * k] = math.sin(pos / denom)
        enc[2 * k + 1] = math.cos(pos / denom)
    return enc


class _TGAT(nn.Module):
    """1-layer GAT on time-augmented features."""

    def __init__(self, feature_dim: int, time_dim: int, hidden: int, heads: int = 4):
        super().__init__()
        augmented_dim = feature_dim + time_dim
        d_per_head = max(4, hidden // heads)
        self.gat1 = _GATLayer(augmented_dim, d_per_head, heads, dropout=0.1)
        self.proj = nn.Linear(d_per_head * heads, hidden)
        self.edge_score = nn.Bilinear(hidden, hidden, 1)

    def forward(self, x_aug: Tensor, edge_index: Tensor) -> Tensor:
        V = x_aug.shape[0]
        h = self.gat1(x_aug, edge_index, V)
        return torch.relu(self.proj(h))

    def score_edges(self, h: Tensor, ei: Tensor) -> Tensor:
        return self.edge_score(h[ei[0]], h[ei[1]]).squeeze(-1)


def train_tgat_per_spell(
    snapshots: list[ETGSnapshot],
    vocab: Vocab,
    config: Config,
    device: torch.device,
) -> list[np.ndarray]:
    """Temporal GAT: node features augmented with sinusoidal spell-index encoding."""
    feature_dim = snapshots[0]["x"].shape[1]
    hidden = config.dgt_hidden_dim
    time_dim = max(8, hidden // 4)  # sinusoidal time encoding dimension
    epochs = config.dgt_epochs
    lr = config.dgt_lr
    V = vocab.size
    out: list[np.ndarray] = []
    for t, snap in enumerate(snapshots):
        model = _TGAT(feature_dim, time_dim, hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        x = snap["x"].to(device)
        ei = snap["edge_index"].to(device)
        # Augment features with time encoding broadcast to all nodes
        t_enc = _sinusoidal_time_enc(t, time_dim, device).unsqueeze(0).expand(V, -1)
        x_aug = torch.cat([x, t_enc], dim=-1)
        E = ei.shape[1]
        for _ in range(epochs):
            model.train()
            h = model(x_aug, ei)
            if E == 0:
                break
            idx = torch.randint(0, E, (min(config.dgt_batch_size_edges, E),), device=device)
            pos_s = model.score_edges(h, ei[:, idx])
            neg_dst = torch.randint(0, V, (idx.shape[0],), device=device)
            neg_ei = torch.stack([ei[0, idx], neg_dst])
            neg_s = model.score_edges(h, neg_ei)
            loss = -torch.log(torch.sigmoid(pos_s - neg_s) + 1e-8).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            h = model(x_aug, ei)
        out.append(h.cpu().numpy().astype(np.float32))
    return _align_sequence(out)


# ---------------------------------------------------------------------------
# EvolveGCN-O
# ---------------------------------------------------------------------------

class _EvolveGCNO(nn.Module):
    """EvolveGCN-O: GRU evolves the GCN weight matrix across spells.

    GRU input at spell t: mean-pooled node embeddings H^{t-1}.
    GRU hidden state: W^t (the GCN weight matrix), reshaped as a vector.
    GCN forward: H^t = ReLU(A_hat · X · W^t).
    """

    def __init__(self, feature_dim: int, hidden: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden = hidden
        # GRU that updates W: input size = hidden (mean pool), hidden size = feature_dim * hidden
        self.gru_cell = nn.GRUCell(input_size=hidden, hidden_size=feature_dim * hidden)
        # Bias for the GCN layer
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.edge_score = nn.Bilinear(hidden, hidden, 1)
        # Initialise W as identity-like
        W_init = torch.randn(feature_dim * hidden) * 0.01
        self.register_buffer("W_state", W_init)

    def reset_W(self, device: torch.device) -> None:
        """Re-initialise weight state at the start of a training run."""
        self.W_state = torch.randn(self.feature_dim * self.hidden, device=device) * 0.01

    def forward_spell(
        self, x: Tensor, A_norm: Tensor, prev_mean: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Run one spell forward.  Returns (H, mean_H)."""
        # Update weight matrix via GRU
        self.W_state = self.gru_cell(prev_mean.unsqueeze(0), self.W_state.unsqueeze(0)).squeeze(0)
        W = self.W_state.view(self.feature_dim, self.hidden)
        H = torch.relu(A_norm @ x @ W + self.bias)
        mean_H = H.mean(dim=0)
        return H, mean_H

    def score_edges(self, h: Tensor, ei: Tensor) -> Tensor:
        return self.edge_score(h[ei[0]], h[ei[1]]).squeeze(-1)


def train_evolvegcn_per_spell(
    snapshots: list[ETGSnapshot],
    vocab: Vocab,
    config: Config,
    device: torch.device,
) -> list[np.ndarray]:
    """EvolveGCN-O: GRU updates GCN weights across spells, BPR loss."""
    feature_dim = snapshots[0]["x"].shape[1]
    hidden = config.dgt_hidden_dim
    epochs = config.dgt_epochs
    lr = config.dgt_lr
    V = vocab.size
    # Pre-compute normalised adjacency matrices (not tracked in grad graph)
    norm_adjs = [make_gcn_norm_adj(s["edge_index"].to(device), V, device) for s in snapshots]
    xs = [s["x"].to(device) for s in snapshots]
    eis = [s["edge_index"].to(device) for s in snapshots]

    model = _EvolveGCNO(feature_dim, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    out_tensors: list[Tensor] = []
    for epoch in range(epochs):
        model.train()
        model.reset_W(device)
        total_loss = torch.tensor(0.0, device=device)
        prev_mean = torch.zeros(hidden, device=device)
        spell_hs: list[Tensor] = []
        for t, (x, A_norm, ei) in enumerate(zip(xs, norm_adjs, eis)):
            H, mean_H = model.forward_spell(x, A_norm, prev_mean)
            spell_hs.append(H)
            prev_mean = mean_H.detach()
            E = ei.shape[1]
            if E > 0:
                idx = torch.randint(0, E, (min(config.dgt_batch_size_edges, E),), device=device)
                pos_s = model.score_edges(H, ei[:, idx])
                neg_dst = torch.randint(0, V, (idx.shape[0],), device=device)
                neg_ei = torch.stack([ei[0, idx], neg_dst])
                neg_s = model.score_edges(H, neg_ei)
                total_loss = total_loss + (-torch.log(torch.sigmoid(pos_s - neg_s) + 1e-8).mean())
        if epochs > 0:
            opt.zero_grad()
            total_loss.backward()
            opt.step()

    # Final forward pass to collect embeddings
    model.eval()
    model.reset_W(device)
    prev_mean = torch.zeros(hidden, device=device)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for x, A_norm, _ in zip(xs, norm_adjs, eis):
            H, mean_H = model.forward_spell(x, A_norm, prev_mean)
            out.append(H.cpu().numpy().astype(np.float32))
            prev_mean = mean_H
    return _align_sequence(out)


# ---------------------------------------------------------------------------
# LDA topic drift
# ---------------------------------------------------------------------------

def lda_topic_drift_series(
    posts_by_spell: list[list[ForumPost]],
    config: Config,
) -> list[float]:
    """Hellinger distance between adjacent LDA topic distributions per spell pair."""
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    n_spells = len(posts_by_spell)
    if n_spells < 2:
        return []

    # Aggregate all texts for fitting
    all_texts: list[str] = []
    spell_slices: list[tuple[int, int]] = []
    for spell_posts in posts_by_spell:
        start = len(all_texts)
        all_texts.extend(p.text for p in spell_posts)
        spell_slices.append((start, len(all_texts)))

    vec = CountVectorizer(max_features=config.vocab_size_cap, min_df=1)
    X_all = vec.fit_transform(all_texts)

    n_topics = min(10, n_spells * 2)
    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=config.seed, max_iter=20
    )
    doc_topics = lda.fit_transform(X_all)  # [total_docs, n_topics]

    # Per-spell mean topic distribution
    spell_dists: list[np.ndarray] = []
    for start, end in spell_slices:
        if end > start:
            dist = doc_topics[start:end].mean(axis=0).astype(np.float64)
        else:
            dist = np.ones(n_topics, dtype=np.float64) / n_topics
        dist /= dist.sum() + 1e-12
        spell_dists.append(dist)

    # Hellinger distance between adjacent spells
    series: list[float] = []
    for t in range(1, n_spells):
        p = spell_dists[t - 1]
        q = spell_dists[t]
        hellinger = math.sqrt(0.5 * float(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))
        series.append(hellinger)
    return series


# ---------------------------------------------------------------------------
# PMI ratio shift
# ---------------------------------------------------------------------------

def pmi_ratio_series(
    snapshots: list[ETGSnapshot],
    vocab: Vocab,
) -> list[float]:
    """Mean absolute log-ratio of PPMI values for edges present in both adjacent spells."""
    V = vocab.size

    def _ppmi_dict(snap: ETGSnapshot) -> dict[tuple[int, int], float]:
        ei = snap["edge_index"]
        ew = snap["edge_weight"]
        if ei.numel() == 0:
            return {}
        rows = ei[0].cpu().numpy().astype(int)
        cols = ei[1].cpu().numpy().astype(int)
        vals = ew.cpu().numpy().astype(np.float64)
        counts = np.zeros((V, V), dtype=np.float64)
        np.add.at(counts, (rows, cols), vals)
        np.add.at(counts, (cols, rows), vals)
        row_s = counts.sum(axis=1, keepdims=True)
        col_s = counts.sum(axis=0, keepdims=True)
        total = counts.sum() + 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(counts * total / (row_s @ col_s + 1e-12) + 1e-12)
        ppmi = np.maximum(pmi, 0.0)
        # Store only edges that appeared in edge_index
        result: dict[tuple[int, int], float] = {}
        for r, c, v in zip(rows, cols, vals):
            if v > 0:
                key = (int(r), int(c))
                result[key] = float(ppmi[r, c])
        return result

    series: list[float] = []
    for t in range(1, len(snapshots)):
        d_prev = _ppmi_dict(snapshots[t - 1])
        d_curr = _ppmi_dict(snapshots[t])
        common_keys = set(d_prev.keys()) & set(d_curr.keys())
        if not common_keys:
            series.append(0.0)
            continue
        diffs = [
            abs(math.log(d_curr[k] + 1e-9) - math.log(d_prev[k] + 1e-9))
            for k in common_keys
        ]
        series.append(float(np.mean(diffs)))
    return series


# ---------------------------------------------------------------------------
# ETS forecast (Holt-Winters double exponential smoothing)
# ---------------------------------------------------------------------------

def ets_forecast_series(
    shift_score_series: list[float],
) -> tuple[list[float], list[float]]:
    """Rolling 1-step-ahead Holt-Winters (additive trend) forecasts.

    Returns (preds, actuals) where each list has length max(0, len(series)-2).
    Fallback to naive-last if fewer than 3 historical points.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ModuleNotFoundError:
        # statsmodels not available — fall back to naive-last throughout
        series = list(shift_score_series)
        n = len(series)
        preds = [series[k - 1] for k in range(2, n)]
        actuals = [series[k] for k in range(2, n)]
        return preds, actuals

    series = list(shift_score_series)
    n = len(series)
    if n < 3:
        # Naive: predict last known value
        preds = [series[k - 1] for k in range(2, n)]
        actuals = [series[k] for k in range(2, n)]
        return preds, actuals

    preds: list[float] = []
    actuals: list[float] = []
    for k in range(2, n):
        history = series[:k]
        actual = series[k]
        if len(history) < 3:
            preds.append(history[-1])
        else:
            try:
                model = ExponentialSmoothing(
                    history, trend="add", seasonal=None, initialization_method="estimated"
                )
                fit = model.fit(optimized=True, disp=False)
                preds.append(float(fit.forecast(1)[0]))
            except Exception:
                preds.append(history[-1])
        actuals.append(actual)
    return preds, actuals


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

def random_shift_series(
    shift_results: list,
    seed: int = 0,
) -> list[float]:
    """Absolute floor: uniform random scores in [0, 1], same length as shift_results."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0.0, 1.0, len(shift_results)).tolist()


# ---------------------------------------------------------------------------
# GloVe-style embeddings (weighted SVD on co-occurrence)
# ---------------------------------------------------------------------------

def train_glove_per_spell(
    posts_by_spell: list[list[ForumPost]],
    vocab: Vocab,
    config: Config,
) -> list[np.ndarray]:
    """GloVe-style: weighted matrix factorisation of co-occurrence, Procrustes-aligned."""
    from sklearn.decomposition import TruncatedSVD

    dim = config.dgt_hidden_dim
    V = vocab.size
    x_max = 100.0
    alpha = 0.75

    # Build a token-to-index lookup from the vocab
    tok2idx: dict[str, int] = {vocab.token(i): i for i in range(V)}

    out: list[np.ndarray] = []
    for posts in posts_by_spell:
        # Build co-occurrence matrix from sliding window over post tokens
        cooc = np.zeros((V, V), dtype=np.float32)
        window = config.window_size
        for post in posts:
            tokens = [t for t in post.text.split() if t in tok2idx]
            indices = [tok2idx[t] for t in tokens]
            for i, idx_i in enumerate(indices):
                start = max(0, i - window)
                end = min(len(indices), i + window + 1)
                for j in range(start, end):
                    if j == i:
                        continue
                    idx_j = indices[j]
                    dist = abs(i - j)
                    cooc[idx_i, idx_j] += 1.0 / dist
        # GloVe weighting: f(x) = min(1, (x/x_max)^alpha)
        weight = np.minimum(1.0, (cooc / x_max) ** alpha)
        # Weighted matrix: W ⊙ C (element-wise), then take sqrt of weights
        W_sqrt = np.sqrt(weight)
        C_weighted = W_sqrt * cooc
        nz_rows = int((cooc.sum(axis=1) > 0).sum())
        k_comp = max(2, min(dim, nz_rows - 1))
        try:
            svd = TruncatedSVD(n_components=k_comp, random_state=config.seed)
            U = svd.fit_transform(C_weighted)   # [V, k_comp]
            S = np.sqrt(np.maximum(svd.singular_values_, 0.0))
            emb = U * S
        except Exception:
            emb = np.zeros((V, dim), dtype=np.float32)
        if emb.shape[1] < dim:
            emb = np.pad(emb, ((0, 0), (0, dim - emb.shape[1])))
        out.append(emb[:V, :dim].astype(np.float32))
    return _align_sequence(out)


# ---------------------------------------------------------------------------
# LapPE direct series (structural PE as embedding, no DGT)
# ---------------------------------------------------------------------------

def lapPE_direct_series(
    snapshots: list[ETGSnapshot],
    config: Config,
) -> np.ndarray:
    """Use raw Laplacian PE as node embeddings, Procrustes-align, return drift series.

    Isolates whether the DGT adds value beyond the structural positional encoding.
    """
    embeddings: list[np.ndarray] = []
    for snap in snapshots:
        lap_pe = snap["lap_pe"]
        if isinstance(lap_pe, Tensor):
            emb = lap_pe.cpu().numpy().astype(np.float32)
        else:
            emb = np.array(lap_pe, dtype=np.float32)
        embeddings.append(emb)
    aligned = _align_sequence(embeddings)
    return embeddings_to_shift_series(aligned)
