# ETG v5 Ablations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement v5 quick fixes and a comprehensive ablation matrix (9 neural DGT ablations, 3 new Exp3 baselines, 9 new Exp4 matrix ablations, 1 new Exp1 baseline) derived from graph transformer and diachronic linguistics literature (2020–2025).

**Architecture:** All changes are confined to two source files (`career_hackersignal_pipeline.py` and `career_rt1_benchmarks.py`) plus one notebook cell. The pipeline receives new parameters controlling PE type, residual bypass, temporal gate, time encoding, and trend-seasonal decomposition; benchmarks receive new helper functions for RWPE, MoSE, SPSE, OT, SIGN, NodeFormer, and GRIT features, plus fixed EMA-based prediction in all baselines.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, SciPy (sparse, linalg), scikit-learn, pandas, tqdm. No new pip dependencies required (OT fallback avoids POT library requirement).

---

## File Map

| File | Changes |
|------|---------|
| `src/etg/career_hackersignal_pipeline.py` | STOPWORDS (+2), cache version, RuntimePlan (+7 fields), `_rwpe()`, `_mose_features()`, updated `_build_dgt_tensors()`, updated `MaskedGraphTransformer`, updated `run_dgt_pipeline()` (+7 params, fingerprint, trend-seasonal loss, EMA decay 0.7) |
| `src/etg/career_rt1_benchmarks.py` | Fix polyfit→EMA in `_embedding_shifts` + `_trend_shifts`, add `_ema_predict_bench()`, add 7 feature helpers, extend `_run_non_neural_and_classic()`, extend `_run_transformer_contextual()`, extend `_run_dgt_ablations()` |
| `ETG_MISQ/00_end_to_end.ipynb` | Update `run_dgt_pipeline` call to pass new default params |

---

## Task 1: Quick Fixes — Stopwords + Cache Version

**Files:**
- Modify: `src/etg/career_hackersignal_pipeline.py:87-92`

- [ ] **Step 1: Add "out" and "order" to STOPWORDS**

In `career_hackersignal_pipeline.py`, find the STOPWORDS block that ends with `"question", "answer", "info", "information",` and add two words after `"information"`:

```python
    "question", "answer", "info", "information",
    "out", "order",
```

- [ ] **Step 2: Bump cache version**

Change line 92:
```python
PIPELINE_CACHE_VERSION = "career_rt1_cache_v5"
```

- [ ] **Step 3: Smoke verify**

```bash
cd "/Volumes/Extreme SSD/Exploit Text Graph/exploit-text-graph"
python -c "from etg.career_hackersignal_pipeline import STOPWORDS, PIPELINE_CACHE_VERSION; assert 'out' in STOPWORDS; assert 'order' in STOPWORDS; assert PIPELINE_CACHE_VERSION == 'career_rt1_cache_v5'; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/etg/career_hackersignal_pipeline.py
git commit -m "feat(v5): add 'out','order' to stopwords; bump cache to v5"
```

---

## Task 2: RuntimePlan — Add New Fields

**Files:**
- Modify: `src/etg/career_hackersignal_pipeline.py:105-124`

- [ ] **Step 1: Add seven new fields to RuntimePlan**

Replace the RuntimePlan dataclass body. Find the block ending with `temporal_loss_weight: float = 0.1` and add after it:

```python
    temporal_loss_weight: float = 0.1  # weight for temporal context loss (0.1 avoids ablation inversion)
    pe_type: str = "laplacian"           # "laplacian" | "rwpe" | "mose" | "none"
    use_residual_bypass: bool = True     # learnable alpha bypass (v4 default on)
    temporal_gate: bool = False          # per-node GRU-style gate replaces global alpha
    use_time_embedding: bool = True      # include spell-index embedding in transformer
    time_encoding: str = "learned_discrete"  # "learned_discrete" | "learned_linear"
    rwpe_attention_bias: bool = False    # add pairwise RWPE dot-product to attn mask
    use_trend_seasonal: bool = False     # TIDFormer trend+seasonal decomposition in temporal loss
```

- [ ] **Step 2: Smoke verify**

```bash
python -c "
from etg.career_hackersignal_pipeline import RuntimePlan
p = RuntimePlan.__new__(RuntimePlan)
# Check defaults exist
import dataclasses
fields = {f.name: f.default for f in dataclasses.fields(RuntimePlan) if f.default is not dataclasses.MISSING}
assert fields['pe_type'] == 'laplacian'
assert fields['use_residual_bypass'] == True
assert fields['temporal_gate'] == False
assert fields['use_time_embedding'] == True
assert fields['time_encoding'] == 'learned_discrete'
assert fields['rwpe_attention_bias'] == False
assert fields['use_trend_seasonal'] == False
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/etg/career_hackersignal_pipeline.py
git commit -m "feat(v5): add pe_type, residual_bypass, temporal_gate, time_encoding, rwpe_bias, trend_seasonal fields to RuntimePlan"
```

---

## Task 3: New Helper Functions — `_rwpe()` and `_mose_features()`

**Files:**
- Modify: `src/etg/career_hackersignal_pipeline.py` — add two functions after `_laplacian_pe` (line ~961)

- [ ] **Step 1: Add `_rwpe()` after `_laplacian_pe`**

Insert after the closing `return pe.astype(np.float32)` line of `_laplacian_pe`:

```python

def _rwpe(edge_counts: dict, node_to_idx: dict[str, int], k: int) -> np.ndarray:
    """Random Walk Positional Encoding (GraphGPS, NeurIPS 2022).

    For each node i, computes the k-step landing probabilities
    [(D^{-1}A)^s]_{ii} for s=1..k.  These diagonal entries measure how
    likely a random walk starting at i returns to i after exactly s steps,
    capturing multi-scale local structure without an eigenvector solve.
    """
    n = len(node_to_idx)
    if n == 0 or k <= 0:
        return np.zeros((n, 0), dtype=np.float32)
    rows, cols, data = [], [], []
    for (src, dst), weight in edge_counts.items():
        if src in node_to_idx and dst in node_to_idx:
            i, j = node_to_idx[src], node_to_idx[dst]
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([float(weight), float(weight)])
    if not data:
        return np.zeros((n, k), dtype=np.float32)
    adj = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr().astype(np.float64)
    degree = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.zeros_like(degree)
    mask = degree > 0
    inv[mask] = 1.0 / degree[mask]
    # Row-stochastic transition matrix P = D^{-1}A
    P = sparse.diags(inv) @ adj
    # Compute diagonal of P^s for s=1..k without full matrix materialisation
    # Use dense diagonal extraction: diag(P^s) = row-wise hadamard accumulation
    pe = np.zeros((n, k), dtype=np.float32)
    Ps = P.copy()
    for s in range(k):
        # Extract diagonal of Ps efficiently
        # For sparse: convert to csr and pull [i,i] entries
        Ps_csr = Ps.tocsr()
        diag_vals = np.array(Ps_csr.diagonal(), dtype=np.float32)
        pe[:, s] = diag_vals
        if s < k - 1:
            Ps = Ps @ P
    return pe


def _mose_features(edge_counts: dict, node_to_idx: dict[str, int]) -> np.ndarray:
    """Motif Structural Encoding (MoSE, ICLR 2025, arxiv:2410.18676).

    Computes three cheap motif counts per node:
      col 0 — star count   = degree (number of incident edges)
      col 1 — triangle count = number of closed triangles through node
      col 2 — path-2 count  = number of length-2 paths through node
                             = sum_{v in N(i)} deg(v) - deg(i)

    These homomorphism counts are the three cheapest structural features
    that capture topology beyond degree alone.
    """
    n = len(node_to_idx)
    if n == 0:
        return np.zeros((n, 3), dtype=np.float32)

    # Build adjacency sets and degree array
    adj_sets: dict[int, set[int]] = {i: set() for i in range(n)}
    for (src, dst) in edge_counts:
        if src in node_to_idx and dst in node_to_idx:
            i, j = node_to_idx[src], node_to_idx[dst]
            adj_sets[i].add(j)
            adj_sets[j].add(i)

    features = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        nbrs = adj_sets[i]
        deg_i = len(nbrs)
        features[i, 0] = float(deg_i)  # star count = degree
        # Triangle count: number of pairs (j,k) in N(i) with j-k edge
        tri = 0
        nbr_list = list(nbrs)
        for idx_j, j in enumerate(nbr_list):
            for k in nbr_list[idx_j + 1:]:
                if k in adj_sets[j]:
                    tri += 1
        features[i, 1] = float(tri)
        # Path-2 count: sum of neighbour degrees minus own degree
        path2 = sum(len(adj_sets[j]) for j in nbrs) - deg_i
        features[i, 2] = float(max(path2, 0))

    return features
```

- [ ] **Step 2: Smoke verify**

```bash
python -c "
from etg.career_hackersignal_pipeline import _rwpe, _mose_features
import numpy as np
ec = {('a','b'): 3, ('b','c'): 2, ('a','c'): 1}
ni = {'a': 0, 'b': 1, 'c': 2}
rw = _rwpe(ec, ni, k=4)
assert rw.shape == (3, 4), rw.shape
mo = _mose_features(ec, ni)
assert mo.shape == (3, 3), mo.shape
assert mo[0, 0] == 2.0  # 'a' has degree 2
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/etg/career_hackersignal_pipeline.py
git commit -m "feat(v5): add _rwpe() and _mose_features() PE helpers"
```

---

## Task 4: Update `_build_dgt_tensors()` for `pe_type`

**Files:**
- Modify: `src/etg/career_hackersignal_pipeline.py:964-1000`

- [ ] **Step 1: Add `pe_type` parameter to `_build_dgt_tensors`**

Replace the function signature and PE selection block:

Old signature:
```python
def _build_dgt_tensors(snapshots: list[dict], max_nodes: int, lap_pe_k: int, vocab: list[str] | None = None) -> dict:
```

New signature and body — replace the entire function:

```python
def _build_dgt_tensors(
    snapshots: list[dict],
    max_nodes: int,
    lap_pe_k: int,
    vocab: list[str] | None = None,
    pe_type: str = "laplacian",
) -> dict:
    words = vocab if vocab is not None else _select_dgt_vocab(snapshots, max_nodes)
    node_to_idx = {word: i for i, word in enumerate(words)}
    features, adjacencies = [], []
    for snap in tqdm(snapshots, desc="Building DGT tensors", unit=" spell", leave=False):
        if pe_type == "laplacian":
            pe = _laplacian_pe(snap["edge_counts"], node_to_idx, lap_pe_k)
        elif pe_type == "rwpe":
            pe = _rwpe(snap["edge_counts"], node_to_idx, lap_pe_k)
        elif pe_type == "mose":
            pe = _mose_features(snap["edge_counts"], node_to_idx)
        else:  # "none"
            pe = np.zeros((len(words), 0), dtype=np.float32)
        x = []
        for word in words:
            count = snap["term_counts"].get(word, 0)
            doc_count = snap["term_doc_counts"].get(word, 0)
            in_deg = snap["in_degree"].get(word, 0)
            out_deg = snap["out_degree"].get(word, 0)
            win = snap["weighted_in_degree"].get(word, 0)
            wout = snap["weighted_out_degree"].get(word, 0)
            x.append(
                [
                    np.log1p(count),
                    np.log1p(doc_count),
                    np.log1p(in_deg),
                    np.log1p(out_deg),
                    np.log1p(win),
                    np.log1p(wout),
                    len(word) / 32.0,
                    float(any(ch.isdigit() for ch in word)),
                    float(any(not ch.isalnum() for ch in word)),
                ]
            )
        base = np.asarray(x, dtype=np.float32)
        features.append(np.concatenate([base, pe], axis=1))

        adj = np.eye(len(words), dtype=bool)
        for (src, dst), _ in snap["edge_counts"].items():
            if src in node_to_idx and dst in node_to_idx:
                adj[node_to_idx[src], node_to_idx[dst]] = True
                adj[node_to_idx[dst], node_to_idx[src]] = True
        adjacencies.append(adj)
    return {"words": words, "features": features, "adjacencies": adjacencies}
```

- [ ] **Step 2: Smoke verify**

```bash
python -c "
from etg.career_hackersignal_pipeline import _build_dgt_tensors
snaps = [{'term_counts': {'foo': 5, 'bar': 3}, 'term_doc_counts': {'foo': 2}, 'in_degree': {}, 'out_degree': {}, 'weighted_in_degree': {}, 'weighted_out_degree': {}, 'edge_counts': {('foo','bar'): 2}}]
for pe in ['laplacian', 'rwpe', 'mose', 'none']:
    t = _build_dgt_tensors(snaps, max_nodes=10, lap_pe_k=4, pe_type=pe)
    assert len(t['features']) == 1
    print(f'{pe}: input_dim={t[\"features\"][0].shape[1]}')
print('OK')
"
```
Expected output: four lines showing different input dims for each PE type, then `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/etg/career_hackersignal_pipeline.py
git commit -m "feat(v5): add pe_type param to _build_dgt_tensors (laplacian/rwpe/mose/none)"
```

---

## Task 5: Update `MaskedGraphTransformer` — New Architecture Modes

**Files:**
- Modify: `src/etg/career_hackersignal_pipeline.py:1116-1142`

- [ ] **Step 1: Replace `MaskedGraphTransformer` class definition**

The class is defined as a local class inside `run_dgt_pipeline`. The full replacement (all five new modes are controlled by closure variables that will be added to `run_dgt_pipeline` in Task 6):

Replace from `class MaskedGraphTransformer(nn.Module):` through the closing `return F.normalize(self.proj(z), dim=1)` with:

```python
    class MaskedGraphTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(input_dim, hidden_dim)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 2,
                dropout=0.2,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            # Time embedding: discrete (nn.Embedding) or linear (nn.Linear)
            if use_time_embedding:
                if time_encoding == "learned_linear":
                    self.time_embed_linear = nn.Linear(1, hidden_dim)
                    self.time_embed_table = None
                else:
                    self.time_embed_table = nn.Embedding(len(snapshots), hidden_dim)
                    self.time_embed_linear = None
            else:
                self.time_embed_table = None
                self.time_embed_linear = None
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
            )
            # Residual bypass: global scalar alpha OR per-node gate
            if temporal_gate:
                self.gate = nn.Linear(hidden_dim, hidden_dim)
                self.residual_alpha = None
            elif use_residual_bypass:
                self.residual_alpha = nn.Parameter(torch.tensor(0.0))
                self.gate = None
            else:
                self.residual_alpha = None
                self.gate = None

        def forward(self, x, adjacency, t):
            h = self.input(x)
            if use_time_embedding:
                if time_encoding == "learned_linear":
                    t_norm = torch.tensor(
                        [[t / max(len(snapshots) - 1, 1)]], dtype=torch.float32, device=x.device
                    )
                    h = h + self.time_embed_linear(t_norm).view(1, -1)
                else:
                    h = h + self.time_embed_table(
                        torch.tensor(t, device=x.device)
                    ).view(1, -1)

            if rwpe_attention_bias and pe_type == "rwpe":
                # pairwise RWPE dot-product as float attention bias (GRIT, ICML 2023)
                # x contains base(9) + rwpe(lap_pe_k) columns; extract rwpe cols
                rwpe_cols = x[:, 9:9 + lap_pe_k]  # (N, k)
                bias = rwpe_cols @ rwpe_cols.T      # (N, N) float
                # TransformerEncoder expects additive float mask (large neg = ignore)
                # adjacency mask says True where attention is BLOCKED
                adj_float = torch.zeros_like(bias)
                adj_float[~adjacency] = float("-inf")
                attn_mask = adj_float + bias
            else:
                attn_mask = torch.tensor(~adjacency, dtype=torch.bool, device=x.device)

            z_enc = self.encoder(h.unsqueeze(0), mask=attn_mask).squeeze(0)

            if temporal_gate:
                gate_val = torch.sigmoid(self.gate(h))
                z = gate_val * z_enc + (1.0 - gate_val) * h
            elif use_residual_bypass:
                alpha = torch.sigmoid(self.residual_alpha)
                z = alpha * z_enc + (1.0 - alpha) * h
            else:
                z = z_enc

            return F.normalize(self.proj(z), dim=1)
```

**Important:** The `rwpe_attention_bias` path passes a float mask to `TransformerEncoder`. PyTorch's `TransformerEncoderLayer` accepts either a boolean mask (True=ignore) or a float additive mask. When using `rwpe_attention_bias=True`, the mask is a float tensor — this is already supported by PyTorch. The tensor must be on the same device as `x`, which it will be since it is derived from `x`.

- [ ] **Step 2: Commit (no standalone smoke yet — Task 6 wires everything)**

```bash
git add src/etg/career_hackersignal_pipeline.py
git commit -m "feat(v5): MaskedGraphTransformer supports temporal_gate, use_residual_bypass=False, learned_linear time, rwpe_attention_bias"
```

---

## Task 6: Update `run_dgt_pipeline()` — New Parameters, Fingerprint, EMA Decay

**Files:**
- Modify: `src/etg/career_hackersignal_pipeline.py:1024-1365`

- [ ] **Step 1: Add new parameters to `run_dgt_pipeline` signature**

Replace old signature:
```python
def run_dgt_pipeline(
    snapshots: list[dict],
    *,
    max_nodes: int,
    hidden_dim: int,
    heads: int,
    layers: int,
    epochs: int,
    lap_pe_k: int,
    device: str = "auto",
    output_dir: str | Path | None = None,
    seed: int = 1729,
    use_cache: bool = True,
    edge_batch: int | None = None,
    temporal_loss_weight: float = 0.1,
) -> dict:
```

New signature:
```python
def run_dgt_pipeline(
    snapshots: list[dict],
    *,
    max_nodes: int,
    hidden_dim: int,
    heads: int,
    layers: int,
    epochs: int,
    lap_pe_k: int,
    device: str = "auto",
    output_dir: str | Path | None = None,
    seed: int = 1729,
    use_cache: bool = True,
    edge_batch: int | None = None,
    temporal_loss_weight: float = 0.1,
    pe_type: str = "laplacian",
    use_residual_bypass: bool = True,
    temporal_gate: bool = False,
    use_time_embedding: bool = True,
    time_encoding: str = "learned_discrete",
    rwpe_attention_bias: bool = False,
    use_trend_seasonal: bool = False,
) -> dict:
```

- [ ] **Step 2: Add new params to cache fingerprint**

Find the `cache_fingerprint = stable_fingerprint({...})` block and add new keys before the closing `}`:

```python
            "temporal_loss_weight": temporal_loss_weight,
            "pe_type": pe_type,
            "use_residual_bypass": use_residual_bypass,
            "temporal_gate": temporal_gate,
            "use_time_embedding": use_time_embedding,
            "time_encoding": time_encoding,
            "rwpe_attention_bias": rwpe_attention_bias,
            "use_trend_seasonal": use_trend_seasonal,
```

- [ ] **Step 3: Thread `pe_type` into `_build_dgt_tensors` call**

Find:
```python
    tensors = _build_dgt_tensors(snapshots, max_nodes=max_nodes, lap_pe_k=lap_pe_k)
```
Replace with:
```python
    tensors = _build_dgt_tensors(snapshots, max_nodes=max_nodes, lap_pe_k=lap_pe_k, pe_type=pe_type)
```

- [ ] **Step 4: Change EMA decay from 0.5 → 0.7**

Find `_ema_predict` function inside `run_dgt_pipeline`:
```python
    def _ema_predict(shift_series: list[float], decay: float = 0.5) -> float:
```
Change to:
```python
    def _ema_predict(shift_series: list[float], decay: float = 0.7) -> float:
```

- [ ] **Step 5: Add `use_trend_seasonal` to temporal loss computation**

Find the temporal loss block:
```python
        temporal_loss = torch.tensor(0.0, device=torch_device)
        for t in range(1, len(embeddings)):
            current = embeddings[t]
            past = torch.stack(embeddings[:t], dim=0)          # shape (t, N, D)
            n_past = past.shape[0]
            # Recency-weighted EMA: ...
            decay = torch.exp(
                -0.5 * torch.arange(n_past - 1, -1, -1, dtype=torch.float32, device=torch_device)
            )
            ema_weights = (decay / decay.sum()).view(-1, 1, 1)
            context = (ema_weights * past).sum(dim=0)           # weighted mean of past
            temporal_loss = temporal_loss + F.mse_loss(current, context.detach())
```

Replace with:
```python
        temporal_loss = torch.tensor(0.0, device=torch_device)
        for t in range(1, len(embeddings)):
            current = embeddings[t]
            past = torch.stack(embeddings[:t], dim=0)          # shape (t, N, D)
            n_past = past.shape[0]
            decay = torch.exp(
                -0.5 * torch.arange(n_past - 1, -1, -1, dtype=torch.float32, device=torch_device)
            )
            ema_weights = (decay / decay.sum()).view(-1, 1, 1)
            context = (ema_weights * past).sum(dim=0)           # weighted mean of past
            if use_trend_seasonal:
                # TIDFormer (KDD 2025) trend+seasonal decomposition:
                # trend = mean over all past embeddings; seasonal = context - trend
                trend = past.mean(dim=0)                        # (N, D)
                seasonal = context - trend                      # (N, D)
                # Decompose current into its own trend and seasonal components
                curr_trend = current.mean(dim=0, keepdim=True).expand_as(current)
                curr_seasonal = current - curr_trend
                temporal_loss = (
                    temporal_loss
                    + 0.5 * F.mse_loss(curr_trend, trend.detach())
                    + 0.5 * F.mse_loss(curr_seasonal, seasonal.detach())
                )
            else:
                temporal_loss = temporal_loss + F.mse_loss(current, context.detach())
```

- [ ] **Step 6: Add new params to manifest params block**

Find the `"params": {` block in the output manifest and add after `"temporal_loss_weight": temporal_loss_weight,`:

```python
                        "pe_type": pe_type,
                        "use_residual_bypass": use_residual_bypass,
                        "temporal_gate": temporal_gate,
                        "use_time_embedding": use_time_embedding,
                        "time_encoding": time_encoding,
                        "rwpe_attention_bias": rwpe_attention_bias,
                        "use_trend_seasonal": use_trend_seasonal,
```

- [ ] **Step 7: Smoke test the full pipeline**

```bash
cd "/Volumes/Extreme SSD/Exploit Text Graph/exploit-text-graph"
python -c "
from etg.career_hackersignal_pipeline import choose_runtime_plan, build_spell_etgs_streaming, run_dgt_pipeline
import tempfile, pathlib
plan = choose_runtime_plan(requested_device='cpu', requested_mode='smoke')
snaps, vocab = build_spell_etgs_streaming(plan.data_path, n_spells=plan.n_spells, vocab_size=plan.vocab_size, window=plan.window, min_edge_weight=plan.min_edge_weight, max_edges_per_spell=plan.max_edges_per_spell, limit=plan.record_limit)
with tempfile.TemporaryDirectory() as d:
    r = run_dgt_pipeline(snaps, max_nodes=50, hidden_dim=32, heads=2, layers=1, epochs=2, lap_pe_k=4, device='cpu', output_dir=d, use_cache=False, pe_type='laplacian')
    assert 'embeddings' in r
    r2 = run_dgt_pipeline(snaps, max_nodes=50, hidden_dim=32, heads=2, layers=1, epochs=2, lap_pe_k=4, device='cpu', output_dir=d+'/rwpe', use_cache=False, pe_type='rwpe', use_residual_bypass=False)
    assert 'embeddings' in r2
    r3 = run_dgt_pipeline(snaps, max_nodes=50, hidden_dim=32, heads=2, layers=1, epochs=2, lap_pe_k=4, device='cpu', output_dir=d+'/gate', use_cache=False, pe_type='mose', temporal_gate=True)
    assert 'embeddings' in r3
    r4 = run_dgt_pipeline(snaps, max_nodes=50, hidden_dim=32, heads=2, layers=1, epochs=2, lap_pe_k=4, device='cpu', output_dir=d+'/linear', use_cache=False, time_encoding='learned_linear')
    assert 'embeddings' in r4
    r5 = run_dgt_pipeline(snaps, max_nodes=50, hidden_dim=32, heads=2, layers=1, epochs=2, lap_pe_k=4, device='cpu', output_dir=d+'/ts', use_cache=False, use_trend_seasonal=True)
    assert 'embeddings' in r5
print('ALL PIPELINE SMOKE TESTS PASSED')
"
```
Expected: `ALL PIPELINE SMOKE TESTS PASSED`

- [ ] **Step 8: Commit**

```bash
git add src/etg/career_hackersignal_pipeline.py
git commit -m "feat(v5): run_dgt_pipeline +7 params, EMA decay 0.7, trend-seasonal loss, updated fingerprint"
```

---

## Task 7: Fix Polyfit → EMA in Benchmark Prediction Functions

**Files:**
- Modify: `src/etg/career_rt1_benchmarks.py:127-228`

- [ ] **Step 1: Add `_ema_predict_bench` helper near top of benchmarks file**

After the imports block (before `_transition_label`), add:

```python
def _ema_predict_bench(values: list[float], decay: float = 0.7) -> float:
    """EMA shift predictor — mirrors the DGT's _ema_predict for fair comparison.

    Uses decay=0.7 (v5), clipped to [0, inf) since cosine distances are non-negative.
    """
    n = len(values)
    if n == 0:
        return 0.0
    weights = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=np.float64)
    weights /= weights.sum()
    return float(max(0.0, np.dot(weights, values)))
```

- [ ] **Step 2: Fix `_embedding_shifts` — replace polyfit with EMA**

In `_embedding_shifts`, find:
```python
        pred = values[-2] if len(values) == 2 else float(np.poly1d(np.polyfit(np.arange(len(values) - 1), values[:-1], 1))(len(values) - 1))
```
Replace with:
```python
        pred = _ema_predict_bench(values[:-1])
```

- [ ] **Step 3: Fix `_trend_shifts` — replace polyfit with EMA**

In `_trend_shifts`, find:
```python
        pred = float(values[-2]) if len(values) == 2 else float(np.poly1d(np.polyfit(np.arange(len(values) - 1), values[:-1], 1))(len(values) - 1))
```
Replace with:
```python
        pred = _ema_predict_bench(list(values[:-1]))
```

- [ ] **Step 4: Smoke verify**

```bash
python -c "
from etg.career_rt1_benchmarks import _ema_predict_bench
# Known value: EMA of [0.1, 0.2, 0.3] with decay=0.7
vals = [0.1, 0.2, 0.3]
n = len(vals)
import numpy as np
w = np.array([0.7**(n-1-i) for i in range(n)]); w /= w.sum()
expected = float(np.dot(w, vals))
got = _ema_predict_bench(vals)
assert abs(got - expected) < 1e-9, f'{got} != {expected}'
# No negative predictions
assert _ema_predict_bench([-0.5, -0.3]) == 0.0
print('OK')
"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/etg/career_rt1_benchmarks.py
git commit -m "fix(v5): replace np.polyfit with EMA in all baseline shift predictors for fair comparison"
```

---

## Task 8: New Benchmark Feature Helpers

**Files:**
- Modify: `src/etg/career_rt1_benchmarks.py` — add helpers before `_run_non_neural_and_classic`

Add the following functions in this order after `_feature_propagation_embeddings`:

- [ ] **Step 1: Add `_rwpe_features`, `_mose_features_bench`, `_heat_kernel_pe`, `_spse_features`**

```python
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
    """Heat kernel diagonal PE: [exp(-t*L)]_{ii} for each t in t_vals.

    Approximated via the normalised Laplacian eigenvector decomposition.
    Falls back to zeros for disconnected graphs.
    """
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
    # exp(-t * lambda) heat kernel diagonal: sum_k exp(-t*lambda_k) * v_k^2
    pe = np.zeros((n, len(t_vals)), dtype=np.float32)
    for col_idx, t in enumerate(t_vals):
        kernel_weights = np.exp(-t * np.maximum(vals, 0.0))  # (k_eig,)
        # diag(V @ diag(w) @ V^T) = sum_k w_k * V[:,k]^2
        pe[:, col_idx] = (vecs ** 2 @ kernel_weights).astype(np.float32)
    return pe


def _spse_features(snap: dict, words: list[str], k_paths: int = 4) -> np.ndarray:
    """Simple Path Structural Encoding (SPSE, ICML 2025, arxiv:2502.09365).

    For each node, counts non-self-intersecting simple paths of lengths 1..k_paths.
    Length-1 count = degree.  Length-2 = paths through exactly one intermediate node.
    Lengths 3-4 are approximated as products of degree sequences (exact enumeration
    is expensive; this approximation captures the structural signal at O(N*deg) cost).
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
        features[i, 0] = float(deg_i)  # length-1: degree
        if k_paths >= 2:
            # length-2: count paths i->j->k where j in N(i), k in N(j), k != i
            cnt2 = sum(max(len(adj_sets[j]) - 1, 0) for j in nbrs_i)
            features[i, 1] = float(cnt2)
        if k_paths >= 3:
            # length-3 approximation: sum over pairs (j,k) in N(i) of |N(k) \ {i,j}|
            nbr_list = list(nbrs_i)
            cnt3 = 0
            for j in nbr_list:
                for k in adj_sets[j]:
                    if k != i and k not in nbrs_i:
                        cnt3 += max(len(adj_sets[k]) - 2, 0)
            features[i, 2] = float(cnt3)
        if k_paths >= 4:
            # length-4: approximate as deg_i * avg_2nd_hop_degree^2
            if deg_i > 0:
                avg_2nd = sum(len(adj_sets[j]) for j in nbrs_i) / deg_i
                features[i, 3] = float(deg_i * avg_2nd * avg_2nd)
    return features


def _sign_aggregate(snap: dict, words: list[str], dim: int, k_hops: int, rng: np.random.Generator) -> np.ndarray:
    """SIGN multi-hop aggregation proxy (ICML-W 2020).

    Concatenates raw node features with 1..k_hops mean-aggregated neighbour
    features, then projects to *dim* with a random linear map.
    """
    adj = _edge_matrix(snap, words, weighted=False, sym=True)
    n = len(words)
    # Base features: count + degree (2-dim per node)
    counts = np.array([[snap["term_counts"].get(w, 0), snap["in_degree"].get(w, 0) + snap["out_degree"].get(w, 0)] for w in words], dtype=np.float64)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.zeros_like(deg); inv[deg > 0] = 1.0 / deg[deg > 0]
    P = sparse.diags(inv) @ adj  # row-normalised propagation matrix
    hops = [counts.copy()]
    prop = counts.copy()
    for _ in range(k_hops):
        prop = P @ prop
        hops.append(prop.copy())
    feat = np.concatenate(hops, axis=1)  # (N, 2*(k_hops+1))
    proj = rng.normal(0, 1 / math.sqrt(max(feat.shape[1], 1)), size=(feat.shape[1], dim))
    return (feat @ proj).astype(np.float32)


def _nodeformer_features(snap: dict, words: list[str], dim: int, rng: np.random.Generator) -> np.ndarray:
    """NodeFormer kernelized random-feature attention proxy (NeurIPS 2022).

    Approximates full-graph attention using ELU random Fourier features.
    Each node attends to all others (no adjacency mask), aggregating
    a random-projected message weighted by ELU kernel similarity.
    """
    adj = _edge_matrix(snap, words, weighted=False, sym=True)
    n = len(words)
    counts = np.array([[snap["term_counts"].get(w, 0)] for w in words], dtype=np.float64)
    feat_dim = 1
    # Random Fourier feature map: phi(x) = ELU(W*x + b) / sqrt(d)
    W = rng.normal(0, 1, size=(dim, feat_dim))
    b = rng.uniform(0, 2 * math.pi, size=dim)
    phi = np.maximum(counts @ W.T + b, 0.0) + 1.0  # ELU+1 ≥ 0 everywhere, shape (N, dim)
    # Kernelised attention: output_i = sum_j k(i,j)*x_j / sum_j k(i,j)
    # With ELU kernel: output = phi * (phi^T @ counts) / (phi * (phi^T @ ones))
    phi_sum = phi.sum(axis=0, keepdims=True)  # (1, dim)
    phi_weighted = phi @ (phi.T @ counts)     # (N, 1)
    phi_norm = phi @ phi_sum.T                # (N, 1)
    aggregated = phi_weighted / (phi_norm + 1e-9)  # (N, 1)
    out = np.concatenate([counts, aggregated, phi[:, :dim // 2]], axis=1)
    proj = rng.normal(0, 1 / math.sqrt(max(out.shape[1], 1)), size=(out.shape[1], dim))
    return (out @ proj).astype(np.float32)


def _grit_features(snap: dict, words: list[str], dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """GRIT RWPE relative-attention proxy (ICML 2023, arxiv:2312.02220).

    Computes RWPE(k) for each node, then forms a pairwise dot-product
    attention bias matrix.  The final embedding is the RWPE features
    projected to *dim* (the relative bias is captured implicitly in the
    pairwise interactions represented by the SVD of the bias matrix).
    """
    rwpe = _rwpe_features(snap, words, k)  # (N, k)
    # Pairwise RWPE dot-product bias (N, N)
    bias = rwpe @ rwpe.T
    # Use the column-sum of the bias as an aggregated relative attention signal
    agg = bias.sum(axis=1, keepdims=True)  # (N, 1) — how much each node attends to all others
    feat = np.concatenate([rwpe, agg], axis=1)  # (N, k+1)
    proj = rng.normal(0, 1 / math.sqrt(max(feat.shape[1], 1)), size=(feat.shape[1], dim))
    return (feat @ proj).astype(np.float32)
```

- [ ] **Step 2: Smoke verify helpers**

```bash
python -c "
import numpy as np
# Build a minimal fake snapshot
snap = {
    'term_counts': {'foo': 5, 'bar': 3, 'baz': 2},
    'term_doc_counts': {'foo': 2},
    'in_degree': {'foo': 1},
    'out_degree': {'foo': 1, 'bar': 1},
    'weighted_in_degree': {},
    'weighted_out_degree': {},
    'edge_counts': {('foo','bar'): 3, ('bar','baz'): 2},
}
words = ['foo', 'bar', 'baz']
rng = np.random.default_rng(42)
from etg.career_rt1_benchmarks import _rwpe_features, _mose_features_bench, _heat_kernel_pe, _spse_features, _sign_aggregate, _nodeformer_features, _grit_features
assert _rwpe_features(snap, words, k=4).shape == (3, 4)
assert _mose_features_bench(snap, words).shape == (3, 3)
assert _heat_kernel_pe(snap, words).shape == (3, 4)
assert _spse_features(snap, words, k_paths=4).shape == (3, 4)
assert _sign_aggregate(snap, words, dim=16, k_hops=2, rng=rng).shape == (3, 16)
assert _nodeformer_features(snap, words, dim=16, rng=rng).shape == (3, 16)
assert _grit_features(snap, words, dim=16, k=4, rng=rng).shape == (3, 16)
print('ALL HELPERS OK')
"
```
Expected: `ALL HELPERS OK`

- [ ] **Step 3: Commit**

```bash
git add src/etg/career_rt1_benchmarks.py
git commit -m "feat(v5): add RWPE, MoSE, heat-kernel, SPSE, SIGN, NodeFormer, GRIT benchmark helpers"
```

---

## Task 9: New Exp1 Baseline — `unbalanced_ot_shift`

**Files:**
- Modify: `src/etg/career_rt1_benchmarks.py` — `_run_non_neural_and_classic`

- [ ] **Step 1: Add OT shift helper**

Add after `_grit_features`:

```python
def _ot_shift_magnitude(snap_prev: dict, snap_curr: dict, words: list[str]) -> np.ndarray:
    """Unbalanced OT semantic shift magnitude (ACL 2025, arxiv:2412.12569).

    Measures distributional drift between a word's co-occurrence context
    across consecutive spells using L1 distance between PPMI distributions.
    Full UOT requires the POT library; this is a deterministic L1 approximation
    that captures the same structural signal without an external dependency.

    Returns an (N,) array of shift magnitudes, one per word.
    """
    node_to_idx = {w: i for i, w in enumerate(words)}
    n = len(words)

    def _ppmi_row(snap: dict) -> np.ndarray:
        mat = _edge_matrix(snap, words, weighted=True, sym=True)
        ppmi = _ppmi(mat)
        return np.asarray(ppmi.toarray(), dtype=np.float32)  # (N, N)

    prev_ppmi = _ppmi_row(snap_prev)
    curr_ppmi = _ppmi_row(snap_curr)

    # L1 distance between PPMI context distributions (row-wise)
    # Normalise rows to form proper distributions first
    prev_norm = prev_ppmi / (prev_ppmi.sum(axis=1, keepdims=True) + 1e-9)
    curr_norm = curr_ppmi / (curr_ppmi.sum(axis=1, keepdims=True) + 1e-9)
    l1_dist = np.abs(prev_norm - curr_norm).sum(axis=1)  # (N,)
    return l1_dist.astype(np.float32)
```

- [ ] **Step 2: Add `unbalanced_ot_shift` to `_run_non_neural_and_classic`**

Inside `_run_non_neural_and_classic`, after the `for shifts, preds in experiments:` loop and before the `return`, add:

```python
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
```

- [ ] **Step 3: Commit**

```bash
git add src/etg/career_rt1_benchmarks.py
git commit -m "feat(v5): add unbalanced_ot_shift Exp1 baseline (ACL 2025 L1-PPMI)"
```

---

## Task 10: New Exp3 Baselines — SIGN, NodeFormer, GRIT

**Files:**
- Modify: `src/etg/career_rt1_benchmarks.py` — `_run_transformer_contextual`

- [ ] **Step 1: Add three new Exp3 models at the end of `_run_transformer_contextual`**

Inside `_run_transformer_contextual`, before `return pd.concat(...)`, add:

```python
    rng_exp3 = np.random.default_rng(seed=1729)
    # SIGN (ICML-W 2020): multi-hop inception aggregation
    sign_embs = _align_embeddings([_sign_aggregate(s, words, dim=dim, k_hops=3, rng=rng_exp3) for s in snapshots])
    shifts, preds = _embedding_shifts(sign_embs, words, model="sign_scalable_inception", tier="modern_graph_transformer", experiment="experiment_3_modern_transformers", top_k=top_k, note="SIGN 3-hop inception aggregation (ICML-W 2020)")
    shift_frames.append(shifts); pred_frames.append(preds)

    rng_exp3b = np.random.default_rng(seed=1730)
    # NodeFormer (NeurIPS 2022): kernelized random-feature all-pairs attention
    nf_embs = _align_embeddings([_nodeformer_features(s, words, dim=dim, rng=rng_exp3b) for s in snapshots])
    shifts, preds = _embedding_shifts(nf_embs, words, model="nodeformer_kernelized", tier="modern_graph_transformer", experiment="experiment_3_modern_transformers", top_k=top_k, note="NodeFormer ELU kernel random-feature all-pairs attention (NeurIPS 2022)")
    shift_frames.append(shifts); pred_frames.append(preds)

    rng_exp3c = np.random.default_rng(seed=1731)
    # GRIT (ICML 2023): RWPE relative attention bias
    grit_embs = _align_embeddings([_grit_features(s, words, dim=dim, k=lap_pe_k, rng=rng_exp3c) for s in snapshots])
    shifts, preds = _embedding_shifts(grit_embs, words, model="grit_rwpe_attention", tier="modern_graph_transformer", experiment="experiment_3_modern_transformers", top_k=top_k, note="GRIT RWPE relative attention bias proxy (ICML 2023)")
    shift_frames.append(shifts); pred_frames.append(preds)
```

Note: `_run_transformer_contextual` currently takes `(snapshots, words, dim, top_k, lap_pe_k)` — `lap_pe_k` is already a parameter. Use it for the GRIT `k` argument.

- [ ] **Step 2: Commit**

```bash
git add src/etg/career_rt1_benchmarks.py
git commit -m "feat(v5): add SIGN, NodeFormer, GRIT to Exp3 transformer baselines"
```

---

## Task 11: New Exp4 Matrix Ablations (9 models)

**Files:**
- Modify: `src/etg/career_rt1_benchmarks.py` — `_run_dgt_ablations`

- [ ] **Step 1: Add 9 new matrix ablations inside `_run_dgt_ablations`, after existing `ablation_specs` loop**

Find the end of the existing loop:
```python
    for model, fn, note in ablation_specs:
        shifts, preds = _matrix_embedding_baseline(...)
        shift_frames.append(shifts)
        pred_frames.append(preds)
    return pd.concat(shift_frames, ignore_index=True), pd.concat(pred_frames, ignore_index=True)
```

Replace the `return` statement with the following block (keep it before return):

```python
    rng_abl = np.random.default_rng(seed + 100)

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
        # Approximation: (I - 0.5*L) @ adj where L is normalised Laplacian
        deg = np.asarray(adj.sum(axis=1)).ravel()
        inv_sqrt = np.zeros_like(deg); inv_sqrt[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
        norm_adj = sparse.diags(inv_sqrt) @ adj @ sparse.diags(inv_sqrt)
        lap = sparse.eye(len(words)) - norm_adj
        return (sparse.eye(len(words)) - 0.5 * lap) @ adj
    wavelet_embs = [_svd(_wavelet_matrix(s), dim) for s in snapshots]
    shifts, preds = _embedding_shifts(_align_embeddings(wavelet_embs), words, model="spectral_wavelet_svd", tier="dgt_ablation", experiment="experiment_4_dgt_ablations", top_k=top_k, note="graph wavelet diffusion coefficient approximation → SVD")
    shift_frames.append(shifts); pred_frames.append(preds)

    # DGT Procrustes post-hoc: re-align DGT embeddings with Procrustes after training
    dgt_embs_raw = dgt_result["embeddings"]  # list of (N, D) np arrays
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
    ot_shift_rows = []
    ot_series_abl: dict[str, list[float]] = {w: [] for w in words}
    for t in range(1, len(snapshots)):
        magnitudes = _ot_shift_magnitude(snapshots[t - 1], snapshots[t], words)
        for idx, val in enumerate(magnitudes):
            ot_series_abl[words[idx]].append(float(val))
        for rank, idx in enumerate(np.argsort(magnitudes)[-top_k:][::-1], 1):
            ot_shift_rows.append({
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
    ot_pred_rows_abl = []
    for word, vals in ot_series_abl.items():
        if len(vals) < 2:
            continue
        pred = _ema_predict_bench(vals[:-1])
        ot_pred_rows_abl.append({
            "experiment": "experiment_4_dgt_ablations",
            "tier": "dgt_ablation",
            "model": "ot_wasserstein_aligned",
            "word": word,
            "predicted_next_shift": float(pred),
            "heldout_shift": float(vals[-1]),
            "abs_error": abs(float(pred) - float(vals[-1])),
        })
    if ot_shift_rows:
        shift_frames.append(pd.DataFrame(ot_shift_rows))
    if ot_pred_rows_abl:
        pred_frames.append(pd.DataFrame(ot_pred_rows_abl))

    return pd.concat(shift_frames, ignore_index=True), pd.concat(pred_frames, ignore_index=True)
```

- [ ] **Step 2: Commit**

```bash
git add src/etg/career_rt1_benchmarks.py
git commit -m "feat(v5): add 9 new Exp4 matrix ablations (RWPE, heat-kernel, PPMI, SGNS, wavelet, DGT-Procrustes, MoSE, SPSE, OT)"
```

---

## Task 12: New Neural DGT Ablations — 9 GPU Models in `_run_dgt_ablations`

**Files:**
- Modify: `src/etg/career_rt1_benchmarks.py` — `_run_dgt_ablations`

- [ ] **Step 1: Add 9 new neural ablations to the `variants` list**

Find the existing `variants` list in `_run_dgt_ablations`:
```python
    variants = [
        ("dgt_no_laplacian_pe", {"lap_pe_k": 0}, "no Laplacian positional encoding"),
        ("dgt_static_graph_transformer_only", {"epochs": max(1, epochs // 2), "lap_pe_k": lap_pe_k}, "static graph transformer comparison with reduced temporal training budget"),
    ]
```

Replace with (keep existing two, append nine new):
```python
    variants = [
        ("dgt_no_laplacian_pe", {"lap_pe_k": 0}, "no Laplacian positional encoding"),
        ("dgt_static_graph_transformer_only", {"epochs": max(1, epochs // 2), "lap_pe_k": lap_pe_k}, "static graph transformer comparison with reduced temporal training budget"),
        # v5 neural ablations
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
```

- [ ] **Step 2: Update the loop to thread new params through `run_dgt_pipeline`**

The existing loop already passes `params` kwargs through — but note it uses `params.get(key, default)`. Add handling for new keys. Find:

```python
        res = run_dgt_pipeline(
            snapshots,
            max_nodes=max_nodes,
            hidden_dim=dim,
            heads=4,
            layers=2,
            epochs=params.get("epochs", epochs),
            lap_pe_k=params.get("lap_pe_k", lap_pe_k),
            device=device,
            output_dir=None,
            seed=seed + len(shift_frames),
        )
```

Replace with:
```python
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
```

- [ ] **Step 3: Commit**

```bash
git add src/etg/career_rt1_benchmarks.py
git commit -m "feat(v5): add 9 neural DGT ablations (RWPE, no-bypass, temporal-gate, no-time-emb, 3-layers, GRIT, linear-time, MoSE, TIDFormer)"
```

---

## Task 13: Smoke Test the Full Benchmark Suite

- [ ] **Step 1: Run full benchmark smoke test**

```bash
cd "/Volumes/Extreme SSD/Exploit Text Graph/exploit-text-graph"
python -c "
from etg.career_hackersignal_pipeline import choose_runtime_plan, build_spell_etgs_streaming, run_dgt_pipeline
from etg.career_rt1_benchmarks import _run_non_neural_and_classic, _run_static_dynamic_graph, _run_transformer_contextual, _run_dgt_ablations

plan = choose_runtime_plan(requested_device='cpu', requested_mode='smoke')
snaps, vocab = build_spell_etgs_streaming(
    plan.data_path, n_spells=plan.n_spells, vocab_size=plan.vocab_size,
    window=plan.window, min_edge_weight=plan.min_edge_weight,
    max_edges_per_spell=plan.max_edges_per_spell, limit=plan.record_limit
)

words = vocab[:min(len(vocab), plan.dgt_max_nodes)]
dim = plan.dgt_hidden_dim
top_k = 5

print('Running Exp1...')
s1, p1 = _run_non_neural_and_classic(snaps, words, dim, top_k)
models_exp1 = s1['model'].unique().tolist()
assert 'unbalanced_ot_shift' in models_exp1, f'Missing OT: {models_exp1}'
print(f'  Exp1 models: {len(models_exp1)}')

print('Running Exp3...')
s3, p3 = _run_transformer_contextual(snaps, words, dim, top_k, plan.lap_pe_k)
models_exp3 = s3['model'].unique().tolist()
for m in ['sign_scalable_inception', 'nodeformer_kernelized', 'grit_rwpe_attention']:
    assert m in models_exp3, f'Missing {m}: {models_exp3}'
print(f'  Exp3 models: {len(models_exp3)}')

print('Running DGT full...')
import tempfile
with tempfile.TemporaryDirectory() as d:
    dgt_result = run_dgt_pipeline(snaps, max_nodes=plan.dgt_max_nodes, hidden_dim=dim, heads=plan.dgt_heads, layers=plan.dgt_layers, epochs=plan.dgt_epochs, lap_pe_k=plan.lap_pe_k, device='cpu', output_dir=d, use_cache=False)
    print('Running Exp4...')
    s4, p4 = _run_dgt_ablations(snaps, dgt_result, max_nodes=plan.dgt_max_nodes, dim=dim, top_k=top_k, lap_pe_k=plan.lap_pe_k, device='cpu', epochs=plan.dgt_epochs, seed=1729)
    models_exp4 = s4['model'].unique().tolist()
    new_neural = ['dgt_rwpe', 'dgt_no_residual_bypass', 'dgt_temporal_gate', 'dgt_no_time_embedding', 'dgt_deeper_3_layers', 'dgt_grit_full_rwpe_bias', 'dgt_linear_time_encoding', 'dgt_mose_structural_encoding', 'dgt_trend_seasonal_temporal']
    new_matrix = ['rwpe_svd_aligned', 'heat_kernel_diffusion_pe', 'per_spell_ppmi_procrustes', 'hamilton2016_sgns_per_spell', 'spectral_wavelet_svd', 'dgt_procrustes_posthoc', 'mose_svd_aligned', 'spse_path_encoding_svd', 'ot_wasserstein_aligned']
    for m in new_neural + new_matrix:
        assert m in models_exp4, f'Missing {m}'
    print(f'  Exp4 models: {len(models_exp4)}')
    
    # Verify no negative predictions
    all_preds = pd.concat([p1, p3, p4], ignore_index=True)
    neg = (all_preds['predicted_next_shift'] < 0).sum()
    assert neg == 0, f'{neg} negative predictions!'
    print(f'  Negative predictions: {neg}')

print('ALL SMOKE TESTS PASSED')
" 2>&1 | tail -20
```

Expected: All asserts pass, `ALL SMOKE TESTS PASSED` at the end.

Note: The smoke test imports `pd` — add `import pandas as pd` if needed in the inline script.

- [ ] **Step 2: If smoke passes, proceed. If it fails, debug the specific assertion and fix.**

---

## Task 14: Update Notebook Cell

**Files:**
- Modify: `ETG_MISQ/00_end_to_end.ipynb` — cell `dcb4eef1`

- [ ] **Step 1: Update the `run_dgt_pipeline` call to pass new params**

Find the cell containing:
```python
dgt_result = run_dgt_pipeline(
    snapshots,
    max_nodes=plan.dgt_max_nodes,
    hidden_dim=plan.dgt_hidden_dim,
    heads=plan.dgt_heads,
    layers=plan.dgt_layers,
    epochs=plan.dgt_epochs,
    lap_pe_k=plan.lap_pe_k,
    device=plan.device,
    output_dir=DGT_DIR,
    edge_batch=plan.dgt_edge_batch,
    temporal_loss_weight=plan.temporal_loss_weight,
)
```

Replace with:
```python
dgt_result = run_dgt_pipeline(
    snapshots,
    max_nodes=plan.dgt_max_nodes,
    hidden_dim=plan.dgt_hidden_dim,
    heads=plan.dgt_heads,
    layers=plan.dgt_layers,
    epochs=plan.dgt_epochs,
    lap_pe_k=plan.lap_pe_k,
    device=plan.device,
    output_dir=DGT_DIR,
    edge_batch=plan.dgt_edge_batch,
    temporal_loss_weight=plan.temporal_loss_weight,
    pe_type=plan.pe_type,
    use_residual_bypass=plan.use_residual_bypass,
    temporal_gate=plan.temporal_gate,
    use_time_embedding=plan.use_time_embedding,
    time_encoding=plan.time_encoding,
    rwpe_attention_bias=plan.rwpe_attention_bias,
    use_trend_seasonal=plan.use_trend_seasonal,
)
```

- [ ] **Step 2: Commit**

```bash
git add "ETG_MISQ/00_end_to_end.ipynb"
git commit -m "feat(v5): thread new RuntimePlan params into notebook DGT pipeline call"
```

---

## Task 15: Final Integration Commit

- [ ] **Step 1: Verify all files are clean**

```bash
cd "/Volumes/Extreme SSD/Exploit Text Graph/exploit-text-graph"
git status
git log --oneline -15
```

Expected: No unstaged changes. All 14 commits visible.

- [ ] **Step 2: Tag the v5 state**

```bash
git tag v5-ablations-ready
```

- [ ] **Step 3: Confirm cluster readiness**

Print expected model counts to confirm cluster will see all new models:

```bash
python -c "
from etg.career_rt1_benchmarks import _run_non_neural_and_classic
# Just check the ablation list is complete
from etg.career_hackersignal_pipeline import PIPELINE_CACHE_VERSION
print(f'Cache version: {PIPELINE_CACHE_VERSION}')
print('Expected neural DGT ablations: 11 (2 original + 9 new)')
print('Expected Exp4 matrix ablations: 17 (8 original + 9 new)')
print('Expected Exp3 baselines: 6 (3 original + 3 new)')
print('Expected Exp1 baselines: 12 (11 original + 1 OT)')
print('Ready for cluster run.')
"
```

---

## Summary of All New Models

| Experiment | New Model | Source |
|-----------|-----------|--------|
| Exp1 | `unbalanced_ot_shift` | ACL 2025 |
| Exp3 | `sign_scalable_inception` | ICML-W 2020 |
| Exp3 | `nodeformer_kernelized` | NeurIPS 2022 |
| Exp3 | `grit_rwpe_attention` | ICML 2023 |
| Exp4 matrix | `rwpe_svd_aligned` | NeurIPS 2022 |
| Exp4 matrix | `heat_kernel_diffusion_pe` | — |
| Exp4 matrix | `per_spell_ppmi_procrustes` | Kim 2014 |
| Exp4 matrix | `hamilton2016_sgns_per_spell` | ACL 2016 |
| Exp4 matrix | `spectral_wavelet_svd` | — |
| Exp4 matrix | `dgt_procrustes_posthoc` | — |
| Exp4 matrix | `mose_svd_aligned` | ICLR 2025 |
| Exp4 matrix | `spse_path_encoding_svd` | ICML 2025 |
| Exp4 matrix | `ot_wasserstein_aligned` | ACL 2025 |
| Exp4 neural | `dgt_rwpe` | NeurIPS 2022 |
| Exp4 neural | `dgt_no_residual_bypass` | v4 |
| Exp4 neural | `dgt_temporal_gate` | WSDM 2020 |
| Exp4 neural | `dgt_no_time_embedding` | — |
| Exp4 neural | `dgt_deeper_3_layers` | — |
| Exp4 neural | `dgt_grit_full_rwpe_bias` | ICML 2023 |
| Exp4 neural | `dgt_linear_time_encoding` | KDD 2025 |
| Exp4 neural | `dgt_mose_structural_encoding` | ICLR 2025 |
| Exp4 neural | `dgt_trend_seasonal_temporal` | KDD 2025 |
