# ETG v5 Ablations Design

**Date:** 2026-05-12  
**Authors:** Ben Ampel  
**Status:** Approved  
**Target files:**
- `src/etg/career_hackersignal_pipeline.py`
- `src/etg/career_rt1_benchmarks.py`
- `ETG_MISQ/00_end_to_end.ipynb`

---

## Context

v4 produced 21 statistically significant wins, 0 significant losses, MAE 0.077 (−39% vs baselines), R² −0.031. Three issues remain: (1) hub-term contamination from "out"/"order", (2) baselines use `np.polyfit` for shift prediction while DGT uses EMA — an unfair comparison, (3) EMA decay=0.5 causes slight underprediction (mean predicted 0.083 vs actual 0.125).

v5 addresses all three quick fixes plus adds a comprehensive ablation matrix informed by graph transformer and diachronic linguistics literature from 2020–2025.

---

## Quick Fixes

### 1. Stopwords
Add `"out"` and `"order"` to `STOPWORDS` in `career_hackersignal_pipeline.py`. Both appear in top-6 hub terms for spells 1–4 and are generic English prepositions/nouns with no signal value.

### 2. EMA decay
Change `decay=0.5` → `decay=0.7` in `_ema_predict()`. Higher decay gives more weight to recent spells, reducing the systematic underprediction gap (mean predicted 0.083 vs actual 0.125).

### 3. Fair baseline predictions
In `career_rt1_benchmarks.py`, functions `_embedding_shifts()` and `_trend_shifts()` currently use `np.polyfit` linear extrapolation for shift prediction. Replace with `_ema_predict()` (same function as DGT's) so all models use the same prediction method. This makes comparison fair and eliminates negative predictions from baselines.

### 4. Cache version
Bump `PIPELINE_CACHE_VERSION` to `"career_rt1_cache_v5"` to invalidate all v4 caches.

---

## Architecture Changes to `run_dgt_pipeline`

### New parameters

```python
pe_type: str = "laplacian"          # "laplacian" | "rwpe" | "mose" | "none"
use_residual_bypass: bool = True     # learnable alpha bypass (v4 default)
temporal_gate: bool = False          # per-node GRU-style gate (replaces global alpha)
use_time_embedding: bool = True      # spell-index embedding in transformer
time_encoding: str = "learned_discrete"  # "learned_discrete" | "learned_linear"
rwpe_attention_bias: bool = False    # pairwise RWPE dot-product added to attn mask
use_trend_seasonal: bool = False     # TIDFormer-style temporal loss decomposition
```

### New helper functions in `career_hackersignal_pipeline.py`

**`_rwpe(edge_counts, node_to_idx, k) -> ndarray (N, k)`**  
Computes random walk positional encoding. For each node i, the k-step landing probabilities `[(D⁻¹A)^k][i,i]` for k=1..K. Normalizes degree matrix from `edge_counts`. Falls back to zeros for isolated nodes.

**`_mose_features(edge_counts, node_to_idx) -> ndarray (N, 3)`**  
Motif Structural Encoding (ICLR 2025). Per node: triangle count (number of closed triangles), star count (degree), path-of-2 count (number of length-2 paths through node). These are the cheapest homomorphism counts that capture local structure beyond degree alone.

### `MaskedGraphTransformer` changes

**`pe_type="rwpe"`**: Concatenate RWPE features (k=`lap_pe_k` dimensions) to input node features before `self.input` projection. RWPE replaces LaplacianPE — both produce the same dimensionality so `input_dim` is unchanged.

**`pe_type="mose"`**: Concatenate MoSE features (3 dimensions) to input node features. `input_dim` increases by 3 — Linear layer resized accordingly.

**`pe_type="none"`**: No positional encoding added.

**`use_residual_bypass=False`**: Remove `self.residual_alpha` parameter; `z = z_enc` directly.

**`temporal_gate=True`**: Replace scalar `self.residual_alpha` with `self.gate = nn.Linear(hidden_dim, hidden_dim)`. Forward: `gate_val = torch.sigmoid(self.gate(h)); z = gate_val * z_enc + (1 - gate_val) * h`. This is a per-node, per-dimension gating (DySAT-inspired, WSDM 2020).

**`use_time_embedding=False`**: Skip the `self.time_embedding` lookup; `h = self.input(x)` only.

**`time_encoding="learned_linear"`**: Replace `nn.Embedding(n_spells, hidden_dim)` with `nn.Linear(1, hidden_dim)`. Input: normalized spell index `t / max(n_spells - 1, 1)` as a scalar. From KDD 2025 workshop (arxiv:2504.08129) — uses 43% fewer parameters than discrete embedding, generalizes to unseen time steps.

**`rwpe_attention_bias=True`**: Compute pairwise RWPE dot products `RWPE @ RWPE.T` (N×N float matrix). Add this to the attention mask float values before passing to TransformerEncoder. Implements the relative PE bias from GRIT (ICML 2023, arxiv:2312.02220).

**`use_trend_seasonal=True`**: In the temporal loss computation, decompose the EMA context into trend (average-pooled across spells) and seasonal (residual = context − trend) components. Compute separate MSE losses for trend alignment and seasonal alignment, weighted equally. Implements the TIDFormer mechanism (KDD 2025, arxiv:2506.00431).

---

## New Benchmark Models

### Exp1 Baselines (non-neural)

**`unbalanced_ot_shift`** (ACL 2025, arxiv:2412.12569)  
Uses unbalanced optimal transport to measure distributional drift between a word's usage contexts across consecutive spells. For each word, computes PPMI-weighted context distributions and measures UOT distance as the shift magnitude. Requires `pot` library (`pip install POT`). If not available, falls back to L1 distance between PPMI distributions.

### Exp3 Transformer/Contextual Baselines (matrix, fast)

**`sign_scalable_inception`** (SIGN, ICML-W 2020)  
Multi-hop aggregation: for each node, concatenate raw features with 1-hop, 2-hop, 3-hop mean-aggregated neighbor features, then project with MLP. Implements the SIGN "inception-style" aggregation without GNN training. One embedding per spell, cosine shift computed as before.

**`nodeformer_kernelized`** (NodeFormer, NeurIPS 2022)  
Kernelized random-feature attention: approximate full-graph attention using random Fourier features (ELU kernel). Nodes attend to all other nodes without adjacency masking. Computes one embedding per spell.

**`grit_rwpe_attention`** (GRIT, ICML 2023)  
RWPE features for each node, pairwise dot-product attention bias added to a simple linear transformer layer. Serves as a proxy for GRIT's relative PE mechanism on our graph snapshots.

### New Exp4 Matrix Ablations (SVD-based, fast)

These follow the existing pattern: build node feature matrix → SVD → Procrustes align across spells → cosine shift.

| Name | Features | Source |
|------|----------|--------|
| `rwpe_svd_aligned` | RWPE (k=8) landing probabilities | GraphGPS, NeurIPS 2022 |
| `heat_kernel_diffusion_pe` | Heat kernel diagonal `exp(-t·L)_{ii}` for t=1,2,4,8 | Standard spectral |
| `per_spell_ppmi_procrustes` | PPMI vectors, Procrustes align | Kim et al. 2014 |
| `hamilton2016_sgns_per_spell` | SGNS word2vec per spell, Procrustes | Hamilton ACL 2016 |
| `spectral_wavelet_svd` | Graph wavelet coefficients (approx) | Spectral GNN literature |
| `dgt_procrustes_posthoc` | DGT embeddings, post-hoc Procrustes | Ablates DGT's learned alignment |
| `mose_svd_aligned` | MoSE motif counts (triangle, star, path-2) → SVD | ICLR 2025 (arxiv:2410.18676) |
| `spse_path_encoding_svd` | Simple path counts (lengths 1–4) → SVD | ICML 2025 (arxiv:2502.09365) |
| `ot_wasserstein_aligned` | UOT context distribution alignment, shift = OT distance | ACL 2025 (arxiv:2412.12569) |

### New Neural DGT Ablations (Exp4, GPU-hours)

All ablations run at v5 full-mode hyperparameters (200 epochs, `temporal_loss_weight=0.1`, EMA decay=0.7). Each varies exactly one mechanism from the DGT full model.

| Ablation name | Change from DGT full | Paper |
|---------------|---------------------|-------|
| `dgt_rwpe` | `pe_type="rwpe"` | GraphGPS NeurIPS 2022 |
| `dgt_no_residual_bypass` | `use_residual_bypass=False` | v4 contribution |
| `dgt_temporal_gate` | `temporal_gate=True` | DySAT WSDM 2020 |
| `dgt_no_time_embedding` | `use_time_embedding=False` | Ablation |
| `dgt_deeper_3_layers` | `layers=3` | Architecture search |
| `dgt_grit_full_rwpe_bias` | `pe_type="rwpe"`, `rwpe_attention_bias=True` | GRIT ICML 2023 |
| `dgt_linear_time_encoding` | `time_encoding="learned_linear"` | KDD 2025 (arxiv:2504.08129) |
| `dgt_mose_structural_encoding` | `pe_type="mose"` | ICLR 2025 (arxiv:2410.18676) |
| `dgt_trend_seasonal_temporal` | `use_trend_seasonal=True` | KDD 2025 (arxiv:2506.00431) |

---

## Notebook Changes (`00_end_to_end.ipynb`)

Update the `run_dgt_pipeline` call in cell `dcb4eef1` to pass new default parameters. For v5 full run, defaults match v4 behavior (`pe_type="laplacian"`, `use_residual_bypass=True`, `temporal_gate=False`, etc.).

Each new neural ablation is a separate `run_dgt_pipeline` call with the appropriate parameter override, same pattern as existing `dgt_no_laplacian_pe` and `dgt_static` ablations.

---

## `RuntimePlan` Changes

Add new fields with defaults that preserve v4 behavior:

```python
pe_type: str = "laplacian"
use_residual_bypass: bool = True
temporal_gate: bool = False
use_time_embedding: bool = True
time_encoding: str = "learned_discrete"
rwpe_attention_bias: bool = False
use_trend_seasonal: bool = False
```

---

## Estimated Runtime

| Phase | Time |
|-------|------|
| Quick fixes + cache bump | 15 min |
| New matrix ablations (Exp3 + Exp4) | ~1.5 hrs |
| 9 neural DGT ablations × ~40 min | ~6 hrs |
| `unbalanced_ot_shift` Exp1 baseline | ~15 min |
| **Total (sequential)** | **~8 hrs** |

Cluster parallelism (3–4 GPUs) reduces wall time to ~3–4 hrs.

---

## Success Criteria

- 0 negative predictions across all models
- At least 5 new significant wins from new ablations (showing which mechanisms help)
- At least one 2024–2026 paper mechanism improves DGT MAE vs v4 full
- All baselines use EMA prediction (fair comparison)
- R² ≥ 0.0 (positive predictive relationship)
- No regressions on existing 21 significant wins

---

## Literature References

1. Müller et al. "Graph-ViT: Transformers are Graph Neural Networks." NeurIPS 2022. (GraphGPS)
2. Ma et al. "GRIT: Graph Relative Inductive Transformer." ICML 2023. arxiv:2312.02220
3. Wu et al. "NodeFormer." NeurIPS 2022.
4. Frasca et al. "SIGN: Scalable Inception Graph Neural Networks." ICML-W 2020.
5. Sankar et al. "DySAT: Dynamic Self-Attention Networks." WSDM 2020.
6. Lim et al. "MoSE: Motif Structural Encoding." ICLR 2025. arxiv:2410.18676
7. Huang et al. "SPSE: Simple Path Structural Encoding." ICML 2025. arxiv:2502.09365
8. Zhao et al. "Unbalanced OT for Semantic Shift." ACL 2025. arxiv:2412.12569
9. Anon. "Linear Time Encoding for Graph Transformers." KDD 2025 workshop. arxiv:2504.08129
10. Anon. "TIDFormer: Trend-Seasonal Decomposition." KDD 2025. arxiv:2506.00431
11. Kim et al. "Temporal Analysis of Language through Neural Language Models." ACL 2014.
12. Hamilton et al. "Diachronic Word Embeddings." ACL 2016.
