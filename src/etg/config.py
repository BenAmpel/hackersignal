"""Global configuration for the ETG pipeline.

Config.smoke() runs in < 2 minutes on a laptop CPU.
Config.full()  targets a GPU cluster and real data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # ---- run control ----
    seed: int = 1729
    cache_dir: str = "cache"
    results_path: str = "cache/results.json"

    # ---- data (synthetic generator) ----
    n_posts: int = 2000
    n_spells: int = 5
    n_forums: int = 3
    posts_per_spell: tuple[int, ...] = field(default_factory=lambda: ())  # derived if empty
    n_ev_pairs: int = 1200
    n_ev_negatives_per_positive: int = 4
    noise_rate: float = 0.08
    shift_subtlety: float = 0.5  # 0 = drastic shift, 1 = very subtle
    inject_n_shift_words: int = 12  # ground truth
    overlap_rate: float = 0.35  # shared vocab between exploit/vuln classes

    # ---- vocab / tokenization ----
    vocab_size_cap: int = 5000
    trigram_buckets: int = 2**14  # 16,384  (smoke-scale; full uses 2**20)
    min_token_len: int = 2

    # ---- ETG construction ----
    window_size: int = 4  # sliding window d
    min_edge_weight: float = 1.0

    # ---- Laplacian PE ----
    lap_pe_k: int = 16
    lap_pe_method: str = "eigsh"  # or "nystrom"
    lap_pe_nystrom_samples: int = 1024

    # ---- DGT (RT1) ----
    dgt_hidden_dim: int = 64
    dgt_num_heads: int = 4
    dgt_num_layers: int = 2
    dgt_dropout: float = 0.1
    dgt_k_hop: int = 2
    dgt_temporal_window: int = 3
    dgt_temporal_lambda: float = 0.5
    dgt_recon_neg_samples: int = 3
    dgt_epochs: int = 6
    dgt_lr: float = 1e-3
    dgt_batch_size_edges: int = 256

    # ---- Shift detection ----
    shift_top_quantile: float = 0.05
    shift_perm_trials: int = 100

    # ---- ARIMA ----
    arima_order: tuple[int, int, int] = (1, 1, 1)
    arima_min_points: int = 8

    # ---- CTE (RT2) ----
    cte_emb_dim: int = 64
    cte_lstm_hidden: int = 64
    cte_cnn_channels: int = 64
    cte_cnn_kernels: tuple[int, ...] = (3, 4, 5)
    cte_attn_dim: int = 64
    cte_dropout: float = 0.1
    cte_max_len: int = 48
    cte_pretrain_epochs: int = 4
    cte_finetune_epochs: int = 4
    cte_lr: float = 1e-3
    cte_batch_size: int = 64
    cte_mask_rate: float = 0.15
    cte_contrastive_temp: float = 0.07
    cte_contrastive_weight: float = 1.0
    cte_mlm_weight: float = 1.0

    # ---- eval ----
    eval_top_k: int = 10
    analogy_hit_at_k: int = 5

    # ---- viz ----
    save_figures: bool = True
    figure_dir: str = "cache/figures"

    # ---- fingerprint (for cache keys) ----
    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def smoke(cls) -> "Config":
        """Defaults — small enough for CPU, realistic enough to show behavior."""
        return cls()

    @classmethod
    def tiny(cls) -> "Config":
        """Ultra-small config for rapid end-to-end verification (< 2 min on MPS/CPU).

        Retains every pipeline stage (DGT training, shift detection, ARIMA, CTE
        pretraining/fine-tuning, all baselines) — just scales everything down so
        that a full run completes in well under two minutes. Use this to sanity-
        check that the pipeline works end-to-end before committing to `smoke()`
        or `full()` runs.
        """
        return cls(
            n_posts=400,
            n_spells=4,
            n_ev_pairs=200,
            n_ev_negatives_per_positive=2,
            vocab_size_cap=400,
            trigram_buckets=2**12,
            dgt_hidden_dim=32,
            dgt_num_heads=2,
            dgt_num_layers=1,
            dgt_epochs=2,
            dgt_batch_size_edges=128,
            lap_pe_k=8,
            cte_emb_dim=32,
            cte_lstm_hidden=32,
            cte_cnn_channels=32,
            cte_attn_dim=32,
            cte_max_len=16,
            cte_pretrain_epochs=1,
            cte_finetune_epochs=1,
            cte_batch_size=32,
            inject_n_shift_words=6,
            analogy_hit_at_k=3,
        )

    @classmethod
    def full(cls) -> "Config":
        """Cluster-scale settings. Override as needed for specific hardware."""
        return cls(
            n_posts=500_000,
            n_spells=12,
            n_forums=17,
            n_ev_pairs=100_000,
            vocab_size_cap=30_000,
            trigram_buckets=2**20,
            lap_pe_k=64,
            dgt_hidden_dim=256,
            dgt_num_heads=8,
            dgt_num_layers=4,
            dgt_epochs=50,
            dgt_batch_size_edges=4096,
            cte_emb_dim=256,
            cte_lstm_hidden=256,
            cte_cnn_channels=128,
            cte_max_len=128,
            cte_pretrain_epochs=20,
            cte_finetune_epochs=10,
            cte_batch_size=512,
        )

    def ensure_dirs(self) -> None:
        for p in (self.cache_dir, self.figure_dir, Path(self.results_path).parent):
            Path(p).mkdir(parents=True, exist_ok=True)
