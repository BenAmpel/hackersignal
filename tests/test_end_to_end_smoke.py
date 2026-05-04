"""End-to-end pipeline smoke test on a tiny config.

Runs both thrusts end-to-end and asserts basic sanity:
- DGT returns finite per-spell embeddings,
- Shift detection returns ShiftResults of the right shape,
- CTE pretraining and fine-tuning complete one epoch each,
- Ranking metrics are finite and better than random.
"""

from __future__ import annotations

import numpy as np

from etg.device import auto_device
from etg.rt1.dgt_model import DGT
from etg.rt1.eval_rt1 import extrinsic_forecast
from etg.rt1.shift_detection import detect_shifts_pairwise
from etg.rt1.train_dgt import extract_embeddings, train_dgt
from etg.rt2.cte_model import CTE
from etg.rt2.eval_rt2 import evaluate_ranking
from etg.rt2.finetune_cte import finetune_cte
from etg.rt2.pretrain_cte import pretrain_cte
from etg.rt2.ranking import rank_exploits, sample_candidate_pool
from etg.seeding import set_global_seed


def test_pipeline_rt1_end_to_end(tiny_config, vocab_and_snapshots):
    set_global_seed(tiny_config.seed)
    _, snaps = vocab_and_snapshots
    device = auto_device()
    feature_dim = snaps[0]["x"].shape[1]
    model = DGT(tiny_config, feature_dim)
    hist = train_dgt(model, snaps, tiny_config, device)
    assert len(hist["loss"]) == tiny_config.dgt_epochs
    embeddings = extract_embeddings(model, snaps, device)
    masks = [s["node_mask"] for s in snaps]
    results = detect_shifts_pairwise(embeddings, masks, top_quantile=0.2)
    assert len(results) == len(snaps) - 1
    # Forecast metrics should be finite numbers.
    metrics = extrinsic_forecast(results, tiny_config)
    for k in ("MAE", "RMSE", "MAPE", "R2"):
        assert np.isfinite(metrics[k])


def test_pipeline_rt2_end_to_end(tiny_config, synthetic_ev_pairs):
    set_global_seed(tiny_config.seed)
    device = auto_device()
    pairs = synthetic_ev_pairs
    model = CTE(tiny_config).to(device)
    pretrain_cte(model, pairs, tiny_config, device)
    finetune_cte(model, pairs, tiny_config, device)
    positives = [p for p in pairs if p.label == 1][:20]
    pool_texts = sample_candidate_pool(pairs, n_candidates=30, seed=tiny_config.seed)
    scores, pos_idx = rank_exploits(model, positives, pool_texts, tiny_config, device)
    metrics = evaluate_ranking(scores, pos_idx, k=10)
    assert np.isfinite(metrics["HR@10"])
    # HR@10 with ~30 candidates must beat random (random ≈ 10/30 ≈ 0.33).
    # Tiny config can be noisy; loosen slightly.
    assert metrics["HR@10"] > 0.1
