"""Run the CAREER RT1/RT2 pipeline on real data artifacts.

Usage:
    PYTHONPATH=src python3 -m etg.career_run --profile tiny
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Config
from .data.career_award import (
    DEFAULT_RT2_FILE,
    default_rt1_paths,
    load_career_rt1_posts,
    load_career_rt2_pairs,
)
from .device import auto_device, device_summary
from .rt1.dgt_model import DGT
from .rt1.etg_builder import build_etg_sequence, group_posts_by_spell
from .rt1.eval_rt1 import extrinsic_forecast
from .rt1.laplacian_pe import attach_lap_pe
from .rt1.shift_detection import detect_shifts_pairwise
from .rt1.train_dgt import extract_embeddings, train_dgt
from .rt1.vocab import build_vocab
from .rt2.cte_model import CTE
from .rt2.eval_rt2 import evaluate_ranking
from .rt2.finetune_cte import finetune_cte
from .rt2.pretrain_cte import pretrain_cte
from .rt2.ranking import rank_exploits, sample_candidate_pool
from .seeding import set_global_seed
from .data.time_spells import TimeSpellIndex


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CAREER-award-aligned RT1/RT2 models on real ETG data."
    )
    parser.add_argument(
        "--profile",
        choices=("tiny", "smoke", "full"),
        default="tiny",
        help="Config preset to use (default: tiny).",
    )
    parser.add_argument(
        "--rt1-source",
        action="append",
        default=None,
        help="RT1 ForumPost JSONL path. Repeat to use multiple community corpora.",
    )
    parser.add_argument(
        "--rt2-source",
        default=str(Path("data") / DEFAULT_RT2_FILE),
        help="RT2 EV-pair JSONL path (default: data/ev_pairs.jsonl).",
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Optional JSON output path. Defaults to cache/career_<profile>_results.json.",
    )
    return parser.parse_args()


def _config_from_profile(profile: str) -> Config:
    return {
        "tiny": Config.tiny,
        "smoke": Config.smoke,
        "full": Config.full,
    }[profile]()


def main() -> None:
    args = _parse_args()
    config = _config_from_profile(args.profile)
    config.ensure_dirs()
    set_global_seed(config.seed)
    device = auto_device()

    rt1_paths = args.rt1_source or [str(path) for path in default_rt1_paths()]
    rt1_posts = load_career_rt1_posts(config, paths=rt1_paths)
    ts_index = TimeSpellIndex.from_posts(rt1_posts, config.n_spells)
    posts_by_spell = group_posts_by_spell(rt1_posts, ts_index)
    vocab = build_vocab(rt1_posts, config)
    snapshots = build_etg_sequence(rt1_posts, posts_by_spell, vocab, config)
    attach_lap_pe(snapshots, config)

    dgt = DGT(config, snapshots[0]["x"].shape[1])
    train_dgt(dgt, snapshots, config, device)
    dgt_embeddings = extract_embeddings(dgt, snapshots, device)
    shift_results = detect_shifts_pairwise(
        dgt_embeddings,
        [snapshot["node_mask"] for snapshot in snapshots],
        top_quantile=config.shift_top_quantile,
    )
    rt1_metrics = extrinsic_forecast(shift_results, config)

    rt2_pairs = load_career_rt2_pairs(config, path=args.rt2_source)
    cte = CTE(config)
    pretrain_cte(cte, rt2_pairs, config, device)
    finetune_cte(cte, rt2_pairs, config, device)

    positive_pairs = [pair for pair in rt2_pairs if pair.label == 1]
    pool = sample_candidate_pool(
        rt2_pairs,
        n_candidates=max(50, len(positive_pairs) // 2),
        seed=config.seed,
    )
    scores, positive_indices = rank_exploits(cte, positive_pairs, pool, config, device)
    rt2_metrics = evaluate_ranking(scores, positive_indices, k=config.eval_top_k)

    results = {
        "profile": args.profile,
        "device": device_summary(device),
        "config_hash": config.hash(),
        "rt1": rt1_metrics,
        "rt2": rt2_metrics,
        "rt1_data": {
            "sources": rt1_paths,
            "posts": len(rt1_posts),
            "forums": len({post.forum_id for post in rt1_posts}),
        },
        "rt2_data": {
            "source": args.rt2_source,
            "pairs": len(rt2_pairs),
            "positives": len(positive_pairs),
            "negatives": sum(1 for pair in rt2_pairs if pair.label == 0),
        },
    }

    results_path = Path(args.results) if args.results else Path("cache") / f"career_{args.profile}_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

    print(f"device: {results['device']}")
    print(f"rt1 sources: {len(rt1_paths)} files  posts={results['rt1_data']['posts']}  forums={results['rt1_data']['forums']}")
    print(
        "rt1 metrics: "
        f"MAE={rt1_metrics['MAE']:.4f} RMSE={rt1_metrics['RMSE']:.4f} "
        f"MAPE={rt1_metrics['MAPE']:.4f} R2={rt1_metrics['R2']:.4f}"
    )
    print(
        "rt2 metrics: "
        f"HR@{config.eval_top_k}={rt2_metrics[f'HR@{config.eval_top_k}']:.4f} "
        f"MRR@{config.eval_top_k}={rt2_metrics[f'MRR@{config.eval_top_k}']:.4f} "
        f"NDCG@{config.eval_top_k}={rt2_metrics[f'NDCG@{config.eval_top_k}']:.4f} "
        f"MAP={rt2_metrics['MAP']:.4f}"
    )
    print(f"saved: {results_path}")


if __name__ == "__main__":
    main()
