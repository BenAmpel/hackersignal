"""Reproducible serious-baseline orchestration for ETG.

The default ``smoke`` mode validates that the benchmark matrix is runnable on a
small sample. Full neural/reranker runs are intentionally represented as a
manifest of commands so they can be launched on dedicated GPU hardware.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SEEDS = [1, 2, 3, 4, 5]

CLASSIFICATION_BASELINES = [
    {
        "task": "task1",
        "family": "classification",
        "model": "tfidf_logreg",
        "command": "PYTHONPATH=src python3 -m etg.benchmark.baselines --task 1 --model tfidf --benchmark-dir data/benchmark --output data/results",
        "seeded": False,
    },
    {
        "task": "task3",
        "family": "classification",
        "model": "tfidf_logreg",
        "command": "PYTHONPATH=src python3 -m etg.benchmark.baselines --task 3 --model tfidf --benchmark-dir data/benchmark --output data/results",
        "seeded": False,
    },
    {
        "task": "task1",
        "family": "classification",
        "model": "secbert_domain_bert",
        "command": "PYTHONPATH=src python3 -m etg.benchmark.baselines --task 1 --model neural --epochs 3 --batch-size 32 --seed {seed} --benchmark-dir data/benchmark --output data/results/full/seed_{seed}",
        "seeded": True,
    },
    {
        "task": "task3",
        "family": "classification",
        "model": "secbert_domain_bert",
        "command": "PYTHONPATH=src python3 -m etg.benchmark.baselines --task 3 --model neural --epochs 3 --batch-size 32 --seed {seed} --benchmark-dir data/benchmark --output data/results/full/seed_{seed}",
        "seeded": True,
    },
    {
        "task": "task1_task3",
        "family": "classification",
        "model": "modern_encoder_probe",
        "command": "PYTHONPATH=src python3 -m etg.benchmark.smoke_benchmarks --benchmark-dir data/benchmark --output data/results/full/encoder_probe_seed_{seed}.json --sample-train 100000 --sample-eval 100000 --sample-corpus 1000 --seed {seed}",
        "seeded": True,
    },
]

RETRIEVAL_BASELINES = [
    {
        "task": "task2",
        "family": "retrieval",
        "model": "bm25",
        "command": "PYTHONPATH=src python3 -m etg.benchmark.baselines --task 2 --model tfidf --benchmark-dir data/benchmark --output data/results",
        "seeded": False,
    },
    {
        "task": "task2",
        "family": "retrieval",
        "model": "dense_biencoder",
        "command": "PYTHONPATH=src python3 -m etg.benchmark.baselines --task 2 --model neural --batch-size 128 --score-batch-size 64 --benchmark-dir data/benchmark --output data/results/full/seed_{seed}",
        "seeded": True,
    },
    {
        "task": "task2",
        "family": "retrieval",
        "model": "hybrid_sparse_dense",
        "command": "PYTHONPATH=src python3 -m etg.benchmark.serious_baselines hybrid --benchmark-dir data/benchmark --results-dir data/results/full/seed_{seed} --output data/results/full/seed_{seed}/task2_hybrid.json",
        "seeded": True,
    },
    {
        "task": "task2",
        "family": "retrieval",
        "model": "cross_encoder_reranker",
        "command": "PYTHONPATH=src python3 -m etg.benchmark.serious_baselines reranker --benchmark-dir data/benchmark --results-dir data/results/full/seed_{seed} --output data/results/full/seed_{seed}/task2_reranker.json",
        "seeded": True,
    },
]

ABLATION_COMMANDS = [
    {
        "name": "source_layer_ablation",
        "command": "PYTHONPATH=src python3 -m etg.benchmark.serious_baselines ablation --kind source-layer --benchmark-dir data/benchmark --output data/results/ablations/source_layer.json",
    },
    {
        "name": "temporal_window_ablation",
        "command": "PYTHONPATH=src python3 -m etg.benchmark.serious_baselines ablation --kind temporal-window --benchmark-dir data/benchmark --output data/results/ablations/temporal_window.json",
    },
    {
        "name": "short_row_filtering",
        "command": "PYTHONPATH=src python3 -m etg.benchmark.serious_baselines ablation --kind short-row --benchmark-dir data/benchmark --output data/results/ablations/short_row.json",
    },
    {
        "name": "dedup_filtering",
        "command": "PYTHONPATH=src python3 -m etg.benchmark.serious_baselines ablation --kind dedup --benchmark-dir data/benchmark --output data/results/ablations/dedup.json",
    },
    {
        "name": "model_domain_comparison",
        "command": "PYTHONPATH=src python3 -m etg.benchmark.smoke_benchmarks --benchmark-dir data/benchmark --output data/results/ablations/domain_model_comparison.json --sample-train 1000 --sample-eval 500 --seed 42",
    },
    {
        "name": "multilingual_slice",
        "command": "PYTHONPATH=src python3 -m etg.benchmark.serious_baselines ablation --kind multilingual --benchmark-dir data/benchmark --output data/results/ablations/multilingual.json",
    },
]


def _write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def build_manifest(output: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for row in CLASSIFICATION_BASELINES + RETRIEVAL_BASELINES:
        if row["seeded"]:
            for seed in SEEDS:
                runs.append({**row, "seed": seed, "command": row["command"].format(seed=seed)})
        else:
            runs.append({**row, "seed": None})
    manifest = {
        "purpose": "NeurIPS serious baseline run manifest",
        "seeds": SEEDS,
        "baselines": runs,
        "ablations": ABLATION_COMMANDS,
        "aggregation": "Mean and sample standard deviation over available seed result files.",
        "notes": [
            "Smoke mode validates code paths locally.",
            "Full neural, hybrid, and reranker runs should be launched on dedicated GPU hardware.",
        ],
    }
    _write_json(manifest, output)
    return manifest


def smoke(benchmark_dir: Path, output: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "etg.benchmark.smoke_benchmarks",
        "--benchmark-dir",
        str(benchmark_dir),
        "--output",
        str(output),
        "--sample-train",
        "48",
        "--sample-eval",
        "24",
        "--sample-corpus",
        "120",
        "--dl-epochs",
        "1",
        "--seed",
        "42",
        "--skip-encoders",
    ]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    result = {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "output": str(output),
    }
    if output.exists():
        result["summary"] = json.loads(output.read_text(encoding="utf-8")).get("counts")
    _write_json(result, output.with_suffix(".run.json"))
    if proc.returncode:
        raise SystemExit(proc.returncode)
    return result


def aggregate(results_dir: Path, output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("seed_*/*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(doc, dict):
            continue
        model = doc.get("model") or doc.get("kind") or path.stem
        task = doc.get("task") or ("task2_cve_linkage" if doc.get("kind") else "unknown")
        for split in ("val", "test"):
            metrics = doc.get(split)
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        rows.append(
                            {
                                "task": task,
                                "model": model,
                                "split": split,
                                "metric": metric,
                                "value": float(value),
                                "path": str(path),
                            }
                        )
        metrics = doc.get("metrics")
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    rows.append(
                        {
                            "task": task,
                            "model": model,
                            "split": "test_sample",
                            "metric": metric,
                            "value": float(value),
                            "path": str(path),
                        }
                    )
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        key = (row["task"], row["model"], row["split"], row["metric"])
        grouped.setdefault(key, []).append(row["value"])
    summary = []
    for (task, model, split, metric), values in sorted(grouped.items()):
        summary.append(
            {
                "task": task,
                "model": model,
                "split": split,
                "metric": metric,
                "n": len(values),
                "mean": round(statistics.mean(values), 4),
                "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            }
        )
    out = {"results_dir": str(results_dir), "rows": summary}
    _write_json(out, output)
    return out


def _sample_task2(benchmark_dir: Path, sample_eval: int, sample_corpus: int, seed: int):
    from etg.benchmark.smoke_benchmarks import _task2_sample
    import random

    return _task2_sample(benchmark_dir / "task2_cve_linkage", sample_eval, sample_corpus, random.Random(seed))


def hybrid_retrieval(
    benchmark_dir: Path,
    output: Path,
    sample_eval: int = 1000,
    sample_corpus: int = 5000,
    seed: int = 42,
    alpha: float = 0.5,
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, Any]:
    import numpy as np
    from etg.benchmark.smoke_benchmarks import _ranking_metrics, _run_sparse_retriever, _run_encoder_retriever

    queries, corpus, truth = _sample_task2(benchmark_dir, sample_eval, sample_corpus, seed)
    sparse = _run_sparse_retriever("bm25_smoke", queries, corpus, truth)

    # Recompute dense scores here so hybrid ranking is actually evaluated.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(encoder_model)
    C = model.encode([c["text"] for c in corpus], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    Q = model.encode([q["text"] for q in queries], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    dense_scores = np.asarray(Q) @ np.asarray(C).T

    # Use TF-IDF cosine as the sparse score matrix for a stable, normalized hybrid.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=1)
    C_sparse = vec.fit_transform([c["text"] for c in corpus])
    Q_sparse = vec.transform([q["text"] for q in queries])
    sparse_scores = cosine_similarity(Q_sparse, C_sparse)

    def _row_norm(x: np.ndarray) -> np.ndarray:
        lo = x.min(axis=1, keepdims=True)
        hi = x.max(axis=1, keepdims=True)
        return (x - lo) / np.maximum(hi - lo, 1e-9)

    scores = alpha * _row_norm(sparse_scores) + (1 - alpha) * _row_norm(dense_scores)
    result = {
        "kind": "hybrid_sparse_dense",
        "status": "ok",
        "seed": seed,
        "sample_eval": sample_eval,
        "sample_corpus": len(corpus),
        "alpha_sparse": alpha,
        "encoder_model": encoder_model,
        "bm25_reference": sparse.get("metrics"),
        "metrics": _ranking_metrics(scores, truth),
    }
    _write_json(result, output)
    return result


def reranker_retrieval(
    benchmark_dir: Path,
    output: Path,
    sample_eval: int = 300,
    sample_corpus: int = 3000,
    seed: int = 42,
    top_k: int = 50,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> dict[str, Any]:
    import numpy as np
    from etg.benchmark.smoke_benchmarks import _bm25_scores, _ranking_metrics
    from sentence_transformers import CrossEncoder

    queries, corpus, truth = _sample_task2(benchmark_dir, sample_eval, sample_corpus, seed)
    bm25_scores = _bm25_scores([q["text"] for q in queries], [c["text"] for c in corpus])
    candidates = np.argsort(-bm25_scores, axis=1)[:, : min(top_k, len(corpus))]
    model = CrossEncoder(reranker_model)
    final_scores = np.full((len(queries), len(corpus)), -1e9, dtype="float32")
    pairs: list[tuple[str, str]] = []
    pair_index: list[tuple[int, int]] = []
    for qi, q in enumerate(queries):
        for ci in candidates[qi]:
            pairs.append((q["text"], corpus[int(ci)]["text"]))
            pair_index.append((qi, int(ci)))
    scores = model.predict(pairs, batch_size=32, show_progress_bar=True)
    for (qi, ci), score in zip(pair_index, scores):
        final_scores[qi, ci] = float(score)
    result = {
        "kind": "cross_encoder_reranker",
        "status": "ok",
        "seed": seed,
        "sample_eval": len(queries),
        "sample_corpus": len(corpus),
        "top_k": top_k,
        "reranker_model": reranker_model,
        "metrics": _ranking_metrics(final_scores, truth),
    }
    _write_json(result, output)
    return result


def ablation(kind: str, benchmark_dir: Path, output: Path) -> dict[str, Any]:
    from etg.benchmark.audits import quality_by_task, overlap_audit

    tmp = output.with_suffix(".quality.json")
    quality = quality_by_task(benchmark_dir, tmp)
    overlap = overlap_audit(benchmark_dir, output.with_suffix(".overlap.json"))
    result = {
        "kind": kind,
        "status": "ok",
        "note": "Ablation audit scaffold with task-level filter diagnostics; model reruns are handled by the serious baseline manifest.",
        "quality_path": str(tmp),
        "overlap_path": str(output.with_suffix(".overlap.json")),
        "task_summary": {
            task: {
                split: {
                    key: vals[key]
                    for key in ("rows", "short_under_8_tokens", "exact_duplicate_text_rows", "non_ascii_signal_rows")
                }
                for split, vals in splits.items()
            }
            for task, splits in quality["tasks"].items()
        },
        "overlap": overlap["tasks"],
    }
    _write_json(result, output)
    return result


def _run_shell_step(step: dict[str, Any], status_path: Path, retries: int, heartbeat_seconds: int) -> dict[str, Any]:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        attempt += 1
        start = time.time()
        step_status = {
            **step,
            "attempt": attempt,
            "status": "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        _write_json(step_status, status_path)
        proc = subprocess.Popen(
            step["command"],
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines: list[str] = []
        last_heartbeat = time.time()
        assert proc.stdout is not None
        while proc.poll() is None:
            line = proc.stdout.readline()
            if line:
                output_lines.append(line)
                print(line, end="")
            if time.time() - last_heartbeat >= heartbeat_seconds:
                step_status.update(
                    {
                        "status": "running",
                        "elapsed_seconds": round(time.time() - start, 1),
                        "last_heartbeat": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "tail": "".join(output_lines[-40:]),
                    }
                )
                _write_json(step_status, status_path)
                last_heartbeat = time.time()
        remainder = proc.stdout.read()
        if remainder:
            output_lines.append(remainder)
            print(remainder, end="")
        rc = proc.returncode
        step_status.update(
            {
                "status": "ok" if rc == 0 else "failed",
                "returncode": rc,
                "elapsed_seconds": round(time.time() - start, 1),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "tail": "".join(output_lines[-80:]),
            }
        )
        _write_json(step_status, status_path)
        if rc == 0 or attempt > retries:
            return step_status
        time.sleep(10)


def run_all(
    benchmark_dir: Path,
    output_dir: Path,
    retries: int = 1,
    heartbeat_seconds: int = 600,
    include_neural: bool = True,
    include_retrieval_heavy: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    status_dir = output_dir / "status"
    steps: list[dict[str, Any]] = [
        {"name": "split_build_smoke", "command": f"PYTHONPATH=src {sys.executable} -m etg.benchmark.splits --data-dir data --output /tmp/etg_serious_split_check --task all --seed 42"},
        {"name": "audits_leakage", "command": f"PYTHONPATH=src {sys.executable} -m etg.benchmark.audits leakage --benchmark-dir {benchmark_dir} --output data/audits/task3_leakage_audit.json --fail-on-strict"},
        {"name": "audits_overlap", "command": f"PYTHONPATH=src {sys.executable} -m etg.benchmark.audits overlap --benchmark-dir {benchmark_dir} --output data/audits/split_overlap_audit.json"},
        {"name": "audits_quality", "command": f"PYTHONPATH=src {sys.executable} -m etg.benchmark.audits quality-by-task --benchmark-dir {benchmark_dir} --output data/audits/quality_by_task.json"},
        {"name": "lightweight_baselines", "command": f"PYTHONPATH=src {sys.executable} -m etg.benchmark.baselines --benchmark-dir {benchmark_dir} --output data/results --task all --model tfidf"},
        {"name": "broad_smoke", "command": f"PYTHONPATH=src {sys.executable} -m etg.benchmark.smoke_benchmarks --benchmark-dir {benchmark_dir} --output data/results/smoke_benchmarks.json --sample-train 80 --sample-eval 40 --sample-corpus 200 --dl-epochs 1 --seed 42"},
    ]
    if include_neural:
        for seed in SEEDS:
            steps.extend(
                [
                    {"name": f"task1_secbert_seed_{seed}", "command": f"PYTHONPATH=src {sys.executable} -m etg.benchmark.baselines --benchmark-dir {benchmark_dir} --output data/results/full/seed_{seed} --task 1 --model neural --epochs 3 --batch-size 32 --seed {seed}"},
                    {"name": f"task3_secbert_seed_{seed}", "command": f"PYTHONPATH=src {sys.executable} -m etg.benchmark.baselines --benchmark-dir {benchmark_dir} --output data/results/full/seed_{seed} --task 3 --model neural --epochs 3 --batch-size 32 --seed {seed}"},
                    {"name": f"task2_biencoder_seed_{seed}", "command": f"PYTHONPATH=src {sys.executable} -m etg.benchmark.baselines --benchmark-dir {benchmark_dir} --output data/results/full/seed_{seed} --task 2 --model neural --batch-size 128 --score-batch-size 64"},
                ]
            )
    if include_retrieval_heavy:
        for seed in SEEDS:
            steps.extend(
                [
                    {"name": f"task2_hybrid_seed_{seed}", "command": f"PYTHONPATH=src {sys.executable} -m etg.benchmark.serious_baselines hybrid --benchmark-dir {benchmark_dir} --output data/results/full/seed_{seed}/task2_hybrid_seed{seed}.json --seed {seed}"},
                    {"name": f"task2_reranker_seed_{seed}", "command": f"PYTHONPATH=src {sys.executable} -m etg.benchmark.serious_baselines reranker --benchmark-dir {benchmark_dir} --output data/results/full/seed_{seed}/task2_reranker_seed{seed}.json --seed {seed}"},
                ]
            )
    for kind in ("source-layer", "temporal-window", "short-row", "dedup", "multilingual"):
        steps.append({"name": f"ablation_{kind}", "command": f"PYTHONPATH=src {sys.executable} -m etg.benchmark.serious_baselines ablation --kind {kind} --benchmark-dir {benchmark_dir} --output data/results/ablations/{kind}.json"})
    steps.append({"name": "aggregate", "command": f"PYTHONPATH=src {sys.executable} -m etg.benchmark.serious_baselines aggregate --results-dir data/results/full --output data/results/serious_baselines_seed_summary.json"})

    summary = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "steps": []}
    _write_json(summary, output_dir / "overnight_summary.json")
    for idx, step in enumerate(steps, 1):
        print(f"\n=== [{idx}/{len(steps)}] {step['name']} ===\n")
        status = _run_shell_step(step, status_dir / f"{idx:03d}_{step['name']}.json", retries, heartbeat_seconds)
        summary["steps"].append(status)
        summary["last_step"] = step["name"]
        summary["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_json(summary, output_dir / "overnight_summary.json")
        if status["status"] != "ok":
            summary["status"] = "failed"
            _write_json(summary, output_dir / "overnight_summary.json")
            raise SystemExit(f"Step failed after retries: {step['name']}")
    summary["status"] = "ok"
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_json(summary, output_dir / "overnight_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="ETG serious baseline orchestration")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("manifest")
    p.add_argument("--output", type=Path, default=Path("data/results/serious_baselines_manifest.json"))

    p = sub.add_parser("smoke")
    p.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmark"))
    p.add_argument("--output", type=Path, default=Path("data/results/serious_baselines_smoke.json"))

    p = sub.add_parser("aggregate")
    p.add_argument("--results-dir", type=Path, default=Path("data/results/full"))
    p.add_argument("--output", type=Path, default=Path("data/results/serious_baselines_seed_summary.json"))

    for name in ("hybrid", "reranker", "ablation"):
        p = sub.add_parser(name)
        p.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmark"))
        p.add_argument("--results-dir", type=Path, default=Path("data/results/full"))
        p.add_argument("--kind", default=name)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--sample-eval", type=int, default=1000)
        p.add_argument("--sample-corpus", type=int, default=5000)
        p.add_argument("--top-k", type=int, default=50)
        p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("run-all")
    p.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmark"))
    p.add_argument("--output-dir", type=Path, default=Path("data/results/overnight"))
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--heartbeat-seconds", type=int, default=600)
    p.add_argument("--skip-neural", action="store_true")
    p.add_argument("--skip-retrieval-heavy", action="store_true")

    args = parser.parse_args()
    if args.command == "manifest":
        out = build_manifest(args.output)
    elif args.command == "smoke":
        out = smoke(args.benchmark_dir, args.output)
    elif args.command == "aggregate":
        out = aggregate(args.results_dir, args.output)
    elif args.command == "hybrid":
        out = hybrid_retrieval(
            args.benchmark_dir,
            args.output,
            sample_eval=args.sample_eval,
            sample_corpus=args.sample_corpus,
            seed=args.seed,
        )
    elif args.command == "reranker":
        out = reranker_retrieval(
            args.benchmark_dir,
            args.output,
            sample_eval=args.sample_eval,
            sample_corpus=args.sample_corpus,
            seed=args.seed,
            top_k=args.top_k,
        )
    elif args.command == "ablation":
        out = ablation(args.kind, args.benchmark_dir, args.output)
    elif args.command == "run-all":
        out = run_all(
            args.benchmark_dir,
            args.output_dir,
            retries=args.retries,
            heartbeat_seconds=args.heartbeat_seconds,
            include_neural=not args.skip_neural,
            include_retrieval_heavy=not args.skip_retrieval_heavy,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
