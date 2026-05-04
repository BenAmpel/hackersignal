"""Run retrieval baselines for Tasks 1 and 3 (memory-efficient for 340K corpus).

Uses chunked encoding and on-disk index for large corpus.
"""

from __future__ import annotations

import json
import time
import gc
import numpy as np
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def recall_at_k(rankings: list[list[str]], gold: list[str], k: int) -> float:
    hits = sum(1 for ranked, g in zip(rankings, gold) if g in ranked[:k])
    return hits / len(gold) if gold else 0.0


def mrr(rankings: list[list[str]], gold: list[str]) -> float:
    total = 0.0
    for ranked, g in zip(rankings, gold):
        for i, doc_id in enumerate(ranked):
            if doc_id == g:
                total += 1.0 / (i + 1)
                break
    return total / len(gold) if gold else 0.0


def run_bm25(test_queries, corpus, k=10):
    from rank_bm25 import BM25Okapi

    print("    Tokenizing corpus...")
    corpus_ids = [doc.get("id", doc.get("cve_id", "")) for doc in corpus]
    corpus_tokenized = [doc["text"].lower().split() for doc in corpus]
    print("    Building BM25 index...")
    bm25 = BM25Okapi(corpus_tokenized)

    print(f"    Retrieving {len(test_queries)} queries...")
    rankings = []
    for i, q in enumerate(test_queries):
        if (i + 1) % 100 == 0:
            print(f"      {i+1}/{len(test_queries)}")
        scores = bm25.get_scores(q["text"].lower().split())
        top_idx = np.argsort(scores)[-k:][::-1]
        rankings.append([corpus_ids[idx] for idx in top_idx])

    return rankings


def run_dense(test_queries, corpus, model_name, k=10, batch_size=512):
    import torch
    from sentence_transformers import SentenceTransformer
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"    Loading {model_name} on {device}...")
    model = SentenceTransformer(model_name, device=device)

    corpus_ids = [doc.get("id", doc.get("cve_id", "")) for doc in corpus]

    # Encode corpus in chunks to manage memory
    chunk_size = 50000
    all_corpus_embs = []
    print(f"    Encoding {len(corpus)} corpus docs in chunks of {chunk_size}...")
    for start in range(0, len(corpus), chunk_size):
        end = min(start + chunk_size, len(corpus))
        texts = [corpus[i]["text"][:384] for i in range(start, end)]
        embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                           normalize_embeddings=True)
        all_corpus_embs.append(embs)
        print(f"      Chunk {start}-{end} done")

    corpus_embs = np.vstack(all_corpus_embs)
    del all_corpus_embs
    gc.collect()

    # Encode queries
    query_texts = [q["text"][:384] for q in test_queries]
    print(f"    Encoding {len(query_texts)} queries...")
    query_embs = model.encode(query_texts, batch_size=batch_size,
                             show_progress_bar=True, normalize_embeddings=True)

    # Compute top-k via batched matmul
    print("    Computing top-k...")
    rankings = []
    qbatch = 50
    for i in range(0, len(query_embs), qbatch):
        batch_q = query_embs[i:i+qbatch]
        sims = np.dot(batch_q, corpus_embs.T)
        for j in range(len(batch_q)):
            top_idx = np.argsort(sims[j])[-k:][::-1]
            rankings.append([corpus_ids[idx] for idx in top_idx])

    del model, corpus_embs, query_embs
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return rankings


def run_hybrid(test_queries, corpus, model_name, k=10, alpha=0.7, batch_size=512):
    """BM25 + dense hybrid with RRF-style fusion."""
    from rank_bm25 import BM25Okapi
    import torch
    from sentence_transformers import SentenceTransformer
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    corpus_ids = [doc.get("id", doc.get("cve_id", "")) for doc in corpus]

    # BM25
    print("    Building BM25 for hybrid...")
    corpus_tokenized = [doc["text"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(corpus_tokenized)
    del corpus_tokenized
    gc.collect()

    # Get BM25 top-100 for each query
    print(f"    BM25 top-100 for {len(test_queries)} queries...")
    bm25_top100 = []
    for i, q in enumerate(test_queries):
        scores = bm25.get_scores(q["text"].lower().split())
        top_idx = np.argsort(scores)[-100:][::-1]
        bm25_top100.append([(corpus_ids[idx], scores[idx]) for idx in top_idx])

    del bm25
    gc.collect()

    # Dense encoding
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"    Loading {model_name} for hybrid...")
    model = SentenceTransformer(model_name, device=device)

    # Only encode candidate docs from BM25 top-100
    candidate_ids = set()
    for top100 in bm25_top100:
        for cid, _ in top100:
            candidate_ids.add(cid)

    id_to_idx = {doc.get("id", doc.get("cve_id", "")): i for i, doc in enumerate(corpus)}
    candidate_list = list(candidate_ids)
    candidate_texts = [corpus[id_to_idx[cid]]["text"][:384] for cid in candidate_list if cid in id_to_idx]
    candidate_list = [cid for cid in candidate_list if cid in id_to_idx]

    print(f"    Encoding {len(candidate_list)} candidate docs...")
    cand_embs = model.encode(candidate_texts, batch_size=batch_size,
                            show_progress_bar=True, normalize_embeddings=True)
    cand_id_to_emb = {cid: cand_embs[i] for i, cid in enumerate(candidate_list)}

    query_texts = [q["text"][:384] for q in test_queries]
    print(f"    Encoding {len(query_texts)} queries...")
    query_embs = model.encode(query_texts, batch_size=batch_size,
                             show_progress_bar=True, normalize_embeddings=True)

    # Hybrid scoring
    print("    Computing hybrid scores...")
    rankings = []
    for i, (q_emb, top100) in enumerate(zip(query_embs, bm25_top100)):
        scores = []
        bm25_max = max(s for _, s in top100) if top100 else 1.0
        for cid, bm25_score in top100:
            bm25_norm = bm25_score / bm25_max if bm25_max > 0 else 0.0
            if cid in cand_id_to_emb:
                dense_score = float(np.dot(q_emb, cand_id_to_emb[cid]))
                dense_norm = (dense_score + 1) / 2
            else:
                dense_norm = 0.0
            combined = alpha * dense_norm + (1 - alpha) * bm25_norm
            scores.append((cid, combined))
        scores.sort(key=lambda x: x[1], reverse=True)
        rankings.append([cid for cid, _ in scores[:k]])

    del model, cand_embs, query_embs, cand_id_to_emb
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return rankings


def evaluate_task(task_name, task_dir, results):
    print(f"\n{'='*60}")
    print(f"  {task_name}")
    print(f"{'='*60}")

    test_data = load_jsonl(task_dir / "test.jsonl")
    corpus = load_jsonl(task_dir / "corpus.jsonl")
    gold_ids = [q["cve_id"] for q in test_data]

    print(f"  Test queries: {len(test_data)}, Corpus: {len(corpus)}")

    task_results = {}

    # BM25
    print("\n  [BM25]")
    t0 = time.time()
    rankings = run_bm25(test_data, corpus, k=10)
    elapsed = time.time() - t0
    task_results["BM25"] = {
        "R@1": recall_at_k(rankings, gold_ids, 1),
        "R@5": recall_at_k(rankings, gold_ids, 5),
        "R@10": recall_at_k(rankings, gold_ids, 10),
        "MRR": mrr(rankings, gold_ids),
        "time_s": round(elapsed, 1),
    }
    print(f"    R@1={task_results['BM25']['R@1']:.4f}, "
          f"R@5={task_results['BM25']['R@5']:.4f}, "
          f"R@10={task_results['BM25']['R@10']:.4f}, "
          f"MRR={task_results['BM25']['MRR']:.4f} ({elapsed:.0f}s)")

    # MiniLM
    print("\n  [all-MiniLM-L6-v2]")
    t0 = time.time()
    rankings = run_dense(test_data, corpus, "sentence-transformers/all-MiniLM-L6-v2", k=10)
    elapsed = time.time() - t0
    task_results["all-MiniLM-L6-v2"] = {
        "R@1": recall_at_k(rankings, gold_ids, 1),
        "R@5": recall_at_k(rankings, gold_ids, 5),
        "R@10": recall_at_k(rankings, gold_ids, 10),
        "MRR": mrr(rankings, gold_ids),
        "time_s": round(elapsed, 1),
    }
    print(f"    R@1={task_results['all-MiniLM-L6-v2']['R@1']:.4f}, "
          f"R@5={task_results['all-MiniLM-L6-v2']['R@5']:.4f}, "
          f"R@10={task_results['all-MiniLM-L6-v2']['R@10']:.4f}, "
          f"MRR={task_results['all-MiniLM-L6-v2']['MRR']:.4f} ({elapsed:.0f}s)")

    # mpnet
    print("\n  [all-mpnet-base-v2]")
    t0 = time.time()
    rankings = run_dense(test_data, corpus, "sentence-transformers/all-mpnet-base-v2", k=10)
    elapsed = time.time() - t0
    task_results["all-mpnet-base-v2"] = {
        "R@1": recall_at_k(rankings, gold_ids, 1),
        "R@5": recall_at_k(rankings, gold_ids, 5),
        "R@10": recall_at_k(rankings, gold_ids, 10),
        "MRR": mrr(rankings, gold_ids),
        "time_s": round(elapsed, 1),
    }
    print(f"    R@1={task_results['all-mpnet-base-v2']['R@1']:.4f}, "
          f"R@5={task_results['all-mpnet-base-v2']['R@5']:.4f}, "
          f"R@10={task_results['all-mpnet-base-v2']['R@10']:.4f}, "
          f"MRR={task_results['all-mpnet-base-v2']['MRR']:.4f} ({elapsed:.0f}s)")

    # Hybrid
    print("\n  [Hybrid BM25+mpnet]")
    t0 = time.time()
    rankings = run_hybrid(test_data, corpus, "sentence-transformers/all-mpnet-base-v2", k=10)
    elapsed = time.time() - t0
    task_results["Hybrid (BM25+mpnet)"] = {
        "R@1": recall_at_k(rankings, gold_ids, 1),
        "R@5": recall_at_k(rankings, gold_ids, 5),
        "R@10": recall_at_k(rankings, gold_ids, 10),
        "MRR": mrr(rankings, gold_ids),
        "time_s": round(elapsed, 1),
    }
    print(f"    R@1={task_results['Hybrid (BM25+mpnet)']['R@1']:.4f}, "
          f"R@5={task_results['Hybrid (BM25+mpnet)']['R@5']:.4f}, "
          f"R@10={task_results['Hybrid (BM25+mpnet)']['R@10']:.4f}, "
          f"MRR={task_results['Hybrid (BM25+mpnet)']['MRR']:.4f} ({elapsed:.0f}s)")

    results[task_name] = task_results
    return task_results


def main():
    data_dir = Path("data/benchmark_v2")

    results = {}
    evaluate_task("task1_cve_linkage", data_dir / "task1_cve_linkage", results)
    evaluate_task("task3_temporal_generalization", data_dir / "task3_temporal_generalization", results)

    out_path = data_dir / "results_retrieval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nSaved to {out_path}")
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for task, tres in results.items():
        print(f"\n{task}:")
        print(f"  {'Model':<25s} {'R@1':>6s} {'R@5':>6s} {'R@10':>7s} {'MRR':>6s}")
        print(f"  {'-'*50}")
        for model, metrics in tres.items():
            print(f"  {model:<25s} {metrics['R@1']:>6.3f} {metrics['R@5']:>6.3f} "
                  f"{metrics['R@10']:>7.3f} {metrics['MRR']:>6.3f}")


if __name__ == "__main__":
    main()
