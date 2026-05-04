"""Run the full model ladder for HackerSignal benchmark v2.

Model ladder:
  Task 1 & 3 (Retrieval):
    - BM25 (lexical)
    - all-MiniLM-L6-v2 (lightweight dense)
    - all-mpnet-base-v2 (stronger dense)
    - SecBERT embedding (domain-specific)
    - Hybrid BM25 + dense (practical strong)

  Task 2 (Classification + Retrieval):
    - Majority baseline
    - TF-IDF + Logistic Regression
    - SecBERT fine-tuned classifier
    - DeBERTa-v3-base classifier
    - GPT-5-mini zero-shot (if API key available)

Usage:
    python3 src/etg/benchmark/run_baselines_v2.py --data-dir data/benchmark_v2/
"""

from __future__ import annotations

import json
import os
import sys
import time
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Optional

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


# ============================================================================
# RETRIEVAL METRICS
# ============================================================================

def recall_at_k(rankings: list[list[str]], gold: list[str], k: int) -> float:
    hits = 0
    for ranked, g in zip(rankings, gold):
        if g in ranked[:k]:
            hits += 1
    return hits / len(gold) if gold else 0.0


def mrr(rankings: list[list[str]], gold: list[str]) -> float:
    total = 0.0
    for ranked, g in zip(rankings, gold):
        for i, doc_id in enumerate(ranked):
            if doc_id == g:
                total += 1.0 / (i + 1)
                break
    return total / len(gold) if gold else 0.0


# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================

def f1_binary(preds: list[int], labels: list[int], pos_label: int = 1) -> dict:
    tp = sum(1 for p, l in zip(preds, labels) if p == pos_label and l == pos_label)
    fp = sum(1 for p, l in zip(preds, labels) if p == pos_label and l != pos_label)
    fn = sum(1 for p, l in zip(preds, labels) if p != pos_label and l == pos_label)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ============================================================================
# BM25 RETRIEVAL
# ============================================================================

def run_bm25_retrieval(queries: list[dict], corpus: list[dict], k: int = 10) -> list[list[str]]:
    from rank_bm25 import BM25Okapi

    print("    Building BM25 index...")
    corpus_texts = [doc["text"].lower().split() for doc in corpus]
    corpus_ids = [doc.get("id", doc.get("cve_id", "")) for doc in corpus]
    bm25 = BM25Okapi(corpus_texts)

    print(f"    Retrieving for {len(queries)} queries...")
    rankings = []
    for i, q in enumerate(queries):
        if (i + 1) % 100 == 0:
            print(f"      ...{i+1}/{len(queries)}")
        tokenized_q = q["text"].lower().split()
        scores = bm25.get_scores(tokenized_q)
        top_k_idx = np.argsort(scores)[-k:][::-1]
        rankings.append([corpus_ids[idx] for idx in top_k_idx])

    return rankings


# ============================================================================
# DENSE RETRIEVAL (sentence-transformers)
# ============================================================================

def run_dense_retrieval(queries: list[dict], corpus: list[dict],
                        model_name: str, k: int = 10,
                        batch_size: int = 256) -> list[list[str]]:
    from sentence_transformers import SentenceTransformer
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"    Loading {model_name} on {device}...")
    model = SentenceTransformer(model_name, device=device)

    corpus_ids = [doc.get("id", doc.get("cve_id", "")) for doc in corpus]
    corpus_texts = [doc["text"][:512] for doc in corpus]

    print(f"    Encoding {len(corpus_texts)} corpus docs...")
    corpus_embs = model.encode(corpus_texts, batch_size=batch_size,
                               show_progress_bar=True, normalize_embeddings=True)

    query_texts = [q["text"][:512] for q in queries]
    print(f"    Encoding {len(query_texts)} queries...")
    query_embs = model.encode(query_texts, batch_size=batch_size,
                              show_progress_bar=True, normalize_embeddings=True)

    print("    Computing similarities...")
    # Batch dot product
    sims = np.dot(query_embs, corpus_embs.T)

    rankings = []
    for i in range(len(queries)):
        top_k_idx = np.argsort(sims[i])[-k:][::-1]
        rankings.append([corpus_ids[idx] for idx in top_k_idx])

    del model, corpus_embs, query_embs, sims
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return rankings


# ============================================================================
# HYBRID BM25 + DENSE
# ============================================================================

def run_hybrid_retrieval(queries: list[dict], corpus: list[dict],
                         model_name: str, k: int = 10, alpha: float = 0.7,
                         batch_size: int = 256) -> list[list[str]]:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    corpus_ids = [doc.get("id", doc.get("cve_id", "")) for doc in corpus]
    corpus_texts_raw = [doc["text"] for doc in corpus]

    # BM25 scores
    print("    Building BM25 index for hybrid...")
    corpus_tokenized = [t.lower().split() for t in corpus_texts_raw]
    bm25 = BM25Okapi(corpus_tokenized)

    # Dense scores
    print(f"    Loading {model_name} on {device}...")
    model = SentenceTransformer(model_name, device=device)
    corpus_texts_trunc = [t[:512] for t in corpus_texts_raw]
    print(f"    Encoding corpus...")
    corpus_embs = model.encode(corpus_texts_trunc, batch_size=batch_size,
                               show_progress_bar=True, normalize_embeddings=True)

    query_texts = [q["text"][:512] for q in queries]
    print(f"    Encoding queries...")
    query_embs = model.encode(query_texts, batch_size=batch_size,
                              show_progress_bar=True, normalize_embeddings=True)

    dense_sims = np.dot(query_embs, corpus_embs.T)

    print(f"    Hybrid retrieval for {len(queries)} queries (alpha={alpha})...")
    rankings = []
    for i, q in enumerate(queries):
        if (i + 1) % 100 == 0:
            print(f"      ...{i+1}/{len(queries)}")
        # BM25
        bm25_scores = bm25.get_scores(q["text"].lower().split())
        # Normalize BM25 scores to [0, 1]
        bm25_max = bm25_scores.max()
        if bm25_max > 0:
            bm25_norm = bm25_scores / bm25_max
        else:
            bm25_norm = bm25_scores

        # Dense scores already in [-1, 1] from cosine sim
        dense_norm = (dense_sims[i] + 1) / 2  # Map to [0, 1]

        # Combine: alpha * dense + (1-alpha) * bm25
        combined = alpha * dense_norm + (1 - alpha) * bm25_norm
        top_k_idx = np.argsort(combined)[-k:][::-1]
        rankings.append([corpus_ids[idx] for idx in top_k_idx])

    del model, corpus_embs, query_embs, dense_sims
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return rankings


# ============================================================================
# TF-IDF + LOGISTIC REGRESSION (Classification)
# ============================================================================

def run_tfidf_lr(train_data: list[dict], test_data: list[dict]) -> list[int]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    print("    Training TF-IDF + LR...")
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform([d["text"][:2000] for d in train_data])
    y_train = [d["label"] for d in train_data]

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(X_train, y_train)

    X_test = vectorizer.transform([d["text"][:2000] for d in test_data])
    preds = clf.predict(X_test).tolist()
    return preds


# ============================================================================
# TRANSFORMER CLASSIFIER (SecBERT / DeBERTa)
# ============================================================================

def run_transformer_classifier(train_data: list[dict], test_data: list[dict],
                               model_name: str, epochs: int = 3,
                               batch_size: int = 16, lr: float = 2e-5,
                               max_len: int = 256) -> list[int]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"    Training {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    class TextDataset(Dataset):
        def __init__(self, data):
            self.texts = [d["text"][:1000] for d in data]
            self.labels = [min(d["label"], 1) for d in data]  # Binary: 0 or 1

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = tokenizer(self.texts[idx], truncation=True, max_length=max_len,
                           padding="max_length", return_tensors="pt")
            return {k: v.squeeze(0) for k, v in enc.items()}, self.labels[idx]

    train_ds = TextDataset(train_data)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels_t = torch.tensor(labels, dtype=torch.long).to(device)

            outputs = model(**inputs, labels=labels_t)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"      Epoch {epoch+1}, batch {batch_idx+1}: loss={loss.item():.4f}")

        print(f"      Epoch {epoch+1} avg loss: {total_loss / len(train_loader):.4f}")

    # Evaluate
    model.eval()
    test_ds = TextDataset(test_data)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2)

    all_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return all_preds


# ============================================================================
# MAIN
# ============================================================================

def evaluate_retrieval_task(task_name: str, task_dir: Path, results: dict):
    """Run all retrieval baselines for a task."""
    print(f"\n{'='*60}")
    print(f"  {task_name}")
    print(f"{'='*60}")

    test_data = load_jsonl(task_dir / "test.jsonl")
    corpus = load_jsonl(task_dir / "corpus.jsonl")
    gold_ids = [q["cve_id"] for q in test_data]

    task_results = {}

    # BM25
    print("\n  [BM25]")
    t0 = time.time()
    bm25_rankings = run_bm25_retrieval(test_data, corpus, k=10)
    t1 = time.time()
    task_results["bm25"] = {
        "recall@1": recall_at_k(bm25_rankings, gold_ids, 1),
        "recall@5": recall_at_k(bm25_rankings, gold_ids, 5),
        "recall@10": recall_at_k(bm25_rankings, gold_ids, 10),
        "mrr": mrr(bm25_rankings, gold_ids),
        "time_s": round(t1 - t0, 1),
    }
    print(f"    R@1={task_results['bm25']['recall@1']:.3f}, "
          f"R@5={task_results['bm25']['recall@5']:.3f}, "
          f"R@10={task_results['bm25']['recall@10']:.3f}, "
          f"MRR={task_results['bm25']['mrr']:.3f}")

    # all-MiniLM-L6-v2
    print("\n  [all-MiniLM-L6-v2]")
    t0 = time.time()
    minilm_rankings = run_dense_retrieval(test_data, corpus,
                                          "sentence-transformers/all-MiniLM-L6-v2", k=10)
    t1 = time.time()
    task_results["all-MiniLM-L6-v2"] = {
        "recall@1": recall_at_k(minilm_rankings, gold_ids, 1),
        "recall@5": recall_at_k(minilm_rankings, gold_ids, 5),
        "recall@10": recall_at_k(minilm_rankings, gold_ids, 10),
        "mrr": mrr(minilm_rankings, gold_ids),
        "time_s": round(t1 - t0, 1),
    }
    print(f"    R@1={task_results['all-MiniLM-L6-v2']['recall@1']:.3f}, "
          f"R@5={task_results['all-MiniLM-L6-v2']['recall@5']:.3f}, "
          f"R@10={task_results['all-MiniLM-L6-v2']['recall@10']:.3f}, "
          f"MRR={task_results['all-MiniLM-L6-v2']['mrr']:.3f}")

    # all-mpnet-base-v2
    print("\n  [all-mpnet-base-v2]")
    t0 = time.time()
    mpnet_rankings = run_dense_retrieval(test_data, corpus,
                                         "sentence-transformers/all-mpnet-base-v2", k=10)
    t1 = time.time()
    task_results["all-mpnet-base-v2"] = {
        "recall@1": recall_at_k(mpnet_rankings, gold_ids, 1),
        "recall@5": recall_at_k(mpnet_rankings, gold_ids, 5),
        "recall@10": recall_at_k(mpnet_rankings, gold_ids, 10),
        "mrr": mrr(mpnet_rankings, gold_ids),
        "time_s": round(t1 - t0, 1),
    }
    print(f"    R@1={task_results['all-mpnet-base-v2']['recall@1']:.3f}, "
          f"R@5={task_results['all-mpnet-base-v2']['recall@5']:.3f}, "
          f"R@10={task_results['all-mpnet-base-v2']['recall@10']:.3f}, "
          f"MRR={task_results['all-mpnet-base-v2']['mrr']:.3f}")

    # Hybrid BM25 + mpnet
    print("\n  [Hybrid BM25+mpnet (alpha=0.7)]")
    t0 = time.time()
    hybrid_rankings = run_hybrid_retrieval(test_data, corpus,
                                           "sentence-transformers/all-mpnet-base-v2",
                                           k=10, alpha=0.7)
    t1 = time.time()
    task_results["hybrid_bm25_mpnet"] = {
        "recall@1": recall_at_k(hybrid_rankings, gold_ids, 1),
        "recall@5": recall_at_k(hybrid_rankings, gold_ids, 5),
        "recall@10": recall_at_k(hybrid_rankings, gold_ids, 10),
        "mrr": mrr(hybrid_rankings, gold_ids),
        "time_s": round(t1 - t0, 1),
    }
    print(f"    R@1={task_results['hybrid_bm25_mpnet']['recall@1']:.3f}, "
          f"R@5={task_results['hybrid_bm25_mpnet']['recall@5']:.3f}, "
          f"R@10={task_results['hybrid_bm25_mpnet']['recall@10']:.3f}, "
          f"MRR={task_results['hybrid_bm25_mpnet']['mrr']:.3f}")

    results[task_name] = task_results


def evaluate_classification_task(task_dir: Path, results: dict):
    """Run all classification baselines for Task 2."""
    print(f"\n{'='*60}")
    print("  Task 2: Signal Detection (Classification)")
    print(f"{'='*60}")

    train_data = load_jsonl(task_dir / "train.jsonl")
    test_data = load_jsonl(task_dir / "test.jsonl")
    test_labels = [min(d["label"], 1) for d in test_data]

    task_results = {}

    # Majority baseline
    train_labels = [min(d["label"], 1) for d in train_data]
    majority = Counter(train_labels).most_common(1)[0][0]
    majority_preds = [majority] * len(test_labels)
    task_results["majority"] = f1_binary(majority_preds, test_labels)
    print(f"\n  [Majority]: F1={task_results['majority']['f1']:.3f}")

    # TF-IDF + LR
    print("\n  [TF-IDF + LR]")
    tfidf_preds = run_tfidf_lr(train_data, test_data)
    tfidf_preds_binary = [min(p, 1) for p in tfidf_preds]
    task_results["tfidf_lr"] = f1_binary(tfidf_preds_binary, test_labels)
    print(f"    F1={task_results['tfidf_lr']['f1']:.3f}, "
          f"P={task_results['tfidf_lr']['precision']:.3f}, "
          f"R={task_results['tfidf_lr']['recall']:.3f}")

    # SecBERT classifier
    print("\n  [SecBERT]")
    try:
        secbert_preds = run_transformer_classifier(
            train_data, test_data, "jackaduma/SecBERT",
            epochs=3, batch_size=16, max_len=256
        )
        task_results["secbert"] = f1_binary(secbert_preds, test_labels)
        print(f"    F1={task_results['secbert']['f1']:.3f}, "
              f"P={task_results['secbert']['precision']:.3f}, "
              f"R={task_results['secbert']['recall']:.3f}")
    except Exception as e:
        print(f"    FAILED: {e}")
        task_results["secbert"] = {"f1": None, "error": str(e)}

    # DeBERTa-v3-base
    print("\n  [DeBERTa-v3-base]")
    try:
        deberta_preds = run_transformer_classifier(
            train_data, test_data, "microsoft/deberta-v3-base",
            epochs=3, batch_size=8, max_len=256
        )
        task_results["deberta_v3_base"] = f1_binary(deberta_preds, test_labels)
        print(f"    F1={task_results['deberta_v3_base']['f1']:.3f}, "
              f"P={task_results['deberta_v3_base']['precision']:.3f}, "
              f"R={task_results['deberta_v3_base']['recall']:.3f}")
    except Exception as e:
        print(f"    FAILED: {e}")
        task_results["deberta_v3_base"] = {"f1": None, "error": str(e)}

    results["task2_signal_detection"] = task_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/benchmark_v2"))
    parser.add_argument("--output", type=Path, default=Path("data/benchmark_v2/results.json"))
    parser.add_argument("--skip-dense", action="store_true", help="Skip dense retrieval (slow)")
    parser.add_argument("--task", choices=["1", "2", "3", "all"], default="all")
    args = parser.parse_args()

    results = {}

    if args.task in ("1", "all"):
        evaluate_retrieval_task("task1_cve_linkage",
                               args.data_dir / "task1_cve_linkage", results)

    if args.task in ("2", "all"):
        evaluate_classification_task(args.data_dir / "task2_signal_detection", results)

    if args.task in ("3", "all"):
        evaluate_retrieval_task("task3_temporal_generalization",
                               args.data_dir / "task3_temporal_generalization", results)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to {args.output}")

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for task_name, task_res in results.items():
        print(f"\n{task_name}:")
        for model_name, metrics in task_res.items():
            if isinstance(metrics, dict):
                metric_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()
                                      if isinstance(v, (int, float)) and v is not None)
                print(f"  {model_name:30s} {metric_str}")


if __name__ == "__main__":
    main()
