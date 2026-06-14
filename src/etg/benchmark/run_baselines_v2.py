"""Run the full model ladder for HackerSignal benchmark v2.

Model ladder:
  Task 1 & 3 (Retrieval):
    - BM25 (lexical)
    - all-MiniLM-L6-v2 (lightweight dense)
    - all-mpnet-base-v2 (stronger dense)
    - SecBERT embedding (domain-specific)
    - Hybrid BM25 + dense (practical strong)

  Task 2 (8-class Exploit Type Classification):
    Bag-of-words classifiers (TF-IDF features):
      - Decision Tree (max_depth=30, class_weight=balanced)
      - TF-IDF + Logistic Regression (C=1.0, class_weight=balanced)
      - Linear SVM (max_iter=2000, class_weight=balanced)
    Recurrent neural networks (word-level, 128-dim emb/hidden):
      - RNN, GRU, LSTM, BiLSTM
    Domain-specific transformer:
      - SecBERT (jackaduma/SecBERT) fine-tuned

  Metrics: macro-F1, weighted-F1, accuracy, and per-class F1.

Usage:
    python3 src/etg/benchmark/run_baselines_v2.py --data-dir data/benchmark_v2/
    python3 src/etg/benchmark/run_baselines_v2.py --task 2          # Task 2 only
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

SEED = 42

# Canonical 8-class exploit-type taxonomy (fixed order for per-class reporting).
EXPLOIT_TYPE_CLASSES = [
    "injection", "xss", "memory_corruption", "dos",
    "file_inclusion", "auth_access", "rce", "info_disclosure",
]


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and (if available) PyTorch for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass


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
# MULTICLASS CLASSIFICATION METRICS
# ============================================================================

def multiclass_metrics(preds: list[str], labels: list[str]) -> dict:
    """Macro-F1, weighted-F1, accuracy, and per-class F1 over EXPLOIT_TYPE_CLASSES."""
    from sklearn.metrics import f1_score, accuracy_score

    macro = f1_score(labels, preds, average="macro", zero_division=0)
    weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    per_class_arr = f1_score(labels, preds, average=None,
                             labels=EXPLOIT_TYPE_CLASSES, zero_division=0)
    per_class = {c: round(float(v), 4) for c, v in zip(EXPLOIT_TYPE_CLASSES, per_class_arr)}
    return {
        "macro_f1": round(float(macro), 4),
        "weighted_f1": round(float(weighted), 4),
        "accuracy": round(float(acc), 4),
        "per_class_f1": per_class,
    }


def _print_clf_metrics(name: str, m: dict) -> None:
    print(f"    {name}: macro-F1={m['macro_f1']:.3f}, "
          f"weighted-F1={m['weighted_f1']:.3f}, acc={m['accuracy']:.3f}")


# ============================================================================
# BAG-OF-WORDS CLASSIFIERS (Decision Tree, TF-IDF + LR, Linear SVM)
# ============================================================================

def run_bow_classifier(train_data: list[dict], test_data: list[dict],
                       kind: str) -> list[str]:
    """Train a TF-IDF bag-of-words classifier and return string-label predictions.

    kind in {"decision_tree", "logreg", "svm"}. Hyperparameters follow the paper
    appendix: TfidfVectorizer(max_features=50000, ngram_range=(1,2)); all
    classifiers use class_weight="balanced".
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform([d["text"][:2000] for d in train_data])
    y_train = [d["label"] for d in train_data]

    if kind == "decision_tree":
        clf = DecisionTreeClassifier(max_depth=30, class_weight="balanced",
                                     random_state=SEED)
    elif kind == "logreg":
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                 random_state=SEED)
    elif kind == "svm":
        clf = LinearSVC(max_iter=2000, class_weight="balanced", random_state=SEED)
    else:
        raise ValueError(f"unknown BoW classifier kind: {kind}")

    clf.fit(X_train, y_train)
    X_test = vectorizer.transform([d["text"][:2000] for d in test_data])
    return clf.predict(X_test).tolist()


# ============================================================================
# RECURRENT NEURAL CLASSIFIERS (RNN / GRU / LSTM / BiLSTM)
# ============================================================================

def _build_vocab(train_data: list[dict], min_freq: int = 3) -> dict:
    """Word-level vocabulary from training texts; index 0=<pad>, 1=<unk>."""
    counts = Counter()
    for d in train_data:
        counts.update(d["text"].lower().split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in counts.items():
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def _encode(text: str, vocab: dict, max_len: int = 300) -> list[int]:
    ids = [vocab.get(tok, 1) for tok in text.lower().split()[:max_len]]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids


def run_rnn_classifier(train_data: list[dict], val_data: list[dict],
                       test_data: list[dict], cell: str, bidirectional: bool,
                       epochs: int = 8, batch_size: int = 128, lr: float = 1e-3,
                       max_len: int = 300, embed_dim: int = 128,
                       hidden_dim: int = 128) -> list[str]:
    """Word-level recurrent classifier with class-weighted CE and best-val-F1 checkpoint.

    cell in {"rnn", "gru", "lstm"}; bidirectional=True gives BiLSTM when cell=="lstm".
    Hyperparameters follow the paper appendix.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import f1_score
    from sklearn.utils.class_weight import compute_class_weight

    set_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    vocab = _build_vocab(train_data, min_freq=3)
    label2id = {c: i for i, c in enumerate(EXPLOIT_TYPE_CLASSES)}
    id2label = {i: c for c, i in label2id.items()}
    num_classes = len(label2id)

    def to_tensors(data):
        X = torch.tensor([_encode(d["text"], vocab, max_len) for d in data],
                         dtype=torch.long)
        y = torch.tensor([label2id[d["label"]] for d in data], dtype=torch.long)
        return X, y

    X_tr, y_tr = to_tensors(train_data)
    X_val, y_val = to_tensors(val_data)
    X_te, y_te = to_tensors(test_data)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size,
                              shuffle=True)

    # Class-weighted cross-entropy (balanced over present classes)
    present = np.array(sorted(set(y_tr.tolist())))
    cw = compute_class_weight("balanced", classes=present, y=y_tr.numpy())
    weight = torch.ones(num_classes)
    for cls, w in zip(present, cw):
        weight[cls] = w
    criterion = nn.CrossEntropyLoss(weight=weight.to(device))

    class RNNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(len(vocab), embed_dim, padding_idx=0)
            rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell]
            self.rnn = rnn_cls(embed_dim, hidden_dim, batch_first=True,
                               bidirectional=bidirectional)
            self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

        def forward(self, x):
            emb = self.embed(x)
            out, hidden = self.rnn(emb)
            if cell == "lstm":
                hidden = hidden[0]  # (h_n, c_n) -> h_n
            if bidirectional:
                last = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                last = hidden[-1]
            return self.fc(last)

    model = RNNClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def predict(X):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = X[i:i + batch_size].to(device)
                logits = model(xb)
                preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        return preds

    best_val_f1 = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        val_preds = predict(X_val)
        val_f1 = f1_score(y_val.tolist(), val_preds, average="macro", zero_division=0)
        print(f"      Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f}, "
              f"val macro-F1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_pred_ids = predict(X_te)

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return [id2label[i] for i in test_pred_ids]


# ============================================================================
# TRANSFORMER CLASSIFIER (SecBERT)
# ============================================================================

def run_transformer_classifier(train_data: list[dict], test_data: list[dict],
                               model_name: str, epochs: int = 3,
                               batch_size: int = 16, lr: float = 2e-5,
                               max_len: int = 256, weight_decay: float = 0.01,
                               warmup_ratio: float = 0.1,
                               subsample: Optional[int] = 15000) -> list[str]:
    """Fine-tune a transformer for 8-class exploit-type classification.

    Follows the paper appendix: AdamW(lr, weight_decay), linear warmup over
    warmup_ratio of steps, optional train subsample. Returns string labels.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_linear_schedule_with_warmup)

    set_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"    Training {model_name} on {device}...")

    label2id = {c: i for i, c in enumerate(EXPLOIT_TYPE_CLASSES)}
    id2label = {i: c for c, i in label2id.items()}

    if subsample and len(train_data) > subsample:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(train_data), size=subsample, replace=False)
        train_data = [train_data[i] for i in idx]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label2id)
    ).to(device)

    class TextDataset(Dataset):
        def __init__(self, data):
            self.texts = [d["text"][:1000] for d in data]
            self.labels = [label2id[d["label"]] for d in data]

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = tokenizer(self.texts[idx], truncation=True, max_length=max_len,
                           padding="max_length", return_tensors="pt")
            return {k: v.squeeze(0) for k, v in enc.items()}, self.labels[idx]

    train_ds = TextDataset(train_data)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps)
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
            scheduler.step()
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

    return [id2label[i] for i in all_preds]


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


def evaluate_classification_task(task_dir: Path, results: dict, skip_neural: bool = False):
    """Run the 8-class exploit-type classification ladder for Task 2."""
    print(f"\n{'='*60}")
    print("  Task 2: Exploit Type Classification (8-class)")
    print(f"{'='*60}")

    train_data = load_jsonl(task_dir / "train.jsonl")
    val_data = load_jsonl(task_dir / "val.jsonl")
    test_data = load_jsonl(task_dir / "test.jsonl")
    test_labels = [d["label"] for d in test_data]

    task_results = {}

    # ── Bag-of-words classifiers ──────────────────────────────────────────────
    for kind, label in [("decision_tree", "Decision Tree"),
                        ("logreg", "TF-IDF + LR"),
                        ("svm", "SVM")]:
        print(f"\n  [{label}]")
        preds = run_bow_classifier(train_data, test_data, kind)
        task_results[label] = multiclass_metrics(preds, test_labels)
        _print_clf_metrics(label, task_results[label])

    # ── Recurrent neural classifiers ──────────────────────────────────────────
    if not skip_neural:
        for cell, bidir, label in [("rnn", False, "RNN"),
                                   ("gru", False, "GRU"),
                                   ("lstm", False, "LSTM"),
                                   ("lstm", True, "BiLSTM")]:
            print(f"\n  [{label}]")
            try:
                preds = run_rnn_classifier(train_data, val_data, test_data,
                                           cell=cell, bidirectional=bidir)
                task_results[label] = multiclass_metrics(preds, test_labels)
                _print_clf_metrics(label, task_results[label])
            except Exception as e:
                print(f"    FAILED: {e}")
                task_results[label] = {"macro_f1": None, "error": str(e)}

        # ── SecBERT transformer ───────────────────────────────────────────────
        print("\n  [SecBERT]")
        try:
            secbert_preds = run_transformer_classifier(
                train_data, test_data, "jackaduma/SecBERT",
                epochs=3, batch_size=16, max_len=256, subsample=15000
            )
            task_results["SecBERT"] = multiclass_metrics(secbert_preds, test_labels)
            _print_clf_metrics("SecBERT", task_results["SecBERT"])
        except Exception as e:
            print(f"    FAILED: {e}")
            task_results["SecBERT"] = {"macro_f1": None, "error": str(e)}

    results["task2_exploit_type"] = task_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/benchmark_v2"))
    parser.add_argument("--output", type=Path, default=Path("data/benchmark_v2/results.json"))
    parser.add_argument("--skip-dense", action="store_true", help="Skip dense retrieval (slow)")
    parser.add_argument("--skip-neural", action="store_true",
                        help="Task 2: skip RNN/GRU/LSTM/BiLSTM/SecBERT (BoW only)")
    parser.add_argument("--task", choices=["1", "2", "3", "all"], default="all")
    args = parser.parse_args()

    set_seed(SEED)
    results = {}

    if args.task in ("1", "all"):
        evaluate_retrieval_task("task1_cve_linkage",
                               args.data_dir / "task1_cve_linkage", results)

    if args.task in ("2", "all"):
        evaluate_classification_task(args.data_dir / "task2_exploit_type", results,
                                     skip_neural=args.skip_neural)

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
