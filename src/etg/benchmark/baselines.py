"""HackerSignal benchmark baseline models.

Three lightweight baselines for each task, designed to run on a CPU or a
single GPU in reasonable time and set a fair lower bound for future work.

Task 1 — Exploit Relevance Classification
    B1a: TF-IDF (unigrams+bigrams, 50k features) + Logistic Regression
    B1b: Fine-tuned SecBERT (jackaduma/SecBERT, 128 max tokens, 3 epochs)

Task 2 — CVE Linkage Retrieval
    B2a: BM25 (rank_bm25) over NVD corpus
    B2b: Bi-encoder (sentence-transformers/all-MiniLM-L6-v2) cosine retrieval

Task 3 — Severity Prediction
    B3a: TF-IDF + Logistic Regression (multiclass)
    B3b: Fine-tuned SecBERT (jackaduma/SecBERT, 128 max tokens, 3 epochs)

Task 4 — Hacker Exploit Labeling
    B4a: TF-IDF + Logistic Regression (ternary)

Task 5 — Hacker Exploit Signal Detection
    B5a: TF-IDF + Logistic Regression for actionability plus BM25 CVE retrieval

Usage
-----
    python -m etg.benchmark.baselines --task 1 --benchmark-dir data/benchmark/ --output data/results/
    python -m etg.benchmark.baselines --task 2 --benchmark-dir data/benchmark/ --output data/results/
    python -m etg.benchmark.baselines --task all --benchmark-dir data/benchmark/ --output data/results/
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Shared I/O helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Results written to %s", path)


# ---------------------------------------------------------------------------
# Task 1 — Exploit Relevance Classification baselines
# ---------------------------------------------------------------------------


def run_task1_tfidf(benchmark_dir: Path, output_dir: Path) -> dict:
    """TF-IDF + Logistic Regression baseline for Task 1."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        classification_report, f1_score, roc_auc_score, precision_score, recall_score
    )
    import numpy as np

    task_dir = benchmark_dir / "task1_exploit_clf"
    train = _load_jsonl(task_dir / "train.jsonl")
    val   = _load_jsonl(task_dir / "val.jsonl")
    test  = _load_jsonl(task_dir / "test.jsonl")

    log.info("task1/tfidf: train=%d val=%d test=%d", len(train), len(val), len(test))

    X_train = [r["text"] for r in train]
    y_train = [r["label"] for r in train]
    X_val   = [r["text"] for r in val]
    y_val   = [r["label"] for r in val]
    X_test  = [r["text"] for r in test]
    y_test  = [r["label"] for r in test]

    log.info("task1/tfidf: fitting TF-IDF vectorizer (50k features, unigrams+bigrams)...")
    t0 = time.time()
    vec = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_tr_vec = vec.fit_transform(X_train)
    X_va_vec = vec.transform(X_val)
    X_te_vec = vec.transform(X_test)
    log.info("task1/tfidf: vectorized in %.1fs", time.time() - t0)

    log.info("task1/tfidf: fitting Logistic Regression...")
    t0 = time.time()
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", n_jobs=-1, random_state=0)
    clf.fit(X_tr_vec, y_train)
    log.info("task1/tfidf: trained in %.1fs", time.time() - t0)

    def _eval(X_vec, y_true: list[int], split: str) -> dict:
        y_pred = clf.predict(X_vec)
        y_prob = clf.predict_proba(X_vec)[:, 1]
        f1m = f1_score(y_true, y_pred, average="macro")
        f1p = f1_score(y_true, y_pred, pos_label=1)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")
        prec = precision_score(y_true, y_pred, pos_label=1)
        rec  = recall_score(y_true, y_pred, pos_label=1)
        log.info("task1/tfidf [%s]: F1-macro=%.4f F1+=%.4f AUC=%.4f P=%.4f R=%.4f",
                 split, f1m, f1p, auc, prec, rec)
        return {
            "split": split,
            "f1_macro": round(f1m, 4),
            "f1_positive": round(f1p, 4),
            "auc_roc": round(auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "n": len(y_true),
        }

    results = {
        "task": "task1_exploit_clf",
        "model": "tfidf_logreg",
        "model_description": "TF-IDF (unigrams+bigrams, 50k features) + Logistic Regression (balanced)",
        "val": _eval(X_va_vec, y_val, "val"),
        "test": _eval(X_te_vec, y_test, "test"),
    }
    _save_results(results, output_dir / "task1_tfidf_logreg.json")
    return results


def run_task1_source_only(benchmark_dir: Path, output_dir: Path) -> dict:
    """Source-ID shortcut diagnostic for Task 1.

    This is not a semantic text model. It estimates how much of the Task 1
    signal can be recovered from source membership alone. On the current
    within-source Task 1 split this should be near chance; elevated
    performance would indicate a regression to source-membership shortcuts.
    """
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    task_dir = benchmark_dir / "task1_exploit_clf"
    train = _load_jsonl(task_dir / "train.jsonl")
    val = _load_jsonl(task_dir / "val.jsonl")
    test = _load_jsonl(task_dir / "test.jsonl")

    def _features(records: list[dict]) -> list[dict[str, str]]:
        return [{"source": str(r.get("source", "unknown"))} for r in records]

    y_train = [r["label"] for r in train]
    y_val = [r["label"] for r in val]
    y_test = [r["label"] for r in test]

    vec = DictVectorizer()
    X_train = vec.fit_transform(_features(train))
    X_val = vec.transform(_features(val))
    X_test = vec.transform(_features(test))

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=0)
    clf.fit(X_train, y_train)

    def _eval(X_vec, y_true: list[int], split: str) -> dict:
        y_pred = clf.predict(X_vec)
        y_prob = clf.predict_proba(X_vec)[:, 1]
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")
        out = {
            "split": split,
            "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
            "f1_positive": round(f1_score(y_true, y_pred, pos_label=1), 4),
            "auc_roc": round(auc, 4),
            "precision": round(precision_score(y_true, y_pred, pos_label=1), 4),
            "recall": round(recall_score(y_true, y_pred, pos_label=1), 4),
            "n": len(y_true),
        }
        log.info(
            "task1/source_only [%s]: F1-macro=%.4f F1+=%.4f AUC=%.4f",
            split,
            out["f1_macro"],
            out["f1_positive"],
            out["auc_roc"],
        )
        return out

    results = {
        "task": "task1_exploit_clf",
        "model": "source_only_logreg",
        "model_description": "Diagnostic logistic regression using only source identifier features",
        "feature_names": vec.get_feature_names_out().tolist(),
        "val": _eval(X_val, y_val, "val"),
        "test": _eval(X_test, y_test, "test"),
        "interpretation": (
            "High performance indicates source-membership confounding. Chance "
            "AUC supports the within-source Task 1 split design."
        ),
    }
    _save_results(results, output_dir / "task1_source_only.json")
    return results


def run_task1_secbert(
    benchmark_dir: Path,
    output_dir: Path,
    model_name: str = "jackaduma/SecBERT",
    max_length: int = 128,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    device: str | None = None,
    seed: int | None = None,
) -> dict:
    """Fine-tuned SecBERT baseline for Task 1."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from sklearn.metrics import f1_score, roc_auc_score

    _set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("task1/secbert: using device=%s", device)

    task_dir = benchmark_dir / "task1_exploit_clf"
    train = _load_jsonl(task_dir / "train.jsonl")
    val   = _load_jsonl(task_dir / "val.jsonl")
    test  = _load_jsonl(task_dir / "test.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = model.to(device)

    class _DS(Dataset):
        def __init__(self, records: list[dict]):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            r = self.records[idx]
            enc = tokenizer(
                r["text"],
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(r["label"], dtype=torch.long),
            }

    train_loader = DataLoader(_DS(train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(_DS(val),   batch_size=batch_size * 2)
    test_loader  = DataLoader(_DS(test),  batch_size=batch_size * 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def _train_epoch():
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["label"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += out.loss.item()
        return total_loss / len(train_loader)

    def _evaluate(loader, split: str) -> dict:
        model.eval()
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for batch in loader:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
                preds = out.logits.argmax(dim=-1).cpu().numpy()
                all_labels.extend(batch["label"].numpy())
                all_preds.extend(preds)
                all_probs.extend(probs)

        f1m = f1_score(all_labels, all_preds, average="macro")
        f1p = f1_score(all_labels, all_preds, pos_label=1)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = float("nan")
        log.info("task1/secbert [%s]: F1-macro=%.4f F1+=%.4f AUC=%.4f", split, f1m, f1p, auc)
        return {
            "split": split,
            "f1_macro": round(float(f1m), 4),
            "f1_positive": round(float(f1p), 4),
            "auc_roc": round(float(auc), 4),
            "n": len(all_labels),
        }

    val_results_by_epoch = []
    for epoch in range(1, epochs + 1):
        loss = _train_epoch()
        log.info("task1/secbert: epoch %d/%d — train_loss=%.4f", epoch, epochs, loss)
        val_results_by_epoch.append(_evaluate(val_loader, f"val_epoch{epoch}"))

    results = {
        "task": "task1_exploit_clf",
        "model": "secbert_finetuned",
        "model_description": f"{model_name} fine-tuned for binary classification, {epochs} epochs, lr={lr}",
        "hyperparams": {
            "model": model_name,
            "max_length": max_length,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
        },
        "val_by_epoch": val_results_by_epoch,
        "val": _evaluate(val_loader, "val"),
        "test": _evaluate(test_loader, "test"),
    }
    suffix = f"_seed{seed}" if seed is not None else ""
    _save_results(results, output_dir / f"task1_secbert{suffix}.json")

    # Save model checkpoint
    ckpt_dir = output_dir / f"task1_secbert{suffix}_checkpoint"
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    log.info("task1/secbert: checkpoint saved to %s", ckpt_dir)

    return results


# ---------------------------------------------------------------------------
# Task 2 — CVE Linkage Retrieval baselines
# ---------------------------------------------------------------------------


def run_task2_bm25(benchmark_dir: Path, output_dir: Path, ks: list[int] = [1, 5, 10]) -> dict:
    """BM25 retrieval baseline for Task 2."""
    import numpy as np
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import CountVectorizer

    task_dir = benchmark_dir / "task2_cve_linkage"
    corpus_records = _load_jsonl(task_dir / "corpus.jsonl")
    val_queries    = _load_jsonl(task_dir / "val.jsonl")
    test_queries   = _load_jsonl(task_dir / "test.jsonl")

    log.info("task2/bm25: corpus=%d queries_val=%d queries_test=%d",
             len(corpus_records), len(val_queries), len(test_queries))

    # Build sparse BM25 document-term matrix. This avoids an optional
    # rank_bm25 dependency and supports batched top-k scoring.
    log.info("task2/bm25: building sparse BM25 index...")
    t0 = time.time()
    corpus_cves  = [r["cve_id"] for r in corpus_records]
    corpus_texts = [r["text"] for r in corpus_records]
    vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b[\w./:-]{2,}\b")
    doc_tf = vectorizer.fit_transform(corpus_texts).astype(np.float32).tocsr()
    n_docs = doc_tf.shape[0]
    doc_len = np.asarray(doc_tf.sum(axis=1)).ravel()
    avg_doc_len = float(doc_len.mean()) if n_docs else 0.0
    df = np.asarray((doc_tf > 0).sum(axis=0)).ravel()
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)

    k1 = 1.5
    b = 0.75
    denom_norm = k1 * (1.0 - b + b * doc_len / max(avg_doc_len, 1e-9))
    bm25 = doc_tf.copy()
    rows, _cols = bm25.nonzero()
    bm25.data = bm25.data * (k1 + 1.0) / (bm25.data + denom_norm[rows])
    bm25 = bm25.multiply(idf).tocsr()
    bm25_t = bm25.T.tocsr()
    log.info("task2/bm25: indexed %d docs in %.1fs", len(corpus_records), time.time() - t0)

    cve_to_indices: dict[str, list[int]] = {}
    for idx, cve in enumerate(corpus_cves):
        cve_to_indices.setdefault(cve, []).append(idx)

    def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
        if len(scores) <= k:
            return np.argsort(-scores)
        candidates = np.argpartition(scores, -k)[-k:]
        return candidates[np.argsort(-scores[candidates])]

    def _eval(queries: list[dict], split: str, batch_size: int = 256) -> dict:
        max_k = max(ks)
        hits_k = {k: 0 for k in ks}
        rr_sum = 0.0
        missing_cve = 0
        texts = [q["text"] for q in queries]
        query_tf = vectorizer.transform(texts).astype(np.float32).tocsr()

        for start in range(0, len(queries), batch_size):
            end = min(start + batch_size, len(queries))
            score_mat = (query_tf[start:end] @ bm25_t).toarray()
            for offset, scores in enumerate(score_mat):
                q = queries[start + offset]
                true_indices = cve_to_indices.get(q["cve_id"])
                if not true_indices:
                    missing_cve += 1
                    continue
                true_set = set(true_indices)
                ranked = _top_indices(scores, max_k)
                rank = next((i for i, idx in enumerate(ranked, 1) if idx in true_set), None)
                if rank is None:
                    continue
                rr_sum += 1.0 / rank
                for k in ks:
                    if rank <= k:
                        hits_k[k] += 1

        out: dict[str, Any] = {"split": split, "n": len(queries), "missing_cve": missing_cve}
        denom = max(len(queries), 1)
        for k in ks:
            out[f"recall_at_{k}"] = round(hits_k[k] / denom, 4)
            log.info("task2/bm25 [%s]: R@%d=%.4f", split, k, out[f"recall_at_{k}"])
        out["mrr"] = round(rr_sum / denom, 4)
        log.info("task2/bm25 [%s]: MRR=%.4f", split, out["mrr"])
        return out

    results = {
        "task": "task2_cve_linkage",
        "model": "bm25",
        "model_description": "BM25 (Okapi) retrieval over NVD CVE corpus",
        "val": _eval(val_queries, "val"),
        "test": _eval(test_queries, "test"),
    }
    _save_results(results, output_dir / "task2_bm25.json")
    return results


def run_task2_biencoder(
    benchmark_dir: Path,
    output_dir: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
    score_batch_size: int = 64,
    ks: list[int] = [1, 5, 10],
) -> dict:
    """Bi-encoder (sentence-transformers) baseline for Task 2."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    task_dir = benchmark_dir / "task2_cve_linkage"
    corpus_records = _load_jsonl(task_dir / "corpus.jsonl")
    val_queries    = _load_jsonl(task_dir / "val.jsonl")
    test_queries   = _load_jsonl(task_dir / "test.jsonl")

    log.info("task2/biencoder: model=%s corpus=%d val=%d test=%d",
             model_name, len(corpus_records), len(val_queries), len(test_queries))

    model = SentenceTransformer(model_name)

    log.info("task2/biencoder: encoding corpus...")
    t0 = time.time()
    corpus_cves   = [r["cve_id"] for r in corpus_records]
    corpus_texts  = [r["text"] for r in corpus_records]
    cache_key = model_name.replace("/", "__")
    emb_cache = output_dir / f"task2_biencoder_{cache_key}_corpus_embs.npy"
    cve_cache = output_dir / f"task2_biencoder_{cache_key}_corpus_cves.json"
    sibling_cache: tuple[Path, Path] | None = None
    for candidate in sorted(output_dir.parent.glob("seed_*/task2_biencoder_*_corpus_embs.npy")):
        candidate_cves = candidate.with_name(candidate.name.replace("_corpus_embs.npy", "_corpus_cves.json"))
        if candidate == emb_cache or not candidate_cves.exists():
            continue
        if candidate.name == emb_cache.name:
            sibling_cache = (candidate, candidate_cves)
            break
    if emb_cache.exists() and cve_cache.exists():
        log.info("task2/biencoder: loading cached corpus embeddings from %s", emb_cache)
        corpus_embs = np.load(emb_cache)
        corpus_cves = json.loads(cve_cache.read_text(encoding="utf-8"))
    elif sibling_cache is not None:
        sibling_emb_cache, sibling_cve_cache = sibling_cache
        log.info("task2/biencoder: reusing sibling corpus embedding cache from %s", sibling_emb_cache)
        corpus_embs = np.load(sibling_emb_cache)
        corpus_cves = json.loads(sibling_cve_cache.read_text(encoding="utf-8"))
    else:
        corpus_embs = model.encode(
            corpus_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_cache, corpus_embs)
        cve_cache.write_text(json.dumps(corpus_cves), encoding="utf-8")
    if len(corpus_cves) != len(corpus_records) or corpus_embs.shape[0] != len(corpus_records):
        raise ValueError(
            "Cached corpus embedding shape does not match current Task 2 corpus: "
            f"embs={corpus_embs.shape[0]} cves={len(corpus_cves)} corpus={len(corpus_records)}"
        )
    log.info("task2/biencoder: corpus encoded in %.1fs", time.time() - t0)

    def _eval(queries: list[dict], split: str) -> dict:
        q_texts = [q["text"] for q in queries]
        q_embs  = model.encode(q_texts, batch_size=batch_size, show_progress_bar=False,
                               convert_to_numpy=True, normalize_embeddings=True)

        out: dict[str, Any] = {"split": split, "n": len(queries)}
        hits_k  = {k: 0 for k in ks}
        rr_sum  = 0.0

        max_k = max(ks)
        for start in range(0, len(queries), score_batch_size):
            end = min(start + score_batch_size, len(queries))
            scores = q_embs[start:end] @ corpus_embs.T
            for offset, q in enumerate(queries[start:end]):
                row = scores[offset]
                if len(row) <= max_k:
                    ranked = np.argsort(-row)
                else:
                    candidates = np.argpartition(row, -max_k)[-max_k:]
                    ranked = candidates[np.argsort(-row[candidates])]
                for rank, idx in enumerate(ranked, 1):
                    if corpus_cves[idx] == q["cve_id"]:
                        rr_sum += 1.0 / rank
                        for k in ks:
                            if rank <= k:
                                hits_k[k] += 1
                        break

        for k in ks:
            rk = hits_k[k] / max(len(queries), 1)
            out[f"recall_at_{k}"] = round(rk, 4)
            log.info("task2/biencoder [%s]: R@%d=%.4f", split, k, rk)
        mrr = rr_sum / max(len(queries), 1)
        out["mrr"] = round(mrr, 4)
        log.info("task2/biencoder [%s]: MRR=%.4f", split, mrr)
        return out

    results = {
        "task": "task2_cve_linkage",
        "model": "biencoder",
        "model_description": f"Bi-encoder ({model_name}) with cosine similarity over NVD corpus",
        "score_batch_size": score_batch_size,
        "val": _eval(val_queries, "val"),
        "test": _eval(test_queries, "test"),
    }
    _save_results(results, output_dir / "task2_biencoder.json")
    return results


# ---------------------------------------------------------------------------
# Task 3 — Severity Prediction baselines
# ---------------------------------------------------------------------------


def run_task3_tfidf(benchmark_dir: Path, output_dir: Path) -> dict:
    """TF-IDF + Logistic Regression baseline for Task 3."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score, accuracy_score

    task_dir = benchmark_dir / "task3_severity"
    train = _load_jsonl(task_dir / "train.jsonl")
    val   = _load_jsonl(task_dir / "val.jsonl")
    test  = _load_jsonl(task_dir / "test.jsonl")

    log.info("task3/tfidf: train=%d val=%d test=%d", len(train), len(val), len(test))

    X_train = [r["text"] for r in train]
    y_train = [r["severity_label"] for r in train]
    X_val   = [r["text"] for r in val]
    y_val   = [r["severity_label"] for r in val]
    X_test  = [r["text"] for r in test]
    y_test  = [r["severity_label"] for r in test]

    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X_tr_vec = vec.fit_transform(X_train)
    X_va_vec = vec.transform(X_val)
    X_te_vec = vec.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", n_jobs=-1, random_state=0)
    clf.fit(X_tr_vec, y_train)

    def _eval(X_vec, y_true: list[int], split: str) -> dict:
        y_pred = clf.predict(X_vec)
        f1m = f1_score(y_true, y_pred, average="macro")
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred,
                                       target_names=["low", "medium", "high", "critical"],
                                       output_dict=True)
        log.info("task3/tfidf [%s]: F1-macro=%.4f acc=%.4f", split, f1m, acc)
        return {
            "split": split,
            "f1_macro": round(f1m, 4),
            "accuracy": round(acc, 4),
            "f1_per_class": {
                k: round(v["f1-score"], 4)
                for k, v in report.items()
                if k in {"low", "medium", "high", "critical"}
            },
            "n": len(y_true),
        }

    results = {
        "task": "task3_severity",
        "model": "tfidf_logreg",
        "model_description": "TF-IDF (unigrams+bigrams, 50k features) + Multinomial Logistic Regression",
        "val": _eval(X_va_vec, y_val, "val"),
        "test": _eval(X_te_vec, y_test, "test"),
    }
    _save_results(results, output_dir / "task3_tfidf_logreg.json")
    return results


def run_task3_secbert(
    benchmark_dir: Path,
    output_dir: Path,
    model_name: str = "jackaduma/SecBERT",
    max_length: int = 128,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    device: str | None = None,
    seed: int | None = None,
) -> dict:
    """Fine-tuned SecBERT baseline for Task 3 (severity prediction)."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from sklearn.metrics import f1_score, accuracy_score

    _set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("task3/secbert: using device=%s", device)

    task_dir = benchmark_dir / "task3_severity"
    train = _load_jsonl(task_dir / "train.jsonl")
    val   = _load_jsonl(task_dir / "val.jsonl")
    test  = _load_jsonl(task_dir / "test.jsonl")

    num_labels = 4  # low / medium / high / critical
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model = model.to(device)

    class _DS(Dataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            r = self.records[idx]
            enc = tokenizer(r["text"], max_length=max_length, truncation=True,
                            padding="max_length", return_tensors="pt")
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(r["severity_label"], dtype=torch.long),
            }

    train_loader = DataLoader(_DS(train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(_DS(val),   batch_size=batch_size * 2)
    test_loader  = DataLoader(_DS(test),  batch_size=batch_size * 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["label"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += out.loss.item()
        log.info("task3/secbert: epoch %d/%d loss=%.4f", epoch, epochs, total_loss / len(train_loader))

    def _evaluate(loader, split: str) -> dict:
        model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for batch in loader:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                preds = out.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["label"].numpy())

        from sklearn.metrics import classification_report
        f1m = f1_score(all_labels, all_preds, average="macro")
        acc = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds,
                                       target_names=["low", "medium", "high", "critical"],
                                       output_dict=True)
        log.info("task3/secbert [%s]: F1-macro=%.4f acc=%.4f", split, f1m, acc)
        return {
            "split": split,
            "f1_macro": round(float(f1m), 4),
            "accuracy": round(float(acc), 4),
            "f1_per_class": {
                k: round(v["f1-score"], 4)
                for k, v in report.items()
                if k in {"low", "medium", "high", "critical"}
            },
            "n": len(all_labels),
        }

    results = {
        "task": "task3_severity",
        "model": "secbert_finetuned",
        "model_description": f"{model_name} fine-tuned for 4-class severity prediction",
        "hyperparams": {
            "model": model_name, "max_length": max_length,
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "seed": seed,
        },
        "val": _evaluate(val_loader, "val"),
        "test": _evaluate(test_loader, "test"),
    }
    suffix = f"_seed{seed}" if seed is not None else ""
    _save_results(results, output_dir / f"task3_secbert{suffix}.json")
    return results


# ---------------------------------------------------------------------------
# Task 4 — Hacker Exploit Labeling baselines
# ---------------------------------------------------------------------------


def run_task4_tfidf(benchmark_dir: Path, output_dir: Path) -> dict:
    """TF-IDF + Logistic Regression baseline for ternary hacker exploit labeling."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    task_dir = benchmark_dir / "task4_hacker_exploit_labeling"
    train = _load_jsonl(task_dir / "train.jsonl")
    val = _load_jsonl(task_dir / "val.jsonl")
    test = _load_jsonl(task_dir / "test.jsonl")
    names = ["non_exploit_noise", "vulnerability_discussion", "actionable_exploit"]

    log.info("task4/tfidf: train=%d val=%d test=%d", len(train), len(val), len(test))

    X_train = [r["text"] for r in train]
    y_train = [r["label"] for r in train]
    X_val = [r["text"] for r in val]
    y_val = [r["label"] for r in val]
    X_test = [r["text"] for r in test]
    y_test = [r["label"] for r in test]

    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X_tr_vec = vec.fit_transform(X_train)
    X_va_vec = vec.transform(X_val)
    X_te_vec = vec.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", n_jobs=-1, random_state=0)
    clf.fit(X_tr_vec, y_train)

    def _eval(X_vec, y_true: list[int], split: str) -> dict:
        y_pred = clf.predict(X_vec)
        f1m = f1_score(y_true, y_pred, average="macro")
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=names, output_dict=True, zero_division=0)
        log.info("task4/tfidf [%s]: F1-macro=%.4f acc=%.4f", split, f1m, acc)
        return {
            "split": split,
            "f1_macro": round(float(f1m), 4),
            "accuracy": round(float(acc), 4),
            "f1_per_class": {
                k: round(v["f1-score"], 4)
                for k, v in report.items()
                if k in set(names)
            },
            "n": len(y_true),
        }

    results = {
        "task": "task4_hacker_exploit_labeling",
        "model": "tfidf_logreg",
        "model_description": "TF-IDF (unigrams+bigrams, 50k features) + balanced multinomial Logistic Regression",
        "val": _eval(X_va_vec, y_val, "val"),
        "test": _eval(X_te_vec, y_test, "test"),
    }
    _save_results(results, output_dir / "task4_tfidf_logreg.json")
    return results


# ---------------------------------------------------------------------------
# Task 5 — Hacker Exploit Signal Detection baselines
# ---------------------------------------------------------------------------


def run_task5_tfidf_bm25(benchmark_dir: Path, output_dir: Path, ks: list[int] = [1, 5, 10]) -> dict:
    """TF-IDF actionability classifier plus BM25 CVE-context retrieval baseline."""
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score

    task_dir = benchmark_dir / "task5_hacker_signal_detection"
    train = _load_jsonl(task_dir / "train.jsonl")
    val = _load_jsonl(task_dir / "val.jsonl")
    test = _load_jsonl(task_dir / "test.jsonl")
    corpus_records = _load_jsonl(task_dir / "corpus.jsonl")

    log.info("task5/tfidf_bm25: train=%d val=%d test=%d corpus=%d", len(train), len(val), len(test), len(corpus_records))

    clf_vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X_train = clf_vec.fit_transform([r["text"] for r in train])
    y_train = [r["actionable_label"] for r in train]
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", n_jobs=-1, random_state=0)
    clf.fit(X_train, y_train)

    corpus_cves = [r["cve_id"] for r in corpus_records]
    corpus_texts = [r["text"] for r in corpus_records]
    bm25_vec = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b[\w./:-]{2,}\b")
    doc_tf = bm25_vec.fit_transform(corpus_texts).astype(np.float32).tocsr()
    n_docs = doc_tf.shape[0]
    doc_len = np.asarray(doc_tf.sum(axis=1)).ravel()
    avg_doc_len = float(doc_len.mean()) if n_docs else 0.0
    df = np.asarray((doc_tf > 0).sum(axis=0)).ravel()
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
    k1 = 1.5
    b = 0.75
    denom_norm = k1 * (1.0 - b + b * doc_len / max(avg_doc_len, 1e-9))
    bm25 = doc_tf.copy()
    rows, _cols = bm25.nonzero()
    bm25.data = bm25.data * (k1 + 1.0) / (bm25.data + denom_norm[rows])
    bm25 = bm25.multiply(idf).tocsr()
    bm25_t = bm25.T.tocsr()
    cve_to_indices: dict[str, list[int]] = {}
    for idx, cve in enumerate(corpus_cves):
        cve_to_indices.setdefault(cve, []).append(idx)

    def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
        if len(scores) <= k:
            return np.argsort(-scores)
        candidates = np.argpartition(scores, -k)[-k:]
        return candidates[np.argsort(-scores[candidates])]

    def _eval(records: list[dict], split: str, batch_size: int = 256) -> dict:
        X = clf_vec.transform([r["text"] for r in records])
        y_true = [r["actionable_label"] for r in records]
        y_pred = clf.predict(X)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

        positives = [r for r in records if r.get("actionable_label") == 1 and r.get("cve_id")]
        query_tf = bm25_vec.transform([r["text"] for r in positives]).astype(np.float32).tocsr()
        max_k = max(ks)
        hits_k = {k: 0 for k in ks}
        joint_hits_k = {k: 0 for k in ks}
        rr_sum = 0.0
        positive_index_to_pred = [int(y_pred[i]) for i, r in enumerate(records) if r.get("actionable_label") == 1 and r.get("cve_id")]

        for start in range(0, len(positives), batch_size):
            end = min(start + batch_size, len(positives))
            score_mat = (query_tf[start:end] @ bm25_t).toarray()
            for offset, scores in enumerate(score_mat):
                i = start + offset
                q = positives[i]
                true_indices = cve_to_indices.get(q["cve_id"], [])
                if not true_indices:
                    continue
                ranked = _top_indices(scores, max_k)
                true_set = set(true_indices)
                rank = next((rank for rank, idx in enumerate(ranked, 1) if idx in true_set), None)
                if rank is None:
                    continue
                rr_sum += 1.0 / rank
                for k in ks:
                    if rank <= k:
                        hits_k[k] += 1
                        if positive_index_to_pred[i] == 1:
                            joint_hits_k[k] += 1

        denom_retrieval = max(len(positives), 1)
        out: dict[str, Any] = {
            "split": split,
            "n": len(records),
            "n_actionable": len(positives),
            "f1_actionable": round(float(f1), 4),
            "precision_actionable": round(float(precision), 4),
            "recall_actionable": round(float(recall), 4),
            "mrr": round(rr_sum / denom_retrieval, 4),
        }
        for k in ks:
            out[f"recall_at_{k}"] = round(hits_k[k] / denom_retrieval, 4)
            out[f"joint_recall_at_{k}"] = round(joint_hits_k[k] / denom_retrieval, 4)
        log.info("task5/tfidf_bm25 [%s]: F1+=%.4f R@5=%.4f joint_R@5=%.4f", split, f1, out["recall_at_5"], out["joint_recall_at_5"])
        return out

    results = {
        "task": "task5_hacker_signal_detection",
        "model": "tfidf_logreg_plus_bm25",
        "model_description": "Balanced TF-IDF Logistic Regression for actionable-signal detection plus BM25 retrieval over NVD CVE corpus",
        "val": _eval(val, "val"),
        "test": _eval(test, "test"),
    }
    _save_results(results, output_dir / "task5_tfidf_bm25.json")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run HackerSignal benchmark baselines")
    parser.add_argument("--benchmark-dir", default="data/benchmark")
    parser.add_argument("--output", default="data/results")
    parser.add_argument("--task", choices=["all", "1", "2", "3", "4", "5"], default="all")
    parser.add_argument("--model", choices=["all", "tfidf", "neural"], default="all",
                        help="'tfidf'=lightweight baselines only, 'neural'=transformer baselines only")
    parser.add_argument("--device", default=None, help="torch device (cuda/mps/cpu)")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs for fine-tuned SecBERT baselines")
    parser.add_argument("--batch-size", type=int, default=32, help="Training/encoding batch size")
    parser.add_argument("--score-batch-size", type=int, default=64, help="Bi-encoder scoring batch size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for neural baselines")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    output_dir    = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_tfidf   = args.model in ("all", "tfidf")
    run_neural  = args.model in ("all", "neural")

    if args.task in ("all", "1"):
        if run_tfidf:
            run_task1_source_only(benchmark_dir, output_dir)
            run_task1_tfidf(benchmark_dir, output_dir)
        if run_neural:
            run_task1_secbert(
                benchmark_dir,
                output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                seed=args.seed,
            )

    if args.task in ("all", "2"):
        if run_tfidf:  run_task2_bm25(benchmark_dir, output_dir)
        if run_neural:
            run_task2_biencoder(
                benchmark_dir,
                output_dir,
                batch_size=args.batch_size,
                score_batch_size=args.score_batch_size,
            )

    if args.task in ("all", "3"):
        if run_tfidf:  run_task3_tfidf(benchmark_dir, output_dir)
        if run_neural:
            run_task3_secbert(
                benchmark_dir,
                output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                seed=args.seed,
            )

    if args.task in ("all", "4"):
        if run_tfidf:
            run_task4_tfidf(benchmark_dir, output_dir)
        if run_neural:
            log.warning("task4: neural baseline is not implemented in this lightweight runner")

    if args.task in ("all", "5"):
        if run_tfidf:
            run_task5_tfidf_bm25(benchmark_dir, output_dir)
        if run_neural:
            log.warning("task5: neural baseline is not implemented in this lightweight runner")


if __name__ == "__main__":
    main()
