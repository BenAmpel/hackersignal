"""Broad smoke benchmarks for the HackerSignal core task suite.

This module is intentionally small-sample and failure-tolerant. It verifies
that a broad family of baselines can run across the original three benchmark tasks
without pretending to be the final leaderboard. Full-scale runs should use the
task-specific baseline scripts and larger hardware.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ML_CLASSIFIERS = [
    "logreg_tfidf",
    "linear_svm_tfidf",
    "complement_nb_tfidf",
    "random_forest_svd",
    "xgboost_svd",
]

DL_CLASSIFIERS = [
    "torch_linear_svd",
    "torch_mlp_svd",
    "torch_dropout_mlp_svd",
    "torch_residual_mlp_svd",
    "torch_wide_mlp_svd",
]

ENCODER_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2",
    "thenlper/gte-small",
]

ML_RETRIEVERS = [
    "tfidf_word_cosine",
    "tfidf_char_wb_cosine",
    "count_word_cosine",
    "hashing_word_cosine",
    "bm25_smoke",
]

DL_RETRIEVERS = [
    "torch_linear_tower",
    "torch_mlp_tower",
    "torch_dropout_tower",
    "torch_residual_tower",
    "torch_wide_tower",
]


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit is not None and len(records) >= limit:
                break
    return records


def _write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _balanced_sample(
    records: list[dict[str, Any]],
    label_key: str,
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if len(records) <= n:
        out = list(records)
        rng.shuffle(out)
        return out
    by_label: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[record.get(label_key)].append(record)
    labels = sorted(by_label, key=lambda x: str(x))
    per_label = max(1, n // max(1, len(labels)))
    picked: list[dict[str, Any]] = []
    for label in labels:
        group = by_label[label]
        rng.shuffle(group)
        picked.extend(group[:per_label])
    remaining = [r for r in records if r not in picked]
    rng.shuffle(remaining)
    picked.extend(remaining[: max(0, n - len(picked))])
    rng.shuffle(picked)
    return picked[:n]


def _classification_metrics(y_true: list[int], y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "n": len(y_true),
    }
    if len(set(y_true)) == 2 and y_score is not None:
        try:
            metrics["auc_roc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        except ValueError:
            metrics["auc_roc"] = None
    return metrics


def _svd_features(train_texts: list[str], eval_texts: list[str], n_components: int = 64):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler

    vec = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X_train = vec.fit_transform(train_texts)
    X_eval = vec.transform(eval_texts)
    k = max(2, min(n_components, min(X_train.shape) - 1))
    svd = TruncatedSVD(n_components=k, random_state=42)
    Z_train = svd.fit_transform(X_train)
    Z_eval = svd.transform(X_eval)
    scaler = StandardScaler()
    return scaler.fit_transform(Z_train).astype("float32"), scaler.transform(Z_eval).astype("float32")


def _run_ml_classifier(name: str, train_texts: list[str], y_train: list[int], eval_texts: list[str], y_eval: list[int]) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import ComplementNB
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression

    start = time.time()
    if name == "logreg_tfidf":
        model = make_pipeline(
            TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), min_df=1, sublinear_tf=True),
            LogisticRegression(max_iter=500, class_weight="balanced"),
        )
    elif name == "linear_svm_tfidf":
        model = make_pipeline(
            TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), min_df=1, sublinear_tf=True),
            LinearSVC(class_weight="balanced"),
        )
    elif name == "complement_nb_tfidf":
        model = make_pipeline(
            TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), min_df=1),
            ComplementNB(),
        )
    else:
        Z_train, Z_eval = _svd_features(train_texts, eval_texts)
        if name == "random_forest_svd":
            model = RandomForestClassifier(n_estimators=60, max_depth=12, class_weight="balanced", random_state=42)
        elif name == "xgboost_svd":
            try:
                from xgboost import XGBClassifier

                model = XGBClassifier(
                    n_estimators=40,
                    max_depth=3,
                    learning_rate=0.1,
                    eval_metric="logloss",
                    random_state=42,
                )
            except Exception:
                model = ExtraTreesClassifier(n_estimators=80, class_weight="balanced", random_state=42)
        else:
            raise ValueError(name)
        model.fit(Z_train, y_train)
        pred = model.predict(Z_eval)
        score = model.predict_proba(Z_eval)[:, 1] if hasattr(model, "predict_proba") and len(set(y_train)) == 2 else None
        return {"status": "ok", "metrics": _classification_metrics(y_eval, pred, score), "seconds": round(time.time() - start, 2)}

    model.fit(train_texts, y_train)
    pred = model.predict(eval_texts)
    score = None
    if hasattr(model[-1], "predict_proba") and len(set(y_train)) == 2:
        score = model.predict_proba(eval_texts)[:, 1]
    return {"status": "ok", "metrics": _classification_metrics(y_eval, pred, score), "seconds": round(time.time() - start, 2)}


def _run_torch_classifier(name: str, train_texts: list[str], y_train: list[int], eval_texts: list[str], y_eval: list[int], epochs: int) -> dict[str, Any]:
    import torch
    from torch import nn

    start = time.time()
    torch.manual_seed(42)
    Z_train, Z_eval = _svd_features(train_texts, eval_texts)
    x_train = torch.tensor(Z_train)
    x_eval = torch.tensor(Z_eval)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    n_classes = len(sorted(set(y_train)))
    in_dim = x_train.shape[1]

    class ResidualMLP(nn.Module):
        def __init__(self, dim: int, hidden: int, out: int):
            super().__init__()
            self.inp = nn.Linear(dim, hidden)
            self.block = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.out = nn.Linear(hidden, out)

        def forward(self, x):
            h = torch.relu(self.inp(x))
            return self.out(torch.relu(h + self.block(h)))

    if name == "torch_linear_svd":
        model: nn.Module = nn.Linear(in_dim, n_classes)
    elif name == "torch_mlp_svd":
        model = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, n_classes))
    elif name == "torch_dropout_mlp_svd":
        model = nn.Sequential(nn.Linear(in_dim, 96), nn.ReLU(), nn.Dropout(0.25), nn.Linear(96, n_classes))
    elif name == "torch_residual_mlp_svd":
        model = ResidualMLP(in_dim, 64, n_classes)
    elif name == "torch_wide_mlp_svd":
        model = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes))
    else:
        raise ValueError(name)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(max(1, epochs)):
        opt.zero_grad()
        loss = loss_fn(model(x_train), y_train_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = model(x_eval)
        pred = logits.argmax(dim=1).cpu().numpy()
        prob = torch.softmax(logits, dim=1).cpu().numpy()
    score = prob[:, 1] if n_classes == 2 else None
    return {"status": "ok", "metrics": _classification_metrics(y_eval, pred, score), "seconds": round(time.time() - start, 2)}


def _run_encoder_classifier(model_name: str, train_texts: list[str], y_train: list[int], eval_texts: list[str], y_eval: list[int]) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    start = time.time()
    model = SentenceTransformer(model_name)
    X_train = model.encode(train_texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
    X_eval = model.encode(eval_texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_eval)
    score = clf.predict_proba(X_eval)[:, 1] if len(set(y_train)) == 2 else None
    return {"status": "ok", "metrics": _classification_metrics(y_eval, pred, score), "seconds": round(time.time() - start, 2)}


def _ranking_metrics(scores: np.ndarray, truth_indices: list[int]) -> dict[str, Any]:
    ranks: list[int] = []
    for i, truth in enumerate(truth_indices):
        order = np.argsort(-scores[i])
        rank = int(np.where(order == truth)[0][0]) + 1
        ranks.append(rank)
    return {
        "recall_at_1": round(float(np.mean([r <= 1 for r in ranks])), 4),
        "recall_at_5": round(float(np.mean([r <= 5 for r in ranks])), 4),
        "recall_at_10": round(float(np.mean([r <= 10 for r in ranks])), 4),
        "mrr": round(float(np.mean([1.0 / r for r in ranks])), 4),
        "n_queries": len(truth_indices),
        "n_corpus": int(scores.shape[1]),
    }


def _task2_sample(task_dir: Path, n_eval: int, n_corpus: int, rng: random.Random) -> tuple[list[dict], list[dict], list[int]]:
    queries = _load_jsonl(task_dir / "test.jsonl")
    rng.shuffle(queries)
    queries = queries[:n_eval]
    true_cves = {q["cve_id"] for q in queries}
    corpus_all = _load_jsonl(task_dir / "corpus.jsonl")
    true_docs = [d for d in corpus_all if d.get("cve_id") in true_cves]
    distractors = [d for d in corpus_all if d.get("cve_id") not in true_cves]
    rng.shuffle(distractors)
    corpus = true_docs + distractors[: max(0, n_corpus - len(true_docs))]
    rng.shuffle(corpus)
    index = {d["cve_id"]: i for i, d in enumerate(corpus)}
    queries = [q for q in queries if q.get("cve_id") in index]
    truth = [index[q["cve_id"]] for q in queries]
    return queries, corpus, truth


def _run_sparse_retriever(name: str, queries: list[dict], corpus: list[dict], truth: list[int]) -> dict[str, Any]:
    from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize

    start = time.time()
    q_texts = [q["text"] for q in queries]
    c_texts = [c["text"] for c in corpus]
    if name == "tfidf_word_cosine":
        vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=1)
        C = vec.fit_transform(c_texts)
        Q = vec.transform(q_texts)
        scores = cosine_similarity(Q, C)
    elif name == "tfidf_char_wb_cosine":
        vec = TfidfVectorizer(max_features=20_000, analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        C = vec.fit_transform(c_texts)
        Q = vec.transform(q_texts)
        scores = cosine_similarity(Q, C)
    elif name == "count_word_cosine":
        vec = CountVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=1)
        C = normalize(vec.fit_transform(c_texts))
        Q = normalize(vec.transform(q_texts))
        scores = Q @ C.T
    elif name == "hashing_word_cosine":
        vec = HashingVectorizer(n_features=2**15, alternate_sign=False, norm="l2")
        C = vec.transform(c_texts)
        Q = vec.transform(q_texts)
        scores = Q @ C.T
    elif name == "bm25_smoke":
        scores = _bm25_scores(q_texts, c_texts)
    else:
        raise ValueError(name)
    scores_arr = scores.toarray() if hasattr(scores, "toarray") else np.asarray(scores)
    return {"status": "ok", "metrics": _ranking_metrics(scores_arr, truth), "seconds": round(time.time() - start, 2)}


def _bm25_scores(queries: list[str], corpus: list[str]) -> np.ndarray:
    import math
    import re
    from collections import Counter

    tok = lambda s: re.findall(r"[A-Za-z0-9_.:-]+", s.lower())
    docs = [tok(d) for d in corpus]
    qs = [tok(q) for q in queries]
    doc_lens = np.array([len(d) for d in docs], dtype="float32")
    avgdl = float(doc_lens.mean()) if len(doc_lens) else 1.0
    df: Counter[str] = Counter()
    tfs = []
    for d in docs:
        c = Counter(d)
        tfs.append(c)
        df.update(c.keys())
    n_docs = len(docs)
    scores = np.zeros((len(qs), n_docs), dtype="float32")
    k1, b = 1.2, 0.75
    for qi, q in enumerate(qs):
        for term in set(q):
            idf = math.log(1 + (n_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
            for di, tf in enumerate(tfs):
                f = tf.get(term, 0)
                if f:
                    denom = f + k1 * (1 - b + b * doc_lens[di] / avgdl)
                    scores[qi, di] += idf * (f * (k1 + 1)) / denom
    return scores


def _run_torch_retriever(name: str, queries: list[dict], corpus: list[dict], truth: list[int], epochs: int) -> dict[str, Any]:
    import torch
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from torch import nn

    start = time.time()
    torch.manual_seed(42)
    q_texts = [q["text"] for q in queries]
    c_texts = [c["text"] for c in corpus]
    vec = TfidfVectorizer(max_features=12_000, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = vec.fit_transform(c_texts + q_texts)
    k = max(2, min(64, min(X.shape) - 1))
    svd = TruncatedSVD(n_components=k, random_state=42)
    Z = svd.fit_transform(X)
    scaler = StandardScaler()
    Z = scaler.fit_transform(Z).astype("float32")
    C = torch.tensor(Z[: len(corpus)])
    Q = torch.tensor(Z[len(corpus) :])
    y = torch.tensor(truth, dtype=torch.long)
    in_dim = C.shape[1]
    out_dim = 64 if name != "torch_wide_tower" else 128

    def tower(dropout: float = 0.0, residual: bool = False):
        if name == "torch_linear_tower":
            return nn.Linear(in_dim, out_dim)
        if residual:
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(out_dim, out_dim))

    q_tower = tower(dropout=0.25 if "dropout" in name else 0.0, residual="residual" in name)
    c_tower = tower(dropout=0.25 if "dropout" in name else 0.0, residual="residual" in name)
    opt = torch.optim.AdamW(list(q_tower.parameters()) + list(c_tower.parameters()), lr=1e-3)
    for _ in range(max(1, epochs)):
        opt.zero_grad()
        qz = torch.nn.functional.normalize(q_tower(Q), dim=1)
        cz = torch.nn.functional.normalize(c_tower(C), dim=1)
        loss = nn.CrossEntropyLoss()(qz @ cz.T, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        scores = (torch.nn.functional.normalize(q_tower(Q), dim=1) @ torch.nn.functional.normalize(c_tower(C), dim=1).T).cpu().numpy()
    return {"status": "ok", "metrics": _ranking_metrics(scores, truth), "seconds": round(time.time() - start, 2)}


def _run_encoder_retriever(model_name: str, queries: list[dict], corpus: list[dict], truth: list[int]) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    start = time.time()
    model = SentenceTransformer(model_name)
    C = model.encode([c["text"] for c in corpus], batch_size=16, normalize_embeddings=True, show_progress_bar=False)
    Q = model.encode([q["text"] for q in queries], batch_size=16, normalize_embeddings=True, show_progress_bar=False)
    scores = np.asarray(Q) @ np.asarray(C).T
    return {"status": "ok", "metrics": _ranking_metrics(scores, truth), "seconds": round(time.time() - start, 2)}


def run_smoke(
    benchmark_dir: Path,
    output: Path,
    sample_train: int,
    sample_eval: int,
    sample_corpus: int,
    dl_epochs: int,
    seed: int,
    encoders: list[str],
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    class_tasks = [
        ("task1", benchmark_dir / "task1_exploit_clf", "label"),
        ("task3", benchmark_dir / "task3_severity", "severity_label"),
    ]
    for task_name, task_dir, label_key in class_tasks:
        train = _balanced_sample(_load_jsonl(task_dir / "train.jsonl"), label_key, sample_train, rng)
        test = _balanced_sample(_load_jsonl(task_dir / "test.jsonl"), label_key, sample_eval, rng)
        train_texts = [r["text"] for r in train]
        test_texts = [r["text"] for r in test]
        y_train = [int(r[label_key]) for r in train]
        y_test = [int(r[label_key]) for r in test]
        for family, names in (("ml", ML_CLASSIFIERS), ("dl", DL_CLASSIFIERS)):
            for name in names:
                try:
                    result = _run_ml_classifier(name, train_texts, y_train, test_texts, y_test) if family == "ml" else _run_torch_classifier(name, train_texts, y_train, test_texts, y_test, dl_epochs)
                except Exception as exc:
                    result = {"status": "failed", "error": repr(exc)}
                rows.append({"task": task_name, "family": family, "model": name, **result})
                _write_json({"rows": rows}, output)
        for model_name in encoders:
            try:
                result = _run_encoder_classifier(model_name, train_texts, y_train, test_texts, y_test)
            except Exception as exc:
                result = {"status": "failed", "error": repr(exc)}
            rows.append({"task": task_name, "family": "encoder", "model": model_name, **result})
            _write_json({"rows": rows}, output)

    queries, corpus, truth = _task2_sample(benchmark_dir / "task2_cve_linkage", sample_eval, sample_corpus, rng)
    for family, names in (("ml", ML_RETRIEVERS), ("dl", DL_RETRIEVERS)):
        for name in names:
            try:
                result = _run_sparse_retriever(name, queries, corpus, truth) if family == "ml" else _run_torch_retriever(name, queries, corpus, truth, dl_epochs)
            except Exception as exc:
                result = {"status": "failed", "error": repr(exc)}
            rows.append({"task": "task2", "family": family, "model": name, **result})
            _write_json({"rows": rows}, output)
    for model_name in encoders:
        try:
            result = _run_encoder_retriever(model_name, queries, corpus, truth)
        except Exception as exc:
            result = {"status": "failed", "error": repr(exc)}
        rows.append({"task": "task2", "family": "encoder", "model": model_name, **result})
        _write_json({"rows": rows}, output)

    summary = {
        "benchmark_dir": str(benchmark_dir),
        "sample_train": sample_train,
        "sample_eval": sample_eval,
        "sample_corpus": sample_corpus,
        "dl_epochs": dl_epochs,
        "seed": seed,
        "rows": rows,
        "counts": {
            "ok": sum(1 for row in rows if row.get("status") == "ok"),
            "failed": sum(1 for row in rows if row.get("status") == "failed"),
            "total": len(rows),
        },
    }
    _write_json(summary, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run broad small-sample HackerSignal smoke benchmarks.")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--output", type=Path, default=Path("data/results/smoke_benchmarks.json"))
    parser.add_argument("--sample-train", type=int, default=120)
    parser.add_argument("--sample-eval", type=int, default=60)
    parser.add_argument("--sample-corpus", type=int, default=300)
    parser.add_argument("--dl-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-encoders", action="store_true")
    args = parser.parse_args()

    encoders = [] if args.skip_encoders else ENCODER_MODELS
    summary = run_smoke(
        benchmark_dir=args.benchmark_dir,
        output=args.output,
        sample_train=args.sample_train,
        sample_eval=args.sample_eval,
        sample_corpus=args.sample_corpus,
        dl_epochs=args.dl_epochs,
        seed=args.seed,
        encoders=encoders,
    )
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
