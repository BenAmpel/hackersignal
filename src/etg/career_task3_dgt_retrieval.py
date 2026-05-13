# src/etg/career_task3_dgt_retrieval.py
"""DGT-augmented Task-3 temporal CVE retrieval evaluation.

Addresses Issue 2 (weak downstream performance): uses DGT's final-spell
term embeddings to represent hacker-post queries and CISA KEV entries as
documents, then retrieves the correct CVE by cosine similarity.

Compared against a BM25 baseline on the same corpus so the comparison is
identical in scope and does not rely on external model weights.

Public API
----------
run_task3_dgt_retrieval(queries, corpus_texts, corpus_cve_ids, emb_dict,
                        spell_key="G_12", stopwords=None)
    → dict with "dgt" and "bm25" sub-dicts each containing MRR@10, R@1,
        R@5, R@10, n_queries.

build_kev_corpus(kev_entries)
    → (corpus_texts: list[str], corpus_cve_ids: list[str])
"""

from __future__ import annotations

import logging
import re
import string
from typing import Optional

import numpy as np


# ── inline IR metrics (no etg.eval dependency required on cluster) ─────────

def _ranks_from_scores(scores: np.ndarray, positive_indices: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty(scores.shape[0], dtype=np.int64)
    for q in range(scores.shape[0]):
        ranks[q] = int(np.where(order[q] == int(positive_indices[q]))[0][0]) + 1
    return ranks


def _mrr_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(np.where(ranks <= k, 1.0 / ranks, 0.0)))


def _hit_rate_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


# ── inline BM25Okapi (no rank_bm25 dependency required on cluster) ─────────

class _BM25Okapi:
    """Minimal BM25Okapi implementation — identical scoring to rank_bm25."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.N, 1)
        # df: number of docs containing each term
        df: dict[str, int] = {}
        for doc in corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        import math
        self.idf: dict[str, float] = {
            term: math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    def get_scores(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float64)
        for term in query:
            idf = self.idf.get(term, 0.0)
            if idf == 0.0:
                continue
            for i, doc in enumerate(self.corpus):
                tf = doc.count(term)
                denom = tf + self.k1 * (1 - self.b + self.b * len(doc) / max(self.avgdl, 1))
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        return scores

log = logging.getLogger(__name__)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "of", "to", "with", "for",
    "on", "by", "is", "are", "was", "be", "as", "this", "that",
    "which", "can", "may", "has", "have", "its", "from", "it", "at",
    "but", "not", "within", "via", "allows", "allow", "could",
}


# ── helpers ────────────────────────────────────────────────────────────────

def _tokenize(text: str, stopwords: set[str]) -> list[str]:
    """Lower-case, remove punctuation, filter stopwords and short tokens."""
    text = text.lower()
    text = re.sub(r"\bcve-\d+-\d+\b", " ", text)
    text = re.sub(r"\bcwe-\d+\b", " ", text)
    text = text.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )
    return [
        t for t in text.split()
        if len(t) >= 3 and not t.isdigit() and t not in stopwords
    ]


def _embed_text(
    text: str,
    vocab: np.ndarray,
    emb: np.ndarray,
    stopwords: set[str],
) -> np.ndarray:
    """TF-weighted average of DGT term embeddings for *text*.

    Unknown words (not in DGT vocab) are skipped.  If no words match,
    returns a zero vector of shape (emb.shape[1],).
    """
    vocab_index = {w: i for i, w in enumerate(vocab.tolist())}
    tokens = _tokenize(text, stopwords)
    vecs = []
    for tok in tokens:
        idx = vocab_index.get(tok)
        if idx is not None:
            vecs.append(emb[idx])
    if not vecs:
        return np.zeros(emb.shape[1], dtype=np.float32)
    mean = np.mean(vecs, axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


def _bm25_ranks(
    query_texts: list[str],
    corpus_texts: list[str],
    positive_indices: np.ndarray,
    stopwords: Optional[set[str]] = None,
) -> np.ndarray:
    """BM25Okapi retrieval; returns 1-indexed ranks for each query."""
    sw = stopwords or _STOPWORDS
    tokenized_corpus = [_tokenize(t, sw) for t in corpus_texts]
    bm25 = _BM25Okapi(tokenized_corpus)
    C = len(corpus_texts)
    Q = len(query_texts)
    scores = np.zeros((Q, C), dtype=np.float64)
    for qi, qtext in enumerate(query_texts):
        qtoks = _tokenize(qtext, sw)
        scores[qi] = bm25.get_scores(qtoks)
    return _ranks_from_scores(scores, positive_indices)


def _dgt_ranks(
    query_texts: list[str],
    corpus_texts: list[str],
    vocab: np.ndarray,
    emb: np.ndarray,
    stopwords: set[str],
    positive_indices: np.ndarray,
) -> np.ndarray:
    """DGT cosine-similarity retrieval; returns 1-indexed ranks."""
    corpus_embs = np.stack(
        [_embed_text(t, vocab, emb, stopwords) for t in corpus_texts]
    )  # (C, D)
    query_embs = np.stack(
        [_embed_text(t, vocab, emb, stopwords) for t in query_texts]
    )  # (Q, D)
    # Cosine similarity (vectors already L2-normalised by _embed_text;
    # zero vectors will give 0 similarity to everything — harmless).
    scores = query_embs @ corpus_embs.T  # (Q, C)
    return _ranks_from_scores(scores, positive_indices)


def build_kev_corpus(kev_entries: list[dict]) -> tuple[list[str], list[str]]:
    """Convert KEV entries to (texts, cve_ids) parallel lists for retrieval."""
    texts, cve_ids = [], []
    for e in kev_entries:
        text = (
            e.get("vulnerabilityName", "")
            + " "
            + e.get("shortDescription", "")
            + " "
            + e.get("vendorProject", "")
            + " "
            + e.get("product", "")
        ).strip()
        if text:
            texts.append(text)
            cve_ids.append(e["cveID"])
    return texts, cve_ids


# ── main public function ────────────────────────────────────────────────────

def run_task3_dgt_retrieval(
    queries: list[dict],
    corpus_texts: list[str],
    corpus_cve_ids: list[str],
    emb_dict: dict[str, np.ndarray],
    spell_key: str = "G_12",
    stopwords: Optional[set[str]] = None,
) -> dict:
    """Evaluate DGT-based and BM25 retrieval on a CVE corpus.

    Parameters
    ----------
    queries : list of dicts with 'text' and 'cve_id' fields.
              Only queries whose cve_id appears in corpus_cve_ids are used.
    corpus_texts : list of document strings (one per CVE).
    corpus_cve_ids : list of CVE IDs parallel to corpus_texts.
    emb_dict : dict with 'words' (str array) and spell_key (float32 N×D).
    spell_key : which spell embedding to use (default 'G_12', most recent).
    stopwords : optional extra stop-words.

    Returns
    -------
    dict with 'dgt' and 'bm25' keys, each a dict of:
        MRR@10, R@1, R@5, R@10, n_queries
    """
    sw = (stopwords or set()) | _STOPWORDS
    vocab: np.ndarray = emb_dict["words"]
    emb: np.ndarray = emb_dict[spell_key].astype(np.float32)

    # Normalise embeddings in case stored un-normalised
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(norms > 0, norms, 1.0)

    cve_to_idx = {cid: i for i, cid in enumerate(corpus_cve_ids)}

    # Filter queries to those whose target CVE is in the corpus
    valid_queries = [q for q in queries if q.get("cve_id") in cve_to_idx]
    if not valid_queries:
        log.warning("run_task3_dgt_retrieval: no queries with CVE IDs found in corpus.")
        empty = {"MRR@10": None, "R@1": None, "R@5": None, "R@10": None, "n_queries": 0}
        return {"dgt": empty, "bm25": empty}

    query_texts = [q["text"] for q in valid_queries]
    positive_indices = np.array([cve_to_idx[q["cve_id"]] for q in valid_queries])

    log.info("Task-3 DGT retrieval: %d queries, %d corpus docs", len(valid_queries), len(corpus_texts))

    dgt_ranks_arr  = _dgt_ranks(query_texts, corpus_texts, vocab, emb, sw, positive_indices)
    bm25_ranks_arr = _bm25_ranks(query_texts, corpus_texts, positive_indices, stopwords=sw)

    def _metrics(ranks: np.ndarray, n: int) -> dict:
        return {
            "MRR@10": _mrr_at_k(ranks, 10),
            "R@1":   _hit_rate_at_k(ranks, 1),
            "R@5":   _hit_rate_at_k(ranks, 5),
            "R@10":  _hit_rate_at_k(ranks, 10),
            "n_queries": n,
        }

    result = {
        "dgt":  _metrics(dgt_ranks_arr,  len(valid_queries)),
        "bm25": _metrics(bm25_ranks_arr, len(valid_queries)),
    }
    log.info(
        "DGT  MRR@10=%.3f  R@1=%.3f  R@5=%.3f  R@10=%.3f",
        result["dgt"]["MRR@10"], result["dgt"]["R@1"],
        result["dgt"]["R@5"], result["dgt"]["R@10"],
    )
    log.info(
        "BM25 MRR@10=%.3f  R@1=%.3f  R@5=%.3f  R@10=%.3f",
        result["bm25"]["MRR@10"], result["bm25"]["R@1"],
        result["bm25"]["R@5"], result["bm25"]["R@10"],
    )
    return result
