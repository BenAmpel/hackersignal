# tests/test_career_task3_dgt_retrieval.py
import numpy as np
import pytest
from etg.career_task3_dgt_retrieval import (
    _embed_text,
    _bm25_ranks,
    _dgt_ranks,
    run_task3_dgt_retrieval,
)

VOCAB = np.array(["heap", "overflow", "sql", "injection", "rce", "exploit"])
EMB = np.eye(6, dtype=np.float32)  # each word is a one-hot basis vector

CORPUS_TEXTS = [
    "heap overflow allows rce exploit",   # idx 0
    "sql injection in login form",        # idx 1
    "buffer overflow stack smash",        # idx 2
]
CORPUS_CVES = ["CVE-2021-001", "CVE-2021-002", "CVE-2021-003"]

QUERIES = [
    {"text": "heap overflow rce", "cve_id": "CVE-2021-001"},  # should match idx 0
    {"text": "sql injection exploit", "cve_id": "CVE-2021-002"},  # should match idx 1
    {"text": "CVE-NOT-THERE bad cve", "cve_id": "CVE-XXXX"},  # not in corpus
]


def test_embed_text_known_words():
    emb = _embed_text("heap overflow", VOCAB, EMB, stopwords=set())
    # heap=idx0, overflow=idx1 → mean of rows 0 and 1
    expected = (EMB[0] + EMB[1]) / 2
    np.testing.assert_allclose(emb, expected / np.linalg.norm(expected), atol=1e-5)


def test_embed_text_all_oov():
    emb = _embed_text("zzzunknown word", VOCAB, EMB, stopwords=set())
    # OOV → zero vector
    assert np.allclose(emb, 0.0)


def test_bm25_ranks_top1():
    """BM25 should rank the exact-match corpus entry first."""
    ranks = _bm25_ranks(
        query_texts=["heap overflow rce"],
        corpus_texts=CORPUS_TEXTS,
        positive_indices=np.array([0]),
    )
    assert ranks[0] == 1  # exact topic match should be rank 1


def test_dgt_ranks_top1():
    """DGT cosine should rank heap-overflow query to heap-overflow corpus first."""
    ranks = _dgt_ranks(
        query_texts=["heap overflow rce"],
        corpus_texts=CORPUS_TEXTS,
        vocab=VOCAB,
        emb=EMB,
        stopwords=set(),
        positive_indices=np.array([0]),
    )
    assert ranks[0] == 1


def test_run_task3_dgt_retrieval_output_keys():
    emb_dict = {"words": VOCAB, "G_12": EMB}
    queries = [q for q in QUERIES if q["cve_id"] in CORPUS_CVES]
    result = run_task3_dgt_retrieval(
        queries=queries,
        corpus_texts=CORPUS_TEXTS,
        corpus_cve_ids=CORPUS_CVES,
        emb_dict=emb_dict,
        spell_key="G_12",
    )
    assert "dgt" in result
    assert "bm25" in result
    for system in ("dgt", "bm25"):
        assert "MRR@10" in result[system]
        assert "R@1" in result[system]
        assert "R@5" in result[system]
        assert "R@10" in result[system]
        assert "n_queries" in result[system]
