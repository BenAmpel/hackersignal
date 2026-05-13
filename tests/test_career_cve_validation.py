# tests/test_career_cve_validation.py
import numpy as np
import pytest
from etg.career_cve_validation import (
    _spell_windows_from_snapshots,
    _assign_kev_to_spells,
    _tokenize_cve_text,
    _dgt_shift_scores,
    run_cve_emergence_validation,
)

FAKE_SNAPS = [
    {"spell": 1, "start": "2020-01-01", "end": "2020-06-30"},
    {"spell": 2, "start": "2020-01-01", "end": "2021-06-30"},
    {"spell": 3, "start": "2020-01-01", "end": "2022-06-30"},
]

FAKE_KEV = [
    {"cveID": "CVE-2020-001", "dateAdded": "2020-04-01",
     "vulnerabilityName": "Remote code execution flaw",
     "shortDescription": "heap overflow allows rce", "cwes": ["CWE-122"]},
    {"cveID": "CVE-2021-002", "dateAdded": "2021-03-15",
     "vulnerabilityName": "SQL injection vuln",
     "shortDescription": "sql injection in login form", "cwes": ["CWE-89"]},
    {"cveID": "CVE-2022-003", "dateAdded": "2022-02-20",
     "vulnerabilityName": "Buffer overflow",
     "shortDescription": "stack overflow in parser", "cwes": ["CWE-121"]},
]


def test_spell_windows_from_snapshots():
    windows = _spell_windows_from_snapshots(FAKE_SNAPS)
    # spell 1: 2020-01-01 to 2020-06-30
    assert windows[1][0] == "2020-01-01"
    assert windows[1][1] == "2020-06-30"
    # spell 2: 2020-06-30 to 2021-06-30
    assert windows[2][0] == "2020-06-30"
    assert windows[2][1] == "2021-06-30"


def test_assign_kev_to_spells_basic():
    windows = _spell_windows_from_snapshots(FAKE_SNAPS)
    assigned = _assign_kev_to_spells(FAKE_KEV, windows)
    # CVE-2020-001 dateAdded 2020-04-01 → spell 1 (2020-01-01 to 2020-06-30)
    assert any(v["cveID"] == "CVE-2020-001" for v in assigned[1])
    # CVE-2021-002 dateAdded 2021-03-15 → spell 2 (2020-06-30 to 2021-06-30)
    assert any(v["cveID"] == "CVE-2021-002" for v in assigned[2])
    # CVE-2022-003 dateAdded 2022-02-20 → spell 3 (2021-06-30 to 2022-06-30)
    assert any(v["cveID"] == "CVE-2022-003" for v in assigned[3])


def test_tokenize_cve_text_basic():
    tokens = _tokenize_cve_text("Remote code execution in heap", stopwords={"in"})
    assert "remote" in tokens
    assert "execution" in tokens
    assert "in" not in tokens
    assert "heap" in tokens
    # Numbers and short tokens stripped
    tokens2 = _tokenize_cve_text("CVE-2020-001 allows RCE via HTTP/2", stopwords=set())
    assert "rce" in tokens2
    assert "http" in tokens2
    # CVE ID pattern itself should not be in tokens
    assert "cve" not in tokens2 or "cve-2020-001" not in tokens2


def test_dgt_shift_scores_shape():
    words = np.array(["heap", "overflow", "sql", "injection"])
    # 3 spells, 4 words, 8-dim embeddings
    emb = {
        "G_01": np.random.randn(4, 8).astype(np.float32),
        "G_02": np.random.randn(4, 8).astype(np.float32),
        "G_03": np.random.randn(4, 8).astype(np.float32),
    }
    # Normalize
    for k in emb:
        norms = np.linalg.norm(emb[k], axis=1, keepdims=True)
        emb[k] = emb[k] / np.where(norms > 0, norms, 1.0)
    shifts = _dgt_shift_scores(emb, n_spells=3)
    # shifts[t] = cosine distances for transition t→t+1, shape (4,)
    assert len(shifts) == 2  # transitions 1→2 and 2→3
    assert shifts[0].shape == (4,)
    assert np.all(shifts[0] >= 0)
    assert np.all(shifts[0] <= 2.0)  # cosine distance ∈ [0, 2]


def test_run_cve_emergence_validation_smoke():
    """End-to-end smoke: tiny fake embeddings and KEV, check output keys."""
    words = np.array(["heap", "overflow", "sql", "injection",
                      "remote", "execution", "buffer", "stack"])
    n = len(words)
    rng = np.random.default_rng(42)
    emb_dict = {"words": words}
    for i in range(1, 4):
        key = f"G_{i:02d}"
        m = rng.standard_normal((n, 4)).astype(np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        emb_dict[key] = m / np.where(norms > 0, norms, 1.0)

    result = run_cve_emergence_validation(
        emb_dict=emb_dict,
        snapshots=FAKE_SNAPS,
        kev_entries=FAKE_KEV,
        top_k=3,
        stopwords=set(),
    )
    assert "mean_precision_at_k" in result
    assert "mean_auc_roc" in result
    assert "mean_spearman_rho" in result
    assert "per_transition" in result
    assert isinstance(result["mean_precision_at_k"], float)
