# src/etg/career_cve_validation.py
"""External CVE validation for DGT shift predictions.

Validates DGT embedding shifts against CISA KEV term emergence.
Fixes the endogenous-evaluation problem (Issue 1) and replaces R² with
externally-grounded metrics (Issue 3).

Public API
----------
run_cve_emergence_validation(emb_dict, snapshots, kev_entries, top_k, stopwords)
    → dict with mean_precision_at_k, mean_auc_roc, mean_spearman_rho,
         per_transition list.

fetch_kev(cache_path=None)
    → list[dict]  (downloads CISA KEV catalog, caches to disk)
"""

from __future__ import annotations

import json
import logging
import re
import string
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

# Tokens to strip regardless of user stopwords
_ALWAYS_STRIP = {
    "cve", "cwe", "via", "the", "a", "an", "and", "or", "in", "of",
    "to", "with", "for", "on", "by", "is", "are", "was", "be", "as",
    "this", "that", "which", "can", "may", "has", "have", "its",
    "from", "it", "at", "but", "not", "within",
}


# ── helpers ────────────────────────────────────────────────────────────────

def _to_date(s: str) -> datetime:
    return datetime.strptime(s[:10], "%Y-%m-%d")


def _spell_windows_from_snapshots(snapshots: list[dict]) -> dict[int, tuple[str, str]]:
    """Return {spell_number: (start_str, end_str)} where start_str for spell T
    is the end of spell T-1 (or the corpus start for spell 1).

    Spells are cumulative in the pipeline (each snapshot covers 2016-01-01 to
    its end date), so the *incremental* window for spell T is
    (prev_end, curr_end).
    """
    snaps = sorted(snapshots, key=lambda s: int(s["spell"]))
    windows: dict[int, tuple[str, str]] = {}
    prev_end = snaps[0]["start"]  # corpus start date
    for s in snaps:
        windows[int(s["spell"])] = (prev_end, s["end"])
        prev_end = s["end"]
    return windows


def _assign_kev_to_spells(
    kev_entries: list[dict],
    windows: dict[int, tuple[str, str]],
) -> dict[int, list[dict]]:
    """Assign each KEV entry to the spell whose incremental window contains
    its ``dateAdded`` field.  Entries outside all windows are dropped."""
    assigned: dict[int, list[dict]] = {spell: [] for spell in windows}
    for entry in kev_entries:
        try:
            d = _to_date(entry["dateAdded"])
        except (KeyError, ValueError):
            continue
        for spell, (lo_str, hi_str) in windows.items():
            lo, hi = _to_date(lo_str), _to_date(hi_str)
            if lo < d <= hi:
                assigned[spell].append(entry)
                break
    return assigned


def _tokenize_cve_text(text: str, stopwords: set[str]) -> set[str]:
    """Lower-case, strip punctuation, remove stop-words and short tokens.

    Also strips tokens that look like CVE/CWE identifiers (digits-heavy) and
    tokens shorter than 3 characters, so only meaningful content words remain.
    """
    combined_stop = _ALWAYS_STRIP | {w.lower() for w in stopwords}
    text = text.lower()
    # Replace hyphens inside CVE IDs (CVE-2020-001) with space so digits split off
    text = re.sub(r"\bcve-\d+-\d+\b", " ", text)
    text = re.sub(r"\bcwe-\d+\b", " ", text)
    # Remove remaining punctuation
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    tokens: set[str] = set()
    for tok in text.split():
        tok = tok.strip()
        if len(tok) < 3:
            continue
        if tok.isdigit():
            continue
        if re.fullmatch(r"\d+\w*", tok):  # starts with digit
            continue
        if tok in combined_stop:
            continue
        tokens.add(tok)
    return tokens


def _dgt_shift_scores(
    emb_dict: dict[str, np.ndarray],
    n_spells: int,
) -> list[np.ndarray]:
    """Compute per-word cosine distance between consecutive spell embeddings.

    Returns a list of length (n_spells - 1).  shifts[t] is a float array of
    shape (N_words,) giving 1 - cos(G_{t+1}, G_{t}) for each word.

    Expects emb_dict to have keys 'G_01', 'G_02', … and embeddings to be
    L2-normalised (dot product = cosine similarity).  If not normalised,
    normalises internally.
    """
    def _key(t: int) -> str:
        return f"G_{t:02d}"

    shifts: list[np.ndarray] = []
    for t in range(1, n_spells):
        prev = emb_dict[_key(t)].astype(np.float64)
        curr = emb_dict[_key(t + 1)].astype(np.float64)
        # Normalise in case embeddings were stored un-normalised
        prev_norm = np.linalg.norm(prev, axis=1, keepdims=True)
        curr_norm = np.linalg.norm(curr, axis=1, keepdims=True)
        prev = prev / np.where(prev_norm > 0, prev_norm, 1.0)
        curr = curr / np.where(curr_norm > 0, curr_norm, 1.0)
        cos_sim = np.sum(prev * curr, axis=1)
        dist = np.clip(1.0 - cos_sim, 0.0, 2.0)
        shifts.append(dist)
    return shifts


# ── main public function ────────────────────────────────────────────────────

def run_cve_emergence_validation(
    emb_dict: dict[str, np.ndarray],
    snapshots: list[dict],
    kev_entries: list[dict],
    top_k: int = 50,
    stopwords: Optional[set[str]] = None,
) -> dict:
    """Validate DGT shift predictions against CISA KEV term emergence.

    For each spell transition T→T+1 that has KEV entries in spell T+1:
      - DGT shift score per word = cosine distance G_T → G_{T+1}
      - KEV emerging terms = tokenised text of KEV entries added in spell T+1
      - Precision@K: fraction of top-K DGT-predicted shifting words that
        appear in KEV emerging terms
      - AUC-ROC: DGT shift score as predictor of binary KEV-term membership
      - Spearman ρ: DGT shift score rank vs KEV-term co-occurrence count rank

    Parameters
    ----------
    emb_dict : dict with keys 'words' (str array, N) and 'G_01'..'G_12'
               (float32, N×D) — typically loaded from dgt_embeddings.npz.
    snapshots : list of 12 spell dicts from the cached snapshots pickle.
    kev_entries : list of KEV dicts (from fetch_kev).
    top_k : int — number of top-predicted terms to evaluate (default 50).
    stopwords : optional extra stop-words for CVE text tokenisation.

    Returns
    -------
    dict with keys:
      mean_precision_at_k, mean_auc_roc, mean_spearman_rho,
      top_k, n_transitions_evaluated,
      per_transition: list of per-spell-transition result dicts.
    """
    if stopwords is None:
        stopwords = set()

    words: np.ndarray = emb_dict["words"]          # (N,)
    n_spells = sum(1 for k in emb_dict if re.match(r"^G_\d+$", k))
    vocab_set = set(words.tolist())

    windows = _spell_windows_from_snapshots(snapshots)
    kev_by_spell = _assign_kev_to_spells(kev_entries, windows)

    # Precompute DGT cosine shifts for every transition
    all_shifts = _dgt_shift_scores(emb_dict, n_spells)  # list len n_spells-1
    # all_shifts[t-1] = shifts for transition t → t+1  (t is 1-indexed spell)

    per_transition: list[dict] = []

    for t in range(1, n_spells):
        next_spell = t + 1
        kev_in_next = kev_by_spell.get(next_spell, [])
        if not kev_in_next:
            log.debug("Spell %d→%d: no KEV entries, skipping.", t, next_spell)
            continue

        # Build set of emerging CVE terms (from KEV entries in spell t+1)
        cve_terms: set[str] = set()
        cve_term_counts: dict[str, int] = {}
        for entry in kev_in_next:
            text = (
                entry.get("vulnerabilityName", "")
                + " "
                + entry.get("shortDescription", "")
                + " "
                + " ".join(entry.get("cwes", []))
            )
            toks = _tokenize_cve_text(text, stopwords) & vocab_set
            cve_terms |= toks
            for tok in toks:
                cve_term_counts[tok] = cve_term_counts.get(tok, 0) + 1

        if not cve_terms:
            log.debug("Spell %d→%d: no CVE terms overlap with DGT vocab.", t, next_spell)
            continue

        shift_scores = all_shifts[t - 1]  # shape (N,)
        N = len(words)

        # Binary label: 1 if word appears in any KEV entry in spell t+1
        labels = np.array([1 if w in cve_terms else 0 for w in words])
        # Continuous KEV relevance: how many KEV entries contain this word
        kev_counts = np.array([cve_term_counts.get(w, 0) for w in words], dtype=float)

        n_positive = int(labels.sum())
        if n_positive == 0 or n_positive == N:
            log.debug("Spell %d→%d: degenerate labels, skipping.", t, next_spell)
            continue

        # Precision@K
        top_k_actual = min(top_k, N)
        top_indices = np.argsort(-shift_scores)[:top_k_actual]
        precision_at_k = float(labels[top_indices].mean())

        # AUC-ROC
        try:
            auc = float(roc_auc_score(labels, shift_scores))
        except Exception:
            auc = float("nan")

        # Spearman ρ between shift scores and KEV co-occurrence counts
        try:
            rho, pval = spearmanr(shift_scores, kev_counts)
            rho = float(rho)
            pval = float(pval)
        except Exception:
            rho, pval = float("nan"), float("nan")

        per_transition.append({
            "transition": f"G{t:02d}_to_G{next_spell:02d}",
            "spell_from": t,
            "spell_to": next_spell,
            "n_kev_entries": len(kev_in_next),
            "n_cve_terms_in_vocab": len(cve_terms),
            "precision_at_k": precision_at_k,
            "auc_roc": auc,
            "spearman_rho": rho,
            "spearman_pval": pval,
            "top_k": top_k_actual,
        })
        log.info(
            "Spell %d→%d | KEV=%d | vocab_overlap=%d | P@%d=%.3f | AUC=%.3f | ρ=%.3f",
            t, next_spell, len(kev_in_next), len(cve_terms),
            top_k_actual, precision_at_k, auc, rho,
        )

    if not per_transition:
        return {
            "mean_precision_at_k": None,
            "mean_auc_roc": None,
            "mean_spearman_rho": None,
            "top_k": top_k,
            "n_transitions_evaluated": 0,
            "per_transition": [],
        }

    valid_prec = [r["precision_at_k"] for r in per_transition if not np.isnan(r["precision_at_k"])]
    valid_auc  = [r["auc_roc"] for r in per_transition if not np.isnan(r["auc_roc"])]
    valid_rho  = [r["spearman_rho"] for r in per_transition if not np.isnan(r["spearman_rho"])]

    return {
        "mean_precision_at_k": float(np.mean(valid_prec)) if valid_prec else None,
        "mean_auc_roc": float(np.mean(valid_auc)) if valid_auc else None,
        "mean_spearman_rho": float(np.mean(valid_rho)) if valid_rho else None,
        "top_k": top_k,
        "n_transitions_evaluated": len(per_transition),
        "per_transition": per_transition,
    }


def fetch_kev(cache_path: Optional[Path] = None) -> list[dict]:
    """Download the CISA KEV catalog. Caches to *cache_path* if provided."""
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            log.info("KEV: loading from cache %s", cache_path)
            with cache_path.open(encoding="utf-8") as fh:
                return json.load(fh)
    log.info("KEV: downloading from CISA")
    resp = requests.get(KEV_URL, timeout=60)
    resp.raise_for_status()
    entries = resp.json()["vulnerabilities"]
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2)
        log.info("KEV: cached %d entries to %s", len(entries), cache_path)
    return entries
