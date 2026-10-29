"""B5: verbatim-overlap strata and the overlap-stripped BM25 bound (Tasks 1 and 3).

Overlap (as defined in the rebuttal to Reviewer LrBh): a test query shares a verbatim span of at least 100
characters with its gold NVD description (raw text, no normalization; 918 of 1,990 Task 1 queries).
1. Per-stratum results for every retrieval model, from the existing as_released per-query ranks
   (raw/overlap_strata.jsonl; CIs are added by finalize.py). Flags are saved to per_query/overlap__<task>.npz.
2. Overlap-stripped BM25 (diagnostic, uses the gold description): every query character covered by a
   30-character span shared with the gold description is replaced by a space; the corpus index is unchanged.
   Saved through run_retrieval.save as condition strip_overlap30.
Usage: python audit/overlap_strata.py
"""
import time

import numpy as np

from common import DATA, PERQ, log, write_row
from run_retrieval import FastBM25, gold_rank, load, save

K_OVERLAP, K_STRIP = 100, 30
MODELS = ["BM25", "all-MiniLM-L6-v2", "all-mpnet-base-v2", "BGE-base-en-v1.5", "E5-base-v2", "Hybrid BM25+mpnet"]


def grams(s, k):
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def shares(q, g, k):
    if len(q) < k or len(g) < k:
        return False
    gg = grams(g, k)
    return any(q[i:i + k] in gg for i in range(len(q) - k + 1))


def strip(q, g, k):
    if len(q) < k or len(g) < k:
        return q
    gg = grams(g, k)
    cover = np.zeros(len(q), bool)
    for i in range(len(q) - k + 1):
        if q[i:i + k] in gg:
            cover[i:i + k] = True
    return "".join(" " if c else ch for ch, c in zip(q, cover))


def metrics(r):
    return dict(n=int(len(r)), r_at_1=float((r <= 1).mean()), r_at_10=float((r <= 10).mean()),
                mrr=float(np.where(r <= 10, 1.0 / r, 0.0).mean()))


def main():
    corpus = load(DATA / "task1_cve_linkage/corpus.jsonl")
    cpos = {d["id"]: i for i, d in enumerate(corpus)}
    craw = [d["text"] for d in corpus]
    tests = {"task1": load(DATA / "task1_cve_linkage/test.jsonl"),
             "task3": load(DATA / "task3_temporal_generalization/test.jsonl")}
    for tk, te in tests.items():
        f = np.array([shares(q["text"], craw[cpos[q["cve_id"]]], K_OVERLAP) for q in te])
        np.savez_compressed(PERQ / f"overlap__{tk}.npz", qid=np.array([q["id"] for q in te]), overlap=f)
        log(f"{tk}: {int(f.sum())} of {len(f)} queries share a >= {K_OVERLAP}-char span with the gold description")
        for m in MODELS:
            tag = f"{tk}__{m}__as_released__primary".replace(" ", "_").replace("+", "plus")
            z = np.load(PERQ / f"{tag}.npz")
            assert (z["qid"] == np.array([q["id"] for q in te])).all(), tag
            for stratum, sel in (("overlap", f), ("no_overlap", ~f)):
                write_row("overlap_strata", task=tk, model=m, stratum=stratum, k_chars=K_OVERLAP,
                          per_query_file=f"per_query/{tag}.npz", flag_file=f"per_query/overlap__{tk}.npz",
                          **metrics(z["rank"][sel]))
    t0 = time.time()
    bm = FastBM25([t.lower().split() for t in craw])
    log(f"BM25 index ({time.time() - t0:.0f}s)")
    for tk, te in tests.items():
        tq = time.time()
        det, opt, pes = [], [], []
        for q in te:
            g = cpos[q["cve_id"]]
            s = bm.scores(strip(q["text"], craw[g], K_STRIP).lower().split())
            a, b, c = gold_rank(s, g)
            det.append(a); opt.append(b); pes.append(c)
        save(tk, "BM25", "strip_overlap30", "primary", [q["id"] for q in te], np.array(det), np.array(opt),
             np.array(pes), extra=dict(k_strip_chars=K_STRIP, note="diagnostic: uses the gold description"), t0=tq)


if __name__ == "__main__":
    main()
