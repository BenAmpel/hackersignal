"""Tasks 1 and 3 (and the FP-CVE discourse pilot): every retrieval number, from the released data.

Spec = paper Appendix D; where the paper is silent, the released repo code
(src/etg/benchmark/run_retrieval_baselines.py at e0c1153). Deviations are run as
named sensitivity variants, never silently.

Primary protocol
  BM25      Okapi k1=1.5 b=0.75 eps=0.25 (rank_bm25 defaults), lowercase whitespace split,
            exact sparse reimplementation, validated against rank_bm25.BM25Okapi in-run.
  dense     text[:384] chars, max_seq_length 512, l2-normalised, cosine, fp32 on MPS, batch 128.
            E5 "query: "/"passage: " prefixes; BGE without query instruction.
  hybrid    BM25 top-100 candidates, 0.7*(cos+1)/2 + 0.3*bm25/max(bm25 top-100).
  ranking   exact rank of the gold doc over the whole corpus, ties broken by corpus index
            (lower index first); optimistic/pessimistic tie bounds stored too.
  metrics   R@k = rank<=k; MRR = mean 1/rank if rank<=10 else 0 (MRR@10, as the repo code).
Every (model, condition, variant) writes its row and per-query ranks as soon as it finishes.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

from common import DATA, PERQ, ROOT, data_hashes, done_keys, log, model_revision, write_row

SEED = 20261027
CVE_RE = re.compile(r"CVE[-\s]?\d{4}[-\s]?\d{4,7}", re.I)
URL_RE = re.compile(r"https?://\S+|\bURL\b")
VER_RE = re.compile(r"\b\d+(?:\.\d+){1,}[a-z0-9\-]*\b", re.I)
BOIL_RE = re.compile(
    r"^\s*[#*\-\s]*(Exploit Title|Date|Exploit Author|Author|Vendor Homepage|Vendor|Software Link|"
    r"Version|Tested on|Tested against|CVE|CWE|Category|Type|Severity|Class|Product|Affected|Impact|"
    r"Solution|Credits?|References?|Advisory ID|Risk|Description|Executive Summary|Platform)\s*:.*$",
    re.I | re.M)
CONDITIONS = [
    ("as_released", {}), ("minus_cve_ids", dict(cve=True)), ("minus_urls", dict(url=True)),
    ("minus_versions", dict(ver=True)), ("minus_boilerplate", dict(boil=True)),
    ("all_removed", dict(cve=True, url=True, ver=True, boil=True)),
]
DENSE = [  # label, hf path, query prefix, passage prefix
    ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", "", ""),
    ("all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2", "", ""),
    ("BGE-base-en-v1.5", "BAAI/bge-base-en-v1.5", "", ""),
    ("E5-base-v2", "intfloat/e5-base-v2", "query: ", "passage: "),
]
BGE_INSTR = "Represent this sentence for searching relevant passages: "
CHARS = 384
STREAM = "retrieval"


def mask(t, cve=False, url=False, ver=False, boil=False):
    if boil: t = BOIL_RE.sub(" ", t)
    if cve: t = CVE_RE.sub(" ", t)
    if url: t = URL_RE.sub(" ", t)
    if ver: t = VER_RE.sub(" ", t)
    return t


class FastBM25:
    """Sparse exact reimplementation of rank_bm25.BM25Okapi.get_scores."""

    def __init__(self, docs, k1=1.5, b=0.75, epsilon=0.25):
        vocab, indptr, indices, data = {}, [0], [], []
        n = len(docs); dl = np.empty(n); nd = Counter()
        for i, toks in enumerate(docs):
            dl[i] = len(toks); c = Counter(toks)
            for w, f in c.items():
                j = vocab.setdefault(w, len(vocab)); indices.append(j); data.append(f)
            nd.update(c.keys()); indptr.append(len(indices))
        self.vocab = vocab
        idf = np.empty(len(vocab)); s = 0.0; neg = []
        for w, j in vocab.items():
            v = math.log(n - nd[w] + 0.5) - math.log(nd[w] + 0.5); idf[j] = v; s += v
            if v < 0: neg.append(j)
        idf[neg] = epsilon * s / len(vocab)
        self.idf = idf
        M = csr_matrix((np.asarray(data, float), np.asarray(indices, np.int32), np.asarray(indptr, np.int64)),
                       shape=(n, len(vocab)))
        norm = k1 * (1 - b + b * dl / dl.mean())
        rows = np.repeat(np.arange(n), np.diff(M.indptr))
        M.data = M.data * (k1 + 1) / (M.data + norm[rows])
        self.M = M

    def scores(self, toks):
        q = np.zeros(len(self.vocab))
        for t in toks:
            j = self.vocab.get(t)
            if j is not None: q[j] += self.idf[j]
        return self.M @ q


def gold_rank(s: np.ndarray, g: int) -> tuple[int, int, int]:
    """(deterministic, optimistic, pessimistic) 1-based rank of index g in score vector s."""
    v = s[g]; gt = int((s > v).sum()); eq = s == v
    return gt + int(eq[:g].sum()) + 1, gt + 1, gt + int(eq.sum())


def topk_det(s: np.ndarray, k: int) -> np.ndarray:
    """Top-k indices, score descending then index ascending."""
    part = np.argpartition(-s, k)[:k]
    thr = s[part].min()
    cand = np.flatnonzero(s >= thr)
    order = np.lexsort((cand, -s[cand]))
    return cand[order][:k]


def summarise(r: np.ndarray) -> dict:
    n = len(r)
    return dict(n=n, r_at_1=float((r <= 1).mean()), r_at_5=float((r <= 5).mean()),
                r_at_10=float((r <= 10).mean()),
                mrr=float(np.where(r <= 10, 1.0 / r, 0.0).mean()),
                mrr_full=float((1.0 / r).mean()))


def save(task, model, cond, variant, qids, det, opt, pes, extra=None, t0=None, **prov):
    tag = f"{task}__{model}__{cond}__{variant}".replace(" ", "_").replace("+", "plus").replace("/", "_")
    np.savez_compressed(PERQ / f"{tag}.npz", qid=np.asarray(qids), rank=det, rank_opt=opt, rank_pes=pes)
    m = summarise(det)
    m_opt, m_pes = summarise(opt), summarise(pes)
    write_row(STREAM, result_type="retrieval", task=task, model=model, condition=cond, variant=variant,
              **m, mrr_tie_opt=m_opt["mrr"], mrr_tie_pes=m_pes["mrr"],
              n_queries_with_gold_tie=int((opt != pes).sum()), seed=SEED,
              seconds=round(time.time() - t0, 1) if t0 else None, per_query_file=f"per_query/{tag}.npz",
              **(extra or {}), **prov)
    log(f"  {task:6s} {model:20s} {cond:17s} {variant:14s} R@1={m['r_at_1']:.4f} R@5={m['r_at_5']:.4f} "
        f"R@10={m['r_at_10']:.4f} MRR={m['mrr']:.4f}")


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list: bm25,dense,hybrid,validate,pilot,sens")
    ap.add_argument("--smoke", action="store_true", help="tiny subset, writes to results/smoke/")
    args = ap.parse_args()
    global PERQ
    if args.smoke:
        import common
        common.RAW = ROOT / "results" / "smoke" / "raw"; common.RAW.mkdir(parents=True, exist_ok=True)
        PERQ = ROOT / "results" / "smoke" / "per_query"; PERQ.mkdir(parents=True, exist_ok=True)
    only = set(filter(None, args.only.split(",")))
    want = lambda k: not only or k in only
    import torch
    from sentence_transformers import SentenceTransformer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED); np.random.seed(SEED)
    hashes = data_hashes()
    prov_data = {"data_sha256_corpus": hashes["task1_cve_linkage/corpus.jsonl"][:16],
                 "data_sha256_task1_test": hashes["task1_cve_linkage/test.jsonl"][:16],
                 "data_sha256_task3_test": hashes["task3_temporal_generalization/test.jsonl"][:16],
                 "device": dev, "torch": torch.__version__}

    corpus = load(DATA / "task1_cve_linkage/corpus.jsonl")
    if args.smoke:
        keep = {q["cve_id"] for t in ("task1_cve_linkage", "task3_temporal_generalization")
                for q in load(DATA / t / "test.jsonl")[:60]}
        corpus = [d for i, d in enumerate(corpus) if i < 5000 or d["id"] in keep]
    cpos = {d["id"]: i for i, d in enumerate(corpus)}
    craw = [d["text"] for d in corpus]
    tests = {"task1": load(DATA / "task1_cve_linkage/test.jsonl"),
             "task3": load(DATA / "task3_temporal_generalization/test.jsonl")}
    pool = []
    idx_dir = ROOT / "data_release" / "cve_index"
    for f in sorted(idx_dir.glob("*_cve_index.jsonl")):
        for l in open(f):
            l = l.strip()
            if not l: continue
            r = json.loads(l)
            if r.get("cve_id") in cpos and len(r.get("exploit_text", "").split()) >= 8:
                pool.append(r)
    pilot = [{"id": f"pilot{i}", "cve_id": r["cve_id"], "text": CVE_RE.sub(" ", r["exploit_text"])}
             for i, r in enumerate(pool)]
    if args.smoke:
        tests = {k: v[:60] for k, v in tests.items()}; pilot = pilot[:60]
    log(f"corpus {len(corpus):,} | task1 {len(tests['task1']):,} | task3 {len(tests['task3']):,} | pilot {len(pilot):,}")
    golds = {k: np.array([cpos[q["cve_id"]] for q in v]) for k, v in tests.items()}
    golds["fp_cve"] = np.array([cpos[q["cve_id"]] for q in pilot])
    qsets = dict(tests, fp_cve=pilot)

    # ---------------- shortcut diagnostics ----------------
    if want("bm25") and ("diag", "-", "-") not in done_keys("diagnostics", ("task", "model", "condition")):
        own = sum(1 for d in corpus if d["id"].lower() in d["text"].lower())
        write_row("diagnostics", task="diag", model="-", condition="-", note="marker")
        write_row("diagnostics", task="corpus", model="-", condition="target_contains_own_cve", n=len(corpus),
                  value=own / len(corpus), count=own)
        for tk, te in tests.items():
            n = len(te)
            for nm, v in [("query_contains_gold_cve", sum(q["cve_id"].lower() in q["text"].lower() for q in te)),
                          ("query_contains_any_cve", sum(bool(CVE_RE.search(q["text"])) for q in te)),
                          ("query_contains_url", sum(bool(URL_RE.search(q["text"])) for q in te))]:
                write_row("diagnostics", task=tk, model="-", condition=nm, n=n, value=v / n, count=v)
            for s, c in Counter(q["source"] for q in te).most_common():
                write_row("diagnostics", task=tk, model="-", condition=f"source::{s}", n=n, value=c / n, count=c)
        cyr = sum(1 for r in pool if sum(1 for ch in r["exploit_text"] if "Ѐ" <= ch <= "ӿ")
                  / max(len(r["exploit_text"]), 1) > 0.05)
        write_row("diagnostics", task="fp_cve", model="-", condition="cyrillic_heavy_share", n=len(pool),
                  value=cyr / len(pool), count=cyr)
        log("diagnostics written")

    done = done_keys(STREAM, ("task", "model", "condition", "variant"))
    is_done = lambda m, c, v: all((t, m, c, v) in done for t in ("task1", "task3"))

    # ---------------- BM25 + validation ----------------
    bm25_top100 = {}
    if want("bm25") or want("hybrid") or want("validate"):
        for cond, kw in CONDITIONS:
            need_bm = want("bm25") and not is_done("BM25", cond, "primary")
            need_hy = want("hybrid") and not is_done("Hybrid BM25+mpnet", cond, "primary")
            need_val = want("validate") and cond == "as_released" and not is_done("BM25", cond, "validated")
            if not (need_bm or need_hy or need_val):
                continue
            t0 = time.time()
            bm = FastBM25([mask(t, **kw).lower().split() for t in craw])
            log(f"BM25 index {cond}: {len(bm.vocab):,} terms ({time.time()-t0:.0f}s)")
            for tk, te in tests.items():
                tq = time.time()
                det, opt, pes, top, bsc = [], [], [], [], []
                for q, g in zip(te, golds[tk]):
                    s = bm.scores(mask(q["text"], **kw).lower().split())
                    a, b_, c = gold_rank(s, g); det.append(a); opt.append(b_); pes.append(c)
                    if need_hy:
                        t_ = topk_det(s, 100); top.append(t_); bsc.append(s[t_])
                det, opt, pes = map(np.array, (det, opt, pes))
                if need_bm:
                    save(tk, "BM25", cond, "primary", [q["id"] for q in te], det, opt, pes, t0=tq, **prov_data)
                if need_hy:
                    bm25_top100[(tk, cond)] = (top, bsc)
            if need_val:
                from rank_bm25 import BM25Okapi
                tv = time.time()
                ref = BM25Okapi([t.lower().split() for t in craw])
                rs = np.random.default_rng(SEED).choice(len(tests["task1"]), 50, replace=False)
                maxdiff, mism = 0.0, 0
                for i in rs:
                    toks = tests["task1"][i]["text"].lower().split()
                    a, b_ = bm.scores(toks), ref.get_scores(toks)
                    maxdiff = max(maxdiff, float(np.abs(a - b_).max()))
                    mism += int(not np.array_equal(topk_det(a, 10), topk_det(b_, 10)))
                write_row(STREAM, result_type="validation", task="task1", model="BM25", condition=cond,
                          variant="validated", n=50, max_abs_score_diff=maxdiff, top10_mismatches=mism,
                          seconds=round(time.time() - tv, 1), **prov_data)
                log(f"FastBM25 vs rank_bm25 on 50 queries: max|diff|={maxdiff:.2e} top-10 mismatches={mism}")
                del ref
            del bm; gc.collect()

    # ---------------- dense (+ hybrid reuse of mpnet) ----------------
    def encode(model, texts, bs=128):
        return model.encode(texts, batch_size=bs, normalize_embeddings=True, convert_to_tensor=True,
                            show_progress_bar=False).float()

    def dense_ranks(qe, ce, gold):
        det, opt, pes = [], [], []
        g_t = torch.as_tensor(gold, device=ce.device)
        ar = torch.arange(ce.shape[0], device=ce.device)
        for s in range(0, qe.shape[0], 64):
            sims = qe[s:s + 64] @ ce.T
            gs = sims.gather(1, g_t[s:s + 64, None])
            gt = (sims > gs).sum(1); eq = sims == gs
            before = (eq & (ar[None, :] < g_t[s:s + 64, None])).sum(1)
            det.append((gt + before + 1).cpu()); opt.append((gt + 1).cpu()); pes.append((gt + eq.sum(1)).cpu())
        return [torch.cat(x).numpy().astype(np.int64) for x in (det, opt, pes)]

    def hybrid_ranks(qe, ce, top, bsc, gold):
        det, opt, pes = [], [], []
        for i, (cand, sc) in enumerate(zip(top, bsc)):
            bn = sc / (sc.max() if sc.max() > 0 else 1.0)
            dn = ((ce[torch.as_tensor(cand, device=ce.device)] @ qe[i]).cpu().numpy() + 1) / 2
            comb = 0.7 * dn + 0.3 * bn
            w = np.flatnonzero(cand == gold[i])
            if len(w) == 0:
                det.append(10 ** 9); opt.append(10 ** 9); pes.append(10 ** 9); continue
            a, b_, c = gold_rank(comb, int(w[0]))  # ties inside candidates: BM25 order already index-ascending
            det.append(a); opt.append(b_); pes.append(c)
        return [np.array(x) for x in (det, opt, pes)]

    if want("dense") or want("hybrid") or want("pilot"):
        for label, path, qp, dp in DENSE:
            todo = [(c, kw) for c, kw in CONDITIONS
                    if (want("dense") and not is_done(label, c, "primary"))
                    or (label == "all-mpnet-base-v2" and want("hybrid") and not is_done("Hybrid BM25+mpnet", c, "primary"))
                    or (c == "as_released" and want("pilot")
                        and ("fp_cve", label, "cve_masked", "primary") not in done)]
            if not todo:
                continue
            tm = time.time()
            model = SentenceTransformer(path, device=dev); model.max_seq_length = 512
            rev = model_revision(path)
            prov = dict(prov_data, hf_model=path, hf_revision=rev, max_seq_length=512, char_trunc=CHARS,
                        query_prefix=qp, passage_prefix=dp)
            log(f"{label} loaded ({rev})")
            # Embeddings depend only on each document's own text, so a masked condition
            # re-encodes just the documents whose first CHARS characters the mask changes.
            base_txt = [dp + t[:CHARS] for t in craw]
            t0 = time.time()
            base = encode(model, base_txt)
            log(f"  corpus encoded as_released {tuple(base.shape)} {time.time()-t0:.0f}s")
            checked = False
            for cond, kw in todo:
                t0 = time.time()
                if cond == "as_released":
                    ce = base; n_new = len(craw)
                else:
                    txt = [dp + mask(t, **kw)[:CHARS] for t in craw]
                    chg = [i for i, (a, b_) in enumerate(zip(txt, base_txt)) if a != b_]
                    ce = base.clone()
                    if chg:
                        ce[torch.as_tensor(chg, device=ce.device)] = encode(model, [txt[i] for i in chg])
                    n_new = len(chg)
                    if not checked:
                        # exactness check: re-encode 256 unchanged docs mixed into a batch of changed ones
                        same = [i for i in np.random.default_rng(SEED).choice(len(craw), 4000, replace=False)
                                if txt[i] == base_txt[i]][:256]
                        mix = same + chg[:256]
                        e = encode(model, [txt[i] for i in mix])[:len(same)]
                        diff = float((e - base[torch.as_tensor(same, device=base.device)]).abs().max())
                        write_row(STREAM, result_type="validation", task="corpus", model=label, condition=cond,
                                  variant="embedding_reuse", n=len(same), max_abs_embedding_diff=diff, **prov)
                        log(f"  reuse check {cond}: max|emb diff| over {len(same)} unchanged docs = {diff:.2e}")
                        checked = True
                log(f"  corpus encoded {cond} {tuple(ce.shape)} ({n_new:,} docs re-encoded) {time.time()-t0:.0f}s")
                for tk, te in tests.items():
                    tq = time.time()
                    qe = encode(model, [qp + mask(q["text"], **kw)[:CHARS] for q in te])
                    if want("dense") and (tk, label, cond, "primary") not in done:
                        save(tk, label, cond, "primary", [q["id"] for q in te], *dense_ranks(qe, ce, golds[tk]),
                             t0=t0 if tk == "task1" else tq, **prov)
                    if label == "all-mpnet-base-v2" and (tk, cond) in bm25_top100:
                        top, bsc = bm25_top100[(tk, cond)]
                        save(tk, "Hybrid BM25+mpnet", cond, "primary", [q["id"] for q in te],
                             *hybrid_ranks(qe, ce, top, bsc, golds[tk]), t0=tq,
                             extra=dict(alpha=0.7, bm25_candidates=100), **prov)
                    if label == "BGE-base-en-v1.5" and cond == "as_released" and want("sens") \
                            and (tk, label, cond, "query_instruction") not in done:
                        qi = encode(model, [BGE_INSTR + q["text"][:CHARS] for q in te])
                        save(tk, label, cond, "query_instruction", [q["id"] for q in te],
                             *dense_ranks(qi, ce, golds[tk]), t0=tq, **dict(prov, query_prefix=BGE_INSTR))
                if cond == "as_released" and want("pilot") and ("fp_cve", label, "cve_masked", "primary") not in done:
                    tq = time.time()
                    qe = encode(model, [qp + q["text"][:CHARS] for q in pilot])
                    save("fp_cve", label, "cve_masked", "primary", [q["id"] for q in pilot],
                         *dense_ranks(qe, ce, golds["fp_cve"]), t0=tq, **prov)
                if ce is not base:
                    del ce
                gc.collect(); torch.mps.empty_cache() if dev == "mps" else None
            del model, base; gc.collect(); torch.mps.empty_cache() if dev == "mps" else None
            log(f"{label} done ({time.time()-tm:.0f}s)")

    if want("pilot") and ("fp_cve", "BM25", "cve_masked", "primary") not in done:
        t0 = time.time()
        bm = FastBM25([t.lower().split() for t in craw])
        r = [gold_rank(bm.scores(q["text"][:CHARS].lower().split()), g) for q, g in zip(pilot, golds["fp_cve"])]
        save("fp_cve", "BM25", "cve_masked", "primary", [q["id"] for q in pilot],
             *[np.array(x) for x in zip(*r)], t0=t0, **prov_data)
        del bm; gc.collect()

    # ---------------- sensitivity: E5 without prefixes, full text up to 512 tokens ----------------
    if want("sens"):
        for label, path, qp, dp in DENSE:
            variants = [("full_text_512tok", qp, dp, None)]
            if label == "E5-base-v2":
                variants.append(("no_prefixes", "", "", CHARS))
            if label == "BGE-base-en-v1.5":
                variants.append(("query_instruction", BGE_INSTR, "", CHARS))
            variants = [v for v in variants if not is_done(label, "as_released", v[0])]
            if not variants:
                continue
            model = SentenceTransformer(path, device=dev); model.max_seq_length = 512
            rev = model_revision(path)
            for var, q_, d_, ch in variants:
                t0 = time.time()
                cut = (lambda t: t[:ch]) if ch else (lambda t: t)
                ce = encode(model, [d_ + cut(t) for t in craw], bs=64)
                for tk, te in tests.items():
                    qe = encode(model, [q_ + cut(q["text"]) for q in te], bs=64)
                    save(tk, label, "as_released", var, [q["id"] for q in te], *dense_ranks(qe, ce, golds[tk]),
                         t0=t0, **dict(prov_data, hf_model=path, hf_revision=rev, max_seq_length=512,
                                       char_trunc=ch, query_prefix=q_, passage_prefix=d_))
                del ce; gc.collect(); torch.mps.empty_cache() if dev == "mps" else None
            del model; gc.collect()
    log("retrieval run complete")


if __name__ == "__main__":
    main()
