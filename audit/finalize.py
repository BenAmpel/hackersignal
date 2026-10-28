"""Build the camera-ready result files from the raw run rows.

Inputs  results/camera_ready/raw/*.jsonl, per_query/*.npz, data_integrity.json,
        reference/july_rerun/hackersignal_corrected_results.csv,
        published values (current tex and pre-10/27 backup, parsed below).
Outputs results.csv (single source of truth), provenance.json, reconciliation.md.
Nothing here trains or encodes; every number is read from a run artefact.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from common import DATA, OUT, PERQ, RAW, ROOT

N_BOOT_MRR, N_BOOT_F1, SEED = 10000, 2000, 20261027
CLASSES = ["injection", "xss", "memory_corruption", "dos", "file_inclusion", "auth_access", "rce",
           "info_disclosure"]
PAPER_NAME = {"all-MiniLM-L6-v2": "MiniLM-L6-v2", "all-mpnet-base-v2": "mpnet-base-v2",
              "BGE-base-en-v1.5": "BGE-base-v1.5", "E5-base-v2": "E5-base-v2", "BM25": "BM25",
              "Hybrid BM25+mpnet": "Hybrid BM25+mpnet"}
T2_MODELS = ["Decision Tree", "TF-IDF + LR", "SVM", "RNN", "GRU", "LSTM", "BiLSTM", "SecBERT",
             "SecureBERT", "CySecBERT"]
# Primary Task 2 variant per family (the protocol proposed for the camera-ready; see discrepancies.md)
T2_PRIMARY = {"bow": "repo_trunc2000", "rnn": "packed_last_token", "tf": "subsample15000"}
FAMILY = {"Decision Tree": "bow", "TF-IDF + LR": "bow", "SVM": "bow", "RNN": "rnn", "GRU": "rnn",
          "LSTM": "rnn", "BiLSTM": "rnn", "SecBERT": "tf", "SecureBERT": "tf", "CySecBERT": "tf"}


def rows(stream):
    p = RAW / f"{stream}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def mrr10(r):
    return np.where(r <= 10, 1.0 / r, 0.0)


# ---------------------------------------------------------------- published values
def parse_tex(path):
    """Table 4/6 rows and Table 5 rows from a tex file -> {(task, model): {metric: value}}."""
    tex = Path(path).read_text()
    out = {}
    for task, label in [("task1", "tab:task1-results"), ("task3", "tab:task3-results")]:
        blk = tex[tex.index(label):]; blk = blk[:blk.index("\\end{tabular}")]
        for line in blk.splitlines():
            m = re.match(r"\s*([A-Za-z0-9\-\.\+ ]+?)\s*&(.*)\\\\", line)
            if m and m.group(1).strip() in PAPER_NAME.values():
                v = [float(re.sub(r"[^0-9.]", "", x)) for x in m.group(2).split("&")]
                out[(task, m.group(1).strip())] = dict(zip(["r_at_1", "r_at_5", "r_at_10", "mrr"], v))
    blk = tex[tex.index("tab:task2-results"):]; blk = blk[:blk.index("\\end{tabular}")]
    for line in blk.splitlines():
        m = re.match(r"\s*&\s*([A-Za-z\-\+ \(\)]+?)\s*&(.*)\\\\", line)
        if m:
            name = m.group(1).replace(" (Linear)", "").strip()
            v = [float("0" + re.sub(r"[^0-9.]", "", x)) for x in m.group(2).split("&")]
            out[("task2", name)] = dict(zip(["macro_f1", "f1_info_disclosure", "f1_memory_corruption",
                                             "f1_auth_access"], v))
    return out


def july():
    out = {}
    for r in csv.DictReader(open(ROOT / "reference/july_rerun/hackersignal_corrected_results.csv")):
        out[(r["result_type"], r["task"], r["model"], r["condition"])] = r
    return out


# ---------------------------------------------------------------- bootstrap
def boot_mean(v, n_boot, rng):
    ix = rng.integers(0, len(v), size=(n_boot, len(v)))
    d = v[ix].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_f1(runs, n_boot, rng, ix=None):
    """runs: list of (y, p) for the seeds of one model; statistic = seed-mean macro-F1."""
    n = len(runs[0][0])
    ix = rng.integers(0, n, size=(n_boot, n)) if ix is None else ix
    d = np.array([np.mean([f1_score(y[i], p[i], average="macro", labels=list(range(8)), zero_division=0)
                           for y, p in runs]) for i in ix])
    return d, ix


def main():
    rng = np.random.default_rng(SEED)
    integ = json.loads((OUT / "data_integrity.json").read_text())
    pub_now = parse_tex(ROOT / "reference/paper/HackerSignal_NeurIPS/HackerSignal_neurips2026.tex")
    pub_old = parse_tex(ROOT / "reference/paper/HackerSignal_NeurIPS.bak-2026-10-27/HackerSignal_neurips2026.tex")
    jul = july()
    res, prov = [], []

    def emit(prov_src=None, **r):
        res.append(r)
        if prov_src is not None:
            prov.append({"row": len(res) - 1, **{k: r.get(k) for k in ("result_type", "task", "model", "condition",
                                                                        "variant")}, **prov_src})

    # ---------- retrieval ----------
    R = rows("retrieval")
    rr = {}
    for r in R:
        if r["result_type"] != "retrieval":
            continue
        z = np.load(OUT / r["per_query_file"])
        rr[(r["task"], r["model"], r["condition"], r["variant"])] = (z["rank"], z["qid"])
        keep = {k: r[k] for k in ("n", "r_at_1", "r_at_5", "r_at_10", "mrr", "mrr_full", "mrr_tie_opt",
                                  "mrr_tie_pes", "n_queries_with_gold_tie")}
        ci = {}
        if r["variant"] == "primary" and r["condition"] in ("as_released", "cve_masked"):
            ci["mrr_ci_lo"], ci["mrr_ci_hi"] = boot_mean(mrr10(z["rank"]), N_BOOT_MRR, rng)
            ci["r_at_1_ci_lo"], ci["r_at_1_ci_hi"] = boot_mean((z["rank"] <= 1).astype(float), N_BOOT_MRR, rng)
        emit(prov_src={k: r.get(k) for k in ("git_sha", "seed", "seconds", "device", "torch", "hf_model",
                                             "hf_revision", "data_sha256_corpus", "data_sha256_task1_test",
                                             "data_sha256_task3_test", "per_query_file", "written_at",
                                             "machine", "max_seq_length", "char_trunc", "query_prefix",
                                             "passage_prefix")},
             result_type="retrieval", task=r["task"], model=r["model"], condition=r["condition"],
             variant=r["variant"], seeds=1, **keep, **ci, note="MRR = MRR@10; ranks over full 340,536 corpus")
    seen_val = set()
    for r in reversed(R):  # latest validation row per (model, condition, variant); BM25 check ran twice
        if r["result_type"] != "validation" or (r["model"], r["condition"], r["variant"]) in seen_val:
            continue
        seen_val.add((r["model"], r["condition"], r["variant"]))
        if r["variant"] == "validated":
            note = f"FastBM25 vs rank_bm25: max|score diff|={r['max_abs_score_diff']:.2e}, top-10 mismatches={r['top10_mismatches']}"
        else:
            note = f"embedding reuse: max|emb diff| over {r['n']} unchanged docs={r['max_abs_embedding_diff']:.2e}"
        emit(prov_src={"git_sha": r["git_sha"], "seconds": r.get("seconds")}, result_type="validation",
             task=r["task"], model=r["model"], condition=r["condition"], variant=r["variant"], n=r["n"], note=note)

    # paired deltas, top two per metric, as_released primary, plus multi-gold grouped sensitivity
    for task in ("task1", "task3"):
        cand = {m: v[0] for (t, m, c, var), v in rr.items() if t == task and c == "as_released" and var == "primary"}
        if len(cand) < 2:
            continue
        for metric, f in [("mrr", mrr10), ("r_at_1", lambda r: (r <= 1).astype(float)),
                          ("r_at_5", lambda r: (r <= 5).astype(float)), ("r_at_10", lambda r: (r <= 10).astype(float))]:
            order = sorted(cand, key=lambda m: -f(cand[m]).mean())
            a, b = f(cand[order[0]]), f(cand[order[1]])
            ix = rng.integers(0, len(a), size=(N_BOOT_MRR, len(a)))
            d = (a[ix] - b[ix]).mean(1)
            lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            emit(prov_src={"n_boot": N_BOOT_MRR, "boot_seed": SEED}, result_type="paired_delta", task=task,
                 model=f"{order[0]} - {order[1]}", condition="as_released", variant=metric, n=len(a),
                 delta=float((a - b).mean()), delta_ci_lo=lo, delta_ci_hi=hi,
                 note="excludes zero" if lo > 0 or hi < 0 else "INCLUDES ZERO")
        # grouped by query id (a multi-CVE query is one query; hit if any gold is found)
        for m, r in cand.items():
            qid = rr[(task, m, "as_released", "primary")][1]
            best = defaultdict(lambda: 10 ** 9)
            for q, k in zip(qid, r):
                best[q] = min(best[q], int(k))
            g = np.array(list(best.values()))
            emit(prov_src={"derived_from": f"per_query rows of {m}"}, result_type="retrieval", task=task, model=m,
                 condition="as_released", variant="grouped_multi_gold", n=len(g),
                 r_at_1=float((g <= 1).mean()), r_at_5=float((g <= 5).mean()), r_at_10=float((g <= 10).mean()),
                 mrr=float(mrr10(g).mean()), note="duplicate query ids merged; any gold counts")
        # per-source MRR
        test = [json.loads(l) for l in open(DATA / ("task1_cve_linkage" if task == "task1" else
                                                    "task3_temporal_generalization") / "test.jsonl")]
        src = np.array([q["source"] for q in test])
        for m, r in cand.items():
            for s in sorted(set(src)):
                sel = src == s
                emit(prov_src={"derived_from": f"per_query rows of {m}"}, result_type="per_source", task=task,
                     model=m, condition=f"source::{s}", variant="primary", n=int(sel.sum()),
                     r_at_1=float((r[sel] <= 1).mean()), mrr=float(mrr10(r[sel]).mean()))

    # ---------- task 2 ----------
    T = rows("task2")
    runs = defaultdict(list)
    for r in T:
        if r.get("variant") == "availability":
            emit(prov_src={"git_sha": r["git_sha"]}, result_type="classification", task="task2", model=r["model"],
                 condition="temporal_ood", variant="not_run", note=r["note"])
            continue
        runs[(r["model"], r["variant"])].append(r)
    preds = {}
    for (m, var), rs in sorted(runs.items()):
        rs = sorted(rs, key=lambda x: x["seed"])
        ps = []
        for r in rs:
            z = np.load(OUT / r["per_query_file"]); ps.append((z["y_true"], z["y_pred"]))
        preds[(m, var)] = ps
        agg = {}
        for k in ["macro_f1", "weighted_f1", "accuracy"] + [f"f1_{c}" for c in CLASSES]:
            v = np.array([r[k] for r in rs])
            agg[k] = float(v.mean()); agg[k + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else None
        d, _ = boot_f1(ps, N_BOOT_F1, rng)
        emit(prov_src={"seeds": [r["seed"] for r in rs], "seconds_per_seed": [r["seconds"] for r in rs],
                       "git_sha": rs[0]["git_sha"], "device": rs[0].get("device"),
                       "hf_model": rs[0].get("hf_model"), "hf_revision": rs[0].get("hf_revision"),
                       "best_epochs": [r.get("best_epoch") for r in rs],
                       "best_val_macro_f1": [r.get("best_val_macro_f1") for r in rs],
                       "train_rows": rs[0].get("train_rows"),
                       "data_sha256": {k: rs[0][k] for k in rs[0] if k.startswith("data_sha256")},
                       "per_query_files": [r["per_query_file"] for r in rs], "n_boot": N_BOOT_F1},
             result_type="classification", task="task2", model=m, condition="temporal_ood", variant=var,
             seeds=len(rs), n=rs[0]["n"], **agg, macro_f1_ci_lo=float(np.percentile(d, 2.5)),
             macro_f1_ci_hi=float(np.percentile(d, 97.5)),
             note="mean (SD) over seeds; CI = test bootstrap of the seed-mean macro-F1")
    # ---------- task 2, full 64K training set on the GSU cluster (NVIDIA L40S; appendix only) ----------
    # Per-query predictions stay on the cluster, so these rows carry mean (SD) over seeds and no bootstrap CI.
    # The two mini (MPS) SecBERT full_64413 runs are superseded by these and kept on record only.
    for r in res:
        if r.get("task") == "task2" and r.get("variant") == "full_64413" and r.get("result_type") == "classification":
            r["note"] = "SUPERSEDED by full_64413_l40s (mini MPS run stopped after 2 seeds); not reported"
    cl = defaultdict(list)
    for r in rows("task2_cluster_tf_full"):
        if r.get("variant") == "full_64413":
            cl[r["model"]].append(r)
    for m in ("SecBERT", "SecureBERT", "CySecBERT"):
        rs = sorted(cl.get(m, []), key=lambda x: x["seed"])
        if not rs:
            continue
        assert len({r["seed"] for r in rs}) == len(rs) and {r.get("gpu") for r in rs} == {"NVIDIA L40S"}
        agg = {}
        for k in ["macro_f1", "weighted_f1", "accuracy"] + [f"f1_{c}" for c in CLASSES]:
            v = np.array([r[k] for r in rs])
            agg[k] = float(v.mean()); agg[k + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else None
        emit(prov_src={"seeds": [r["seed"] for r in rs], "seconds_per_seed": [r["seconds"] for r in rs],
                       "git_sha": rs[0]["git_sha"], "device": "cuda", "gpu": rs[0]["gpu"], "torch": rs[0]["torch"],
                       "hf_model": rs[0]["hf_model"], "hf_revision": rs[0]["hf_revision"],
                       "best_epochs": [r["best_epoch"] for r in rs], "train_rows": rs[0]["train_rows"],
                       "data_sha256": {k: rs[0][k] for k in rs[0] if k.startswith("data_sha256")},
                       "source": "GSU ACIDS array 4777484 (raw/task2_cluster_tf_full.jsonl)"},
             result_type="classification", task="task2", model=m, condition="temporal_ood",
             variant="full_64413_l40s", seeds=len(rs), n=rs[0]["n"], **agg,
             note="NVIDIA L40S; all 64,413 training rows; best epoch by validation; mean (SD) over seeds, no CI")

    # ---------- B6: bi-encoder fine-tuned on Task 3 train (pre-2022 CVEs), evaluated on the CVE-disjoint test ----------
    b6_dir = ROOT.parent / "b6-finetune" / "results" / "b6"
    b6_rows = [json.loads(x) for x in open(b6_dir / "rows.jsonl")] if (b6_dir / "rows.jsonl").exists() else []
    t3_bge = next((r for r in res if r["result_type"] == "retrieval" and r["task"] == "task3" and
                   r["model"] == "BGE-base-en-v1.5" and r["condition"] == "as_released" and r["variant"] == "primary"), None)
    for m in sorted({r["model"] for r in b6_rows}):
        zs = [r for r in b6_rows if r["model"] == m and r["variant"] == "zero_shot"]
        ft = sorted((r for r in b6_rows if r["model"] == m and r["variant"] == "finetuned"), key=lambda x: x["seed"])
        tag = m.replace(" ", "")
        if zs:
            emit(prov_src={k: zs[-1].get(k) for k in ("hf_model", "seconds", "written_at")},
                 result_type="retrieval", task="task3", model=m, condition="b6_zero_shot", variant="primary",
                 seeds=1, n=zs[-1]["test_n"], r_at_1=zs[-1]["test_r_at_1"], r_at_5=zs[-1]["test_r_at_5"],
                 r_at_10=zs[-1]["test_r_at_10"], mrr=zs[-1]["test_mrr"],
                 note="B6 harness zero-shot (must reproduce the audit's as_released row)")
            if m == "BGE-base-en-v1.5" and t3_bge is not None:
                assert abs(zs[-1]["test_mrr"] - t3_bge["mrr"]) < 5e-4, (zs[-1]["test_mrr"], t3_bge["mrr"])
        if len(ft) < 3:
            continue  # fold a model in only when all three seeds are done
        z0 = np.load(b6_dir / "per_query" / f"task3__{tag}__zero_shot.npz")
        zr = [np.load(b6_dir / "per_query" / f"task3__{tag}__finetuned__seed{r['seed']}.npz") for r in ft]
        assert all((z["qid"] == z0["qid"]).all() for z in zr)
        per_q_ft = np.mean([mrr10(z["rank"]) for z in zr], axis=0)  # seed-mean MRR@10 per query
        per_q_zs = mrr10(z0["rank"])
        lo, hi = boot_mean(per_q_ft, N_BOOT_MRR, rng)
        agg = {}
        for k in ("r_at_1", "r_at_5", "r_at_10", "mrr"):
            v = np.array([r[f"test_{k}"] for r in ft])
            agg[k] = float(v.mean())
        sd = float(np.std([r["test_mrr"] for r in ft], ddof=1))
        emit(prov_src={"seeds": [r["seed"] for r in ft], "train_seconds": [r["train_seconds"] for r in ft],
                       "hf_model": ft[0]["hf_model"], "saved_models": [r.get("saved_model") for r in ft],
                       "loss": "CachedMultipleNegativesRankingLoss (scale 20, mini-batch 16; batch-64 objective)",
                       "device": "mps", "worktree": "b6-finetune"},
             result_type="retrieval", task="task3", model=m, condition="b6_finetuned", variant="primary",
             seeds=len(ft), n=ft[0]["test_n"], **agg, mrr_ci_lo=lo, mrr_ci_hi=hi,
             note=f"fine-tuned on Task 3 train; mean over seeds (MRR SD {sd:.4f}); CI = bootstrap of seed-mean MRR@10")
        idx = rng.integers(0, len(per_q_ft), size=(N_BOOT_MRR, len(per_q_ft)))
        dd = per_q_ft[idx].mean(1) - per_q_zs[idx].mean(1)
        dlo, dhi = float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
        emit(prov_src={"n_boot": N_BOOT_MRR, "boot_seed": SEED}, result_type="paired_delta", task="task3",
             model=f"{m} fine-tuned - zero-shot", condition="b6", variant="mrr", n=len(per_q_ft),
             delta=float(per_q_ft.mean() - per_q_zs.mean()), delta_ci_lo=dlo, delta_ci_hi=dhi,
             note="excludes zero" if dlo > 0 or dhi < 0 else "INCLUDES ZERO")

    prim = {m: (m, T2_PRIMARY[FAMILY[m]]) for m in T2_MODELS if (m, T2_PRIMARY[FAMILY[m]]) in preds}
    if len(prim) >= 2:
        f1m = {m: np.mean([f1_score(y, p, average="macro") for y, p in preds[k]]) for m, k in prim.items()}
        order = sorted(f1m, key=lambda m: -f1m[m])
        da, ix = boot_f1(preds[prim[order[0]]], N_BOOT_F1, rng)
        db, _ = boot_f1(preds[prim[order[1]]], N_BOOT_F1, rng, ix=ix)
        lo, hi = float(np.percentile(da - db, 2.5)), float(np.percentile(da - db, 97.5))
        emit(prov_src={"n_boot": N_BOOT_F1, "boot_seed": SEED}, result_type="paired_delta", task="task2",
             model=f"{order[0]} - {order[1]}", condition="temporal_ood", variant="macro_f1", n=len(ix[0]),
             delta=float(f1m[order[0]] - f1m[order[1]]), delta_ci_lo=lo, delta_ci_hi=hi,
             note="excludes zero" if lo > 0 or hi < 0 else "INCLUDES ZERO")

    # ---------- diagnostics ----------
    for r in rows("diagnostics"):
        if r.get("note") == "marker":
            continue
        emit(prov_src={"git_sha": r["git_sha"]}, result_type="diagnostic", task=r["task"], model="-",
             condition=r["condition"], variant="-", n=r.get("n"), value=r.get("value"), count=r.get("count"))
    for t in ("task1_cve_linkage", "task3_temporal_generalization"):
        c = integ["checks"][t]
        for k in ("cve_overlap_train_test", "cve_overlap_train_val", "cve_overlap_val_test",
                  "test_exact_dup_of_train_text", "test_internal_exact_dups", "duplicate_ids_in_test"):
            emit(prov_src={"source": "data_integrity.json"}, result_type="data_check", task=t, model="-",
                 condition=k, variant="-", value=c[k])

    # ---------- write results.csv ----------
    cols = ["result_type", "task", "model", "condition", "variant", "seeds", "n", "r_at_1", "r_at_1_ci_lo",
            "r_at_1_ci_hi", "r_at_5", "r_at_10", "mrr", "mrr_ci_lo", "mrr_ci_hi", "mrr_full", "mrr_tie_opt",
            "mrr_tie_pes", "n_queries_with_gold_tie", "macro_f1", "macro_f1_sd", "macro_f1_ci_lo",
            "macro_f1_ci_hi", "weighted_f1", "weighted_f1_sd", "accuracy", "accuracy_sd"] + \
           [x for c in CLASSES for x in (f"f1_{c}", f"f1_{c}_sd")] + \
           ["delta", "delta_ci_lo", "delta_ci_hi", "value", "count", "note"]
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row_id"] + cols, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(res):
            w.writerow({"row_id": i, **{k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()}})
    (OUT / "provenance.json").write_text(json.dumps({
        "generated_by": "audit/finalize.py", "data_files": integ["files"], "hf_repo_sha": integ["hf_repo_sha"],
        "bootstrap": {"mrr_resamples": N_BOOT_MRR, "macro_f1_resamples": N_BOOT_F1, "seed": SEED},
        "rows": prov}, indent=1, default=str))

    # ---------- reconciliation ----------
    L = ["# Three-way reconciliation", "",
         "Published = camera-ready tex as of 2026-10-27 (Tables 4/6 hold the July values) and, in brackets,",
         "the submitted tex (`HackerSignal_NeurIPS.bak-2026-10-27`). July = `hackersignal_corrected_results.csv`.",
         "Rerun = this audit (`results.csv`). Δ = rerun − published (current tex). Flag: |Δ| > 0.005.", ""]
    for task, tname in [("task1", "Table 4 — Task 1"), ("task3", "Table 6 — Task 3")]:
        L += [f"## {tname}", "", "| model | metric | published [submitted] | July | rerun | Δ | verdict |",
              "|---|---|---|---|---|---|---|"]
        for model, pname in PAPER_NAME.items():
            rr_ = next((r for r in res if r["result_type"] == "retrieval" and r["task"] == task and
                        r["model"] == model and r["condition"] == "as_released" and r["variant"] == "primary"), None)
            j = jul.get(("retrieval", task, model, "as_released"))
            for met in ("r_at_1", "r_at_5", "r_at_10", "mrr"):
                p = pub_now.get((task, pname), {}).get(met); po = pub_old.get((task, pname), {}).get(met)
                v = rr_[met] if rr_ else None; jv = float(j[met]) if j else None
                dlt = (v - p) if (v is not None and p is not None) else None
                verdict = "NOT RUN" if v is None else ("match" if abs(dlt) <= 0.005 else "**DIFFERS**")
                if v is not None and jv is not None and abs(v - jv) > 0.005:
                    verdict += " (rerun≠July)"
                L.append(f"| {pname} | {met} | {p} [{po}] | {'' if jv is None else f'{jv:.4f}'} | "
                         f"{'' if v is None else f'{v:.4f}'} | {'' if dlt is None else f'{dlt:+.4f}'} | {verdict} |")
        L.append("")
    L += ["## Table 5 — Task 2 (rerun = primary variant, mean over seeds)", "",
          "| model | metric | published | July | rerun (variant) | Δ | verdict |", "|---|---|---|---|---|---|---|"]
    for m in T2_MODELS:
        var = T2_PRIMARY[FAMILY[m]]
        rr_ = next((r for r in res if r["result_type"] == "classification" and r["model"] == m and
                    r["variant"] == var), None)
        for met, jcol in [("macro_f1", "macro_f1"), ("f1_info_disclosure", "info_disclosure"),
                          ("f1_memory_corruption", "memory_corruption"), ("f1_auth_access", "auth_access")]:
            p = pub_now.get(("task2", m), {}).get(met)
            if met == "macro_f1":
                j = jul.get(("classification", "task2", m, "temporal_ood")); jv = float(j["macro_f1"]) if j and j["macro_f1"] else None
            else:
                j = jul.get(("per_class_f1", "task2", m, jcol)); jv = float(j["f1"]) if j else None
            v = rr_[met] if rr_ else None
            tol = 0.005 if met == "macro_f1" else 0.005 + 0.005  # per-class published to 2 dp
            dlt = (v - p) if (v is not None and p is not None) else None
            verdict = "NOT RUN" if v is None else ("new row" if p is None else
                                                    ("match" if abs(dlt) <= tol else "**DIFFERS**"))
            L.append(f"| {m} | {met} | {'' if p is None else p} | {'' if jv is None else f'{jv:.4f}'} | "
                     f"{'' if v is None else f'{v:.4f}'} ({var}) | {'' if dlt is None else f'{dlt:+.4f}'} | {verdict} |")
    L += ["", "Per-class published values have two decimals, so per-class rows use tolerance 0.01.",
          "Explanations for every DIFFERS row are in discrepancies.md (cause column)."]
    (OUT / "reconciliation.md").write_text("\n".join(L) + "\n")
    print(f"results.csv: {len(res)} rows; provenance rows {len(prov)}")


if __name__ == "__main__":
    main()
