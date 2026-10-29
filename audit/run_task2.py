"""Task 2 (8-class exploit type): every classification number, from the released data.

Spec = paper Appendix D; where the paper is silent, the released repo code
(src/etg/benchmark/run_baselines_v2.py at e0c1153, i.e. after the 086db52 8-class fix).
Departures are separate, named variants. Model selection uses validation only.

Stages
  bow      DT / TF-IDF+LR / LinearSVC; variants repo_trunc2000 (primary) and full_text (July)
  rnn      RNN / GRU / LSTM / BiLSTM; variants
             repo_last_hidden   h_n of a right-padded 300-step sequence (as released)
             packed_last_token  pack_padded_sequence, h_n = last real token (corrected)
             mean_pool_july     mean of outputs over all 300 positions (July notebook)
  tf       SecBERT / SecureBERT / CySecBERT on the 15K subsample (repo protocol), best epoch by val
  tf_full  same, on all 64,413 training rows
  tf_july  July notebook protocol (class-weighted CE, grad clip, no char cut), seed 42 only
Each run writes its row, per-class F1 and test predictions as soon as it ends.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from collections import Counter

import numpy as np

from common import DATA, PERQ, ROOT, data_hashes, done_keys, log, model_revision, write_row

CLASSES = ["injection", "xss", "memory_corruption", "dos", "file_inclusion", "auth_access", "rce",
           "info_disclosure"]
L2I = {c: i for i, c in enumerate(CLASSES)}
SEEDS = [42, 43, 44]
STREAM = "task2"
TFS = [("SecBERT", "jackaduma/SecBERT"), ("SecureBERT", "ehsanaghaei/SecureBERT"),
       ("CySecBERT", "markusbayer/CySecBERT")]
GATED = [("DarkBERT", "s2w-ai/DarkBERT")]


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def set_seed(seed):
    import random
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def record(model, variant, seed, y, p, ids, t0, prov, **extra):
    from sklearn.metrics import accuracy_score, f1_score
    y, p = np.asarray(y), np.asarray(p)
    per = f1_score(y, p, labels=list(range(8)), average=None, zero_division=0)
    tag = f"task2__{model}__{variant}__seed{seed}".replace(" ", "_").replace("+", "plus")
    np.savez_compressed(PERQ / f"{tag}.npz", id=np.asarray(ids), y_true=y, y_pred=p)
    row = dict(result_type="classification", task="task2", model=model, condition="temporal_ood",
               variant=variant, seed=seed, n=len(y),
               macro_f1=float(f1_score(y, p, average="macro", zero_division=0)),
               weighted_f1=float(f1_score(y, p, average="weighted", zero_division=0)),
               accuracy=float(accuracy_score(y, p)),
               **{f"f1_{c}": float(per[i]) for i, c in enumerate(CLASSES)},
               seconds=round(time.time() - t0, 1), per_query_file=f"per_query/{tag}.npz", **extra, **prov)
    write_row(STREAM, **row)
    log(f"  {model:14s} {variant:18s} seed={seed} macroF1={row['macro_f1']:.4f} wF1={row['weighted_f1']:.4f} "
        f"acc={row['accuracy']:.4f} | info={per[L2I['info_disclosure']]:.2f} mem={per[L2I['memory_corruption']]:.2f} "
        f"auth={per[L2I['auth_access']]:.2f}")


def stage_bow(d, prov, done):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    ytr = [L2I[r["label"]] for r in d["train"]]; yte = [L2I[r["label"]] for r in d["test"]]
    for variant, cut in [("repo_trunc2000", 2000), ("full_text", None)]:
        f = (lambda t: t[:cut]) if cut else (lambda t: t)
        t0 = time.time()
        vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
        Xtr = vec.fit_transform([f(r["text"]) for r in d["train"]]); Xte = vec.transform([f(r["text"]) for r in d["test"]])
        log(f"TF-IDF {variant}: {Xtr.shape} ({time.time()-t0:.0f}s)")
        for name, mk in [("Decision Tree", lambda s: DecisionTreeClassifier(max_depth=30, class_weight="balanced", random_state=s)),
                         ("TF-IDF + LR", lambda s: LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=s)),
                         ("SVM", lambda s: LinearSVC(max_iter=2000, class_weight="balanced", random_state=s))]:
            for s in SEEDS:
                if (name, variant, s) in done:
                    continue
                t1 = time.time(); clf = mk(s); clf.fit(Xtr, ytr)
                extra = {}
                if hasattr(clf, "n_iter_"):
                    extra["n_iter"] = int(np.max(clf.n_iter_))
                record(name, variant, s, yte, clf.predict(Xte), [r["id"] for r in d["test"]], t1,
                       dict(prov, char_trunc=cut), **extra)


def stage_rnn(d, prov, done, dev, smoke=False, variants=None):
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score
    from sklearn.utils.class_weight import compute_class_weight

    counts = Counter()
    for r in d["train"]:
        counts.update(r["text"].lower().split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in counts.items():
        if c >= 3:
            vocab[tok] = len(vocab)
    log(f"RNN vocabulary (min freq 3): {len(vocab):,} entries (paper: ~19K)")
    write_row("diagnostics", task="task2", model="-", condition="rnn_vocab_size_minfreq3", value=len(vocab))
    ML = 300

    def tens(rows):
        X = np.zeros((len(rows), ML), np.int64); L = np.zeros(len(rows), np.int64)
        for i, r in enumerate(rows):
            ids = [vocab.get(t, 1) for t in r["text"].lower().split()[:ML]]
            X[i, :len(ids)] = ids; L[i] = max(len(ids), 1)
        return torch.from_numpy(X), torch.from_numpy(L), torch.tensor([L2I[r["label"]] for r in rows])

    (Xtr, Ltr, ytr), (Xva, Lva, yva), (Xte, Lte, yte) = map(tens, (d["train"], d["val"], d["test"]))
    write_row("diagnostics", task="task2", model="-", condition="share_truncated_at_300_tokens_test",
              value=float((np.array([len(r["text"].split()) for r in d["test"]]) > ML).mean()))
    cw = compute_class_weight("balanced", classes=np.arange(8), y=ytr.numpy())
    epochs = 1 if smoke else 8

    class Net(nn.Module):
        def __init__(self, cell, bidir, pool):
            super().__init__()
            self.cell, self.bidir, self.pool = cell, bidir, pool
            self.embed = nn.Embedding(len(vocab), 128, padding_idx=0)
            self.rnn = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell](128, 128, batch_first=True, bidirectional=bidir)
            self.fc = nn.Linear(128 * (2 if bidir else 1), 8)

        def forward(self, x, lens):
            e = self.embed(x)
            if self.pool == "packed_last_token" and not self.bidir:
                # Unidirectional: the output at the last real token is exactly the packed h_n
                # (right padding comes after it); verified on MPS to <6e-7, and much faster than packing.
                out, _ = self.rnn(e)
                return self.fc(out[torch.arange(len(lens), device=out.device), (lens - 1).to(out.device)])
            if self.pool == "packed_last_token":
                pk = nn.utils.rnn.pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
                _, h = self.rnn(pk)
            else:
                out, h = self.rnn(e)
                if self.pool == "mean_pool_july":
                    return self.fc(out.mean(dim=1))
            if self.cell == "lstm":
                h = h[0]
            last = torch.cat([h[-2], h[-1]], dim=1) if self.bidir else h[-1]
            return self.fc(last)

    # corrected variant first; July mean-pool is a reconciliation-only variant, seed 42 only
    plan = [("packed_last_token", SEEDS), ("repo_last_hidden", SEEDS), ("mean_pool_july", [42])]
    for variant, seeds in [v for v in plan if variants is None or v[0] in variants]:
        for name, cell, bidir in [("RNN", "rnn", False), ("GRU", "gru", False), ("LSTM", "lstm", False),
                                  ("BiLSTM", "lstm", True)]:
            for seed in seeds:
                if (name, variant, seed) in done:
                    continue
                t0 = time.time(); set_seed(seed)
                m = Net(cell, bidir, variant).to(dev)
                opt = torch.optim.Adam(m.parameters(), lr=1e-3)
                lossf = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=dev))
                g = torch.Generator().manual_seed(seed)

                def predict(X, L):
                    m.eval(); out = []
                    with torch.no_grad():
                        for i in range(0, len(X), 512):
                            out.append(m(X[i:i + 512].to(dev), L[i:i + 512]).argmax(1).cpu())
                    return torch.cat(out).numpy()

                best, best_state, best_ep, hist = -1.0, None, 0, []
                for ep in range(1, epochs + 1):
                    m.train(); perm = torch.randperm(len(Xtr), generator=g); tot = 0.0
                    for i in range(0, len(Xtr), 128):
                        ix = perm[i:i + 128]
                        opt.zero_grad()
                        loss = lossf(m(Xtr[ix].to(dev), Ltr[ix]), ytr[ix].to(dev))
                        loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                        tot += loss.item()
                    vf = f1_score(yva.numpy(), predict(Xva, Lva), average="macro", zero_division=0)
                    hist.append(round(float(vf), 4))
                    if vf > best:
                        best, best_ep = vf, ep
                        best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                m.load_state_dict(best_state)
                record(name, variant, seed, yte.numpy(), predict(Xte, Lte), [r["id"] for r in d["test"]], t0,
                       dict(prov, epochs=epochs, max_len=ML, vocab_size=len(vocab)),
                       best_val_macro_f1=float(best), best_epoch=best_ep, val_history=hist)
                del m, opt; gc.collect()
                if dev == "mps":
                    torch.mps.empty_cache()


def stage_tf(d, prov, done, dev, full=False, july=False, smoke=False, models=None, seeds=None):
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    for name, path in GATED:
        if (name, "availability", 0) not in done:
            rev = model_revision(path)
            write_row(STREAM, result_type="classification", task="task2", model=name, condition="temporal_ood",
                      variant="availability", seed=0, note=f"not run: gated repo, licence not accepted ({rev})")
            log(f"{name}: recorded as unavailable ({rev})")

    variant = "full_64413" if full else ("july_protocol" if july else "subsample15000")
    run_seeds = [s for s in ([42] if july else SEEDS) if seeds is None or s in seeds]
    epochs = 1 if smoke else 3
    for name, path in [t for t in TFS if models is None or t[0] in models]:
        for seed in run_seeds:
            if (name, variant, seed) in done and (name, variant + "_final_epoch", seed) in done:
                continue
            t0 = time.time(); set_seed(seed)
            train = d["train"]
            sub = None if full else (200 if smoke else 15000)
            if sub and len(train) > sub:
                idx = np.random.default_rng(seed).choice(len(train), size=sub, replace=False)
                train = [train[i] for i in idx]
            cut = (lambda t: t) if july else (lambda t: t[:1000])
            tok = AutoTokenizer.from_pretrained(path)
            m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=8).to(dev)
            rev = model_revision(path)

            def enc(rows):
                e = tok([cut(r["text"]) for r in rows], truncation=True, max_length=256, padding="max_length",
                        return_tensors="pt")
                return e["input_ids"], e["attention_mask"], torch.tensor([L2I[r["label"]] for r in rows])

            Itr, Atr, ytr = enc(train); Iva, Ava, yva = enc(d["val"]); Ite, Ate, yte = enc(d["test"])
            opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=0.01)
            steps = math.ceil(len(Itr) / 16) * epochs
            sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
            lossf = None
            if july:
                w = compute_class_weight("balanced", classes=np.arange(8), y=np.array([L2I[r["label"]] for r in d["train"]]))
                lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
            g = torch.Generator().manual_seed(seed)

            def predict(I, A):
                m.eval(); out = []
                with torch.no_grad():
                    for i in range(0, len(I), 64):
                        out.append(m(input_ids=I[i:i + 64].to(dev), attention_mask=A[i:i + 64].to(dev)).logits.argmax(1).cpu())
                return torch.cat(out).numpy()

            best, best_ep, best_pred, hist = -1.0, 0, None, []
            for ep in range(1, epochs + 1):
                m.train(); perm = torch.randperm(len(Itr), generator=g); te = time.time()
                for i in range(0, len(Itr), 16):
                    ix = perm[i:i + 16]
                    out = m(input_ids=Itr[ix].to(dev), attention_mask=Atr[ix].to(dev),
                            labels=None if july else ytr[ix].to(dev))
                    loss = lossf(out.logits, ytr[ix].to(dev)) if july else out.loss
                    loss.backward()
                    if july:
                        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step(); sch.step(); opt.zero_grad()
                vf = f1_score(yva.numpy(), predict(Iva, Ava), average="macro", zero_division=0)
                hist.append(round(float(vf), 4))
                log(f"  {name} {variant} seed={seed} epoch {ep}: val macro-F1 {vf:.4f} ({time.time()-te:.0f}s)")
                pt = predict(Ite, Ate)  # stored; selection below uses validation only
                if vf > best:
                    best, best_ep, best_pred = vf, ep, pt
            p_ = dict(prov, hf_model=path, hf_revision=rev, train_rows=len(train), epochs=epochs, max_len=256,
                      char_trunc=None if july else 1000)
            ids = [r["id"] for r in d["test"]]
            record(name, variant, seed, yte.numpy(), best_pred, ids, t0, p_, best_val_macro_f1=float(best),
                   best_epoch=best_ep, val_history=hist, selection="best epoch by validation macro-F1")
            record(name, variant + "_final_epoch", seed, yte.numpy(), pt, ids, t0, p_, val_history=hist,
                   selection="final epoch (repo behaviour, no selection)")
            del m, opt; gc.collect()
            if dev == "mps":
                torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["bow", "rnn", "tf", "tf_full", "tf_july"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rnn-variants", default="", help="comma list to restrict the rnn stage")
    ap.add_argument("--models", default="", help="comma list to restrict the tf stages, e.g. SecBERT")
    ap.add_argument("--seeds", default="", help="comma list to restrict the tf stages, e.g. 42")
    ap.add_argument("--out", default="", help="write raw/ and per_query/ here instead (one folder per cluster job)")
    args = ap.parse_args()
    global PERQ
    import common
    import torch
    from pathlib import Path
    out = Path(args.out) if args.out else (ROOT / "results" / "smoke" if args.smoke else None)
    if out:
        common.RAW = out / "raw"; common.RAW.mkdir(parents=True, exist_ok=True)
        PERQ = out / "per_query"; PERQ.mkdir(parents=True, exist_ok=True)
    unknown = set(args.models.split(",")) - {n for n, _ in TFS} - {""}
    if unknown:
        ap.error(f"unknown --models {sorted(unknown)}")
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    d = {s: load(DATA / "task2_exploit_type" / f"{s}.jsonl") for s in ("train", "val", "test")}
    if args.smoke:
        d = {"train": d["train"][::40], "val": d["val"][::10], "test": d["test"][::5]}
    h = data_hashes()
    prov = dict(device=dev, torch=torch.__version__,
                **({"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda} if dev == "cuda" else {}),
                **{f"data_sha256_task2_{s}": h[f"task2_exploit_type/{s}.jsonl"][:16] for s in ("train", "val", "test")})
    done = done_keys(STREAM, ("model", "variant", "seed"))
    log(f"stage {args.stage}: train {len(d['train']):,} val {len(d['val']):,} test {len(d['test']):,} on {dev}")
    if args.stage == "bow":
        stage_bow(d, prov, done)
    elif args.stage == "rnn":
        stage_rnn(d, prov, done, dev, smoke=args.smoke,
                  variants=set(args.rnn_variants.split(",")) if args.rnn_variants else None)
    else:
        stage_tf(d, prov, done, dev, full=args.stage == "tf_full", july=args.stage == "tf_july", smoke=args.smoke,
                 models=set(args.models.split(",")) if args.models else None,
                 seeds={int(s) for s in args.seeds.split(",")} if args.seeds else None)
    log(f"stage {args.stage} complete")


if __name__ == "__main__":
    main()
