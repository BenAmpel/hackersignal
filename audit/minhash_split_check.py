"""Train/test (and train/val) near-duplicate check on the released benchmark_v2 splits, as App. B describes it:
MinHash, 128 permutations, LSH at Jaccard >= 0.8, over the repo's character 3-shingles (etg.data.preprocessing._shingles).
Every LSH candidate is confirmed with exact Jaccard on the shingle sets.
Usage: python minhash_split_check.py <benchmark_v2 dir> <out.json>"""
import json, re, sys, time
from pathlib import Path
from datasketch import MinHash, MinHashLSH

TH, PERM = 0.8, 128
root, out = Path(sys.argv[1]), Path(sys.argv[2])


def shingles(text, k=3):
    s = re.sub(r"\s+", " ", text.lower())
    return {s[i:i + k].encode("utf-8") for i in range(len(s) - k + 1)}


def sig(sh):
    m = MinHash(num_perm=PERM, seed=1)
    m.update_batch(list(sh))
    return m


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


res = {"method": "MinHash (datasketch), 128 permutations, LSH threshold 0.8, char 3-shingles; candidates confirmed by exact Jaccard >= 0.8"}
for task in ("task1_cve_linkage", "task2_exploit_type", "task3_temporal_generalization"):
    t0 = time.time()
    train = load(root / task / "train.jsonl")
    lsh = MinHashLSH(threshold=TH, num_perm=PERM)
    tsh = {}
    for i, r in enumerate(train):
        sh = shingles(r["text"])
        if not sh:
            continue
        tsh[i] = sh
        lsh.insert(str(i), sig(sh))
    task_res = {"train_rows": len(train)}
    for split in ("val", "test"):
        rows = load(root / task / f"{split}.jsonl")
        hits, cand, examples = 0, 0, []
        for r in rows:
            sh = shingles(r["text"])
            if not sh:
                continue
            best = 0.0
            for c in lsh.query(sig(sh)):
                cand += 1
                o = tsh[int(c)]
                j = len(sh & o) / len(sh | o)
                best = max(best, j)
            if best >= TH:
                hits += 1
                if len(examples) < 5:
                    examples.append({"id": r.get("id"), "jaccard": round(best, 3)})
        task_res[split] = {"rows": len(rows), "lsh_candidates": cand, "rows_with_train_match_ge_0.8": hits, "examples": examples}
    task_res["seconds"] = round(time.time() - t0)
    res[task] = task_res
    print(task, json.dumps(task_res), flush=True)
json.dump(res, open(out, "w"), indent=2)
