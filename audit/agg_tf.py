"""Mean ± SD (ddof=1, as audit/finalize.py) of Task 2 transformer rows for one variant.

usage: python3 agg_tf.py <task2.jsonl> <variant>   e.g. subsample15000 or full_64413 (best epoch by validation)
"""
import json
import statistics
import sys
from collections import defaultdict

path, variant = sys.argv[1], sys.argv[2]
KEYS = ["macro_f1", "f1_info_disclosure", "f1_memory_corruption", "f1_auth_access"]
rows = defaultdict(dict)
for line in open(path):
    r = json.loads(line)
    if r.get("variant") == variant and r.get("model") in ("SecBERT", "SecureBERT", "CySecBERT"):
        rows[r["model"]][r["seed"]] = r  # last row per (model, seed) wins
for m in ("SecBERT", "SecureBERT", "CySecBERT"):
    seeds = sorted(rows[m])
    out = [f"{m:11s} seeds={seeds}"]
    for k in KEYS:
        v = [rows[m][s][k] for s in seeds]
        sd = statistics.stdev(v) if len(v) > 1 else float("nan")
        out.append(f"{k.replace('f1_', '')}={statistics.mean(v):.3f}±{sd:.3f}" if v else f"{k}=n/a")
    out.append("best_epochs=" + ",".join(str(rows[m][s].get("best_epoch")) for s in seeds))
    out.append("device=" + ",".join(sorted({str(rows[m][s].get('gpu') or rows[m][s].get('device')) for s in seeds})))
    print("  ".join(out))
