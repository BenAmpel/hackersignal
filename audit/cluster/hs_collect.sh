#!/bin/bash
# Collect array 4777484: per-task state, error lines, then the transformer rows between markers (DarkBERT availability rows dropped).
set -u
BASE="/data/temp-scratch/${USER:?}/hackersignal"
date -u
sacct -j 4777484 -X -o JobID%16,State,Elapsed,ExitCode,NodeList 2>/dev/null
grep -l -E "Traceback|Error" "$BASE"/logs/hs-tf-full_4777484_*.log 2>/dev/null && echo "^^ logs with errors"
echo "ROWS_BEGIN"
for d in "$BASE"/results_cluster/*_seed*; do grep -h -v '"variant": "availability"' "$d/raw/task2.jsonl"; done
echo "ROWS_END"
echo COLLECT_DONE
