#!/bin/bash
set -u
W="/Volumes/Extreme SSD/_worktrees/exploit-text-graph/camera-ready-audit"
cd "$W/audit"
export HF_HOME="$W/.cache/hf" TOKENIZERS_PARALLELISM=false
run() { echo "=== $* start $(date)"; caffeinate -i "$W/.venv/bin/python" run_task2.py "$@"; echo "=== $* exit=$? $(date)"; }
run --stage rnn --rnn-variants packed_last_token
run --stage tf
run --stage rnn --rnn-variants repo_last_hidden,mean_pool_july
run --stage tf_july
run --stage tf_full
echo "TASK2_ALL_DONE $(date)"
