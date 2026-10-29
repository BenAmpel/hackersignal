#!/bin/bash
set -u
W="/Volumes/Extreme SSD/_worktrees/exploit-text-graph/camera-ready-audit"
cd "$W/audit"
export HF_HOME="$W/.cache/hf" TOKENIZERS_PARALLELISM=false
for st in bow rnn tf tf_july tf_full; do
  echo "=== stage $st start $(date)"
  caffeinate -i "$W/.venv/bin/python" run_task2.py --stage "$st"
  echo "=== stage $st exit=$? $(date)"
done
echo "TASK2_ALL_DONE $(date)"
