#!/bin/bash
set -u
W="/Volumes/Extreme SSD/_worktrees/exploit-text-graph/camera-ready-audit"
while pgrep -f "launch_retrieval.sh" > /dev/null; do sleep 60; done
cd "$W/audit"
export HF_HOME="$W/.cache/hf" TOKENIZERS_PARALLELISM=false
caffeinate -i "$W/.venv/bin/python" run_retrieval.py --only sens
echo "BGE_INSTRUCTION_DONE $(date)"
