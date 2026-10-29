#!/bin/bash
set -u
W="/Volumes/Extreme SSD/_worktrees/exploit-text-graph/camera-ready-audit"
cd "$W/audit"
export HF_HOME="$W/.cache/hf" TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=0
# validate + BM25 ladder + dense ladder + hybrid ladder + pilot, then sensitivity variants
caffeinate -i "$W/.venv/bin/python" run_retrieval.py --only validate,bm25,dense,hybrid,pilot
caffeinate -i "$W/.venv/bin/python" run_retrieval.py --only sens
echo "RETRIEVAL_ALL_DONE $(date)"
