#!/bin/bash
#SBATCH --job-name=hs-smoke
#SBATCH --account=cis1671s260
#SBATCH --partition=qGPU48,qGPU24
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
# One-model smoke run before the tf_full array: transformers imports, CUDA is used, a row is written.
# Submit from /data/temp-scratch/$USER/hackersignal:  sbatch camera-ready-audit/audit/cluster/smoke_gpu.sbatch
set -eu
BASE="/data/temp-scratch/${USER:?}/hackersignal"
VENV="/data/temp-scratch/${USER:?}/phishgpt/venv-py312-cu126"
export HF_HOME="/data/temp-scratch/${USER:?}/phishgpt/hf_cache"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export AUDIT_GIT_SHA="$(cat "$BASE/camera-ready-audit/GIT_SHA")"
module purge >/dev/null 2>&1 || true
module load python/3.12.7
module load cuda/12.1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$VENV/bin/python" -c "import torch, transformers, sklearn; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0), '| transformers', transformers.__version__, '| sklearn', sklearn.__version__)"
cd "$BASE/camera-ready-audit/audit"
"$VENV/bin/python" run_task2.py --stage tf_full --smoke --models SecBERT --seeds 42 --out "$BASE/results_cluster/smoke_${SLURM_JOB_ID}"
echo "SMOKE DONE"
