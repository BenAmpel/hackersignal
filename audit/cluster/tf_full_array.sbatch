#!/bin/bash
#SBATCH --job-name=hs-tf-full
#SBATCH --account=cis1671s260
#SBATCH --partition=qGPU48,qGPU24
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/%x_%A_%a.log
# Task 2 tf_full on the GSU ACIDS cluster: 3 models x 3 seeds, one per array task, fp32 (no AMP) to match the 15K runs.
# Submit from /data/temp-scratch/$USER/hackersignal:  sbatch camera-ready-audit/audit/cluster/tf_full_array.sbatch
# Each task writes results_cluster/<model>_seed<seed>/{raw,per_query}; merge them afterwards.
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
MODELS=(SecBERT SecureBERT CySecBERT)
SEEDS=(42 43 44)
M=${MODELS[$((SLURM_ARRAY_TASK_ID / 3))]}
S=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "task $SLURM_ARRAY_TASK_ID: $M seed $S on $(hostname), code $AUDIT_GIT_SHA"
cd "$BASE/camera-ready-audit/audit"
"$VENV/bin/python" run_task2.py --stage tf_full --models "$M" --seeds "$S" --out "$BASE/results_cluster/${M}_seed${S}"
