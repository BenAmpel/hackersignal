#!/bin/bash
#SBATCH --job-name=etg_walk_forward
#SBATCH --output=logs/wf_%j.out
#SBATCH --error=logs/wf_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ── environment ────────────────────────────────────────────────────────────
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "====== Walk-Forward Validation ======"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Start:    $(date)"
echo ""

mkdir -p logs ETG_MISQ/output/walk_forward

# Activate conda/venv — adjust to your cluster environment
# conda activate etg
# source venv/bin/activate

python ETG_MISQ/run_walk_forward_validation.py \
    --snapshots  ETG_MISQ/output/cache/rt11_etg_snapshots.pkl \
    --output     ETG_MISQ/output/walk_forward \
    --dgt-epochs 100 \
    --max-nodes  6000 \
    --hidden-dim 128 \
    --heads      4 \
    --layers     2 \
    --lap-pe-k   16 \
    --dim        64 \
    --folds      7 8 9 10 11 \
    --device     cuda

echo ""
echo "End: $(date)"
echo "====== Done ======"
