#!/bin/bash
# ============================================================
# run_worker.sh — sbatch worker script
# Do NOT submit this directly. Use submit.sh instead.
# ============================================================
#SBATCH --partition=all
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

cd ~/eyal

echo "=========================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Job Name : $SLURM_JOB_NAME"
echo "Config   : $CONFIG"
echo "Node     : $SLURM_NODELIST"
echo "Started  : $(date)"
echo "=========================================="

# Activate conda environment
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate conda-env

python src/main.py --config "$CONFIG"
EXIT_CODE=$?

echo "=========================================="
echo "Finished : $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
