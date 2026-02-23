#!/bin/bash
# ============================================================
# submit.sh — submit a training experiment to Slurm
#
# Usage:
#   ./scripts/submit.sh configs/alz_pointnetpp.json
#   ./scripts/submit.sh configs/alz_convnext_base.json
#
# Each run gets its own log directory:
#   logs/<checkpoint_name>_<YYYYMMDD_HHMMSS>/
#     slurm.out       — full stdout/stderr from the job
#     config_used.json — exact config that was run
# ============================================================

set -e

CONFIG=${1:?"Usage: ./scripts/submit.sh <config.json>"}

if [ ! -f "$CONFIG" ]; then
    echo "Error: config file not found: $CONFIG"
    exit 1
fi

# Extract job name from checkpoint_name in config (use basename to strip directory prefix)
CKPT_NAME=$(python3 -c "import json; c=json.load(open('$CONFIG')); print(c.get('checkpoint_name', 'experiment'))")
JOB_NAME=$(basename "$CKPT_NAME")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/${CKPT_NAME}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Save exact config used for this run
cp "$CONFIG" "$LOG_DIR/config_used.json"

echo "Submitting: $JOB_NAME"
echo "Config    : $CONFIG"
echo "Log dir   : $LOG_DIR/"

RESULT=$(sbatch \
    --job-name="$JOB_NAME" \
    -o "$LOG_DIR/slurm.out" \
    --export=CONFIG="$CONFIG" \
    scripts/run_worker.sh)

JOB_ID=$(echo "$RESULT" | awk '{print $NF}')
echo "Job ID    : $JOB_ID"
echo ""
echo "Monitor:  tail -f $LOG_DIR/slurm.out"
echo "Status:   squeue -j $JOB_ID"
