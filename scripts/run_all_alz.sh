#!/bin/bash
# ============================================================
# run_all_alz.sh — submit all 15 Alzheimer experiments sequentially
#
# Uses Slurm dependency chaining so each job starts only after
# the previous one finishes. Safe to leave running overnight.
#
# Usage:
#   ./scripts/run_all_alz.sh
# ============================================================

set -e

CONFIGS=(
    # PointNet2D
    configs/alz_pointnet2d_partial.json

    # PointNet++
    configs/alz_pointnetpp_partial.json

    # ResNet Fusion
    configs/alz_resnet_fusion_partial.json


    # ConvNeXt Base Fusion
    configs/alz_convnext_base_fusion_partial.json
)

echo "========================================"
echo "Submitting ${#CONFIGS[@]} Alzheimer experiments"
echo "========================================"
echo ""

PREV_JOB=""

for CONFIG in "${CONFIGS[@]}"; do
    if [ ! -f "$CONFIG" ]; then
        echo "WARNING: Config not found, skipping: $CONFIG"
        continue
    fi

    CKPT_NAME=$(python3 -c "import json; c=json.load(open('$CONFIG')); print(c.get('checkpoint_name', 'experiment'))")
    JOB_NAME=$(basename "$CKPT_NAME")
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="logs/${CKPT_NAME}_${TIMESTAMP}"
    mkdir -p "$LOG_DIR"
    cp "$CONFIG" "$LOG_DIR/config_used.json"

    if [ -z "$PREV_JOB" ]; then
        # First job — no dependency
        RESULT=$(sbatch \
            --job-name="$JOB_NAME" \
            -o "$LOG_DIR/slurm.out" \
            --export=CONFIG="$CONFIG" \
            scripts/run_worker.sh)
    else
        # Chain after previous job
        RESULT=$(sbatch \
            --dependency=afterany:"$PREV_JOB" \
            --job-name="$JOB_NAME" \
            -o "$LOG_DIR/slurm.out" \
            --export=CONFIG="$CONFIG" \
            scripts/run_worker.sh)
    fi

    PREV_JOB=$(echo "$RESULT" | awk '{print $NF}')
    echo "[$PREV_JOB] $JOB_NAME  →  $LOG_DIR/"
done

echo ""
echo "========================================"
echo "All ${#CONFIGS[@]} jobs submitted."
echo "Monitor: squeue -u \$USER"
echo "========================================"
