#!/bin/bash
# =============================================================================
# Train Terminator with BCE loss
# =============================================================================
# Usage:
#   bash train_terminator_qwen3-14b.sh
#
# Activate your Python environment before running.
# Required packages: torch, transformers, wandb (optional), scikit-learn
# =============================================================================

set -euo pipefail

export HF_HOME="${HF_HOME:-.}"

python train_terminator.py \
    --train-path ./data/outputs/Qwen/Qwen3-14B/train.jsonl \
    --val-path ./data/outputs/Qwen/Qwen3-14B/val.jsonl \
    --model-name "Qwen/Qwen3-14B" \
    --layer-idx -1 \
    --num-extra-layers 1 \
    --extra-layers-init "copy_last" \
    --ffn-layers 1 \
    --lag-size 0 \
    --use-class-weights \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --epochs 2 \
    --lr 2e-4 \
    --weight-decay 1e-2 \
    --dropout 0. \
    --use-wandb \
    --wandb-project bce_qwen3-14b \
    --wandb-run-name "terminator" \
    --log-freq 8 \
    --val-log-freq 64
