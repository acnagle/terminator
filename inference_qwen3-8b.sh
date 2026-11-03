#!/bin/bash
# =============================================================================
# Run Terminator inference (Qwen3-8B)
# =============================================================================
# Usage:
#   bash inference_qwen3-8b.sh
#
# Activate your Python environment before running.
# Required packages: torch, transformers
# =============================================================================

set -euo pipefail

export HF_HOME="${HF_HOME:-.}"

python inference_terminator.py \
    --checkpoint-dir ./path/to/weights/dir \
    --input-file ./data/outputs/Qwen/Qwen3-8B/test_cots.jsonl \
    --data-sources math-500 aime_2025 humaneval gpqa \
    --loss-type "bce" \
    --mode 2 \
    --decoding-strategy "sliding_window" \
    --window-size 10 \
    --threshold 0.7
