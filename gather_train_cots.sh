#!/bin/bash
# =============================================================================
# Configuration for gathering training CoTs with log probs and self-certainty
# =============================================================================
# Usage:
#   bash gather_train_cots.sh
#
# Activate your Python environment before running.
# Required packages: vllm, torch, transformers, aiohttp, httpx, openai, tqdm
# =============================================================================

set -euo pipefail

export HF_HOME="${HF_HOME:-.}"

# =============================================================================
# Configuration - UPDATE THESE BEFORE RUNNING!
# =============================================================================

# Model configuration
export MODEL_NAME="Qwen/Qwen3-8B"        # Model name for output directory
export VLLM_MODEL="Qwen/Qwen3-8B"        # vLLM model to serve
export VLLM_PORT=8000
export VLLM_TP_SIZE=1                    # Tensor parallel size (use 1 for single GPU)
export VLLM_MAX_MODEL_LEN=32768          # Maximum sequence length
export VLLM_GPU_UTIL=0.90                # GPU memory utilization
export VLLM_DTYPE=auto
export VLLM_DOWNLOAD_DIR=$HF_HOME
export REASONING_PARSER=qwen3            # Use 'qwen3' for Qwen models with <think> tokens
export VLLM_SERVED_NAME=vllm

# Inference parameters
export TEMPERATURE=0.6
export TOP_P=0.95
export TOP_K=20                          # Also used for self-certainty computation
export MIN_P=0.0
export PRESENCE_PENALTY=1.5
export MAX_TOKENS=24576                  # Maximum tokens to generate per response
export ENABLE_THINKING=true              # Set to 'false' to disable thinking mode
export BATCH_SIZE=16                     # Number of problems to process concurrently
export SAMPLES_PER_PROBLEM=3             # Number of successful solutions per problem
export MAX_FAILURES=5                    # Max failed attempts before stopping for a problem
export MAX_RETRIES=3                     # Max retries for failed requests
export SERVER_TIMEOUT=600                # Max time to wait for server (seconds)
export LIMIT=""                          # Leave empty to process all, or set a number for testing

# Working directory
export WORK_DIR="${WORK_DIR:-$(pwd)}"

# Input file
export INPUT_FILE="$WORK_DIR/data/inputs/dataset.jsonl"

# Output will be saved to: $WORK_DIR/data/outputs/${MODEL_NAME}/dataset_cots.jsonl

# =============================================================================
# Run the CoT generation script
# =============================================================================

cd "$WORK_DIR"

echo ""
echo "========================================"
echo "Starting CoT generation job..."
echo "Start time: $(date)"
echo "========================================"

bash run_gather_cots.sh

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "========================================"
