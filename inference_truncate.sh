#!/bin/bash
# =============================================================================
# Configuration for truncation inference
# =============================================================================
# Usage:
#   bash inference_truncate.sh
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
export MODEL_NAME="Qwen/Qwen3-8B"        # Model name for input/output directory paths
export TOKENIZER="Qwen/Qwen3-8B"         # Tokenizer for token counting
export VLLM_MODEL="Qwen/Qwen3-8B"        # vLLM model to serve
export VLLM_PORT=8000
export VLLM_TP_SIZE=1                    # Tensor parallel size (use 1 for single GPU)
export VLLM_MAX_MODEL_LEN=32768          # Maximum sequence length
export VLLM_GPU_UTIL=0.90                # GPU memory utilization
export VLLM_DTYPE=auto
export VLLM_DOWNLOAD_DIR=$HF_HOME
export REASONING_PARSER=qwen3            # Use 'qwen3' for Qwen models with <think> tokens
export VLLM_SERVED_NAME=vllm

# Truncation parameters
export CUTOFFS_BEFORE="0.1 0.2 0.4 0.6 0.8 0.9 0.95 0.97 0.99"  # Space-separated cutoff ratios BEFORE span_final_end_token
export CUTOFFS_AFTER="0.0 0.2 0.4 0.6 0.8"              # Space-separated cutoff ratios AFTER span_final_end_token
export DATA_SOURCES="math-500 aime_2025 gpqa humaneval" # Space-separated data sources to process

# Inference parameters
export TEMPERATURE=0.6
export TOP_P=0.95
export TOP_K=20
export MIN_P=0.0
export PRESENCE_PENALTY=1.5
export MAX_TOKENS=4096                  # Maximum tokens to generate per response
export BATCH_SIZE=16                     # Number of samples to process concurrently
export MAX_RETRIES=3                     # Max retries for failed requests
export SERVER_TIMEOUT=600                # Max time to wait for server (seconds)
export LIMIT=""                          # Leave empty to process all, or set a number for testing

# Working directory
export WORK_DIR="${WORK_DIR:-$(pwd)}"

# Input file
export INPUT_FILE="$WORK_DIR/data/outputs/${MODEL_NAME}/test_cots_extracted.jsonl"

# Output will be saved to: $WORK_DIR/data/outputs/${MODEL_NAME}/test_cots_extracted_truncate_{before|after}_{ratio}.jsonl

# =============================================================================
# Run the truncation inference script
# =============================================================================

cd "$WORK_DIR"

echo ""
echo "========================================"
echo "Starting truncation inference job..."
echo "Start time: $(date)"
echo "========================================"

bash run_inference_truncate.sh

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "========================================"
