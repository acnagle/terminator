#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# CoT Generation Runner with vLLM
# =============================================================================
# This script:
# 1. Starts a local vLLM server
# 2. Waits for it to be ready (handled by Python script)
# 3. Runs gather_cots.py to generate CoTs with log probs and self-certainty
# 4. Cleans up vLLM server on exit
# =============================================================================

# Required environment variables
: ${MODEL_NAME:?"ERROR: MODEL_NAME must be set (e.g., 'Qwen/Qwen3-8B')"}
: ${VLLM_MODEL:?"ERROR: VLLM_MODEL must be set (e.g., 'Qwen/Qwen3-8B')"}

# Optional environment variables with defaults
: ${INPUT_FILE:=./data/inputs/dataset.jsonl}
: ${WORK_DIR:=$(pwd)}
: ${VLLM_PORT:=8000}
: ${VLLM_HOST:=0.0.0.0}
: ${VLLM_TP_SIZE:=1}
: ${VLLM_GPU_UTIL:=0.90}
: ${VLLM_DTYPE:=auto}
: ${VLLM_MAX_MODEL_LEN:=32768}
: ${VLLM_DOWNLOAD_DIR:=${HF_HOME:-$SCRATCH/hf_cache}}
: ${REASONING_PARSER:=qwen3}
: ${VLLM_SERVED_NAME:=vllm}

# Inference parameters
: ${TEMPERATURE:=0.6}
: ${TOP_P:=0.95}
: ${TOP_K:=20}
: ${MIN_P:=0.0}
: ${PRESENCE_PENALTY:=1.5}
: ${MAX_TOKENS:=24576}
: ${ENABLE_THINKING:=true}
: ${BATCH_SIZE:=16}
: ${SAMPLES_PER_PROBLEM:=3}
: ${AIME_2025_OVERRIDE:=""}
: ${MAX_FAILURES:=5}
: ${MAX_RETRIES:=3}
: ${SERVER_TIMEOUT:=600}
: ${REQUEST_TIMEOUT:=600}
: ${LIMIT:=""}

echo "========================================"
echo "CoT Generation with vLLM"
echo "========================================"
echo "Model name: $MODEL_NAME"
echo "Input file: $INPUT_FILE"
echo "Output file: ./data/outputs/${MODEL_NAME}/dataset_cots.jsonl"
echo "vLLM model: $VLLM_MODEL"
echo "vLLM server: http://${VLLM_HOST}:${VLLM_PORT}"
echo "Date: $(date)"
echo "========================================"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

# =============================================================================
# STEP 1: Kill any existing vLLM processes
# =============================================================================

echo "Step 1: Cleaning up old vLLM processes..."
pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
sleep 2
echo "Cleanup complete"
echo "========================================"

# =============================================================================
# STEP 2: Start vLLM server
# =============================================================================

echo "Step 2: Starting vLLM server on port $VLLM_PORT..."

# Find host.py - try both current directory and vllm subdirectory
if [ -f "${WORK_DIR}/host.py" ]; then
    HOST_SCRIPT="${WORK_DIR}/host.py"
elif [ -f "${WORK_DIR}/vllm/host.py" ]; then
    HOST_SCRIPT="${WORK_DIR}/vllm/host.py"
else
    echo "ERROR: host.py not found in $WORK_DIR or $WORK_DIR/vllm"
    exit 1
fi

echo "Using host script: $HOST_SCRIPT"

VLLM_HOST=$VLLM_HOST \
VLLM_PORT=$VLLM_PORT \
VLLM_TP_SIZE=$VLLM_TP_SIZE \
VLLM_MODEL=$VLLM_MODEL \
VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN \
VLLM_GPU_UTIL=$VLLM_GPU_UTIL \
VLLM_DTYPE=$VLLM_DTYPE \
VLLM_DOWNLOAD_DIR=$VLLM_DOWNLOAD_DIR \
REASONING_PARSER=$REASONING_PARSER \
VLLM_SERVED_NAME=$VLLM_SERVED_NAME \
python "$HOST_SCRIPT" > vllm_server.log 2>&1 &

VLLM_PID=$!
echo "vLLM server started (PID: $VLLM_PID)"
echo "Server logs: vllm_server.log"
echo "========================================"

# =============================================================================
# Cleanup function
# =============================================================================

cleanup() {
    echo ""
    echo "Shutting down vLLM server..."
    kill $VLLM_PID 2>/dev/null || true
    pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
    echo "Cleanup complete"
}

trap cleanup EXIT INT TERM

# =============================================================================
# STEP 3: Run inference
# =============================================================================

echo "Step 3: Running CoT generation..."
echo "The Python script will wait for the server to be ready (timeout: ${SERVER_TIMEOUT}s)"
echo ""

# Build the inference command
INFERENCE_CMD="python ${WORK_DIR}/gather_cots.py \
    --model-name $MODEL_NAME \
    --input-file $INPUT_FILE \
    --api-base http://localhost:${VLLM_PORT}/v1 \
    --served-name $VLLM_SERVED_NAME \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --min-p $MIN_P \
    --presence-penalty $PRESENCE_PENALTY \
    --max-tokens $MAX_TOKENS \
    --batch-size $BATCH_SIZE \
    --samples-per-problem $SAMPLES_PER_PROBLEM \
    --max-failures $MAX_FAILURES \
    --max-retries $MAX_RETRIES \
    --server-timeout $SERVER_TIMEOUT \
    --request-timeout $REQUEST_TIMEOUT"

# Add thinking mode flag
if [ "$ENABLE_THINKING" = "true" ]; then
    INFERENCE_CMD="$INFERENCE_CMD --enable-thinking"
else
    INFERENCE_CMD="$INFERENCE_CMD --disable-thinking"
fi

# Add AIME 2025 override if specified
if [ -n "$AIME_2025_OVERRIDE" ]; then
    INFERENCE_CMD="$INFERENCE_CMD --aime-2025-samples-override $AIME_2025_OVERRIDE"
fi

# Add limit if specified
if [ -n "$LIMIT" ]; then
    INFERENCE_CMD="$INFERENCE_CMD --limit $LIMIT"
fi

# Run inference
eval $INFERENCE_CMD

INFERENCE_EXIT_CODE=$?

echo ""
echo "========================================"
echo "CoT generation complete!"
echo "========================================"

if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
    echo "✓ CoT generation completed successfully"
    echo "Results saved to: ./data/outputs/${MODEL_NAME}/dataset_cots.jsonl"
else
    echo "✗ CoT generation failed with exit code: $INFERENCE_EXIT_CODE"
    echo "Check vllm_server.log for server logs"
    exit $INFERENCE_EXIT_CODE
fi

echo "========================================"
