#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# Multi-Node Span Extraction Runner
# =============================================================================
# This script:
# 1. Splits input JSONL into N partitions (filtering completed samples)
# 2. Starts vLLM on each node (all using port 8000 on localhost)
# 3. Runs extract_and_verify_spans.py on each node with its partition
# 4. No load balancer needed - each node connects to its own local vLLM
# =============================================================================

# Required environment variables (set in SLURM script or manually)
: ${INPUT_JSONL:?"ERROR: INPUT_JSONL must be set (path to input JSONL file)"}
: ${VLLM_MODEL:?"ERROR: VLLM_MODEL must be set"}

# Optional environment variables with defaults
: ${WORK_DIR:=$(pwd)}
: ${PARTITION_DIR:=${WORK_DIR}/partitions}
: ${VLLM_PORT:=8000}  # All nodes use the same port on localhost
: ${VLLM_TP_SIZE:=1}
: ${VLLM_GPU_UTIL:=0.90}
: ${VLLM_DTYPE:=auto}
: ${VLLM_MAX_MODEL_LEN:=131072}
: ${VLLM_DOWNLOAD_DIR:=$SCRATCH/hf_cache}
: ${REASONING_PARSER:=qwen3}
: ${VLLM_SERVED_NAME:=vllm}
: ${VLLM_HOST:=0.0.0.0}
: ${TOKENIZER:=Qwen/Qwen3-8B}
: ${NUM_SOLUTIONS:=1}
: ${BATCH_SIZE:=8}
: ${TEMPERATURE:=0.6}
: ${TOP_P:=0.95}
: ${MIN_P:=0.0}
: ${TOP_K:=20}
: ${MAX_TOKENS:=8192}
: ${MAX_SPAN_RETRIES:=3}
: ${TIMEOUT:=1200}

echo "========================================"
echo "Multi-Node Span Extraction Setup"
echo "========================================"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "SLURM_NODELIST: ${SLURM_NODELIST:-N/A}"
echo "SLURM_NNODES: ${SLURM_NNODES:-N/A}"
echo "INPUT_JSONL: $INPUT_JSONL"
echo "VLLM_MODEL: $VLLM_MODEL"
echo "WORK_DIR: $WORK_DIR"
echo "Date: $(date)"
echo "========================================"

# Get all nodes as an array
if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
else
    # Running locally, just use localhost
    NODES=("localhost")
fi

NUM_NODES=${#NODES[@]}

echo "Detected ${NUM_NODES} node(s):"
for i in "${!NODES[@]}"; do
    echo "  Node $((i+1)): ${NODES[$i]}"
done
echo "========================================"

# =============================================================================
# STEP 1: Split input into partitions
# =============================================================================

echo "Step 1: Splitting input into $NUM_NODES partitions..."
mkdir -p "$PARTITION_DIR"

# Determine output file to check for completed samples
INPUT_BASENAME=$(basename "$INPUT_JSONL" .jsonl)
INPUT_DIR=$(dirname "$INPUT_JSONL")
FINAL_OUTPUT="${INPUT_DIR}/${INPUT_BASENAME}_extracted.jsonl"

if [ -f "$FINAL_OUTPUT" ]; then
    echo "Found existing output file: $FINAL_OUTPUT"
    echo "Will filter out already-completed samples..."
    SPLIT_CMD="python ${WORK_DIR}/split_for_extract_and_verify_spans.py \
        --input-file $INPUT_JSONL \
        --num-partitions $NUM_NODES \
        --output-file $FINAL_OUTPUT \
        --output-dir $PARTITION_DIR"
else
    echo "No existing output file found, processing all samples"
    SPLIT_CMD="python ${WORK_DIR}/split_for_extract_and_verify_spans.py \
        --input-file $INPUT_JSONL \
        --num-partitions $NUM_NODES \
        --output-dir $PARTITION_DIR"
fi

eval $SPLIT_CMD

echo "Partitions created in: $PARTITION_DIR"
echo "========================================"

# =============================================================================
# STEP 2: Kill any existing vLLM processes
# =============================================================================

echo "Step 2: Cleaning up old vLLM processes..."
for node in "${NODES[@]}"; do
    if [ "$node" = "localhost" ] || [ "$node" = "$(hostname)" ]; then
        pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
    else
        ssh "$node" "pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true" &
    fi
done
wait
sleep 2
echo "Cleanup complete"
echo "========================================"

# =============================================================================
# STEP 3: Start vLLM on each node
# =============================================================================

echo "Step 3: Starting vLLM instances (all on port $VLLM_PORT)..."
VLLM_PIDS=()

for i in "${!NODES[@]}"; do
    NODE=${NODES[$i]}

    echo "[$((i+1))/${NUM_NODES}] Starting vLLM on $NODE:$VLLM_PORT"

    if [ "$NODE" = "localhost" ] || [ "$NODE" = "$(hostname)" ]; then
        # Local node
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
        python ${WORK_DIR}/vllm/host.py &
        VLLM_PIDS+=($!)
    else
        # Remote node
        ssh "$NODE" "cd $WORK_DIR && \
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
            python host.py" &
        VLLM_PIDS+=($!)
    fi
done

echo "Waiting for vLLM instances to initialize (300s)..."
sleep 300

# Check health
echo "Checking vLLM instance health..."
for i in "${!NODES[@]}"; do
    NODE=${NODES[$i]}
    if [ "$NODE" = "localhost" ] || [ "$NODE" = "$(hostname)" ]; then
        if curl -s -f "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo "  ✓ $NODE:$VLLM_PORT - healthy"
        else
            echo "  ✗ $NODE:$VLLM_PORT - not responding yet (will continue anyway)"
        fi
    else
        if ssh "$NODE" "curl -s -f http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo "  ✓ $NODE:$VLLM_PORT - healthy"
        else
            echo "  ✗ $NODE:$VLLM_PORT - not responding yet (will continue anyway)"
        fi
    fi
done
echo "========================================"

# =============================================================================
# STEP 4: Start processing on each node
# =============================================================================

echo "Step 4: Starting span extraction on each node..."
PROCESSING_PIDS=()
PROCESSING_LOGS=()

for i in "${!NODES[@]}"; do
    NODE=${NODES[$i]}
    PARTITION_FILE="${PARTITION_DIR}/${INPUT_BASENAME}_partition_${i}.jsonl"
    LOG_FILE="${PARTITION_DIR}/node_${i}_processing.log"
    PROCESSING_LOGS+=("$LOG_FILE")

    if [ ! -f "$PARTITION_FILE" ]; then
        echo "  ⚠ Partition file not found: $PARTITION_FILE (skipping)"
        continue
    fi

    echo "[$((i+1))/${NUM_NODES}] Starting processing on $NODE with partition $i"

    PROCESSING_CMD="python ${WORK_DIR}/extract_and_verify_spans.py \
        --input-file $PARTITION_FILE \
        --server-url http://localhost:$VLLM_PORT \
        --model $VLLM_SERVED_NAME \
        --batch-size $BATCH_SIZE \
        --tokenizer $TOKENIZER \
        --num-solutions $NUM_SOLUTIONS \
        --max-span-retries $MAX_SPAN_RETRIES \
        --temperature $TEMPERATURE \
        --top-p $TOP_P \
        --min-p $MIN_P \
        --top-k $TOP_K \
        --max-tokens $MAX_TOKENS \
        --timeout $TIMEOUT"

    if [ "$NODE" = "localhost" ] || [ "$NODE" = "$(hostname)" ]; then
        # Local node
        {
            echo "========================================"
            echo "Processing on node: $NODE (partition $i)"
            echo "Start time: $(date)"
            echo "========================================"
            eval "$PROCESSING_CMD"
        } > "$LOG_FILE" 2>&1 &
        PROCESSING_PIDS+=($!)
    else
        # Remote node
        ssh "$NODE" "cd $WORK_DIR && \
            echo ======================================== && \
            echo Processing on node: $NODE '(partition $i)' && \
            echo Start time: \$(date) && \
            echo ======================================== && \
            $PROCESSING_CMD" > "$LOG_FILE" 2>&1 &
        PROCESSING_PIDS+=($!)
    fi
done

echo "All processing jobs started"
echo "========================================"

# =============================================================================
# STEP 5: Monitor progress
# =============================================================================

echo "Step 5: Monitoring processing jobs..."
echo "Logs are being written to: $PARTITION_DIR/node_*_processing.log"
echo ""
echo "To monitor progress, you can run:"
echo "  tail -f $PARTITION_DIR/node_*_processing.log"
echo ""
echo "Waiting for all processing jobs to complete..."

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."

    # Kill processing jobs
    for pid in "${PROCESSING_PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done

    # Kill vLLM processes
    for pid in "${VLLM_PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done

    # Kill remote processes
    for node in "${NODES[@]}"; do
        if [ "$node" != "localhost" ] && [ "$node" != "$(hostname)" ]; then
            ssh "$node" "pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true" &
            ssh "$node" "pkill -9 -f 'extract_and_verify_spans' 2>/dev/null || true" &
        fi
    done
    wait
    echo "Cleanup complete"
}

trap cleanup EXIT INT TERM

# Wait for all processing jobs
FAILED_NODES=()
for i in "${!PROCESSING_PIDS[@]}"; do
    pid=${PROCESSING_PIDS[$i]}
    if wait $pid; then
        echo "✓ Node $i completed successfully"
    else
        echo "✗ Node $i failed (exit code: $?)"
        FAILED_NODES+=($i)
    fi
done

echo "========================================"
echo "Processing complete!"
echo "========================================"

if [ ${#FAILED_NODES[@]} -eq 0 ]; then
    echo "✓ All nodes completed successfully"
else
    echo "⚠ ${#FAILED_NODES[@]} node(s) failed: ${FAILED_NODES[*]}"
    echo "Check logs for details:"
    for i in "${FAILED_NODES[@]}"; do
        echo "  ${PROCESSING_LOGS[$i]}"
    done
fi

echo ""
echo "Partition outputs are in: $PARTITION_DIR"
echo "Output files: ${INPUT_BASENAME}_partition_*_extracted.jsonl"
echo ""
echo "To collate results, run:"
echo "  python ${WORK_DIR}/collate_partition_outputs.py \\"
echo "    --input-pattern \"${PARTITION_DIR}/${INPUT_BASENAME}_partition_*_extracted.jsonl\" \\"
echo "    --output-file \"${INPUT_DIR}/${INPUT_BASENAME}_extracted.jsonl\" \\"
echo "    --original-input \"$INPUT_JSONL\""
echo "========================================"
