#!/usr/bin/env python3
import os
import shlex
import subprocess
import sys

def env(name, default=None, required=False):
    v = os.environ.get(name, default)
    if required and (v is None or v == ""):
        print(f"Missing required env var: {name}", file=sys.stderr)
        sys.exit(2)
    return v

def main():
    # Required
    model = env("VLLM_MODEL", required=True)

    # Optional tuning
    host = env("VLLM_HOST", "0.0.0.0")
    port = env("VLLM_PORT", "8000")
    tp = env("VLLM_TP_SIZE", "1")  # tensor parallelism; 1 for single GPU
    max_len = env("VLLM_MAX_MODEL_LEN", None)
    gpu_util = env("VLLM_GPU_UTIL", "0.90")  # fraction of GPU VRAM to use
    download_dir = env("VLLM_DOWNLOAD_DIR", None)
    served_name = env("VLLM_SERVED_NAME", None)
    dtype = env("VLLM_DTYPE", "auto")  # e.g., "auto", "float16", "bfloat16"
    max_num_seqs = env("VLLM_MAX_NUM_SEQS", None)
    api_key = env("VLLM_API_KEY", None)  # if set, server will require this
    trust_remote_code = env("VLLM_TRUST_REMOTE_CODE", "0")
    no_prefix_caching = env("NO_PREFIX_CACHING", "0")
    reasoning_parser = env("REASONING_PARSER", None)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tp),
        "--dtype", dtype,
        "--gpu-memory-utilization", str(gpu_util),
    ]

    if served_name:
        cmd += ["--served-model-name", served_name]
    if download_dir:
        cmd += ["--download-dir", download_dir]
    if max_len:
        cmd += ["--max-model-len", str(max_len)]
    if max_num_seqs:
        cmd += ["--max-num-seqs", str(max_num_seqs)]
    if api_key:
        cmd += ["--api-key", api_key]
    if trust_remote_code == "1":
        cmd += ["--trust-remote-code"]
    if no_prefix_caching == "1":
        cmd += ["--enable-prefix-caching", "False"]
    if reasoning_parser:
        cmd += ["--reasoning_parser", reasoning_parser]

    print("Launching vLLM API server with:\n  " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    # Hand off to the server (so Ctrl+C & signals propagate cleanly)
    os.execvp(cmd[0], cmd)

if __name__ == "__main__":
    main()

