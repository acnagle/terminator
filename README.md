# Terminator

[Project page](https://terminator-llm.github.io/) | [Paper](https://arxiv.org/abs/2603.12529) | Trained weights: [Qwen3-8B](https://huggingface.co/acnagle/Terminator-Qwen3-8B), [Qwen3-14B](https://huggingface.co/acnagle/Terminator-Qwen3-14B) | Training data: [Qwen3-8B](https://huggingface.co/datasets/acnagle/Terminator-Qwen3-8B-Training-Data), [Qwen3-14B](https://huggingface.co/datasets/acnagle/Terminator-Qwen3-14B-Training-Data)

Code for training and evaluating **Terminator**, a neural network that predicts when a language model has produced its final answer during chain-of-thought (CoT) reasoning. The trained predictor enables early termination of reasoning, reducing inference cost without sacrificing answer quality.

## Requirements

- Python 3.12
- CUDA 12.x with a GPU that has sufficient VRAM to load the target model (e.g., 24 GB+ for Qwen3-8B, 48 GB+ for Qwen3-14B)
- [uv](https://docs.astral.sh/uv/) (recommended for package installation)

## Environment Setup

Create a Python 3.12 environment using conda or your preferred environment manager:

```bash
conda create -n terminator python=3.12
conda activate terminator
```

Install vLLM following the [official installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/). We recommend using `uv` for installation, which will also install PyTorch, Transformers, and other core dependencies automatically:

```bash
pip install uv
uv pip install vllm --torch-backend=auto
```

Then install the remaining dependencies:

```bash
uv pip install accelerate datasets scikit-learn httpx aiohttp openai wandb
```

Optionally, set the HuggingFace cache directory (useful when disk space is limited on the default path):

```bash
export HF_HOME=/path/to/hf_cache
```

## Pipeline Overview

The full pipeline has six stages. All intermediate data is stored as JSONL files under `data/`.

```
1. Gather data         -->  data/inputs/dataset.jsonl, test.jsonl
2. Generate CoTs       -->  data/outputs/{MODEL}/dataset_cots.jsonl
3. Extract spans       -->  data/outputs/{MODEL}/dataset_cots_extracted.jsonl
4. Prepare training    -->  data/outputs/{MODEL}/train.jsonl, val.jsonl
5. Train Terminator    -->  models/
6. Evaluate            -->  inference outputs
```

Replace `{MODEL}` with `Qwen/Qwen3-8B` or `Qwen/Qwen3-14B` throughout.

### Skip Steps 1-4: Download the Training Data

To train Terminator without generating CoTs, download the prepared `train.jsonl` and `val.jsonl` and go straight to [Step 5](#step-5-train-terminator). This is the tokenized, labeled data used to train the published weights:

- [acnagle/Terminator-Qwen3-8B-Training-Data](https://huggingface.co/datasets/acnagle/Terminator-Qwen3-8B-Training-Data)
- [acnagle/Terminator-Qwen3-14B-Training-Data](https://huggingface.co/datasets/acnagle/Terminator-Qwen3-14B-Training-Data)

These commands save the files where the training scripts expect them:

```bash
# Qwen3-8B
hf download acnagle/Terminator-Qwen3-8B-Training-Data train.jsonl val.jsonl \
    --repo-type dataset --local-dir data/outputs/Qwen/Qwen3-8B

# Qwen3-14B
hf download acnagle/Terminator-Qwen3-14B-Training-Data train.jsonl val.jsonl \
    --repo-type dataset --local-dir data/outputs/Qwen/Qwen3-14B
```

This data covers training only. [Step 6](#step-6-inference-and-evaluation) still needs test CoTs, so run `gather_test_mix.py` from Step 1 and the test commands from Step 2 before evaluating.

### Step 1: Gather Data

Create the training and test data mixtures from HuggingFace datasets:

```bash
python gather_data_mix.py      # --> data/inputs/dataset.jsonl
python gather_test_mix.py      # --> data/inputs/test.jsonl
```

Training data sources: MATH (PRM800K), AIME, OpenCoder, OpenScience.
Test data sources: MATH-500, AIME 2025, GPQA, HumanEval.

### Step 2: Generate CoTs

Generate chain-of-thought solutions with per-token log probabilities and self-certainty scores. This step launches a local vLLM server and runs async batched inference.

For training data:

```bash
bash gather_train_cots.sh
```

For test data:

```bash
# Qwen3-8B
bash gather_test_cots_qwen3-8b.sh

# Qwen3-14B
bash gather_test_cots_qwen3-14b.sh
```

These scripts configure model and inference parameters, then call `run_gather_cots.sh`, which manages the vLLM server lifecycle and runs `gather_cots.py`.

### Step 3: Extract and Verify Spans

Identify the earliest token position where the model's final answer appears in each CoT. Uses a verifier model (Qwen3-30B) to confirm span correctness. The commands below process all three CoTs of each problem. The default of 1 processes only the first, which gives about a third of the training data.

**Single-node:**

```bash
python extract_and_verify_spans.py \
    --input-file data/outputs/Qwen/Qwen3-8B/dataset_cots.jsonl \
    --server-url http://localhost:8000 \
    --model vllm \
    --tokenizer Qwen/Qwen3-8B \
    --batch-size 16 \
    --num-solutions 3
```

**Multi-node** (for large datasets):

```bash
# Set configuration, then run:
NUM_SOLUTIONS=3 bash run_extract_and_verify_spans_multinode.sh
```

This partitions the input via `split_for_extract_and_verify_spans.py`, runs extraction in parallel across nodes, and results are merged with `collate_partition_outputs.py`.

### Step 4: Prepare Training Data

Tokenize reasoning traces with prompts and create binary labels. Within the reasoning, each token is labeled `0` up to and including the last token of the model's final answer span, and `1` after it. Tokens outside the reasoning are labeled `-100` and ignored during training.

```bash
# Use all verified solutions (RECOMMENDED)
python prepare_training_data.py \
    --input-file data/outputs/Qwen/Qwen3-8B/dataset_cots_extracted.jsonl

# Limit solutions per problem and validation size (If you want extra control)
python prepare_training_data.py \
    --input-file data/outputs/Qwen/Qwen3-8B/dataset_cots_extracted.jsonl \
    --max-solutions-per-problem 2 \
    --val-size-per-source 200

# For Qwen3-14B (uses matching tokenizer) (RECOMMENDED for Qwen3-14B)
python prepare_training_data.py \
    --input-file data/outputs/Qwen/Qwen3-14B/dataset_cots_extracted.jsonl \
    --model-name Qwen/Qwen3-14B
```

### Step 5: Train Terminator

Train a binary classifier that attaches to the base model's hidden states. The base model is frozen; only the extra transformer layer(s) and FFN head are trained.

```bash
# Qwen3-8B
bash train_terminator_qwen3-8b.sh

# Qwen3-14B
bash train_terminator_qwen3-14b.sh
```

The training scripts log to [Weights & Biases](https://wandb.ai) through `--use-wandb`. Run `wandb login` first, set `WANDB_MODE=offline` to keep the logs local, or remove the flag from the script.

Key training arguments (see scripts for full configuration):
- `--layer-idx -1`: attach to the last transformer layer
- `--num-extra-layers 1 --extra-layers-init copy_last`: add one trainable transformer layer initialized by copying the base model's final layer
- `--use-class-weights`: weight the BCE loss by inverse class frequency

### Step 6: Inference and Evaluation

**Run the trained Terminator** on test CoTs:

```bash
# Qwen3-8B
bash inference_qwen3-8b.sh

# Qwen3-14B
bash inference_qwen3-14b.sh
```

Set `--checkpoint-dir` in the script first. After training, point it at the run's directory under `models/`.

To evaluate the published weights instead, download a checkpoint and rename it, because `inference_terminator.py` looks for `layer_*.pt`. Then point `--checkpoint-dir` at the download directory:

```bash
# Qwen3-8B
hf download acnagle/Terminator-Qwen3-8B terminator.pt --local-dir models/Terminator-Qwen3-8B
mv models/Terminator-Qwen3-8B/terminator.pt models/Terminator-Qwen3-8B/layer_35.pt

# Qwen3-14B
hf download acnagle/Terminator-Qwen3-14B terminator.pt --local-dir models/Terminator-Qwen3-14B
mv models/Terminator-Qwen3-14B/terminator.pt models/Terminator-Qwen3-14B/layer_39.pt
```

To run Terminator on new prompts, follow the setup and serving instructions on the model pages: [Qwen3-8B](https://huggingface.co/acnagle/Terminator-Qwen3-8B), [Qwen3-14B](https://huggingface.co/acnagle/Terminator-Qwen3-14B).

**Truncation evaluation** -- test whether truncating reasoning at various points relative to the predicted span preserves answer correctness.

This needs `test_cots_extracted.jsonl`, so first run Step 3 on the test CoTs with Qwen3-30B served at `--server-url`. `--num-solutions` sets how many CoTs to process per problem and defaults to 1:

```bash
python extract_and_verify_spans.py \
    --input-file data/outputs/Qwen/Qwen3-8B/test_cots.jsonl \
    --server-url http://localhost:8000 \
    --model vllm \
    --tokenizer Qwen/Qwen3-8B \
    --batch-size 16 \
    --num-solutions 10
```

Then run the evaluation:

```bash
bash inference_truncate.sh
```

This launches a vLLM server, truncates each test sample's reasoning at configurable ratios (before and after the predicted span endpoint), and generates final answers from the truncated context.

## Project Structure

|                    File                   |                           Description                          |
|-------------------------------------------|----------------------------------------------------------------|
| `gather_data_mix.py`                      | Assemble training data mixture                                 |
| `gather_test_mix.py`                      | Assemble test data mixture                                     |
| `gather_cots.py`                          | Generate CoTs with log probs and self-certainty via vLLM       |
| `extract_and_verify_spans.py`             | Extract and verify final answer span positions                 |
| `split_for_extract_and_verify_spans.py`   | Partition data for multi-node span extraction                  |
| `collate_partition_outputs.py`            | Merge partition outputs after multi-node extraction            |
| `prepare_training_data.py`                | Tokenize and create binary labels for training                 |
| `train_terminator.py`                     | Train the Terminator predictor (BCE loss)                      |
| `inference_terminator.py`                 | Run trained Terminator on test data                            |
| `inference_truncate.py`                   | Evaluate truncated reasoning via vLLM                          |
| `terminator_utils.py`                     | Shared model components (`LayerFFN`, `ExtraTransformerLayers`) |
| `host.py`                                 | vLLM API server launcher (env-var driven)                      |
