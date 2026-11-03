#!/usr/bin/env python3
"""
Prepare training and validation datasets from dataset_cots_extracted.jsonl files.

This script reads analyzed CoT responses with verified answer spans, tokenizes them,
creates labels for span prediction, and splits into train/val sets.
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import logging

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_prompt(problem: str, data_source: str) -> str:
    """
    Format the problem with instructions (same as gather_cots.py).

    Args:
        problem: The problem text
        data_source: Source of the data (opencoder, math, etc.)

    Returns:
        Formatted prompt string
    """
    if data_source == "opencoder":
        return f"{problem}\n\nPlease reason step by step and give your code in Python. Use Python's standard library unless otherwise stated."
    else:
        return f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def load_and_verify_tokenizer(model_name: str) -> Tuple[AutoTokenizer, int, int]:
    """
    Load tokenizer and verify it has <think> and </think> tokens.

    Args:
        model_name: Name or path of the model

    Returns:
        Tuple of (tokenizer, think_token_id, think_end_token_id)

    Raises:
        ValueError: If tokenizer doesn't have <think> tokens
    """
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Get <think> and </think> token IDs
    think_token_id = tokenizer.convert_tokens_to_ids('<think>')
    think_end_token_id = tokenizer.convert_tokens_to_ids('</think>')

    # Verify tokens exist
    if think_token_id == tokenizer.unk_token_id or think_end_token_id == tokenizer.unk_token_id:
        raise ValueError(
            f"ERROR: <think> and </think> tokens not found in tokenizer vocabulary!\n"
            f"These special tokens are required for training but were not found.\n"
            f"This model is not supported. Please use Qwen3 models which include <think> tokens.\n"
            f"<think> token ID: {think_token_id} (unk: {tokenizer.unk_token_id})\n"
            f"</think> token ID: {think_end_token_id} (unk: {tokenizer.unk_token_id})"
        )

    logger.info(f"✓ Tokenizer verified: <think> token ID = {think_token_id}, </think> token ID = {think_end_token_id}")
    return tokenizer, think_token_id, think_end_token_id


def load_and_filter_samples(
    input_file: Path,
    max_solutions_per_problem: int = None
) -> List[Dict[str, Any]]:
    """
    Load samples from dataset_cots_extracted.jsonl and filter for verified spans.

    Args:
        input_file: Path to the dataset_cots_extracted.jsonl file
        max_solutions_per_problem: Maximum number of solutions to keep per problem (None = all)

    Returns:
        List of processed samples ready for tokenization
    """
    samples = []
    total_solutions = 0
    filtered_solutions = 0

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {line_num}: JSON decode error: {e}")
                continue

            # Extract base fields
            base_fields = {
                'sample_idx': data.get('sample_idx'),
                'problem': data.get('problem'),
                'solution': data.get('solution'),
                'answer': data.get('answer'),
                'data_source': data.get('data_source'),
                'data_fields': data.get('data_fields', {}),
            }

            # Process solution_analyses
            solution_analyses = data.get('solution_analyses', [])
            kept_count = 0

            for analysis in solution_analyses:
                total_solutions += 1

                # Filter: Only keep if span is verified
                if not analysis.get('llm_span_final_verified', False):
                    filtered_solutions += 1
                    continue

                # Limit: Check if we've already kept enough solutions for this problem
                if max_solutions_per_problem is not None and kept_count >= max_solutions_per_problem:
                    filtered_solutions += 1
                    continue

                # Create sample with all required fields
                sample = {
                    **base_fields,
                    'reasoning_content': analysis.get('reasoning_content'),
                    'final_content': analysis.get('final_content'),
                    'span_final_start_token': analysis.get('span_final_start_token'),
                    'span_final_end_token': analysis.get('span_final_end_token'),
                }

                samples.append(sample)
                kept_count += 1

    logger.info(f"Loaded {len(samples)} valid samples from {input_file}")
    logger.info(f"  Total solutions in file: {total_solutions}")
    logger.info(f"  Filtered out: {filtered_solutions} ({100*filtered_solutions/total_solutions:.1f}%)")
    logger.info(f"  Kept: {len(samples)} ({100*len(samples)/total_solutions:.1f}%)")

    return samples


def tokenize_and_label_sample(
    sample: Dict[str, Any],
    tokenizer: AutoTokenizer,
    think_token_id: int,
    think_end_token_id: int,
    max_length: int
) -> Dict[str, Any]:
    """
    Tokenize a sample and create labels.

    Args:
        sample: Sample dictionary with required fields
        tokenizer: The tokenizer to use
        think_token_id: Token ID for <think>
        think_end_token_id: Token ID for </think>
        max_length: Maximum sequence length (truncate if needed)

    Returns:
        Updated sample with tokenization fields added
    """
    problem = sample.get('problem', '')
    reasoning_content = sample.get('reasoning_content', '')
    data_source = sample.get('data_source', 'unknown')
    span_end_token = sample.get('span_final_end_token')

    # Format prompt using data-source-specific template
    formatted_prompt = format_prompt(problem, data_source)

    # Apply chat template to get the full prompt
    messages = [{"role": "user", "content": formatted_prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize each part separately to ensure <think> and </think> are single tokens
    # 1. Tokenize the prompt
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors='pt'
    )['input_ids'].squeeze(0)

    # 2. Tokenize the reasoning content
    reasoning_ids = tokenizer(
        reasoning_content,
        add_special_tokens=False,
        return_tensors='pt'
    )['input_ids'].squeeze(0)

    # 3. Construct full sequence: prompt + <think> + reasoning + </think>
    input_ids = torch.cat([
        prompt_ids,
        torch.tensor([think_token_id], dtype=prompt_ids.dtype),
        reasoning_ids,
        torch.tensor([think_end_token_id], dtype=prompt_ids.dtype)
    ])

    # Track if truncated
    was_truncated = False
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        was_truncated = True

    # Find positions of <think> and </think> tokens
    think_positions = (input_ids == think_token_id).nonzero(as_tuple=True)[0]
    think_end_positions = (input_ids == think_end_token_id).nonzero(as_tuple=True)[0]

    # Sanity check: these should always be found since we explicitly inserted them
    assert len(think_positions) > 0 and len(think_end_positions) > 0, \
        f"BUG: Could not find <think> or </think> tokens in tokenized sequence for sample {sample.get('sample_idx', 'unknown')}. " \
        f"This should never happen since we explicitly insert these tokens."

    think_start_pos = think_positions[0].item()
    think_end_pos = think_end_positions[-1].item()  # Use last </think> in case of multiple

    # Create labels: only valid between <think> and </think>
    # Initialize all positions as ignored (-100 for PyTorch cross-entropy)
    labels = torch.full((len(input_ids),), -100, dtype=torch.long)

    # Set labels only in the thinking region (exclusive of <think>, inclusive of </think>)
    thinking_start = think_start_pos + 1
    thinking_end = think_end_pos + 1  # Inclusive of </think>

    # Map span_end_token (relative to reasoning_content) to full sequence
    # span_end_token is relative to start of reasoning_content
    # reasoning_content starts at position think_start_pos + 1
    absolute_span_end = thinking_start + span_end_token

    # Ensure we don't go beyond thinking region
    absolute_span_end = min(absolute_span_end, thinking_end - 1)

    # Labels: 0 up to and including absolute_span_end, 1 after
    labels[thinking_start:absolute_span_end + 1] = 0
    labels[absolute_span_end + 1:thinking_end] = 1

    # Create valid_mask: 1 for positions within thinking region, 0 elsewhere
    valid_mask = torch.zeros(len(input_ids), dtype=torch.long)
    valid_mask[thinking_start:thinking_end] = 1

    # Add tokenization fields to sample
    sample['input_ids'] = input_ids.tolist()
    sample['labels'] = labels.tolist()
    sample['valid_mask'] = valid_mask.tolist()
    sample['was_truncated'] = was_truncated

    return sample


def tokenize_samples(
    samples: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    think_token_id: int,
    think_end_token_id: int,
    max_length: int
) -> List[Dict[str, Any]]:
    """
    Tokenize all samples and create labels.

    Args:
        samples: List of samples
        tokenizer: The tokenizer to use
        think_token_id: Token ID for <think>
        think_end_token_id: Token ID for </think>
        max_length: Maximum sequence length

    Returns:
        List of tokenized samples
    """
    logger.info(f"Tokenizing {len(samples)} samples (max_length={max_length})...")

    tokenized_samples = []
    skipped = 0
    truncated_count = 0
    seq_lengths = []

    for sample in tqdm(samples, desc="Tokenizing"):
        try:
            tokenized_sample = tokenize_and_label_sample(
                sample, tokenizer, think_token_id, think_end_token_id, max_length
            )
            tokenized_samples.append(tokenized_sample)
            seq_lengths.append(len(tokenized_sample['input_ids']))
            if tokenized_sample['was_truncated']:
                truncated_count += 1
        except Exception as e:
            logger.warning(f"Skipping sample {sample.get('sample_idx', 'unknown')}: {e}")
            skipped += 1

    # Log statistics
    logger.info(f"✓ Tokenization complete:")
    logger.info(f"  Processed: {len(tokenized_samples)}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Truncated: {truncated_count}")
    if seq_lengths:
        logger.info(f"  Sequence lengths - Min: {min(seq_lengths)}, Max: {max(seq_lengths)}, Avg: {sum(seq_lengths)/len(seq_lengths):.1f}")

    return tokenized_samples


def add_cot_indices(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add cot_idx field to each sample (sequential: 0, 1, 2, ...).

    Args:
        samples: List of samples

    Returns:
        List of samples with cot_idx added
    """
    for idx, sample in enumerate(samples):
        sample['cot_idx'] = idx

    logger.info(f"Added cot_idx to {len(samples)} samples")
    return samples


def get_base_data_source(data_source: str) -> str:
    """
    Extract base data source from full data_source string.

    Examples:
        aime_2004 -> aime
        aime_1989 -> aime
        math_algebra -> math
        opencoder -> opencoder

    Args:
        data_source: Full data source string

    Returns:
        Base data source (first part before underscore, or full string if no underscore)
    """
    if not data_source:
        return 'unknown'

    # Split on underscore and take first part
    return data_source.split('_')[0]


def split_by_data_source(
    samples: List[Dict[str, Any]],
    val_size_per_source: int,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset into train/val by base data_source, sampling at sample_idx level.

    Base data sources are extracted by taking the first part before underscore.
    For example: aime_2004, aime_1989 -> both treated as 'aime'

    Args:
        samples: List of all samples
        val_size_per_source: Number of sample_idx to select for validation per base data_source
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_samples, val_samples)
    """
    random.seed(seed)

    # Group samples by base_data_source and sample_idx
    base_source_to_sample_idx_to_samples = defaultdict(lambda: defaultdict(list))

    for sample in samples:
        data_source = sample.get('data_source', 'unknown')
        base_source = get_base_data_source(data_source)
        sample_idx = sample.get('sample_idx')
        base_source_to_sample_idx_to_samples[base_source][sample_idx].append(sample)

    train_samples = []
    val_samples = []

    # Process each base data_source
    for base_source, sample_idx_to_samples in base_source_to_sample_idx_to_samples.items():
        # Get all unique sample_idx for this base_source
        all_sample_indices = list(sample_idx_to_samples.keys())
        random.shuffle(all_sample_indices)

        # Select up to val_size_per_source for validation
        num_val = min(val_size_per_source, len(all_sample_indices))
        val_sample_indices = set(all_sample_indices[:num_val])
        train_sample_indices = set(all_sample_indices[num_val:])

        # Assign samples to train or val
        for sample_idx, sample_list in sample_idx_to_samples.items():
            if sample_idx in val_sample_indices:
                val_samples.extend(sample_list)
            else:
                train_samples.extend(sample_list)

        logger.info(f"Base data source '{base_source}':")
        logger.info(f"  Total unique problems (sample_idx): {len(all_sample_indices)}")
        logger.info(f"  Val problems: {len(val_sample_indices)}")
        logger.info(f"  Train problems: {len(train_sample_indices)}")
        logger.info(f"  Val samples (cots): {sum(1 for s in val_samples if get_base_data_source(s.get('data_source', '')) == base_source)}")
        logger.info(f"  Train samples (cots): {sum(1 for s in train_samples if get_base_data_source(s.get('data_source', '')) == base_source)}")

    logger.info(f"\nTotal split: {len(train_samples)} train, {len(val_samples)} val")
    return train_samples, val_samples


def save_jsonl(samples: List[Dict[str, Any]], output_file: Path):
    """Save samples to a JSONL file with only specified fields."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Fields to keep in output
    output_fields = [
        'cot_idx',
        'sample_idx',
        'problem',
        'solution',
        'answer',
        'data_source',
        'data_fields',
        'input_ids',
        'labels',
        'valid_mask',
    ]

    with open(output_file, 'w') as f:
        for sample in samples:
            # Only keep specified fields
            output_sample = {k: sample.get(k) for k in output_fields if k in sample}
            json.dump(output_sample, f)
            f.write('\n')

    logger.info(f"Saved {len(samples)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training and validation datasets from dataset_cots_extracted.jsonl"
    )

    parser.add_argument(
        '--input-file',
        type=Path,
        default=Path('data/outputs/Qwen/Qwen3-8B/dataset_cots_extracted.jsonl'),
        help='Path to dataset_cots_extracted.jsonl file'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='Qwen/Qwen3-8B',
        help='Model name for tokenizer (must support <think> tokens)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=32768,
        help='Maximum sequence length (truncate if needed)'
    )
    parser.add_argument(
        '--max-solutions-per-problem',
        type=int,
        default=None,
        help='Maximum number of solutions to keep per problem (default: None = all)'
    )
    parser.add_argument(
        '--val-size-per-source',
        type=int,
        default=50,
        help='Number of sample_idx (problems) for validation per base data_source (e.g., all aime_* variants treated as "aime") (default: 50)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1337,
        help='Random seed for shuffling'
    )

    args = parser.parse_args()

    # Load tokenizer and verify it supports <think> tokens
    tokenizer, think_token_id, think_end_token_id = load_and_verify_tokenizer(args.model_name)

    # Load and filter samples
    logger.info(f"Loading samples from {args.input_file}")
    samples = load_and_filter_samples(args.input_file, args.max_solutions_per_problem)

    if not samples:
        logger.error("No valid samples found!")
        return

    # Tokenize samples and create labels
    samples = tokenize_samples(samples, tokenizer, think_token_id, think_end_token_id, args.max_length)

    if not samples:
        logger.error("No samples remaining after tokenization!")
        return

    # Add cot_idx to each sample
    samples = add_cot_indices(samples)

    # Split dataset by data_source
    train_samples, val_samples = split_by_data_source(
        samples, args.val_size_per_source, args.seed
    )

    # Determine output file names
    output_dir = args.input_file.parent
    if args.max_solutions_per_problem is not None:
        suffix = f"_ss{args.max_solutions_per_problem}"
    else:
        suffix = ""

    train_file = output_dir / f'train{suffix}.jsonl'
    val_file = output_dir / f'val{suffix}.jsonl'

    # Save to output files
    save_jsonl(train_samples, train_file)
    save_jsonl(val_samples, val_file)

    logger.info("\nDataset preparation complete!")
    logger.info(f"  Training samples: {len(train_samples)}")
    logger.info(f"  Validation samples: {len(val_samples)}")
    logger.info(f"  Total samples: {len(samples)}")
    logger.info(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()
