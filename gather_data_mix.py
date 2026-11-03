#!/usr/bin/env python3
"""
Script to gather training data from multiple sources into a unified dataset.

Pools together samples from MATH, AIME, OpenCoder, and OpenScience datasets
into a single JSONL file with standardized fields.
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def extract_boxed_solution(text: str) -> Optional[str]:
    """
    Extract content from LaTeX \\boxed{...} with proper brace matching.
    Returns the content inside the last \\boxed{} in the text.
    """
    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, text))

    if not matches:
        return None

    # Start from the last \boxed{ occurrence
    last_match = matches[-1]
    start_idx = last_match.end()

    # Count braces to find matching closing brace
    brace_count = 1
    idx = start_idx

    while idx < len(text) and brace_count > 0:
        if text[idx] == '{':
            brace_count += 1
        elif text[idx] == '}':
            brace_count -= 1
        idx += 1

    if brace_count == 0:
        return text[start_idx:idx-1]

    return None


def load_math_dataset(data_path: str, max_samples: int) -> List[Dict]:
    """Load and process the MATH dataset."""
    print(f"Loading MATH dataset from {data_path}...")

    samples = []
    with open(data_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    # Sample if needed
    if max_samples < len(samples):
        samples = random.sample(samples, max_samples)

    processed = []
    for sample in tqdm(samples, desc="Processing MATH"):
        processed.append({
            'problem': sample['problem'],
            'solution': sample['solution'],
            'answer': sample['answer'],
            'data_source': 'math',
            'data_fields': {
                'unique_id': sample['unique_id'],
                'subject': sample['subject'],
                'level': sample['level']
            }
        })

    return processed


def load_aime_dataset(max_samples: int) -> List[Dict]:
    """Load and process the AIME dataset from di-zhang-fdu/AIME_1983_2024."""
    print("Loading AIME dataset from HuggingFace (di-zhang-fdu/AIME_1983_2024)...")

    dataset = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")

    # Sample if needed
    if max_samples < len(dataset):
        indices = random.sample(range(len(dataset)), max_samples)
        dataset = dataset.select(indices)

    processed = []
    for sample in tqdm(dataset, desc="Processing AIME"):
        # Extract year (it's int64 by default)
        year = int(sample['Year'])
        data_source = f"aime_{year}"

        processed.append({
            'problem': sample['Question'],
            'solution': None,  # New AIME dataset doesn't have solutions
            'answer': sample['Answer'],
            'data_source': data_source,
            'data_fields': {
                'ID': sample['ID'],
                'Year': year,
                'Problem Number': int(sample['Problem Number']),
                'Part': sample.get('Part'),  # May be None
            }
        })

    return processed


def load_opencoder_dataset(max_samples: int) -> List[Dict]:
    """
    Load and process the OpenCoder dataset with special sampling strategy.

    Samples in rounds: each round goes through all entry_point types (ordered by
    frequency distribution), picking one unused example per type. This ensures
    maximum diversity - every type gets one sample before any type gets a second.
    """
    print("Loading OpenCoder dataset from HuggingFace...")

    dataset = load_dataset("OpenCoder-LLM/opc-sft-stage2", "educational_instruct", split="train")

    # Group examples by entry_point and count occurrences
    type_to_indices = defaultdict(list)
    type_counts = defaultdict(int)

    for idx, sample in enumerate(dataset):
        entry_point = sample['entry_point']
        type_to_indices[entry_point].append(idx)
        type_counts[entry_point] += 1

    # Create probability distribution based on type frequencies
    types = list(type_counts.keys())
    counts = np.array([type_counts[t] for t in types])
    type_probs = counts / counts.sum()

    print(f"Found {len(types)} unique entry_point types")

    # Determine how many samples to actually draw
    num_samples = min(max_samples, len(dataset))

    # Sample examples in rounds
    selected_indices = []
    used_indices = set()

    with tqdm(total=num_samples, desc="Processing OpenCoder") as pbar:
        round_num = 0
        while len(selected_indices) < num_samples:
            round_num += 1
            samples_this_round = 0

            # Sample types for this round according to probability distribution
            # This gives us an ordering of types based on their importance
            sampled_type_order = np.random.choice(
                types,
                size=len(types),
                replace=False,
                p=type_probs
            )

            # Go through each type in this round
            for entry_type in sampled_type_order:
                if len(selected_indices) >= num_samples:
                    break

                # Get unused examples for this type
                unused_for_type = [
                    idx for idx in type_to_indices[entry_type]
                    if idx not in used_indices
                ]

                if unused_for_type:
                    # Randomly pick one unused example
                    selected_idx = random.choice(unused_for_type)
                    selected_indices.append(selected_idx)
                    used_indices.add(selected_idx)
                    samples_this_round += 1
                    pbar.update(1)

            # If we didn't sample any examples this round, all examples are exhausted
            if samples_this_round == 0:
                break

    # Process selected samples
    processed = []
    for idx in selected_indices:
        sample = dataset[idx]
        processed.append({
            'problem': sample['instruction'],
            'solution': sample['output'],
            'answer': sample['code'],
            'data_source': 'opencoder',
            'data_fields': {
                'seq_id': sample['seq_id'],
                'testcase': sample['testcase']
            }
        })

    return processed


def load_openscience_dataset(max_samples: int) -> List[Dict]:
    """Load and process the OpenScience dataset."""
    print("Loading OpenScience dataset from HuggingFace...")

    dataset = load_dataset("nvidia/OpenScience", "OS-Q3-235B-4", split="train")

    # Determine how many samples to draw
    num_samples = min(max_samples, len(dataset))

    # We'll sample with replacement if we encounter extraction errors
    processed = []
    attempts = 0
    max_attempts = num_samples * 3  # Prevent infinite loop

    available_indices = list(range(len(dataset)))
    random.shuffle(available_indices)

    pbar = tqdm(total=num_samples, desc="Processing OpenScience")

    while len(processed) < num_samples and attempts < max_attempts:
        if not available_indices:
            print("\nWarning: Ran out of samples in OpenScience dataset")
            break

        idx = available_indices.pop()
        sample = dataset[idx]
        attempts += 1

        # Extract answer from \boxed{} at the end
        output = sample['output']
        answer = extract_boxed_solution(output)

        if answer is None:
            print(f"\nWarning: Failed to extract answer from sample {idx}, trying another sample...")
            continue

        processed.append({
            'problem': sample['input'],
            'solution': sample['output'],
            'answer': answer,
            'data_source': 'openscience',
            'data_fields': {}
        })
        pbar.update(1)

    pbar.close()

    if len(processed) < num_samples:
        print(f"Warning: Only collected {len(processed)}/{num_samples} samples from OpenScience")

    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Gather training data from multiple sources into a unified dataset"
    )
    parser.add_argument(
        '--samples-per-dataset',
        type=int,
        default=12000,
        help='Maximum number of samples to draw from each dataset'
    )
    parser.add_argument(
        '--math-path',
        type=str,
        default='../data/math-500/train.jsonl',
        help='Path to the MATH dataset JSONL file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/inputs',
        help='Directory to save the output dataset'
    )
    parser.add_argument(
        '--output-filename',
        type=str,
        default='dataset.jsonl',
        help='Filename for the output dataset'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1337,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and process each dataset
    all_samples = []

    # 1. MATH dataset
    math_samples = load_math_dataset(args.math_path, args.samples_per_dataset)
    all_samples.extend(math_samples)
    print(f"Collected {len(math_samples)} samples from MATH")

    # 2. AIME dataset
    aime_samples = load_aime_dataset(args.samples_per_dataset)
    all_samples.extend(aime_samples)
    print(f"Collected {len(aime_samples)} samples from AIME")

    # 3. OpenCoder dataset
    opencoder_samples = load_opencoder_dataset(args.samples_per_dataset)
    all_samples.extend(opencoder_samples)
    print(f"Collected {len(opencoder_samples)} samples from OpenCoder")

    # 4. OpenScience dataset
    openscience_samples = load_openscience_dataset(args.samples_per_dataset)
    all_samples.extend(openscience_samples)
    print(f"Collected {len(openscience_samples)} samples from OpenScience")

    # Save to JSONL with sample_idx as first field
    print(f"\nTotal samples collected: {len(all_samples)}")
    output_path = output_dir / args.output_filename
    print(f"Saving dataset to {output_path}...")

    with open(output_path, 'w') as f:
        for idx, sample in enumerate(tqdm(all_samples, desc="Writing JSONL")):
            # Create ordered dict with sample_idx first
            ordered_sample = {
                'sample_idx': idx,
                **sample  # Unpack remaining fields
            }
            f.write(json.dumps(ordered_sample) + '\n')

    print(f"\nDataset creation complete!")
    print(f"Total samples: {len(all_samples)}")
    print(f"Output saved to: {output_path}")

    # Print distribution by data source
    print("\nDistribution by data source:")
    source_counts = defaultdict(int)
    for sample in all_samples:
        source_counts[sample['data_source']] += 1

    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")


if __name__ == '__main__':
    main()
