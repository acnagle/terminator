#!/usr/bin/env python3
"""
Script to gather test data from multiple sources into a unified dataset.

Pools together samples from MATH-500, AIME 2025, GPQA, HumanEval, and BigCodeBench
datasets into a single JSONL file with standardized fields.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)


def load_math500_dataset() -> List[Dict]:
    """Load and process the MATH-500 dataset."""
    print("Loading MATH-500 dataset from HuggingFace (HuggingFaceH4/MATH-500)...")

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    processed = []
    for sample in tqdm(dataset, desc="Processing MATH-500"):
        processed.append({
            'problem': sample['problem'],
            'solution': sample['solution'],
            'answer': sample['answer'],
            'data_source': 'math-500',
            'data_fields': {
                'unique_id': sample['unique_id'],
                'level': sample['level'],
                'subject': sample['subject']
            }
        })

    return processed


def load_aime25_dataset() -> List[Dict]:
    """Load and process the AIME 2025 dataset."""
    print("Loading AIME 2025 dataset from HuggingFace (math-ai/aime25)...")

    dataset = load_dataset("math-ai/aime25", split="test")

    processed = []
    for sample in tqdm(dataset, desc="Processing AIME 2025"):
        processed.append({
            'problem': sample['problem'],
            'solution': None,  # AIME doesn't have solutions
            'answer': sample['answer'],
            'data_source': 'aime_2025',
            'data_fields': {
                'id': sample['id']
            }
        })

    return processed


def load_gpqa_dataset() -> List[Dict]:
    """Load and process the GPQA dataset with shuffled multiple choice options."""
    print("Loading GPQA dataset from HuggingFace (Idavidrein/gpqa, gpqa_main subset)...")

    dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")

    processed = []
    for sample in tqdm(dataset, desc="Processing GPQA"):
        # Collect all answer options
        correct_answer = sample['Correct Answer']
        incorrect_answers = [
            sample['Incorrect Answer 1'],
            sample['Incorrect Answer 2'],
            sample['Incorrect Answer 3']
        ]

        # Create list of all options and shuffle them
        all_options = [correct_answer] + incorrect_answers
        random.shuffle(all_options)

        # Assign letters A, B, C, D
        options_text = "\n".join([
            f"{chr(65+i)}. {option}" for i, option in enumerate(all_options)
        ])

        # Format the problem with the question and options
        problem_text = f"{sample['Question']}\n\n{options_text}"

        processed.append({
            'problem': problem_text,
            'solution': sample['Explanation'],
            'answer': correct_answer,
            'data_source': 'gpqa',
            'data_fields': {
                'subdomain': sample['Subdomain'],
                'high_level_domain': sample['High-level domain'],
                'incorrect_answers': incorrect_answers
            }
        })

    return processed


def load_humaneval_dataset() -> List[Dict]:
    """Load and process the HumanEval dataset."""
    print("Loading HumanEval dataset from HuggingFace (openai/openai_humaneval)...")

    dataset = load_dataset("openai/openai_humaneval", split="test")

    processed = []
    for sample in tqdm(dataset, desc="Processing HumanEval"):
        processed.append({
            'problem': sample['prompt'],
            'solution': sample['canonical_solution'],
            'answer': sample['canonical_solution'],
            'data_source': 'humaneval',
            'data_fields': {
                'task_id': sample['task_id'],
                'entry_point': sample['entry_point'],
                'test': sample['test']
            }
        })

    return processed


def load_bigcodebench_dataset() -> List[Dict]:
    """Load and process the BigCodeBench dataset."""
    print("Loading BigCodeBench dataset from HuggingFace (bigcode/bigcodebench, v0.1.4)...")

    dataset = load_dataset("bigcode/bigcodebench", split="v0.1.4")

    processed = []
    for sample in tqdm(dataset, desc="Processing BigCodeBench"):
        # Combine complete_prompt with canonical_solution
        complete_code = sample['complete_prompt'] + sample['canonical_solution']

        processed.append({
            'problem': sample['complete_prompt'],
            'solution': complete_code,
            'answer': complete_code,
            'data_source': 'bigcodebench',
            'data_fields': {
                'task_id': sample['task_id'],
                'entry_point': sample['entry_point'],
                'libs': sample['libs'],
                'test': sample['test']
            }
        })

    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Gather test data from multiple sources into a unified dataset"
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
        default='test.jsonl',
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

    # 1. MATH-500 dataset
    math500_samples = load_math500_dataset()
    all_samples.extend(math500_samples)
    print(f"Collected {len(math500_samples)} samples from MATH-500")

    # 2. AIME 2025 dataset
    aime25_samples = load_aime25_dataset()
    all_samples.extend(aime25_samples)
    print(f"Collected {len(aime25_samples)} samples from AIME 2025")

    # 3. GPQA dataset
    gpqa_samples = load_gpqa_dataset()
    all_samples.extend(gpqa_samples)
    print(f"Collected {len(gpqa_samples)} samples from GPQA")

    # 4. HumanEval dataset
    humaneval_samples = load_humaneval_dataset()
    all_samples.extend(humaneval_samples)
    print(f"Collected {len(humaneval_samples)} samples from HumanEval")

#    # 5. BigCodeBench dataset
#    bigcodebench_samples = load_bigcodebench_dataset()
#    all_samples.extend(bigcodebench_samples)
#    print(f"Collected {len(bigcodebench_samples)} samples from BigCodeBench")

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
    from collections import defaultdict
    source_counts = defaultdict(int)
    for sample in all_samples:
        source_counts[sample['data_source']] += 1

    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")


if __name__ == '__main__':
    main()
