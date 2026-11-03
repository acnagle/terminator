#!/usr/bin/env python3
"""
Split input JSONL file into N partitions for multi-node processing.

This script:
1. Reads input JSONL file
2. Optionally checks output file for already-completed samples
3. Filters out completed samples
4. Sorts remaining samples by sample_idx (for deterministic partitioning)
5. Splits into N equal-sized partitions
6. Writes partition files: {input_stem}_partition_{i}.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Set


def load_completed_ids(output_path: Path) -> Set[int]:
    """Load the set of sample_idx values that have already been processed."""
    if not output_path.exists():
        return set()

    processed_ids = set()
    with open(output_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    sample_idx = data.get("sample_idx")
                    if sample_idx is not None and "solution_analyses" in data:
                        processed_ids.add(sample_idx)
                except json.JSONDecodeError:
                    continue

    return processed_ids


def load_samples(input_path: Path) -> List[Dict[str, Any]]:
    """Load all samples from input JSONL file."""
    samples = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}", file=sys.stderr)
                    continue
    return samples


def split_into_partitions(samples: List[Dict[str, Any]], num_partitions: int) -> List[List[Dict[str, Any]]]:
    """
    Split samples into N equal-sized partitions using round-robin distribution.

    Samples are first sorted by sample_idx for deterministic partitioning.
    Round-robin ensures balanced distribution:
    - For 1000 samples and 8 partitions: [125, 125, 125, 125, 125, 125, 125, 125]
    """
    # Sort for deterministic partitioning
    samples_sorted = sorted(samples, key=lambda x: x.get("sample_idx", 0))

    # Initialize partitions
    partitions = [[] for _ in range(num_partitions)]

    # Round-robin distribution
    for i, sample in enumerate(samples_sorted):
        partition_idx = i % num_partitions
        partitions[partition_idx].append(sample)

    return partitions


def write_partition(partition: List[Dict[str, Any]], output_path: Path):
    """Write partition to JSONL file."""
    with open(output_path, "w") as f:
        for sample in partition:
            f.write(json.dumps(sample) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Split input JSONL into N partitions for multi-node processing"
    )
    ap.add_argument("--input-file", type=str, required=True,
                    help="Path to input JSONL file")
    ap.add_argument("--num-partitions", type=int, required=True,
                    help="Number of partitions to create (typically equals number of nodes)")
    ap.add_argument("--output-file", type=str, default=None,
                    help="Optional: Path to check for already-completed samples (e.g., existing _extracted.jsonl)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Directory to write partition files (default: same as input file)")

    args = ap.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_path.parent

    # Load samples
    print(f"Loading samples from {input_path}...")
    all_samples = load_samples(input_path)
    print(f"Loaded {len(all_samples)} samples")

    # Filter out completed samples if output file provided
    if args.output_file:
        output_path = Path(args.output_file)
        print(f"Checking for completed samples in {output_path}...")
        completed_ids = load_completed_ids(output_path)
        print(f"Found {len(completed_ids)} completed samples")

        remaining_samples = [
            sample for sample in all_samples
            if sample.get("sample_idx") not in completed_ids
        ]
        print(f"Remaining samples to process: {len(remaining_samples)}")
    else:
        remaining_samples = all_samples
        print("No output file specified, processing all samples")

    if not remaining_samples:
        print("No samples to process!")
        sys.exit(0)

    # Split into partitions
    print(f"\nSplitting into {args.num_partitions} partitions...")
    partitions = split_into_partitions(remaining_samples, args.num_partitions)

    # Write partition files
    partition_paths = []
    for i, partition in enumerate(partitions):
        partition_filename = f"{input_path.stem}_partition_{i}.jsonl"
        partition_path = output_dir / partition_filename
        write_partition(partition, partition_path)
        partition_paths.append(partition_path)
        print(f"  Partition {i}: {len(partition)} samples → {partition_path}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Total samples in input: {len(all_samples)}")
    if args.output_file:
        print(f"Completed samples: {len(completed_ids)}")
    print(f"Samples to process: {len(remaining_samples)}")
    print(f"Partitions created: {args.num_partitions}")
    print(f"Samples per partition: min={min(len(p) for p in partitions)}, max={max(len(p) for p in partitions)}")
    print(f"\nPartition files written to: {output_dir}")


if __name__ == "__main__":
    main()
