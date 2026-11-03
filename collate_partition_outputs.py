#!/usr/bin/env python3
"""
Collate partition outputs from multi-node span extraction.

This script:
1. Reads all partition output files (_partition_*_extracted.jsonl)
2. Merges them into a single output file
3. Optionally sorts by original input order (if --original-input provided)
4. Checks for duplicate sample_idx values and reports them
"""

import argparse
import json
import sys
from glob import glob
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_partition_outputs(pattern: str) -> List[Dict[str, Any]]:
    """
    Load all partition outputs matching the pattern.

    Args:
        pattern: Glob pattern for partition files (e.g., "partitions/*_partition_*_analyzed.jsonl")

    Returns:
        List of all samples from all partitions
    """
    partition_files = sorted(glob(pattern))

    if not partition_files:
        print(f"Error: No files found matching pattern: {pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(partition_files)} partition files:")
    for pf in partition_files:
        print(f"  {pf}")

    all_samples = []
    sample_ids_seen = {}  # Track duplicates

    for partition_file in partition_files:
        print(f"\nReading {partition_file}...")
        count = 0

        with open(partition_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        sample = json.loads(line)
                        sample_idx = sample.get("sample_idx")

                        # Check for duplicates
                        if sample_idx is not None and sample_idx in sample_ids_seen:
                            print(f"  WARNING: Duplicate sample_idx found: {sample_idx}", file=sys.stderr)
                            print(f"    First seen in: {sample_ids_seen[sample_idx]}", file=sys.stderr)
                            print(f"    Duplicate in: {partition_file}", file=sys.stderr)
                        elif sample_idx is not None:
                            sample_ids_seen[sample_idx] = partition_file

                        all_samples.append(sample)
                        count += 1
                    except json.JSONDecodeError as e:
                        print(f"  Warning: Failed to parse line in {partition_file}: {e}", file=sys.stderr)

        print(f"  Loaded {count} samples")

    print(f"\nTotal samples loaded: {len(all_samples)}")
    return all_samples


def load_original_order(original_input: Path) -> Dict[int, int]:
    """
    Load the original order of sample_idx values from the input file.

    Args:
        original_input: Path to original input JSONL file

    Returns:
        Dict mapping sample_idx to its original position
    """
    order_map = {}
    with open(original_input, "r") as f:
        for idx, line in enumerate(f):
            if line.strip():
                try:
                    sample = json.loads(line)
                    sample_idx = sample.get("sample_idx")
                    if sample_idx is not None:
                        order_map[sample_idx] = idx
                except json.JSONDecodeError:
                    continue

    return order_map


def write_output(samples: List[Dict[str, Any]], output_path: Path, sort_order: Optional[Dict[int, int]] = None):
    """
    Write samples to output file.

    Args:
        samples: List of samples to write
        output_path: Path to output file
        sort_order: Optional dict mapping sample_idx to original position (for sorting)
    """
    # Sort if order provided
    if sort_order is not None:
        print(f"\nSorting by original input order...")
        # Samples not in original order go to the end
        samples_sorted = sorted(
            samples,
            key=lambda s: sort_order.get(s.get("sample_idx"), float('inf'))
        )
    else:
        samples_sorted = samples

    # Write output
    print(f"\nWriting {len(samples_sorted)} samples to {output_path}...")
    with open(output_path, "w") as f:
        for sample in samples_sorted:
            f.write(json.dumps(sample) + "\n")

    print(f"✓ Output written: {output_path}")


def print_statistics(samples: List[Dict[str, Any]]):
    """Print summary statistics from collated results."""
    total_solutions = sum(len(entry.get("solution_analyses", [])) for entry in samples)

    final_correct = sum(
        1 for entry in samples
        for analysis in entry.get("solution_analyses", [])
        if analysis.get("llm_final_answer_correct", False)
    )

    # Final answer span statistics
    span_final_verified = sum(
        1 for entry in samples
        for analysis in entry.get("solution_analyses", [])
        if analysis.get("llm_span_final_verified", False)
    )

    spans_final_with_tokens = sum(
        1 for entry in samples
        for analysis in entry.get("solution_analyses", [])
        if analysis.get("span_final_start_token") is not None
    )

    print("\n=== Summary Statistics ===")
    print(f"Total problems: {len(samples)}")
    print(f"Total solutions: {total_solutions}")
    if total_solutions > 0:
        print(f"Final answers correct: {final_correct}/{total_solutions} ({100*final_correct/total_solutions:.1f}%)")
        print(f"\nFinal Answer Spans:")
        print(f"  Spans verified: {span_final_verified}/{total_solutions} ({100*span_final_verified/total_solutions:.1f}%)")
        print(f"  Spans with token positions: {spans_final_with_tokens}/{total_solutions} ({100*spans_final_with_tokens/total_solutions:.1f}%)")
    else:
        print("No solutions found in output!")


def main():
    ap = argparse.ArgumentParser(
        description="Collate partition outputs from multi-node span extraction"
    )

    # Input specification (mutually exclusive)
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-pattern", type=str,
                            help="Glob pattern for partition files (e.g., 'partitions/*_partition_*_analyzed.jsonl')")
    input_group.add_argument("--input-files", type=str, nargs="+",
                            help="List of partition files to collate")

    # Output
    ap.add_argument("--output-file", type=str, required=True,
                    help="Path to output file")

    # Optional: original input for sorting
    ap.add_argument("--original-input", type=str, default=None,
                    help="Optional: Path to original input JSONL file (for sorting by original order)")

    args = ap.parse_args()

    # Load partition outputs
    if args.input_pattern:
        print(f"Loading partition files matching: {args.input_pattern}")
        all_samples = load_partition_outputs(args.input_pattern)
    else:
        print(f"Loading {len(args.input_files)} partition files...")
        # Convert file list to temporary pattern that glob can handle
        all_samples = []
        for pf in args.input_files:
            print(f"  Reading {pf}...")
            with open(pf, "r") as f:
                for line in f:
                    if line.strip():
                        all_samples.append(json.loads(line))

    # Load original order if provided
    sort_order = None
    if args.original_input:
        original_path = Path(args.original_input)
        if not original_path.exists():
            print(f"Warning: Original input file not found: {original_path}", file=sys.stderr)
            print("Proceeding without sorting...", file=sys.stderr)
        else:
            print(f"\nLoading original order from {original_path}...")
            sort_order = load_original_order(original_path)
            print(f"Loaded order for {len(sort_order)} sample_idx values")

    # Write output
    output_path = Path(args.output_file)
    write_output(all_samples, output_path, sort_order)

    # Print statistics
    print_statistics(all_samples)


if __name__ == "__main__":
    main()
