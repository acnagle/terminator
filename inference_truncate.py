#!/usr/bin/env python3
"""
Truncation inference script that truncates CoT reasoning relative to span_final_end_token.

This script processes solution_analyses from test_cots_extracted.jsonl, truncates the
reasoning_content at positions relative to span_final_end_token (either before or after),
and generates new final answers using vLLM.
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from transformers import AutoTokenizer
import httpx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TruncationTask:
    """Represents a single truncation task."""
    sample_idx: int
    solution_index: int
    problem: str
    answer: str
    solution: str
    data_source: str
    data_fields: Dict[str, Any]
    reasoning_content: str
    span_final_end_token: int
    cutoff_ratio: float
    cutoff_type: str  # "before" or "after"
    unique_id: str  # For tracking completion


def format_prompt(problem: str, data_source: str) -> str:
    """Format the problem with instructions (from gather_cots.py)."""
    if data_source == "opencoder":
        return f"{problem}\n\nPlease reason step by step and give your code in Python. Use Python's standard library unless otherwise stated."
    elif data_source == "humaneval" or data_source == "bigcodebench":
        return f"{problem}\n\nPlease complete the following function. Please reason step by step and give your code in Python. Use Python's standard library unless otherwise stated."
    else:
        return f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


class TruncationInferenceRunner:
    """Handles truncated reasoning inference."""

    def __init__(
        self,
        api_base: str,
        served_name: str,
        output_file: Path,
        tokenizer_name: str,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 1.5,
        max_tokens: int = 32768,
        batch_size: int = 10,
        max_retries: int = 3,
        exit_message: Optional[str] = None,
    ):
        self.client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=api_base,
        )
        self.served_name = served_name
        self.output_file = output_file
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.exit_message = exit_message

        # Load tokenizer for token counting
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        # For tracking progress
        self.completed_ids = self._load_completed_ids()

    def _load_completed_ids(self) -> set:
        """Load already completed task IDs from output file."""
        if not self.output_file.exists():
            return set()

        completed = set()
        try:
            with open(self.output_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        completed.add(data['unique_id'])
            logger.info(f"Loaded {len(completed)} completed tasks from {self.output_file}")
        except Exception as e:
            logger.warning(f"Error loading completed IDs: {e}")

        return completed

    def _truncate_reasoning(
        self,
        reasoning_content: str,
        span_final_end_token: int,
        cutoff_ratio: float,
        cutoff_type: str
    ) -> tuple[str, int, int]:
        """
        Truncate reasoning content relative to span_final_end_token.

        Args:
            reasoning_content: Full reasoning text
            span_final_end_token: Token index marking the reference point
            cutoff_ratio: Float between 0 and 1
            cutoff_type: "before" or "after"

        Returns:
            Tuple of (truncated_text, truncation_token_idx, total_tokens)
        """
        # Tokenize the full reasoning
        tokens = self.tokenizer.encode(reasoning_content, add_special_tokens=False)
        total_tokens = len(tokens)

        # Calculate truncation point
        if cutoff_type == "before":
            # Cut off at cutoff_ratio% of the way from start to span_final_end_token
            truncation_idx = int(cutoff_ratio * span_final_end_token)
        else:  # "after"
            # Cut off at cutoff_ratio% of the way from span_final_end_token to end
            remaining = total_tokens - span_final_end_token
            truncation_idx = span_final_end_token + int(cutoff_ratio * remaining)

        # Ensure truncation_idx is within bounds
        truncation_idx = max(0, min(truncation_idx, total_tokens))

        # Truncate and decode
        truncated_tokens = tokens[:truncation_idx]
        truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=False)

        return truncated_text, truncation_idx, total_tokens

    async def _generate_continuation(
        self,
        problem: str,
        data_source: str,
        truncated_reasoning: str,
        retry_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Generate continuation from truncated reasoning with retry logic."""
        try:
            # Format the prompt (from gather_cots.py)
            formatted_prompt = format_prompt(problem, data_source)

            # Construct the full prompt with truncated reasoning
            # The reasoning content is inside <think> tags, then we add </think> for the model to continue
            if self.exit_message:
                thinking_content = f"<think>\n{truncated_reasoning}\n{self.exit_message}\n</think>"
            else:
                thinking_content = f"<think>\n{truncated_reasoning}\n</think>"
            full_prompt = f"{formatted_prompt}\n{thinking_content}"

            extra_body = {
                "top_k": self.top_k,
                "min_p": self.min_p,
                # Disable thinking mode since we're providing the thinking content manually
                "chat_template_kwargs": {"enable_thinking": False},
            }

            response = await self.client.chat.completions.create(
                model=self.served_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                presence_penalty=self.presence_penalty,
                extra_body=extra_body,
            )

            # Extract response content
            message = response.choices[0].message
            result = {
                "content": message.content,
                "finish_reason": response.choices[0].finish_reason,
            }

            return result

        except Exception as e:
            logger.error(f"Error generating continuation (attempt {retry_count + 1}): {e}")

            if retry_count < self.max_retries:
                logger.info(f"Retrying...")
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                return await self._generate_continuation(
                    problem, data_source, truncated_reasoning, retry_count + 1
                )
            else:
                logger.error(f"Max retries reached")
                return None

    async def _process_single_task(self, task: TruncationTask) -> Dict[str, Any]:
        """Process a single truncation task."""
        logger.info(f"Processing {task.unique_id}")

        # Truncate the reasoning
        truncated_reasoning, truncation_idx, total_tokens = self._truncate_reasoning(
            task.reasoning_content,
            task.span_final_end_token,
            task.cutoff_ratio,
            task.cutoff_type
        )

        logger.info(
            f"  {task.unique_id}: Truncated at token {truncation_idx}/{total_tokens} "
            f"(span_final_end={task.span_final_end_token}, {task.cutoff_type} {task.cutoff_ratio:.1%})"
        )

        # Generate continuation
        result = await self._generate_continuation(
            task.problem,
            task.data_source,
            truncated_reasoning
        )

        if result is None:
            logger.error(f"  Failed to generate continuation for {task.unique_id}")
            content = None
            finish_reason = "error"
        else:
            content = result["content"]
            finish_reason = result["finish_reason"]
            logger.info(f"  ✓ {task.unique_id}: {finish_reason}")

        # Construct output record (following inference_terminator.py pattern)
        output = {
            # Preserved fields
            "sample_idx": task.sample_idx,
            "solution_index": task.solution_index,
            "problem": task.problem,
            "answer": task.answer,
            "solution": task.solution,
            "data_source": task.data_source,
            "data_fields": task.data_fields,

            # Truncation info
            "unique_id": task.unique_id,
            "cutoff_type": task.cutoff_type,
            "cutoff_ratio": task.cutoff_ratio,
            "span_final_end_token": task.span_final_end_token,
            "truncation_token_idx": truncation_idx,
            "original_reasoning_tokens": total_tokens,
            "exit_message": self.exit_message,

            # Generated fields
            "reasoning_content": truncated_reasoning,
            "content": content,
            "finish_reason": finish_reason,
        }

        return output

    def _save_result(self, result: Dict[str, Any]):
        """Save a single result to the output file."""
        with open(self.output_file, 'a') as f:
            json.dump(result, f)
            f.write('\n')

    async def run_inference(self, tasks: List[TruncationTask]):
        """Run inference on all tasks with a worker pool pattern."""
        # Filter out already completed tasks
        tasks_to_process = [
            t for t in tasks
            if t.unique_id not in self.completed_ids
        ]

        if not tasks_to_process:
            logger.info("All tasks already completed!")
            return

        logger.info(
            f"Processing {len(tasks_to_process)} tasks "
            f"({len(self.completed_ids)} already completed)"
        )

        # Use a semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(self.batch_size)

        async def process_with_semaphore(task: TruncationTask) -> Dict[str, Any]:
            """Process a single task with semaphore control."""
            async with semaphore:
                result = await self._process_single_task(task)
                # Save immediately after processing
                self._save_result(result)
                self.completed_ids.add(result['unique_id'])
                logger.info(
                    f"Completed {result['unique_id']} - "
                    f"Progress: {len(self.completed_ids)}/{len(tasks)}"
                )
                return result

        # Create tasks for all samples - the semaphore ensures only batch_size run at once
        async_tasks = [process_with_semaphore(task) for task in tasks_to_process]

        # Process all tasks with the semaphore controlling concurrency
        await asyncio.gather(*async_tasks)

        logger.info(f"All tasks completed! Total: {len(self.completed_ids)}/{len(tasks)}")


def load_samples_and_create_tasks(
    input_file: Path,
    cutoffs_before: List[float],
    cutoffs_after: List[float],
    data_sources: List[str],
    limit: Optional[int] = None
) -> Dict[str, List[TruncationTask]]:
    """
    Load samples from JSONL file and create truncation tasks.

    Returns:
        Dictionary mapping cutoff description to list of tasks
    """
    all_tasks = {}

    # Initialize task lists for each cutoff
    for ratio in cutoffs_before:
        key = f"before_{ratio}"
        all_tasks[key] = []

    for ratio in cutoffs_after:
        key = f"after_{ratio}"
        all_tasks[key] = []

    # Load samples
    samples_processed = 0
    samples_filtered = 0
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            if not line.strip():
                continue

            data = json.loads(line)
            sample_idx = data.get('sample_idx', i)

            # Filter by data source
            if data.get('data_source') not in data_sources:
                samples_filtered += 1
                continue

            # Process all solution_analyses
            solution_analyses = data.get('solution_analyses', [])

            for sol_analysis in solution_analyses:
                solution_index = sol_analysis.get('solution_index', 0)
                span_final_end_token = sol_analysis.get('span_final_end_token')

                # Check if span_final_end_token exists
                if span_final_end_token is None or span_final_end_token < 0:
                    logger.warning(
                        f"Skipping sample_idx={sample_idx}, solution_index={solution_index}: "
                        f"span_final_end_token is {span_final_end_token}"
                    )
                    continue

                reasoning_content = sol_analysis.get('reasoning_content', '')
                if not reasoning_content:
                    logger.warning(
                        f"Skipping sample_idx={sample_idx}, solution_index={solution_index}: "
                        f"reasoning_content is empty"
                    )
                    continue

                # Create tasks for "before" cutoffs
                for ratio in cutoffs_before:
                    unique_id = f"{sample_idx}_{solution_index}_before_{ratio}"
                    task = TruncationTask(
                        sample_idx=sample_idx,
                        solution_index=solution_index,
                        problem=data.get('problem', ''),
                        answer=data.get('answer', ''),
                        solution=data.get('solution', ''),
                        data_source=data.get('data_source', 'unknown'),
                        data_fields=data.get('data_fields', {}),
                        reasoning_content=reasoning_content,
                        span_final_end_token=span_final_end_token,
                        cutoff_ratio=ratio,
                        cutoff_type="before",
                        unique_id=unique_id
                    )
                    all_tasks[f"before_{ratio}"].append(task)

                # Create tasks for "after" cutoffs
                for ratio in cutoffs_after:
                    unique_id = f"{sample_idx}_{solution_index}_after_{ratio}"
                    task = TruncationTask(
                        sample_idx=sample_idx,
                        solution_index=solution_index,
                        problem=data.get('problem', ''),
                        answer=data.get('answer', ''),
                        solution=data.get('solution', ''),
                        data_source=data.get('data_source', 'unknown'),
                        data_fields=data.get('data_fields', {}),
                        reasoning_content=reasoning_content,
                        span_final_end_token=span_final_end_token,
                        cutoff_ratio=ratio,
                        cutoff_type="after",
                        unique_id=unique_id
                    )
                    all_tasks[f"after_{ratio}"].append(task)

            samples_processed += 1

    logger.info(f"Loaded {samples_processed} samples from {input_file}")
    if samples_filtered > 0:
        logger.info(f"Filtered out {samples_filtered} samples not matching data sources: {data_sources}")
    for key, tasks in all_tasks.items():
        logger.info(f"  {key}: {len(tasks)} tasks")

    return all_tasks


def wait_for_server(api_base: str, timeout: int = 300) -> bool:
    """
    Wait for vLLM server to be ready.

    Args:
        api_base: Base URL of the vLLM server (e.g., 'http://localhost:8000/v1')
        timeout: Maximum time to wait in seconds (default: 300 = 5 minutes)

    Returns:
        True if server is ready, False if timeout occurred
    """
    # Extract base URL without /v1 suffix for health endpoint
    health_url = api_base.rstrip('/v1').rstrip('/') + '/health'

    logger.info(f"Waiting for vLLM server at {health_url} (timeout: {timeout}s)")

    start_time = time.time()
    attempt = 0

    while time.time() - start_time < timeout:
        attempt += 1
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(health_url)
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    logger.info(f"✓ Server is ready! (took {elapsed:.1f}s, {attempt} attempts)")
                    return True
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout) as e:
            # Server not ready yet, continue waiting
            pass
        except Exception as e:
            logger.warning(f"Unexpected error checking server health: {e}")

        # Log progress every 30 seconds
        elapsed = time.time() - start_time
        if attempt == 1 or (attempt % 6 == 0):  # First attempt and every ~30s
            remaining = timeout - elapsed
            logger.info(f"Server not ready yet... (elapsed: {elapsed:.1f}s, remaining: {remaining:.1f}s)")

        time.sleep(5)  # Wait 5 seconds between checks

    logger.error(f"✗ Server failed to become ready within {timeout}s")
    return False


def generate_output_filename(
    input_file: Path,
    cutoff_type: str,
    cutoff_ratio: float,
    has_exit_message: bool = True
) -> Path:
    """
    Generate output filename based on input file and cutoff parameters.

    Examples:
        test_cots_extracted.jsonl + before + 0.5 + exit_msg -> test_cots_extracted_truncate_before_0.5_exitmsg.jsonl
        test_cots_extracted.jsonl + after + 0.25 + no exit_msg -> test_cots_extracted_truncate_after_0.25.jsonl
    """
    stem = input_file.stem  # e.g., "test_cots_extracted"
    parent = input_file.parent

    if has_exit_message:
        output_name = f"{stem}_truncate_{cutoff_type}_{cutoff_ratio}_exitmsg.jsonl"
    else:
        output_name = f"{stem}_truncate_{cutoff_type}_{cutoff_ratio}.jsonl"

    return parent / output_name


def main():
    parser = argparse.ArgumentParser(
        description="Run truncation inference relative to span_final_end_token"
    )

    # Required arguments
    parser.add_argument(
        '--input-file',
        type=Path,
        required=True,
        help='Path to input JSONL file (e.g., test_cots_extracted.jsonl)'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        required=True,
        help='Tokenizer to use for counting tokens (e.g., "Qwen/Qwen3-8B")'
    )
    parser.add_argument(
        '--data-sources',
        type=str,
        nargs='+',
        required=True,
        choices=['math-500', 'aime_2025', 'humaneval', 'gpqa', 'bigcodebench'],
        help='List of data sources to process (e.g., --data-sources math-500 aime_2025)'
    )

    # Cutoff arguments
    parser.add_argument(
        '--cutoffs-before',
        type=float,
        nargs='*',
        default=[],
        help='List of cutoff ratios (0-1) BEFORE span_final_end_token (e.g., --cutoffs-before 0.25 0.5 0.75)'
    )
    parser.add_argument(
        '--cutoffs-after',
        type=float,
        nargs='*',
        default=[],
        help='List of cutoff ratios (0-1) AFTER span_final_end_token (e.g., --cutoffs-after 0.25 0.5 0.75)'
    )

    # API configuration
    parser.add_argument(
        '--api-base',
        type=str,
        default='http://localhost:8000/v1',
        help='vLLM API base URL'
    )
    parser.add_argument(
        '--served-name',
        type=str,
        default='vllm',
        help='Served name of the model in vLLM'
    )

    # Sampling parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.6,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Top-p sampling parameter'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=20,
        help='Top-k sampling parameter'
    )
    parser.add_argument(
        '--min-p',
        type=float,
        default=0.0,
        help='Min-p sampling parameter'
    )
    parser.add_argument(
        '--presence-penalty',
        type=float,
        default=1.5,
        help='Presence penalty to reduce repetitions'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=16384,
        help='Maximum tokens to generate'
    )

    # Inference configuration
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of samples to process concurrently'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum retries for failed requests'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of samples to process (for testing)'
    )
    parser.add_argument(
        '--server-timeout',
        type=int,
        default=600,
        help='Maximum time to wait for vLLM server to be ready (in seconds)'
    )
    parser.add_argument(
        '--exit-message',
        type=str,
        default="\nI've run out of thinking tokens. I need to commit to a final answer.",
        help='Message to insert before </think> in truncated reasoning. Use "" or "None" to disable.'
    )

    args = parser.parse_args()

    # Handle exit_message: convert empty string or "None" to Python None
    if args.exit_message == "" or args.exit_message == "None":
        args.exit_message = None

    # Validate cutoffs
    if not args.cutoffs_before and not args.cutoffs_after:
        parser.error("At least one of --cutoffs-before or --cutoffs-after must be specified")

    # Validate cutoff ranges
    for ratio in args.cutoffs_before + args.cutoffs_after:
        if not 0 <= ratio <= 1:
            parser.error(f"Cutoff ratio {ratio} must be between 0 and 1")

    # Log exit message status
    logger.info(f"Exit message: {repr(args.exit_message)}")

    # Wait for server to be ready
    if not wait_for_server(args.api_base, timeout=args.server_timeout):
        logger.error("Server is not ready. Exiting.")
        return 1

    # Load samples and create tasks
    logger.info(f"Loading samples from {args.input_file}")
    logger.info(f"Filtering for data sources: {args.data_sources}")
    all_tasks = load_samples_and_create_tasks(
        args.input_file,
        args.cutoffs_before,
        args.cutoffs_after,
        args.data_sources,
        args.limit
    )

    if not all_tasks or all(len(tasks) == 0 for tasks in all_tasks.values()):
        logger.error("No tasks created!")
        return 1

    # Process each cutoff separately (creates separate output files)
    for cutoff_key, tasks in all_tasks.items():
        if not tasks:
            logger.info(f"Skipping {cutoff_key}: no tasks")
            continue

        # Parse cutoff_key (e.g., "before_0.5" -> "before", 0.5)
        cutoff_type, cutoff_ratio_str = cutoff_key.split('_')
        cutoff_ratio = float(cutoff_ratio_str)

        # Generate output filename
        output_file = generate_output_filename(
            args.input_file, cutoff_type, cutoff_ratio,
            has_exit_message=(args.exit_message is not None)
        )
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {cutoff_type} {cutoff_ratio:.1%} cutoff")
        logger.info(f"Output will be saved to: {output_file}")
        logger.info(f"{'='*80}\n")

        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Create runner
        runner = TruncationInferenceRunner(
            api_base=args.api_base,
            served_name=args.served_name,
            output_file=output_file,
            tokenizer_name=args.tokenizer,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            exit_message=args.exit_message,
        )

        # Run inference
        logger.info(f"Starting inference for {cutoff_type} {cutoff_ratio:.1%}...")
        asyncio.run(runner.run_inference(tasks))

        logger.info(f"Inference complete for {cutoff_type} {cutoff_ratio:.1%}! Results saved to {output_file}\n")

    logger.info(f"\n{'='*80}")
    logger.info("All cutoffs processed successfully!")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
