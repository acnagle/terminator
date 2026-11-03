#!/usr/bin/env python3
"""
Generate chain-of-thought (CoT) outputs with log probabilities and self-certainty scores.

Reads problems from ./data/inputs/dataset.jsonl and writes outputs to
./data/outputs/{model_name}/dataset_cots.jsonl with log probabilities and
self-certainty scores for each token.
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
import httpx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Problem:
    """Represents a single problem from the dataset."""
    sample_idx: int
    problem: str
    solution: Optional[str]
    answer: str
    data_source: str
    data_fields: Dict[str, Any]
    original_data: Dict[str, Any]  # Store all original fields


@dataclass
class InferenceResult:
    """Represents inference results for a problem."""
    sample_idx: int
    sampled_solutions: List[Dict[str, Any]]
    failed_solutions: List[Dict[str, Any]]
    original_data: Dict[str, Any]  # All original fields from input


class CoTInferenceRunner:
    """Handles batched async inference with log probabilities."""

    def __init__(
        self,
        api_base: str,
        served_name: str,
        output_file: Path,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 1.5,
        max_tokens: int = 32768,
        enable_thinking: bool = True,
        batch_size: int = 10,
        samples_per_problem: int = 3,
        max_failures: int = 10,
        max_retries: int = 3,
        aime_2025_samples_override: Optional[int] = None,
        request_timeout: float = 600.0,
    ):
        self.client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=api_base,
            timeout=request_timeout,
        )
        self.served_name = served_name
        self.output_file = output_file
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.batch_size = batch_size
        self.samples_per_problem = samples_per_problem
        self.max_failures = max_failures
        self.max_retries = max_retries
        self.aime_2025_samples_override = aime_2025_samples_override

        # For tracking progress
        self.completed_indices = self._load_completed_indices()

    def _load_completed_indices(self) -> set:
        """Load already completed problem indices from output file."""
        if not self.output_file.exists():
            return set()

        completed = set()
        try:
            with open(self.output_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        completed.add(data['sample_idx'])
            logger.info(f"Loaded {len(completed)} completed problems from {self.output_file}")
        except Exception as e:
            logger.warning(f"Error loading completed indices: {e}")

        return completed

    def _format_prompt(self, problem: Problem) -> str:
        """Format the problem with instructions for the model."""
        if problem.data_source == "opencoder":
            return f"{problem.problem}\n\nPlease reason step by step and give your code in Python. Use Python's standard library unless otherwise stated."
        elif problem.data_source == "humaneval" or problem.data_source == "bigcodebench":
            return f"{problem.problem}\n\nPlease complete the following function. Please reason step by step and give your code in Python. Use Python's standard library unless otherwise stated."
        else:
            return f"{problem.problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

    def _format_prompt_nothink(self, problem: Problem) -> str:
        """Format the problem with instructions for the model (no reasoning instruction)."""
        if problem.data_source == "opencoder":
            return f"{problem.problem}\n\nGive your code in Python. Use Python's standard library unless otherwise stated."
        elif problem.data_source == "humaneval" or problem.data_source == "bigcodebench":
            return f"{problem.problem}\n\nPlease complete the following function. Give your code in Python. Use Python's standard library unless otherwise stated."
        else:
            return f"{problem.problem}\n\nPut your final answer within \\boxed{{}}."

    def _compute_self_certainty(self, top_logprobs_list: List[Dict[str, float]]) -> List[float]:
        """
        Compute self-certainty score for each token.

        self_certainty[i] = -(1/top_k) * sum(log p_i(j) for j in top_k)

        Args:
            top_logprobs_list: List of dicts where each dict maps token -> logprob

        Returns:
            List of self-certainty scores (one per token)
        """
        self_certainty = []

        for top_logprobs_dict in top_logprobs_list:
            # Get the top-k log probabilities (they're already limited to top_k by the API)
            logprobs = list(top_logprobs_dict.values())

            # Compute the uniform average of the negative log probs
            # Note: logprobs are already log probabilities (negative values)
            # So we need to negate them to get negative log probs
            neg_logprobs = [-lp for lp in logprobs]

            # Uniform average
            avg_neg_logprob = sum(neg_logprobs) / len(neg_logprobs)

            self_certainty.append(avg_neg_logprob)

        return self_certainty

    async def _generate_single_solution(
        self,
        problem: Problem,
        retry_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Generate a single solution for a problem with retry logic."""
        try:
            if self.enable_thinking:
                prompt = self._format_prompt(problem)
            else:
                prompt = self._format_prompt_nothink(problem)

            extra_body = {
                "top_k": self.top_k,
                "min_p": self.min_p,
            }

            # Add thinking mode if needed
            extra_body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

            response = await self.client.chat.completions.create(
                model=self.served_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                presence_penalty=self.presence_penalty,
                logprobs=True,  # Request log probabilities
                top_logprobs=self.top_k,  # Request top-k log probs
                extra_body=extra_body,
            )

            # Extract response content
            message = response.choices[0].message
            result = {
                "content": message.content,
                "finish_reason": response.choices[0].finish_reason,
            }

            # Add reasoning content if available (when using --reasoning-parser qwen3)
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                result["reasoning_content"] = message.reasoning_content

            # Extract log probabilities
            logprobs_data = response.choices[0].logprobs
            if logprobs_data and logprobs_data.content:
                # Extract negative log probs for each token (selected token)
                log_probs = []
                top_logprobs_list = []

                for token_data in logprobs_data.content:
                    # Negative log prob of the selected token
                    log_probs.append(-token_data.logprob)

                    # Top-k log probs for this position
                    top_logprobs_dict = {}
                    if token_data.top_logprobs:
                        for top_token in token_data.top_logprobs:
                            top_logprobs_dict[top_token.token] = top_token.logprob

                    top_logprobs_list.append(top_logprobs_dict)

                result["log_probs"] = log_probs

                # Compute self-certainty scores
                result["self_certainty"] = self._compute_self_certainty(top_logprobs_list)
            else:
                # If logprobs not available, set to None
                result["log_probs"] = None
                result["self_certainty"] = None
                logger.warning(f"No log probabilities returned for sample_idx {problem.sample_idx}")

            return result

        except Exception as e:
            logger.error(f"Error generating solution for sample_idx {problem.sample_idx} (attempt {retry_count + 1}): {e}")

            if retry_count < self.max_retries:
                logger.info(f"Retrying sample_idx {problem.sample_idx}...")
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                return await self._generate_single_solution(problem, retry_count + 1)
            else:
                logger.error(f"Max retries reached for sample_idx {problem.sample_idx}")
                return None

    async def _process_single_problem(self, problem: Problem) -> InferenceResult:
        """Process a single problem with multiple samples in parallel."""
        successful_solutions = []
        failed_solutions = []
        pending_tasks = set()

        # Determine samples_per_problem for this specific problem
        samples_per_problem = self.samples_per_problem
        if problem.data_source == 'aime_2025' and self.aime_2025_samples_override is not None:
            samples_per_problem = self.aime_2025_samples_override
            logger.info(f"sample_idx {problem.sample_idx}: Using override for aime_2025 - {samples_per_problem} samples")

        # Launch initial batch of tasks
        for _ in range(samples_per_problem):
            task = asyncio.create_task(self._generate_single_solution(problem))
            pending_tasks.add(task)

        # Process completed tasks and launch new ones as needed
        while pending_tasks:
            # Wait for at least one task to complete
            done, pending_tasks = await asyncio.wait(
                pending_tasks,
                return_when=asyncio.FIRST_COMPLETED
            )

            # Process completed tasks
            for task in done:
                solution = await task

                if solution is None:
                    # Complete failure (couldn't get response after retries)
                    failed_solutions.append({
                        "content": None,
                        "finish_reason": "error",
                        "log_probs": None,
                        "self_certainty": None,
                    })
                elif solution.get("finish_reason") == "stop":
                    # Successful completion
                    successful_solutions.append(solution)
                else:
                    # Failed completion (e.g., length limit, other finish reasons)
                    failed_solutions.append(solution)

            # Check if we've hit either limit
            if len(successful_solutions) >= samples_per_problem:
                # Got enough successes, cancel pending tasks
                for task in pending_tasks:
                    task.cancel()
                break

            if len(failed_solutions) >= self.max_failures:
                # Hit failure limit, cancel pending tasks
                for task in pending_tasks:
                    task.cancel()
                break

            # Launch new tasks to maintain parallelism
            # Calculate how many more we could potentially need
            max_possible_successes = len(successful_solutions) + len(pending_tasks)
            max_possible_failures = len(failed_solutions) + len(pending_tasks)

            # Keep launching tasks until we'd hit a limit if they all went one way
            while (max_possible_successes < samples_per_problem and
                   max_possible_failures < self.max_failures):
                task = asyncio.create_task(self._generate_single_solution(problem))
                pending_tasks.add(task)
                max_possible_successes += 1
                max_possible_failures += 1

        # Log final status
        if len(successful_solutions) < samples_per_problem:
            logger.warning(
                f"sample_idx {problem.sample_idx}: Stopped early with {len(successful_solutions)}/{samples_per_problem} "
                f"successful solutions due to reaching {len(failed_solutions)} failures (max: {self.max_failures})"
            )
        else:
            logger.info(
                f"sample_idx {problem.sample_idx}: Successfully generated {len(successful_solutions)} solutions "
                f"with {len(failed_solutions)} failures"
            )

        return InferenceResult(
            sample_idx=problem.sample_idx,
            sampled_solutions=successful_solutions,
            failed_solutions=failed_solutions,
            original_data=problem.original_data,
        )

    def _save_result(self, result: InferenceResult):
        """Save a single result to the output file, preserving all original fields."""
        # Merge original data with new fields
        output_data = {**result.original_data}
        output_data['sampled_solutions'] = result.sampled_solutions
        output_data['failed_solutions'] = result.failed_solutions

        with open(self.output_file, 'a') as f:
            json.dump(output_data, f)
            f.write('\n')

    async def run_inference(self, problems: List[Problem]):
        """Run inference on all problems with a worker pool pattern."""
        # Filter out already completed problems
        problems_to_process = [
            p for p in problems
            if p.sample_idx not in self.completed_indices
        ]

        if not problems_to_process:
            logger.info("All problems already completed!")
            return

        logger.info(
            f"Processing {len(problems_to_process)} problems "
            f"({len(self.completed_indices)} already completed)"
        )

        # Use a semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(self.batch_size)

        async def process_with_semaphore(problem: Problem) -> InferenceResult:
            """Process a single problem with semaphore control."""
            async with semaphore:
                result = await self._process_single_problem(problem)
                # Save immediately after processing
                self._save_result(result)
                self.completed_indices.add(result.sample_idx)
                logger.info(f"Completed sample_idx {result.sample_idx} - Progress: {len(self.completed_indices)}/{len(problems)}")
                return result

        # Create tasks for all problems - the semaphore ensures only batch_size run at once
        tasks = [process_with_semaphore(problem) for problem in problems_to_process]

        # Process all tasks with the semaphore controlling concurrency
        await asyncio.gather(*tasks)

        logger.info(f"All problems completed! Total: {len(self.completed_indices)}/{len(problems)}")


def load_problems(input_file: Path, limit: Optional[int] = None) -> List[Problem]:
    """Load problems from JSONL file, preserving all fields."""
    problems = []

    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            if not line.strip():
                continue

            data = json.loads(line)

            # Extract required fields
            problem = Problem(
                sample_idx=data.get('sample_idx', i),
                problem=data['problem'],
                solution=data.get('solution'),
                answer=data.get('answer', ''),
                data_source=data.get('data_source', 'unknown'),
                data_fields=data.get('data_fields', {}),
                original_data=data,  # Preserve all original fields
            )
            problems.append(problem)

    logger.info(f"Loaded {len(problems)} problems from {input_file}")
    return problems


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


def main():
    parser = argparse.ArgumentParser(
        description="Generate CoTs with log probabilities and self-certainty scores"
    )

    # Required arguments
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Model name (used for output directory path)'
    )
    parser.add_argument(
        '--input-file',
        type=Path,
        default=Path('./data/inputs/dataset.jsonl'),
        help='Path to input JSONL file (default: ./data/inputs/dataset.jsonl)'
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
        help='Served name of the model in vLLM. This is the same as the --served-model-name used when starting vLLM.'
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
        help='Top-k sampling parameter (also used for self-certainty computation)'
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
    parser.add_argument(
        '--enable-thinking',
        action='store_true',
        default=True,
        help='Enable thinking mode'
    )
    parser.add_argument(
        '--disable-thinking',
        action='store_true',
        help='Disable thinking mode'
    )

    # Inference configuration
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Number of problems to process concurrently'
    )
    parser.add_argument(
        '--samples-per-problem',
        type=int,
        default=3,
        help='Number of successful solutions to generate per problem'
    )
    parser.add_argument(
        '--aime-2025-samples-override',
        type=int,
        default=None,
        help='Override --samples-per-problem when data_source is aime_2025 (default: None, no override)'
    )
    parser.add_argument(
        '--max-failures',
        type=int,
        default=10,
        help='Maximum number of failed solutions before stopping sampling for a problem'
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
        help='Limit number of problems to process (for testing)'
    )
    parser.add_argument(
        '--server-timeout',
        type=int,
        default=300,
        help='Maximum time to wait for vLLM server to be ready (in seconds, default: 300 = 5 minutes)'
    )
    parser.add_argument(
        '--request-timeout',
        type=float,
        default=600.0,
        help='HTTP request timeout for inference requests (in seconds, default: 600 = 10 minutes)'
    )

    args = parser.parse_args()

    # Handle thinking mode flag
    enable_thinking = args.enable_thinking and not args.disable_thinking

    # Construct output path based on model name and input filename
    output_dir = Path(f'./data/outputs/{args.model_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    # Derive output filename from input filename by appending _cots
    input_stem = args.input_file.stem  # e.g., "dataset" from "dataset.jsonl"
    if enable_thinking:
        output_filename = f"{input_stem}_cots.jsonl"  # e.g., "dataset_cots.jsonl"
    else:
        output_filename = f"{input_stem}_cots_nothink.jsonl"  # e.g., "dataset_cots_nothink.jsonl"
    output_file = output_dir / output_filename

    # Wait for server to be ready
    if not wait_for_server(args.api_base, timeout=args.server_timeout):
        logger.error("Server is not ready. Exiting.")
        return 1

    # Load problems
    problems = load_problems(args.input_file, args.limit)

    if not problems:
        logger.error("No problems loaded!")
        return 1

    # Create runner
    runner = CoTInferenceRunner(
        api_base=args.api_base,
        served_name=args.served_name,
        output_file=output_file,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        max_tokens=args.max_tokens,
        enable_thinking=enable_thinking,
        batch_size=args.batch_size,
        samples_per_problem=args.samples_per_problem,
        max_failures=args.max_failures,
        max_retries=args.max_retries,
        aime_2025_samples_override=args.aime_2025_samples_override,
        request_timeout=args.request_timeout,
    )

    # Run inference
    logger.info("Starting inference...")
    logger.info(f"Configuration:")
    logger.info(f"  Model name: {args.model_name}")
    logger.info(f"  Served name: {args.served_name}")
    logger.info(f"  Input file: {args.input_file}")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Top-p: {args.top_p}")
    logger.info(f"  Top-k: {args.top_k}")
    logger.info(f"  Enable thinking: {enable_thinking}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Samples per problem: {args.samples_per_problem}")
    if args.aime_2025_samples_override is not None:
        logger.info(f"  AIME 2025 samples override: {args.aime_2025_samples_override}")
    logger.info(f"  Max failures per problem: {args.max_failures}")
    logger.info(f"  Request timeout: {args.request_timeout}s")

    asyncio.run(runner.run_inference(problems))

    logger.info(f"Inference complete! Results saved to {output_file}")


if __name__ == '__main__':
    main()
