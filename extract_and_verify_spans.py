#!/usr/bin/env python3
"""
Extract and verify final answer spans from dataset_cots.jsonl.

This script:
1. Verifies final_content with majority vote (2-3 attempts)
2. Extracts final answer span: where reasoning first arrives at the final answer
3. Pattern matches span to get token positions

Supports multiple dataset types:
- MATH/AIME: Mathematical problems with boxed answers
- OpenScience: Multiple choice STEM questions (A, B, C, D)
- OpenCoder: Coding problems (extract Python code)
"""

import argparse
import asyncio
import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

import aiohttp
from aiohttp import ClientSession, ClientTimeout, ClientError
from tqdm import tqdm
from transformers import AutoTokenizer


# ============================================================================
# PROMPTS
# ============================================================================

VERIFICATION_SYSTEM_PROMPT_MATH = """You are a mathematical answer verification assistant.

Your task: Determine if a given text contains a specific mathematical answer.

You will be given:
1. **Answer**: The answer (may be in LaTeX, plain text, or other formats)
2. **Text to Check**: An excerpt from a solution

**Answer Equivalence Guidelines:**
- Mathematical equivalence matters: "1/2" = "0.5" = "\\frac{1}{2}"
- LaTeX formatting: "\\frac{a}{b}" is a fraction, "\\sqrt{x}" is square root
- Matrices: "\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}" = "[[a, b], [c, d]]"
- The answer might appear mid-calculation: "z = i/5" contains "\\frac{i}{5}"
- Equivalent algebraic forms: "x^2 - 1" = "(x-1)(x+1)"
- Be GENEROUS: if the answer is mathematically present in any reasonable form, it counts

IMPORTANT: YOU ARE GIVEN A LIMITED TOKEN BUDGET, SO PLEASE BE CONCISE IN YOUR REASONING. This task is easy, so there is no need to spend much time thinking about it.

Once you have arrived to a final answer, output only: true or false"""


VERIFICATION_SYSTEM_PROMPT_MULTIPLE_CHOICE = """You are an answer verification assistant for multiple choice questions.

Your task: Determine if a given text contains a specific answer choice.

You will be given:
1. **Answer**: The answer (should be one of: A, B, C, or D)
2. **Text to Check**: An excerpt from a solution

**Answer Matching Guidelines:**
- The answer should be one of the letters: A, B, C, or D
- Be flexible with formatting: "A", "A.", "Option A", "(A)" all count as "A"
- The letter might appear in context like "The answer is B" or "Therefore, choice C"
- Be GENEROUS: if the answer letter appears as a final answer choice, it counts

IMPORTANT: YOU ARE GIVEN A LIMITED TOKEN BUDGET, SO PLEASE BE CONCISE IN YOUR REASONING.

Once you have arrived to a final answer, output only: true or false"""


VERIFICATION_SYSTEM_PROMPT_CODE = """You are a code verification assistant.

Your task: Determine if a given text contains specific Python code.

You will be given:
1. **Code**: The Python code
2. **Text to Check**: An excerpt that may contain Code

**Code Matching Guidelines:**
- The code should be functionally equivalent (variable names can differ)
- Whitespace differences are acceptable
- Comments are not required to match
- The code might be in a ```python code block or plain text
- Be GENEROUS: if the core logic matches, it counts

IMPORTANT: YOU ARE GIVEN A LIMITED TOKEN BUDGET, SO PLEASE BE CONCISE IN YOUR REASONING.

Once you have arrived to a final answer, output only: true or false"""


EXTRACTION_SYSTEM_PROMPT_MATH = """You are a precise extractor for mathematical problems.

You are given:
- ANSWER: the final answer the model arrived at (may appear in mathematically equivalent forms)
- COT_TEXT: the model's original chain-of-thought text (verbatim)

Task:
1) Find the FIRST occurrence in COT_TEXT where the reasoning LOGICALLY ARRIVES at ANSWER.
   - Treat decimals/fractions as equivalent (e.g., 0.25 == 1/4)
   - Treat algebraically equivalent forms as matches
   - Treat LaTeX vs non-LaTeX formatting as equivalent
   - The answer may appear by coincidence - find where it's DERIVED, not just mentioned

2) OUTPUT a single contiguous substring copied EXACTLY from COT_TEXT that:
   - Contains the answer occurrence
   - Includes context leading up to it (aim for the full sentence, or previous sentence if answer starts a sentence)
   - Enough uniqueness to locate it later via pattern matching
   CRITICAL: Your output MUST be a verbatim substring of COT_TEXT. No reformatting.

3) If no occurrence exists, output exactly "None"

4) Be concise - limited token budget. Output ONLY the extracted span or "None" -- nothing else."""


EXTRACTION_SYSTEM_PROMPT_MULTIPLE_CHOICE = """You are a precise extractor for multiple choice questions.

You are given:
- ANSWER: the final answer choice the model selected (A, B, C, or D)
- COT_TEXT: the model's original reasoning text (verbatim)

Task:
1) Find the FIRST occurrence in COT_TEXT where the reasoning LOGICALLY ARRIVES at selecting ANSWER.
   - Look for phrases like "The answer is A" or "Therefore, choice B"
   - The letter might appear in various formats: "A", "A.", "Option A", "(A)"
   - Find where the choice is MADE, not just where options are listed

2) OUTPUT a single contiguous substring copied EXACTLY from COT_TEXT that:
   - Contains the answer choice selection
   - Includes context leading up to it
   - Enough uniqueness to locate it later via pattern matching
   CRITICAL: Your output MUST be a verbatim substring of COT_TEXT. No reformatting.

3) If no occurrence exists, output exactly "None"

4) Be concise - limited token budget. Output ONLY the extracted span or "None" -- nothing else."""


EXTRACTION_SYSTEM_PROMPT_CODE = """You are a precise extractor for coding problems.

You are given:
- ANSWER: the Python code the model generated
- COT_TEXT: the model's original reasoning and code (verbatim)

Task:
1) Find the FIRST occurrence in COT_TEXT where the complete code solution appears.
   - The code could be in a ```python...``` block, but might be plain text
   - Find where the COMPLETE solution is present, not partial snippets
   - Function/variable names might differ slightly - look for functional equivalence

2) OUTPUT a single contiguous substring copied EXACTLY from COT_TEXT that:
   - Contains the complete code solution
   - If in a code block, include the ```python and ``` markers
   - Enough context to locate it later via pattern matching
   CRITICAL: Your output MUST be a verbatim substring of COT_TEXT. No reformatting.

3) If no occurrence exists, output exactly "None"

4) Be concise - limited token budget. Output ONLY the extracted span or "None" -- nothing else."""


ANSWER_EXTRACTION_SYSTEM_PROMPT_MATH = """You are a precise answer extractor for mathematical problems.

Your task: Extract ONLY the final mathematical answer from a given text.

Rules:
1. Extract ONLY the answer value itself (number, expression, formula, etc.)
2. DO NOT include surrounding words like "Therefore", "So", "The answer is", etc.
3. Look towards the END of the text for the final answer
4. Preserve the exact format (LaTeX, plain text, fraction, decimal, etc.)
5. If the answer is in \\boxed{...}, extract what's inside the box
6. Be concise - you have a limited token budget

Examples:
- Input: "Therefore, the answer is 42"
  Output: 42
- Input: "So we get x = \\frac{3}{4}"
  Output: \\frac{3}{4}
- Input: "The solution is \\boxed{2x + 5}"
  Output: 2x + 5

Output ONLY the extracted answer -- nothing else."""


ANSWER_EXTRACTION_SYSTEM_PROMPT_MULTIPLE_CHOICE = """You are a precise answer extractor for multiple choice questions.

Your task: Extract ONLY the final answer choice from a given text.

Rules:
1. Extract ONLY the letter: A, B, C, or D
2. DO NOT include surrounding text like "The answer is", "Option", etc.
3. Look towards the END of the text for the final answer
4. Be concise - you have a limited token budget

Examples:
- Input: "Therefore, the answer is B"
  Output: B
- Input: "So the correct option is (C)"
  Output: C
- Input: "The answer is A."
  Output: A

Output ONLY the letter (A, B, C, or D) -- nothing else."""


ANSWER_EXTRACTION_SYSTEM_PROMPT_CODE = """You are a precise code extractor.

Your task: Extract ONLY the Python code from a given text.

Rules:
1. Extract ONLY the Python code itself
2. If the code is in a ```python...``` block, extract what's inside (without the markers)
3. DO NOT include explanatory text, comments outside the code, or usage examples
4. Look for the main solution code
5. Be concise - you have a limited token budget

Examples:
- Input: "Here's the solution:\\n```python\\ndef solve(x):\\n    return x * 2\\n```\\nThis works by..."
  Output: def solve(x):\\n    return x * 2

Output ONLY the Python code -- nothing else."""


# ============================================================================
# RESPONSE PARSING
# ============================================================================

def parse_reasoning_response(response_data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], str]:
    """
    Parse response from vLLM with --reasoning-parser qwen3 enabled.
    Handles case where reasoning_content is None but </think> tag is in content.

    Returns:
        (reasoning_content, final_content, finish_reason)
    """
    choice = response_data.get("choices", [{}])[0]
    message = choice.get("message", {})

    reasoning_content = message.get("reasoning_content")
    final_content = message.get("content", "").strip()
    finish_reason = choice.get("finish_reason", "")

    # Fallback: if reasoning_content is None but </think> is in content, split manually
    if (reasoning_content is None or reasoning_content == "") and "</think>" in final_content:
        parts = final_content.split("</think>", 1)
        reasoning_content = parts[0].strip()
        final_content = parts[1].strip() if len(parts) > 1 else ""

    return reasoning_content, final_content, finish_reason


def get_prompts_for_data_source(data_source: str) -> Tuple[str, str, str]:
    """
    Get the appropriate prompts based on data source.

    Returns:
        (verification_prompt, extraction_prompt, answer_extraction_prompt)
    """
    data_source_lower = data_source.lower()

    if data_source_lower == "opencoder":
        return (
            VERIFICATION_SYSTEM_PROMPT_CODE,
            EXTRACTION_SYSTEM_PROMPT_CODE,
            ANSWER_EXTRACTION_SYSTEM_PROMPT_CODE
        )
    elif data_source_lower == "openscience":
        return (
            VERIFICATION_SYSTEM_PROMPT_MULTIPLE_CHOICE,
            EXTRACTION_SYSTEM_PROMPT_MULTIPLE_CHOICE,
            ANSWER_EXTRACTION_SYSTEM_PROMPT_MULTIPLE_CHOICE
        )
    else:
        # Default: math-style prompts (for MATH, AIME, etc.)
        return (
            VERIFICATION_SYSTEM_PROMPT_MATH,
            EXTRACTION_SYSTEM_PROMPT_MATH,
            ANSWER_EXTRACTION_SYSTEM_PROMPT_MATH
        )


# ============================================================================
# VERIFICATION
# ============================================================================

def is_network_error(status_code: Optional[int] = None, exception: Optional[Exception] = None) -> bool:
    """
    Determine if an error is a network/infrastructure issue (should retry without counting).
    """
    if exception is not None:
        return True

    if status_code is not None:
        return status_code >= 500 or status_code == 429

    return False


async def call_api_with_network_retry(
    session: ClientSession,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[int], Optional[str]]:
    """
    Call API with automatic retry for network errors (infinite retries with exponential backoff).

    Returns:
        (success, response_data, status_code, error_message)
    """
    delay = initial_delay
    attempt = 0

    while True:
        attempt += 1

        try:
            async with session.post(url, headers=headers, json=payload) as resp:
                status_code = resp.status

                if is_network_error(status_code=status_code):
                    error_text = await resp.text()
                    print(f"Network error HTTP {status_code}, retrying in {delay:.1f}s... (attempt {attempt})", flush=True)
                    await asyncio.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
                    continue

                if status_code != 200:
                    error_text = await resp.text()
                    return False, None, status_code, f"HTTP {status_code}: {error_text}"

                data = await resp.json()
                return True, data, status_code, ""

        except (asyncio.TimeoutError, aiohttp.ClientError, ConnectionError) as e:
            print(f"Network exception {type(e).__name__}: {e}, retrying in {delay:.1f}s... (attempt {attempt})", flush=True)
            await asyncio.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)
            continue

        except Exception as e:
            print(f"Unknown exception {type(e).__name__}: {e}, retrying in {delay:.1f}s... (attempt {attempt})", flush=True)
            await asyncio.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)
            continue


async def verify_answer(
    session: ClientSession,
    base_url: str,
    model: str,
    api_key: Optional[str],
    ground_truth: str,
    text_to_check: str,
    verification_prompt: str,
    *,
    temperature: float = 0.6,
    top_p: float = 0.95,
    min_p: float = 0.0,
    top_k: int = 20,
    max_tokens: int = 2048,
    max_retries: int = 10,
) -> Tuple[bool, str, str, str]:
    """
    Use LLM to verify if text contains the answer.

    Returns:
        (contains_answer, reasoning, finish_reason, error_message)
    """
    if not text_to_check or not ground_truth:
        return False, "Empty text or ground truth", "error", ""

    user_message = f"""Ground Truth Answer: {ground_truth}

Text to Check:
{text_to_check}

Does the text contain the ground truth answer?"""

    messages = [
        {"role": "system", "content": verification_prompt},
        {"role": "user", "content": user_message}
    ]

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "min_p": min_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
    }

    url = base_url.rstrip("/") + "/v1/chat/completions"

    for attempt in range(max_retries):
        success, data, status_code, error_msg = await call_api_with_network_retry(
            session, url, headers, payload
        )

        if not success:
            return False, f"API error after retries", "error", error_msg

        reasoning_content, final_content, finish_reason = parse_reasoning_response(data)

        if finish_reason == "length" and attempt < max_retries - 1:
            new_max_tokens = max_tokens * (2 ** (attempt + 1))
            payload["max_tokens"] = min(new_max_tokens, 8192)

            if new_max_tokens >= 8192 and payload["max_tokens"] == 8192:
                if finish_reason == "length":
                    break
            continue

        final_lower = final_content.lower()
        contains = final_lower.startswith("true")

        reasoning = reasoning_content if reasoning_content else final_content

        return contains, reasoning, finish_reason, ""

    return False, "", "max_retries_exceeded", "Max retries exceeded for model issues"


async def extract_final_answer(
    session: ClientSession,
    base_url: str,
    model: str,
    api_key: Optional[str],
    final_content: str,
    answer_extraction_prompt: str,
    **kwargs
) -> Optional[str]:
    """
    Extract just the final answer from final_content using LLM.

    Returns:
        The extracted answer, or None if extraction fails
    """
    if not final_content:
        return None

    user_message = f"""Extract the final answer from this text:

{final_content}

Remember: Output ONLY the answer value itself, nothing else."""

    messages = [
        {"role": "system", "content": answer_extraction_prompt},
        {"role": "user", "content": user_message}
    ]

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": kwargs.get("temperature", 0.6),
        "top_p": kwargs.get("top_p", 0.95),
        "min_p": kwargs.get("min_p", 0.0),
        "top_k": kwargs.get("top_k", 20),
        "max_tokens": kwargs.get("max_tokens", 2048),
    }

    url = base_url.rstrip("/") + "/v1/chat/completions"

    success, data, status_code, error_msg = await call_api_with_network_retry(
        session, url, headers, payload
    )

    if not success:
        return None

    reasoning_content, final_content_extracted, finish_reason = parse_reasoning_response(data)

    return final_content_extracted.strip() if final_content_extracted else None


async def verify_final_with_majority_vote(
    session: ClientSession,
    base_url: str,
    model: str,
    api_key: Optional[str],
    ground_truth: str,
    final_content: str,
    verification_prompt: str,
    **kwargs
) -> Tuple[bool, str, int]:
    """
    Verify final_content with majority vote (2-3 attempts).

    Returns:
        (is_correct, successful_reasoning, num_attempts)
    """
    if not final_content:
        return False, "No final content", 1

    # First verification
    contains1, reasoning1, _, _ = await verify_answer(
        session, base_url, model, api_key,
        ground_truth, final_content, verification_prompt,
        **kwargs
    )

    if contains1:
        return True, reasoning1, 1

    # Second verification
    contains2, reasoning2, _, _ = await verify_answer(
        session, base_url, model, api_key,
        ground_truth, final_content, verification_prompt,
        **kwargs
    )

    if not contains1 and not contains2:
        return False, reasoning1, 2

    # Tie-breaker
    contains3, reasoning3, _, _ = await verify_answer(
        session, base_url, model, api_key,
        ground_truth, final_content, verification_prompt,
        **kwargs
    )

    votes = [contains1, contains2, contains3]
    is_correct = sum(votes) >= 2

    if is_correct:
        if contains1:
            successful_reasoning = reasoning1
        elif contains2:
            successful_reasoning = reasoning2
        else:
            successful_reasoning = reasoning3
    else:
        successful_reasoning = reasoning1

    return is_correct, successful_reasoning, 3


# ============================================================================
# SPAN EXTRACTION
# ============================================================================

def truncate_text(text: str, tokenizer, max_tokens: int = 1000) -> str:
    """Truncate text to approximately max_tokens."""
    if not text:
        return text

    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=True)


async def extract_and_verify_span_with_feedback(
    session: ClientSession,
    base_url: str,
    model: str,
    api_key: Optional[str],
    answer_to_find: str,
    reasoning_content: str,
    extraction_prompt: str,
    verification_prompt: str,
    tokenizer,
    max_retries: int = 3,
    **kwargs
) -> Tuple[Optional[str], bool, str, int]:
    """
    Extract span with verification feedback loop.

    Returns:
        (extracted_span, verified, reasoning, num_attempts)
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    previous_failed_spans = []
    previous_was_length_truncated = False
    current_max_tokens = kwargs.get("max_tokens", 98304)

    for attempt in range(max_retries):
        # Build extraction prompt
        if attempt == 0:
            user_message = (
                "ANSWER:\n"
                f"{answer_to_find}\n\n"
                "COT_TEXT (verbatim):\n"
                f"{reasoning_content}\n\n"
                "Find the first occurrence of the answer and output it after your reasoning."
            )
        elif previous_was_length_truncated:
            user_message = (
                "ANSWER:\n"
                f"{answer_to_find}\n\n"
                "COT_TEXT (verbatim):\n"
                f"{reasoning_content}\n\n"
                f"WARNING: Your previous response was cut off. Please be more concise. "
                f"Simply find where the reasoning arrives at the answer and extract that span."
            )
        else:
            previous_text = "\n".join([
                f"Attempt {i+1}: {truncate_text(span, tokenizer, 500)}"
                for i, span in enumerate(previous_failed_spans)
            ])

            user_message = (
                "ANSWER:\n"
                f"{answer_to_find}\n\n"
                "COT_TEXT (verbatim):\n"
                f"{reasoning_content}\n\n"
                f"Your previous extraction(s) were:\n{previous_text}\n\n"
                f"These did not contain the correct answer. Try again or return None."
            )

        messages = [
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": user_message}
        ]

        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.6),
            "top_p": kwargs.get("top_p", 0.95),
            "min_p": kwargs.get("min_p", 0.0),
            "top_k": kwargs.get("top_k", 20),
            "max_tokens": current_max_tokens,
        }

        success, response_data, status_code, error_msg = await call_api_with_network_retry(
            session, url, headers, payload
        )

        if not success:
            return None, False, f"API error: {error_msg}", attempt + 1

        reasoning, final_content, finish_reason = parse_reasoning_response(response_data)

        if finish_reason == "length":
            current_max_tokens = min(current_max_tokens * 2, 65536)
            previous_was_length_truncated = True
            continue

        previous_was_length_truncated = False

        if final_content == "None" or final_content.startswith("None"):
            return None, False, "Model returned None", attempt + 1

        if not final_content or final_content not in reasoning_content:
            previous_failed_spans.append(final_content)
            continue

        # Verify extracted span
        contains, verify_reasoning, _, _ = await verify_answer(
            session, base_url, model, api_key,
            answer_to_find, final_content, verification_prompt,
            **kwargs
        )

        if contains:
            return final_content, True, verify_reasoning, attempt + 1

        previous_failed_spans.append(final_content)

    return None, False, "Max retries exceeded", max_retries


# ============================================================================
# PATTERN MATCHING
# ============================================================================

def fuzzy_find_span(span: str, text: str, min_similarity: float = 0.85) -> Optional[int]:
    """Find span in text using fuzzy string matching."""
    if not span or not text:
        return None

    span_len = len(span)
    if span_len > len(text):
        return None

    best_match_pos = None
    best_similarity = 0.0

    step = max(1, span_len // 10)
    for i in range(0, len(text) - span_len + 1, step):
        window = text[i:i + span_len]
        similarity = SequenceMatcher(None, span, window).ratio()

        if similarity > best_similarity:
            best_similarity = similarity
            best_match_pos = i

    if best_similarity >= min_similarity and best_match_pos is not None:
        return best_match_pos

    return None


def find_span_in_reasoning(span: str, reasoning_text: str) -> Optional[int]:
    """
    Find the span in the reasoning text with various matching strategies.

    Returns:
        Character position of span start, or None if not found
    """
    if not span or not reasoning_text:
        return None

    # Strategy 1: Exact match
    pos = reasoning_text.find(span)
    if pos != -1:
        return pos

    # Strategy 2: Normalize whitespace
    normalized_span = re.sub(r'\s+', ' ', span).strip()
    normalized_text = re.sub(r'\s+', ' ', reasoning_text).strip()

    pos = normalized_text.find(normalized_span)
    if pos != -1:
        # Map back to original position
        original_pos = 0
        norm_pos = 0
        for i, char in enumerate(reasoning_text):
            if not char.isspace() or reasoning_text[i:i+1] == ' ':
                if norm_pos >= pos:
                    original_pos = i
                    break
                if char.isspace():
                    if i == 0 or not reasoning_text[i-1].isspace():
                        norm_pos += 1
                else:
                    norm_pos += 1
        return original_pos

    # Strategy 3: Fuzzy matching
    pos = fuzzy_find_span(span, reasoning_text, min_similarity=0.90)
    if pos is not None:
        return pos

    # Strategy 4: LaTeX normalization
    span_no_latex = re.sub(r'\\[a-zA-Z]+', '', span)
    text_no_latex = re.sub(r'\\[a-zA-Z]+', '', reasoning_text)

    pos = text_no_latex.find(span_no_latex)
    if pos != -1:
        return pos

    # Strategy 5: Lower threshold fuzzy
    pos = fuzzy_find_span(span, reasoning_text, min_similarity=0.80)
    if pos is not None:
        return pos

    return None


def get_token_positions(
    text: str,
    start_char: int,
    end_char: int,
    tokenizer
) -> Tuple[int, int]:
    """Convert character positions to token positions."""
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    offsets = encoding['offset_mapping']

    start_token = None
    end_token = None

    for i, (token_start, token_end) in enumerate(offsets):
        if start_token is None and token_end > start_char:
            start_token = i

        if token_start < end_char:
            end_token = i
        else:
            break

    if start_token is None:
        start_token = 0
    if end_token is None:
        end_token = len(offsets) - 1

    return start_token, end_token


# ============================================================================
# MAIN PROCESSING
# ============================================================================

async def process_solution(
    session: ClientSession,
    base_url: str,
    model: str,
    api_key: Optional[str],
    solution_index: int,
    ground_truth: str,
    reasoning_content: str,
    final_content: str,
    data_source: str,
    tokenizer,
    max_span_retries: int,
    **kwargs
) -> Dict[str, Any]:
    """
    Process a single solution: extract answer → verify → extract span.

    Steps:
    1. Extract just the answer value from final_content
    2. Verify final_content contains ground truth
    3. Extract final answer span from reasoning

    Returns:
        Complete analysis dict for this solution
    """
    # Get appropriate prompts for this data source
    verification_prompt, extraction_prompt, answer_extraction_prompt = get_prompts_for_data_source(data_source)

    analysis = {
        "solution_index": solution_index,
        "reasoning_content": reasoning_content,
        "final_content": final_content,
        "data_source": data_source,

        # Extracted final answer (just the value)
        "extracted_final_answer": None,

        # Final verification results
        "llm_final_answer_correct": False,
        "llm_final_reasoning": "",
        "final_verification_attempts": 0,

        # Final answer span extraction results
        "extracted_final_span": None,
        "llm_span_final_verified": False,
        "llm_span_final_reasoning": "",
        "span_final_extraction_attempts": 0,
        "span_final_start_token": None,
        "span_final_end_token": None,
        "span_final_char_start": None,
        "span_final_char_end": None,
    }

    # STEP 1: Extract the final answer value from final_content
    extracted_answer = await extract_final_answer(
        session, base_url, model, api_key,
        final_content, answer_extraction_prompt,
        **kwargs
    )
    analysis["extracted_final_answer"] = extracted_answer

    # STEP 2: Verify final_content with majority vote
    final_correct, final_reasoning, final_attempts = await verify_final_with_majority_vote(
        session, base_url, model, api_key,
        ground_truth, final_content, verification_prompt,
        **kwargs
    )

    analysis["llm_final_answer_correct"] = final_correct
    analysis["llm_final_reasoning"] = final_reasoning
    analysis["final_verification_attempts"] = final_attempts

    # STEP 3: Extract final answer span
    if not reasoning_content or not extracted_answer:
        return analysis

    extracted_final_span, span_final_verified, span_final_reasoning, span_final_attempts = \
        await extract_and_verify_span_with_feedback(
            session, base_url, model, api_key,
            extracted_answer, reasoning_content,
            extraction_prompt, verification_prompt,
            tokenizer,
            max_retries=max_span_retries,
            **kwargs
        )

    analysis["extracted_final_span"] = extracted_final_span
    analysis["llm_span_final_verified"] = span_final_verified
    analysis["llm_span_final_reasoning"] = span_final_reasoning
    analysis["span_final_extraction_attempts"] = span_final_attempts

    # Get token positions for final answer span
    if extracted_final_span and span_final_verified:
        span_char_pos = find_span_in_reasoning(extracted_final_span, reasoning_content)

        if span_char_pos is not None:
            span_char_end = span_char_pos + len(extracted_final_span)

            try:
                start_token, end_token = get_token_positions(
                    reasoning_content,
                    span_char_pos,
                    span_char_end,
                    tokenizer
                )

                analysis["span_final_start_token"] = start_token
                analysis["span_final_end_token"] = end_token
                analysis["span_final_char_start"] = span_char_pos
                analysis["span_final_char_end"] = span_char_end

            except Exception as e:
                # Token position calculation failed, but we keep the span
                pass

    return analysis


async def process_sample(
    sem: asyncio.Semaphore,
    session: ClientSession,
    base_url: str,
    model: str,
    api_key: Optional[str],
    sample: Dict[str, Any],
    num_solutions_to_process: int,
    tokenizer,
    max_span_retries: int,
    pbar: tqdm,
    **kwargs
) -> Dict[str, Any]:
    """Process a single sample from the input JSONL."""
    async with sem:
        sample_idx = sample.get("sample_idx")
        answer = sample.get("answer", "")
        data_source = sample.get("data_source", "unknown")
        sampled_solutions = sample.get("sampled_solutions", [])

        num_to_process = min(num_solutions_to_process, len(sampled_solutions))

        solution_analyses = []

        for idx in range(num_to_process):
            solution = sampled_solutions[idx]
            reasoning_content = solution.get("reasoning_content", "")
            final_content = solution.get("content", "")

            # Skip if both are empty
            if not reasoning_content and not final_content:
                solution_analyses.append({
                    "solution_index": idx,
                    "reasoning_content": reasoning_content,
                    "final_content": final_content,
                    "data_source": data_source,
                    "extracted_final_answer": None,
                    "llm_final_answer_correct": False,
                    "llm_final_reasoning": "Empty content",
                    "final_verification_attempts": 0,
                    "extracted_final_span": None,
                    "llm_span_final_verified": False,
                    "llm_span_final_reasoning": "",
                    "span_final_extraction_attempts": 0,
                    "span_final_start_token": None,
                    "span_final_end_token": None,
                    "span_final_char_start": None,
                    "span_final_char_end": None,
                })
                continue

            analysis = await process_solution(
                session, base_url, model, api_key,
                idx, answer, reasoning_content, final_content,
                data_source, tokenizer, max_span_retries,
                **kwargs
            )

            solution_analyses.append(analysis)
            pbar.update(1)

        # Build output
        output = sample.copy()
        output["solution_analyses"] = solution_analyses

        return output


def load_processed_indices(output_path: Path) -> Set[int]:
    """Load the set of sample_idx that have already been processed."""
    if not output_path.exists():
        return set()

    processed_indices = set()
    with open(output_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    sample_idx = data.get("sample_idx")
                    if sample_idx is not None and "solution_analyses" in data:
                        processed_indices.add(sample_idx)
                except json.JSONDecodeError:
                    continue

    return processed_indices


async def process_file_async(
    input_path: Path,
    base_url: str,
    model: str,
    api_key: Optional[str],
    batch_size: int,
    timeout_s: float,
    num_solutions_to_process: int,
    tokenizer,
    max_span_retries: int,
    *,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
    max_tokens: int,
) -> Path:
    """Process a JSONL file and write results incrementally."""

    # Output to same directory as input
    output_path = input_path.with_name(input_path.stem + "_extracted.jsonl")

    # Load already processed indices
    processed_indices = load_processed_indices(output_path)
    print(f"Found {len(processed_indices)} already processed samples. Resuming...", flush=True)

    # Load input samples
    samples = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                sample_idx = sample.get("sample_idx")
                if sample_idx not in processed_indices:
                    samples.append(sample)

    if not samples:
        print("All samples already processed!", flush=True)
        return output_path

    print(f"Processing {len(samples)} samples...", flush=True)

    # Count total solutions to process
    total_solutions = sum(
        min(num_solutions_to_process, len(sample.get("sampled_solutions", [])))
        for sample in samples
    )
    print(f"Total solutions to process: {total_solutions}", flush=True)

    sem = asyncio.Semaphore(batch_size)
    timeout = ClientTimeout(total=timeout_s)

    # Open output file in append mode
    output_file = open(output_path, "a")

    kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "min_p": min_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
    }

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            with tqdm(total=total_solutions, desc=f"Processing {input_path.name}", unit="solution") as pbar:
                tasks = [
                    asyncio.create_task(
                        process_sample(
                            sem, session, base_url, model, api_key, sample,
                            num_solutions_to_process, tokenizer, max_span_retries,
                            pbar, **kwargs
                        )
                    )
                    for sample in samples
                ]

                for fut in asyncio.as_completed(tasks):
                    result = await fut
                    # Write immediately to file
                    output_file.write(json.dumps(result) + "\n")
                    output_file.flush()

    finally:
        output_file.close()

    # Print summary statistics
    print_summary(output_path)

    return output_path


def print_summary(output_path: Path):
    """Print summary statistics from processed file."""
    all_results = []
    with open(output_path, "r") as f:
        for line in f:
            if line.strip():
                all_results.append(json.loads(line))

    total_solutions = sum(len(entry.get("solution_analyses", [])) for entry in all_results)

    final_correct = sum(
        1 for entry in all_results
        for analysis in entry.get("solution_analyses", [])
        if analysis.get("llm_final_answer_correct", False)
    )

    span_final_correct = sum(
        1 for entry in all_results
        for analysis in entry.get("solution_analyses", [])
        if analysis.get("llm_span_final_verified", False)
    )

    spans_final_with_tokens = sum(
        1 for entry in all_results
        for analysis in entry.get("solution_analyses", [])
        if analysis.get("span_final_start_token") is not None
    )

    print("\n=== Summary Statistics ===")
    print(f"Total problems: {len(all_results)}")
    print(f"Total solutions: {total_solutions}")
    print(f"Final answers correct: {final_correct}/{total_solutions} ({100*final_correct/total_solutions:.1f}%)")
    print(f"\nFinal Answer Spans:")
    print(f"  Spans verified correct: {span_final_correct}/{total_solutions} ({100*span_final_correct/total_solutions:.1f}%)")
    print(f"  Spans with token positions: {spans_final_with_tokens}/{total_solutions} ({100*spans_final_with_tokens/total_solutions:.1f}%)")


async def _run(args) -> List[str]:
    input_path = Path(args.input_file)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
        use_fast=True
    )

    output_path = await process_file_async(
        input_path=input_path,
        base_url=args.server_url,
        model=args.model,
        api_key=args.api_key,
        batch_size=args.batch_size,
        timeout_s=args.timeout,
        num_solutions_to_process=args.num_solutions,
        tokenizer=tokenizer,
        max_span_retries=args.max_span_retries,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    return [str(output_path)]


def main():
    ap = argparse.ArgumentParser(
        description="Extract and verify final answer spans from dataset_cots.jsonl"
    )
    ap.add_argument("--input-file", type=str, required=True,
                    help="Path to dataset_cots.jsonl file")
    ap.add_argument("--server-url", type=str, default="http://localhost:8000",
                    help="Base URL of the vLLM server")
    ap.add_argument("--model", type=str, default="vllm",
                    help="Served model name")
    ap.add_argument("--api-key", type=str, default=os.environ.get("VLLM_API_KEY", None),
                    help="API key for vLLM server")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Concurrent requests cap")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="Per-request timeout (seconds)")

    # Tokenizer
    ap.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-30B-A3B-Thinking-2507",
                    help="HF tokenizer for token position calculation")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="Trust remote code for tokenizer")

    # Processing control
    ap.add_argument("--num-solutions", type=int, default=1,
                    help="Number of sampled_solutions to process per sample")
    ap.add_argument("--max-span-retries", type=int, default=3,
                    help="Max retry attempts for span extraction")

    # Sampler controls
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="Max tokens to generate per request")

    args = ap.parse_args()

    produced = asyncio.run(_run(args))
    print("\nWrote:")
    for p in produced:
        print("  ", p)


if __name__ == "__main__":
    main()
