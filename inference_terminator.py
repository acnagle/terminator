#!/usr/bin/env python3
"""
Inference script for trained Terminator models.

This script runs inference using a trained FFN (+ optional extra layers) to predict
when the model has generated the final answer in its chain-of-thought reasoning.

Supports models trained with either BCE loss or Focal loss.

Two modes:
1. Generative (mode=1): Generate reasoning token-by-token, exit when model predicts completion
2. Parallel (mode=2): Use existing reasoning trace, predict all positions in parallel

Supports multiple decoding strategies for determining when to exit.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TopKLogitsWarper, TopPLogitsWarper, TemperatureLogitsWarper
from transformers.generation.logits_process import LogitsProcessorList
from tqdm import tqdm

from terminator_utils import ExtraTransformerLayers, LayerFFN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MinPLogitsWarper:
    """
    Logits warper that masks tokens with probability < min_p * max_probability.

    This implements the min-p sampling strategy where tokens are filtered
    based on their probability relative to the most likely token.
    """
    def __init__(self, min_p: float):
        self.min_p = min_p

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.min_p <= 0:
            return scores

        # Get probabilities
        probs = F.softmax(scores, dim=-1)

        # Find max probability for each sequence
        max_probs = probs.max(dim=-1, keepdim=True)[0]

        # Mask tokens with prob < min_p * max_prob
        threshold = self.min_p * max_probs
        mask = probs < threshold
        scores = scores.masked_fill(mask, float('-inf'))

        return scores


def load_checkpoint_for_inference(
    checkpoint_dir: Path,
    device: torch.device,
    base_model: Optional[AutoModelForCausalLM] = None
) -> Tuple[LayerFFN, Optional[ExtraTransformerLayers], Dict[str, Any], int]:
    """
    Load trained model from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        device: Device to load model on
        base_model: Optional base model to get norm layer class from (recommended)

    Returns:
        Tuple of (ffn, extra_layers, config, layer_idx)
    """
    checkpoint_dir = Path(checkpoint_dir)

    # First, try to load the final model (layer_{idx}.pt)
    layer_files = list(checkpoint_dir.glob('layer_*.pt'))

    if layer_files:
        # Use the final saved model
        model_path = layer_files[0]  # Should only be one
        logger.info(f"Loading final model from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
    else:
        # Fall back to checkpoint.pt
        checkpoint_path = checkpoint_dir / 'checkpoint.pt'
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No model files found in {checkpoint_dir}. "
                f"Expected either layer_*.pt or checkpoint.pt"
            )
        logger.info(f"Loading from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Rename keys for consistency
        if 'ffn_state_dict' in checkpoint:
            checkpoint['state_dict'] = checkpoint.pop('ffn_state_dict')

    # Get config and layer_idx
    config = checkpoint['config']
    layer_idx = checkpoint['layer_idx']

    logger.info(f"Model trained at layer {layer_idx}")
    logger.info(f"Model: {config['model_name']}")

    # Reconstruct FFN
    logger.info("Reconstructing FFN...")

    # Get normalization layer - match training behavior
    # During training, the norm class is copied from the base model
    from torch import nn

    if base_model is not None and hasattr(base_model, 'model') and hasattr(base_model.model, 'norm'):
        # Use same norm class as the base model (e.g., Qwen2RMSNorm)
        norm_class = base_model.model.norm.__class__
        eps = getattr(base_model.config, 'rms_norm_eps',
                     getattr(base_model.config, 'layer_norm_eps', 1e-6))
        norm_layer = norm_class(config['hidden_size'], eps=eps)
        logger.info(f"  Using {norm_class.__name__} for pre-FFN normalization (eps={eps})")
    else:
        # Fallback to LayerNorm without bias to match common training setup
        norm_layer = nn.LayerNorm(config['hidden_size'], elementwise_affine=True)
        logger.warning("  Base model not provided - using LayerNorm fallback (may cause loading issues)")

    ffn = LayerFFN(
        hidden_size=config['hidden_size'],
        num_hidden_layers=config.get('ffn_layers', 1),
        activation=config.get('ffn_activation', 'gelu'),
        intermediate_size=config.get('ffn_intermediate_size', None),
        dropout=config.get('dropout', 0.0),
        norm_layer=norm_layer
    )

    # Load FFN weights
    ffn.load_state_dict(checkpoint['state_dict'])
    ffn.to(device)
    ffn.eval()

    logger.info(f"✓ FFN loaded ({sum(p.numel() for p in ffn.parameters()):,} parameters)")

    # Load extra layers if present
    extra_layers = None
    num_extra_layers = config.get('num_extra_layers', 0)

    if num_extra_layers > 0 and 'extra_layers_state_dict' in checkpoint:
        logger.info(f"Loading {num_extra_layers} extra transformer layer(s)...")

        # We need the base model to get layer class and config
        # For now, we'll defer this to after base model is loaded
        # Store the state dict for later
        extra_layers_state_dict = checkpoint['extra_layers_state_dict']
    else:
        extra_layers_state_dict = None

    return ffn, extra_layers_state_dict, config, layer_idx


def load_base_model_and_tokenizer(
    model_name: str,
    device: torch.device
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, int, int]:
    """
    Load base model and tokenizer.

    Returns:
        Tuple of (model, tokenizer, think_token_id, think_end_token_id)
    """
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get <think> and </think> token IDs
    think_token_id = tokenizer.convert_tokens_to_ids('<think>')
    think_end_token_id = tokenizer.convert_tokens_to_ids('</think>')

    if think_token_id == tokenizer.unk_token_id or think_end_token_id == tokenizer.unk_token_id:
        raise ValueError(
            f"<think> and </think> tokens not found in tokenizer!\n"
            f"<think> ID: {think_token_id}, </think> ID: {think_end_token_id}"
        )

    logger.info(f"✓ <think> token ID: {think_token_id}, </think> token ID: {think_end_token_id}")

    logger.info(f"Loading base model: {model_name}")
    # Use explicit device map to avoid dispatching issues with .to()
    # device_map={"": device} puts entire model on the specified device
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True
    )

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    logger.info(f"✓ Model loaded: {model.config.num_hidden_layers} layers, "
                f"hidden size {model.config.hidden_size}")

    return model, tokenizer, think_token_id, think_end_token_id


def reconstruct_extra_layers(
    base_model: AutoModelForCausalLM,
    extra_layers_state_dict: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    device: torch.device
) -> ExtraTransformerLayers:
    """Reconstruct extra transformer layers from state dict."""
    num_extra_layers = config.get('num_extra_layers', 0)

    if num_extra_layers == 0:
        return None

    logger.info(f"Reconstructing {num_extra_layers} extra transformer layer(s)...")

    # Get layer class and rotary embedding from base model
    base_layer_class = base_model.model.layers[0].__class__
    model_config = base_model.config
    rotary_emb = getattr(base_model.model, 'rotary_emb', None)

    extra_layers = ExtraTransformerLayers(
        base_layer_class,
        num_extra_layers,
        model_config,
        rotary_emb=rotary_emb
    ).to(device)

    # Load weights
    extra_layers.load_state_dict(extra_layers_state_dict)
    extra_layers.eval()

    logger.info(f"✓ Extra layers loaded ({sum(p.numel() for p in extra_layers.parameters()):,} parameters)")

    return extra_layers


def format_prompt(problem: str, data_source: str = "math") -> str:
    """Format the problem with instructions for the model."""
    if data_source == "opencoder":
        return f"{problem}\n\nPlease reason step by step and give your code in Python. Use Python's standard library unless otherwise stated."
    elif data_source == "humaneval" or data_source == "bigcodebench":
        return f"{problem}\n\nPlease complete the following function. Please reason step by step and give your code in Python. Use Python's standard library unless otherwise stated."
    else:
        return f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def apply_predictor(
    hidden_states: torch.Tensor,
    ffn: LayerFFN,
    extra_layers: Optional[ExtraTransformerLayers],
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply predictor (extra layers + FFN) to hidden states.

    Args:
        hidden_states: [batch, seq_len, hidden_size] in BF16
        ffn: The prediction FFN
        extra_layers: Optional extra transformer layers
        attention_mask: [batch, seq_len] attention mask

    Returns:
        predictions: [batch, seq_len] probabilities (after sigmoid)
    """
    # Cast to FP32
    hidden_states_fp32 = hidden_states.float()

    # Pass through extra layers if present
    if extra_layers is not None:
        hidden_states_fp32 = extra_layers(hidden_states_fp32, attention_mask=attention_mask)

    # Get logits from FFN
    logits = ffn(hidden_states_fp32)  # [batch, seq_len]

    # Convert to probabilities
    predictions = torch.sigmoid(logits)

    return predictions


class DecodingStrategy:
    """Base class for decoding strategies."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def should_exit(self, predictions: torch.Tensor, position: int) -> bool:
        """
        Determine if we should exit generation.

        Args:
            predictions: [seq_len] prediction probabilities (0 to 1)
            position: Current position in sequence

        Returns:
            True if we should exit, False otherwise
        """
        raise NotImplementedError


class SlidingWindowStrategy(DecodingStrategy):
    """Exit when majority vote in sliding window predicts 1."""

    def __init__(self, window_size: int = 5, threshold: float = 0.5):
        super().__init__("sliding_window", window_size=window_size, threshold=threshold)
        self.window_size = window_size
        self.threshold = threshold

    def should_exit(self, predictions: torch.Tensor, position: int) -> bool:
        # Need at least window_size predictions
        if position < self.window_size - 1:
            return False

        # Get last window_size predictions
        window_start = position - self.window_size + 1
        window = predictions[window_start:position + 1]

        # Majority vote (using configurable threshold)
        binary = (window > self.threshold).float()
        majority = binary.mean() > 0.5

        return majority


class ConfidenceThresholdStrategy(DecodingStrategy):
    """Exit when P(class=1) > threshold for N consecutive tokens."""

    def __init__(self, threshold: float = 0.5, consecutive: int = 3):
        super().__init__("confidence_threshold", threshold=threshold, consecutive=consecutive)
        self.threshold = threshold
        self.consecutive = consecutive
        self.count = 0

    def should_exit(self, predictions: torch.Tensor, position: int) -> bool:
        # Check if current prediction is above threshold
        if predictions[position] > self.threshold:
            self.count += 1
        else:
            self.count = 0

        return self.count >= self.consecutive


class CumulativeStrategy(DecodingStrategy):
    """Exit when cumulative sum of P(class=1) over window > threshold."""

    def __init__(self, window_size: int = 5, threshold: float = 3.5):
        super().__init__("cumulative", window_size=window_size, threshold=threshold)
        self.window_size = window_size
        self.threshold = threshold

        # Warning if threshold is too low
        if threshold < window_size * 0.5:
            logger.warning(
                f"CumulativeStrategy: threshold ({threshold:.2f}) < window_size * 0.5 "
                f"({window_size * 0.5:.2f}). This means the model only requires "
                f"< 0.5 average confidence to terminate."
            )

    def should_exit(self, predictions: torch.Tensor, position: int) -> bool:
        # Need at least window_size predictions
        if position < self.window_size - 1:
            return False

        # Get last window_size predictions
        window_start = position - self.window_size + 1
        window = predictions[window_start:position + 1]

        # Check if cumulative sum exceeds threshold
        cumsum = window.sum()

        return cumsum > self.threshold


class FirstFlipStrategy(DecodingStrategy):
    """Exit immediately at first prediction > threshold."""

    def __init__(self, threshold: float = 0.5):
        super().__init__("first_flip", threshold=threshold)
        self.threshold = threshold

    def should_exit(self, predictions: torch.Tensor, position: int) -> bool:
        return predictions[position] > self.threshold


def create_decoding_strategy(strategy: str, **kwargs) -> DecodingStrategy:
    """Factory function to create decoding strategy."""
    if strategy == "sliding_window":
        return SlidingWindowStrategy(
            window_size=kwargs.get('window_size', 5),
            threshold=kwargs.get('threshold', 0.5)
        )
    elif strategy == "confidence_threshold":
        return ConfidenceThresholdStrategy(
            threshold=kwargs.get('threshold', 0.5),
            consecutive=kwargs.get('consecutive', 3)
        )
    elif strategy == "cumulative":
        return CumulativeStrategy(
            window_size=kwargs.get('window_size', 5),
            threshold=kwargs.get('threshold', 3.5)
        )
    elif strategy == "first_flip":
        return FirstFlipStrategy(threshold=kwargs.get('threshold', 0.5))
    else:
        raise ValueError(f"Unknown decoding strategy: {strategy}")


def get_strategy_params_string(strategy: DecodingStrategy) -> str:
    """
    Generate a string representation of strategy parameters for filename.

    Args:
        strategy: The decoding strategy instance

    Returns:
        String like "window5" or "thresh0.5_consec3"
    """
    params = []
    for key, value in strategy.kwargs.items():
        # Shorten common parameter names
        key_short = key
        if key == "window_size":
            key_short = "window"
        elif key == "threshold":
            key_short = "thresh"
        elif key == "consecutive":
            key_short = "consec"

        # Format value
        if isinstance(value, float):
            value_str = f"{value:.1f}".replace('.', 'p')
        else:
            value_str = str(value)

        params.append(f"{key_short}{value_str}")

    return "_".join(params)


def mode1_generate(
    sample: Dict[str, Any],
    solution_index: int,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    ffn: LayerFFN,
    extra_layers: Optional[ExtraTransformerLayers],
    layer_idx: int,
    lag_size: int,
    think_token_id: int,
    think_end_token_id: int,
    decoding_strategy: DecodingStrategy,
    exit_message: str,
    max_reasoning_tokens: int,
    max_final_tokens: int,
    device: torch.device
) -> Dict[str, Any]:
    """
    Mode 1: Generate reasoning token-by-token with early exit.

    Returns:
        Dictionary with results including reasoning_content, content, predictions_stream
    """
    problem = sample['problem']
    data_source = sample.get('data_source', 'math')

    # Format prompt
    formatted_prompt = format_prompt(problem, data_source)
    messages = [{"role": "user", "content": formatted_prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize prompt
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors='pt'
    )['input_ids'].to(device).long()

    # Start with prompt + <think>
    input_ids = torch.cat([
        prompt_ids,
        torch.tensor([[think_token_id]], dtype=torch.long, device=device)
    ], dim=1).long()  # Ensure final tensor is long dtype

    # Track predictions and tokens
    predictions_list = []
    reasoning_tokens = []

    # Setup sampling (temperature=0.6, top_p=0.95, top_k=20, min_p=0)
    logits_processor = LogitsProcessorList([
        TemperatureLogitsWarper(temperature=0.6),
        TopKLogitsWarper(top_k=20),
        TopPLogitsWarper(top_p=0.95),
        MinPLogitsWarper(min_p=0.0)  # 0 means no filtering
    ])

    # Track exit reason
    early_exit = False
    exit_reason = 'max_length'

    # Generate reasoning token-by-token
    for step in range(max_reasoning_tokens):
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)

        # Get hidden states from specific layer using forward hook
        captured_hidden_states = {}

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured_hidden_states['hidden'] = output[0].detach()
            else:
                captured_hidden_states['hidden'] = output.detach()

        target_layer = model.model.layers[layer_idx]
        hook_handle = target_layer.register_forward_hook(hook_fn)

        # Forward pass through base model
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False
            )

        hook_handle.remove()

        # Get hidden states
        hidden_states = captured_hidden_states['hidden']  # [1, seq_len, hidden_size]

        # Make prediction if we have generated at least one thinking token
        # The hidden state at position -1 is for the last token in input_ids
        if len(reasoning_tokens) > 0:
            # Current thinking position (0-indexed position of last generated token)
            current_thinking_pos = len(reasoning_tokens) - 1

            # Check if we can make a prediction
            # With lag_size L, we need at least L+1 tokens to make the first prediction
            # We predict for position (current_thinking_pos - lag_size) using h_current
            if lag_size == 0 or current_thinking_pos >= lag_size:
                # For extra layers, pass the FULL sequence (prompt + all reasoning so far)
                # to match training where extra layers see complete context with proper position IDs
                if extra_layers is not None:
                    # Pass entire hidden state sequence through extra layers with full attention mask
                    # This preserves absolute position IDs for rotary embeddings and gives full context
                    hidden_states_fp32 = hidden_states.float()
                    hidden_states_processed = extra_layers(hidden_states_fp32, attention_mask=attention_mask)

                    # Extract last position for prediction
                    last_hidden = hidden_states_processed[:, -1:, :]

                    # Pass through FFN only (extra layers already applied)
                    logits = ffn(last_hidden)
                    prediction = torch.sigmoid(logits)
                else:
                    # No extra layers: use last hidden state directly
                    last_hidden = hidden_states[:, -1:, :]
                    prediction = apply_predictor(last_hidden, ffn, None, None)

                predictions_list.append(prediction[0, 0].item())

                # Check exit condition immediately after adding prediction
                predictions_tensor = torch.tensor(predictions_list, device=device)
                current_pred_pos = len(predictions_list) - 1

                if decoding_strategy.should_exit(predictions_tensor, current_pred_pos):
                    early_exit = True
                    exit_reason = 'predictor'
                    break

        # Get next token logits
        next_token_logits = outputs.logits[:, -1:, :]  # [1, 1, vocab_size]

        # Apply sampling
        next_token_logits = logits_processor(input_ids, next_token_logits)

        # Sample next token
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs.squeeze(1), num_samples=1)  # [1, 1]

        # Check if model naturally ended thinking (generated </think>)
        if next_token.item() == think_end_token_id:
            exit_reason = 'natural_end'
            break

        # Add to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
        reasoning_tokens.append(next_token.item())

    # Decode reasoning content (excluding <think>)
    reasoning_content = tokenizer.decode(reasoning_tokens, skip_special_tokens=False)

    # Add exit message + </think> (only if early exit via predictor)
    if early_exit:
        # Only tokenize exit_message if it's not None and not empty
        if exit_message:
            exit_message_ids = tokenizer(
                exit_message,
                add_special_tokens=False,
                return_tensors='pt'
            )['input_ids'].to(device).long()

            input_ids = torch.cat([
                input_ids,
                exit_message_ids,
                torch.tensor([[think_end_token_id]], dtype=torch.long, device=device)
            ], dim=1).long()  # Ensure final tensor is long dtype
        else:
            # No exit message, just add </think>
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[think_end_token_id]], dtype=torch.long, device=device)
            ], dim=1).long()  # Ensure final tensor is long dtype
    else:
        # Natural end or max_length: just add </think>
        input_ids = torch.cat([
            input_ids,
            torch.tensor([[think_end_token_id]], dtype=torch.long, device=device)
        ], dim=1).long()  # Ensure final tensor is long dtype

    # Generate final answer with sampling
    attention_mask = torch.ones_like(input_ids)

    # Final safety check: ensure input_ids is long dtype
    if input_ids.dtype != torch.long:
        logger.warning(f"Converting input_ids from {input_ids.dtype} to torch.long")
        input_ids = input_ids.long()

    with torch.no_grad():
        final_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_final_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Extract final content (everything after </think>)
    full_sequence = final_outputs[0]
    think_end_positions = (full_sequence == think_end_token_id).nonzero(as_tuple=True)[0]

    if len(think_end_positions) > 0:
        think_end_pos = think_end_positions[-1].item()
        final_tokens = full_sequence[think_end_pos + 1:]
        final_content = tokenizer.decode(final_tokens, skip_special_tokens=True)
    else:
        final_content = ""

    return {
        'reasoning_content': reasoning_content,
        'content': final_content,
        'predictions_stream': predictions_list,
        'exit_step': len(reasoning_tokens),
        'exit_reason': exit_reason
    }


def mode2_parallel(
    sample: Dict[str, Any],
    solution_index: int,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    ffn: LayerFFN,
    extra_layers: Optional[ExtraTransformerLayers],
    layer_idx: int,
    lag_size: int,
    think_token_id: int,
    think_end_token_id: int,
    decoding_strategy: DecodingStrategy,
    exit_message: str,
    max_final_tokens: int,
    device: torch.device
) -> Dict[str, Any]:
    """
    Mode 2: Use existing reasoning trace, predict all positions in parallel.

    Returns:
        Dictionary with results including reasoning_content, content, predictions_stream,
        token_savings statistics
    """
    # Get reasoning content from sampled_solutions
    if 'sampled_solutions' in sample and len(sample['sampled_solutions']) > solution_index:
        reasoning_content = sample['sampled_solutions'][solution_index]['reasoning_content']
    elif 'reasoning_content' in sample:
        reasoning_content = sample['reasoning_content']
    else:
        raise ValueError(f"Sample does not contain reasoning_content or sampled_solutions[{solution_index}]")

    problem = sample['problem']
    data_source = sample.get('data_source', 'math')

    # Format prompt (same as training)
    formatted_prompt = format_prompt(problem, data_source)
    messages = [{"role": "user", "content": formatted_prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize each part
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors='pt'
    )['input_ids'].to(device).long()

    reasoning_ids = tokenizer(
        reasoning_content,
        add_special_tokens=False,
        return_tensors='pt'
    )['input_ids'].to(device).long()

    # Construct full sequence: prompt + <think> + reasoning + </think>
    input_ids = torch.cat([
        prompt_ids,
        torch.tensor([[think_token_id]], dtype=torch.long, device=device),
        reasoning_ids,
        torch.tensor([[think_end_token_id]], dtype=torch.long, device=device)
    ], dim=1).long()  # Ensure final tensor is long dtype

    # Create attention mask
    attention_mask = torch.ones_like(input_ids)

    # Safety check: ensure input_ids is long dtype
    if input_ids.dtype != torch.long:
        logger.warning(f"Converting input_ids from {input_ids.dtype} to torch.long")
        input_ids = input_ids.long()

    # Get hidden states from specific layer
    captured_hidden_states = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured_hidden_states['hidden'] = output[0].detach()
        else:
            captured_hidden_states['hidden'] = output.detach()

    target_layer = model.model.layers[layer_idx]
    hook_handle = target_layer.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        )

    hook_handle.remove()

    # Get hidden states
    hidden_states = captured_hidden_states['hidden']  # [1, seq_len, hidden_size]

    # Apply extra layers to FULL sequence first (if present) to match training
    # During training, extra layers see the complete sequence with proper position IDs
    if extra_layers is not None:
        hidden_states_fp32 = hidden_states.float()
        hidden_states = extra_layers(hidden_states_fp32, attention_mask=attention_mask)
        # Note: keep as FP32 after extra layers, will be used by FFN

    # Find <think> and </think> positions
    think_positions = (input_ids[0] == think_token_id).nonzero(as_tuple=True)[0]
    think_end_positions = (input_ids[0] == think_end_token_id).nonzero(as_tuple=True)[0]

    think_start_pos = think_positions[0].item()
    think_end_pos = think_end_positions[-1].item()

    # Valid region: tokens inside <think>...</think>, excluding both markers
    # Training labels only cover this interior region, not the </think> token
    thinking_start = think_start_pos + 1
    thinking_end = think_end_pos  # Exclude </think>

    # Extract hidden states for thinking region AFTER extra layers processed full sequence
    thinking_hidden = hidden_states[:, thinking_start:thinking_end, :]
    thinking_attention = attention_mask[:, thinking_start:thinking_end]

    # Apply lag if specified
    if lag_size > 0:
        # Shift: use h_{t+lag_size} to predict at position t
        thinking_hidden = thinking_hidden[:, lag_size:, :]
        thinking_attention = thinking_attention[:, :-lag_size]

    # Get predictions for all positions in parallel
    # Pass extra_layers=None since we already applied them above
    predictions = apply_predictor(thinking_hidden, ffn, extra_layers=None, attention_mask=thinking_attention)
    predictions = predictions[0]  # [seq_len]

    predictions_list = predictions.cpu().tolist()

    # Find exit position using decoding strategy
    exit_position = None
    predictions_tensor = predictions.cpu()

    for pos in range(len(predictions_list)):
        if decoding_strategy.should_exit(predictions_tensor, pos):
            exit_position = pos
            break

    # Calculate token savings
    original_tokens = len(reasoning_content.split())  # Rough estimate
    original_token_ids = reasoning_ids.size(1)

    if exit_position is not None:
        # Map back to original sequence position
        if lag_size > 0:
            # Adjust for lag
            actual_exit_pos = exit_position
        else:
            actual_exit_pos = exit_position

        # Truncate reasoning content at exit position
        # Map token position back to character position (rough approximation)
        exit_token_idx = actual_exit_pos

        # Decode tokens up to exit position
        reasoning_tokens_up_to_exit = reasoning_ids[0, :exit_token_idx + 1]
        truncated_reasoning = tokenizer.decode(reasoning_tokens_up_to_exit, skip_special_tokens=False)

        new_token_count = exit_token_idx + 1
    else:
        # No exit found, use full reasoning
        truncated_reasoning = reasoning_content
        new_token_count = original_token_ids
        exit_position = len(predictions_list) - 1

    # Reconstruct input for final answer generation
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors='pt'
    )['input_ids'].to(device).long()

    truncated_reasoning_ids = tokenizer(
        truncated_reasoning,
        add_special_tokens=False,
        return_tensors='pt'
    )['input_ids'].to(device).long()

    # Build input_ids_for_final: only include exit_message if it's not None/empty
    if exit_message:
        exit_message_ids = tokenizer(
            exit_message,
            add_special_tokens=False,
            return_tensors='pt'
        )['input_ids'].to(device).long()

        input_ids_for_final = torch.cat([
            prompt_ids,
            torch.tensor([[think_token_id]], dtype=torch.long, device=device),
            truncated_reasoning_ids,
            exit_message_ids,
            torch.tensor([[think_end_token_id]], dtype=torch.long, device=device)
        ], dim=1).long()  # Ensure final tensor is long dtype

        exit_message_token_count = exit_message_ids.size(1)
    else:
        # No exit message
        input_ids_for_final = torch.cat([
            prompt_ids,
            torch.tensor([[think_token_id]], dtype=torch.long, device=device),
            truncated_reasoning_ids,
            torch.tensor([[think_end_token_id]], dtype=torch.long, device=device)
        ], dim=1).long()  # Ensure final tensor is long dtype

        exit_message_token_count = 0

    # Generate final answer with sampling
    attention_mask = torch.ones_like(input_ids_for_final)

    # Final safety check: ensure input_ids is long dtype
    if input_ids_for_final.dtype != torch.long:
        logger.warning(f"Converting input_ids from {input_ids_for_final.dtype} to torch.long")
        input_ids_for_final = input_ids_for_final.long()

    with torch.no_grad():
        final_outputs = model.generate(
            input_ids=input_ids_for_final,
            attention_mask=attention_mask,
            max_new_tokens=max_final_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Extract final content
    full_sequence = final_outputs[0]
    think_end_positions = (full_sequence == think_end_token_id).nonzero(as_tuple=True)[0]

    if len(think_end_positions) > 0:
        think_end_pos = think_end_positions[-1].item()
        final_tokens = full_sequence[think_end_pos + 1:]
        final_content = tokenizer.decode(final_tokens, skip_special_tokens=True)
    else:
        final_content = ""

    # Calculate savings ratio (not including exit message)
    # Note: exit_message_token_count was already set above based on whether exit_message was provided
    savings_ratio = (new_token_count) / original_token_ids if original_token_ids > 0 else 1.0

    return {
        'reasoning_content': truncated_reasoning,
        'content': final_content,
        'predictions_stream': predictions_list,
        'exit_position': exit_position,
        'original_reasoning_tokens': original_token_ids,
        'new_reasoning_tokens': new_token_count,
        'exit_message_tokens': exit_message_token_count,
        'savings_ratio': savings_ratio,
        'tokens_saved': original_token_ids - new_token_count
    }


def run_inference(
    input_file: Path,
    checkpoint_dir: Path,
    output_file: Path,
    mode: int,
    decoding_strategy: str,
    exit_message: str,
    max_reasoning_tokens: int,
    max_final_tokens: int,
    data_sources: List[str],
    device: torch.device,
    **strategy_kwargs
) -> None:
    """Main inference loop."""

    # First, load checkpoint to get config and model name
    checkpoint_dir = Path(checkpoint_dir)
    layer_files = list(checkpoint_dir.glob('layer_*.pt'))
    if layer_files:
        checkpoint_path = layer_files[0]
    else:
        checkpoint_path = checkpoint_dir / 'checkpoint.pt'

    logger.info(f"Loading config from {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
    model_name = checkpoint_data['config']['model_name']

    # Load base model and tokenizer FIRST
    logger.info(f"Loading base model: {model_name}")
    model, tokenizer, think_token_id, think_end_token_id = load_base_model_and_tokenizer(
        model_name, device
    )

    # Now load checkpoint with base model available for correct norm layer
    logger.info(f"Loading checkpoint from {checkpoint_dir}")
    ffn, extra_layers_state_dict, config, layer_idx = load_checkpoint_for_inference(
        checkpoint_dir, device, base_model=model
    )

    # Reconstruct extra layers if present
    extra_layers = None
    if extra_layers_state_dict is not None:
        extra_layers = reconstruct_extra_layers(
            model, extra_layers_state_dict, config, device
        )

    # Get lag_size from config
    lag_size = config.get('lag_size', 0)
    logger.info(f"Using lag_size: {lag_size}")

    # Create decoding strategy
    strategy = create_decoding_strategy(decoding_strategy, **strategy_kwargs)
    logger.info(f"Decoding strategy: {strategy.name} with params {strategy.kwargs}")

    # Load input data
    logger.info(f"Loading input data from {input_file}")
    samples = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    logger.info(f"Loaded {len(samples)} samples")

    # Filter by data sources
    logger.info(f"Filtering for data sources: {data_sources}")
    samples_before_filter = len(samples)
    samples = [s for s in samples if s.get('data_source') in data_sources]
    samples_filtered = samples_before_filter - len(samples)
    if samples_filtered > 0:
        logger.info(f"Filtered out {samples_filtered} samples not matching data sources")
    logger.info(f"Remaining samples after filtering: {len(samples)}")

    # Prepare output
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if output file exists and read already-processed samples
    # Track (sample_idx, solution_index) tuples for resume capability
    processed_ids = set()
    file_mode = 'w'

    if output_file.exists():
        logger.info(f"Output file exists: {output_file}")
        logger.info("Reading already-processed samples to resume...")

        try:
            with open(output_file, 'r') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        sample_idx = record.get('sample_idx')
                        solution_index = record.get('solution_index', 0)
                        if sample_idx is not None:
                            processed_ids.add((sample_idx, solution_index))

            logger.info(f"Found {len(processed_ids)} already-processed (sample, solution) pairs")
            logger.info("Resuming inference - will skip already-processed pairs")
            file_mode = 'a'  # Append mode
        except Exception as e:
            logger.error(f"Error reading existing output file: {e}")
            logger.info("Starting fresh (will overwrite existing file)")
            processed_ids = set()
            file_mode = 'w'
    else:
        logger.info("Starting fresh inference")

    # Open output file for writing or appending
    outfile = open(output_file, file_mode)

    # Track statistics for summary
    num_processed = 0
    if mode == 2:
        total_savings_ratio = 0.0
        total_tokens_saved = 0
        total_original_tokens = 0

    logger.info(f"Running inference in mode {mode}...")
    try:
        for sample in tqdm(samples, desc="Processing samples"):
            sample_idx = sample.get('sample_idx')

            # Determine number of solutions to process
            if 'sampled_solutions' in sample:
                num_solutions = len(sample['sampled_solutions'])
            else:
                num_solutions = 1

            # Process each solution
            for solution_index in range(num_solutions):
                # Skip if already processed
                if (sample_idx, solution_index) in processed_ids:
                    continue

                try:
                    if mode == 1:
                        result = mode1_generate(
                            sample, solution_index, model, tokenizer, ffn, extra_layers,
                            layer_idx, lag_size, think_token_id, think_end_token_id,
                            strategy, exit_message, max_reasoning_tokens, max_final_tokens,
                            device
                        )
                    else:  # mode == 2
                        result = mode2_parallel(
                            sample, solution_index, model, tokenizer, ffn, extra_layers,
                            layer_idx, lag_size, think_token_id, think_end_token_id,
                            strategy, exit_message, max_final_tokens,
                            device
                        )

                    # Prepare output record
                    output_record = {
                        # Preserved fields
                        'sample_idx': sample_idx,
                        'solution_index': solution_index,
                        'problem': sample.get('problem'),
                        'answer': sample.get('answer'),
                        'data_source': sample.get('data_source'),
                        'data_fields': sample.get('data_fields'),

                        # New fields from inference
                        'reasoning_content': result['reasoning_content'],
                        'content': result['content'],
                        'predictions_stream': result['predictions_stream'],

                        # Inference configuration
                        'inference_config': {
                            'mode': mode,
                            'decoding_strategy': strategy.name,
                            'strategy_params': strategy.kwargs,
                            'exit_message': exit_message,
                            'checkpoint_dir': str(checkpoint_dir),
                            'model_name': model_name,
                            'layer_idx': layer_idx,
                            'lag_size': lag_size,
                        }
                    }

                    # Add mode-specific fields
                    if mode == 1:
                        output_record['exit_step'] = result.get('exit_step')
                        output_record['exit_reason'] = result.get('exit_reason')
                    else:  # mode 2
                        output_record['exit_position'] = result.get('exit_position')
                        output_record['original_reasoning_tokens'] = result.get('original_reasoning_tokens')
                        output_record['new_reasoning_tokens'] = result.get('new_reasoning_tokens')
                        output_record['exit_message_tokens'] = result.get('exit_message_tokens')
                        output_record['savings_ratio'] = result.get('savings_ratio')
                        output_record['tokens_saved'] = result.get('tokens_saved')

                        # Update statistics for mode 2
                        total_savings_ratio += result.get('savings_ratio', 1.0)
                        total_tokens_saved += result.get('tokens_saved', 0)
                        total_original_tokens += result.get('original_reasoning_tokens', 0)

                    # Write result immediately to file
                    json.dump(output_record, outfile)
                    outfile.write('\n')
                    outfile.flush()  # Ensure it's written to disk

                    num_processed += 1

                except Exception as e:
                    logger.error(f"Error processing sample {sample_idx}, solution {solution_index}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    finally:
        # Close output file
        outfile.close()

    logger.info(f"✓ Inference complete! Processed {num_processed} samples")
    logger.info(f"Results written to {output_file}")

    # Print summary statistics for mode 2
    if mode == 2 and num_processed > 0:
        avg_savings_ratio = total_savings_ratio / num_processed

        logger.info(f"\nToken Savings Summary (Mode 2):")
        logger.info(f"  Average savings ratio: {avg_savings_ratio:.3f} (new/original)")
        logger.info(f"  Total tokens saved: {total_tokens_saved:,}")
        logger.info(f"  Total original tokens: {total_original_tokens:,}")
        logger.info(f"  Overall compression: {1 - avg_savings_ratio:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with trained Terminator models (BCE or Focal loss)"
    )

    # Required arguments
    parser.add_argument('--checkpoint-dir', type=Path, required=True,
                        help='Directory containing trained model checkpoint')
    parser.add_argument('--input-file', type=Path, required=True,
                        help='Input JSONL file with samples')
    parser.add_argument('--loss-type', type=str, required=True,
                        choices=['bce', 'focal'],
                        help='Loss type used during training (used in output filename)')
    parser.add_argument('--data-sources', type=str, nargs='+', required=True,
                        choices=['math-500', 'aime_2025', 'humaneval', 'gpqa', 'bigcodebench'],
                        help='List of data sources to process (e.g., --data-sources math-500 aime_2025)')
    parser.add_argument('--output-file', type=Path, default=None,
                        help='Output JSONL file (default: auto-generated based on input and strategy)')

    # Mode selection
    parser.add_argument('--mode', type=int, choices=[1, 2], default=1,
                        help='Inference mode: 1=generative (token-by-token), 2=parallel (existing trace)')

    # Decoding strategy
    parser.add_argument('--decoding-strategy', type=str,
                        choices=['sliding_window', 'confidence_threshold', 'cumulative', 'first_flip'],
                        default='sliding_window',
                        help='Strategy for determining when to exit generation')

    # Strategy-specific parameters
    parser.add_argument('--window-size', type=int, default=5,
                        help='Window size for sliding_window and cumulative strategies')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for sliding_window (default 0.5), confidence_threshold, cumulative, and first_flip strategies')
    parser.add_argument('--consecutive', type=int, default=3,
                        help='Number of consecutive tokens for confidence_threshold strategy')

    # Generation parameters
    parser.add_argument('--exit-message', type=str,
                        default="\nI've run out of thinking tokens. I need to commit to a final answer.",
                        help='Message to insert before </think> when exiting early')
    parser.add_argument('--max-reasoning-tokens', type=int, default=24576,
                        help='Maximum number of reasoning tokens to generate (mode 1 only)')
    parser.add_argument('--max-final-tokens', type=int, default=8192,
                        help='Maximum number of tokens for final answer generation')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on')

    args = parser.parse_args()

    # Handle exit_message: convert empty string or "None" to Python None
    if args.exit_message == "" or args.exit_message == "None":
        args.exit_message = None

    # Auto-generate output file if not specified
    if args.output_file is None:
        input_stem = args.input_file.stem
        input_dir = args.input_file.parent

        # Collect strategy kwargs for naming
        strategy_kwargs = {
            'window_size': args.window_size,
            'threshold': args.threshold,
            'consecutive': args.consecutive,
        }

        # Create strategy to get params string
        strategy = create_decoding_strategy(args.decoding_strategy, **strategy_kwargs)
        strategy_params_str = get_strategy_params_string(strategy)

        # Build output filename
        output_name = f"{input_stem}_{args.decoding_strategy}_{strategy_params_str}_{args.loss_type}"

        # Append data source name if only one is specified
        if len(args.data_sources) == 1:
            output_name += f"_{args.data_sources[0]}"

        output_name += ".jsonl"
        args.output_file = input_dir / output_name

    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Data sources: {args.data_sources}")
    logger.info(f"Decoding strategy: {args.decoding_strategy}")
    logger.info(f"Exit message: {repr(args.exit_message)}")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Collect strategy kwargs
    strategy_kwargs = {
        'window_size': args.window_size,
        'threshold': args.threshold,
        'consecutive': args.consecutive,
    }

    # Run inference
    run_inference(
        input_file=args.input_file,
        checkpoint_dir=args.checkpoint_dir,
        output_file=args.output_file,
        mode=args.mode,
        decoding_strategy=args.decoding_strategy,
        exit_message=args.exit_message,
        max_reasoning_tokens=args.max_reasoning_tokens,
        max_final_tokens=args.max_final_tokens,
        data_sources=args.data_sources,
        device=device,
        **strategy_kwargs
    )


if __name__ == '__main__':
    main()
