#!/usr/bin/env python3
"""
Train a single feed-forward network at a specific layer to predict answer span positions.

This script trains one FFN attached to a specific layer of the base model
to predict whether the final answer has been generated at each token position.

Uses weighted Binary Cross-Entropy Loss with optional class weighting.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import f1_score

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from terminator_utils import ExtraTransformerLayers, LayerFFN, load_samples


def filter_samples_by_data_source(
    samples: List[Dict[str, Any]],
    allowed_sources: List[str],
    split_name: str = "train"
) -> List[Dict[str, Any]]:
    """
    Filter samples by data_source field.

    Args:
        samples: List of samples to filter
        allowed_sources: List of allowed data source names
        split_name: Name of the split for logging (e.g., "train", "val")

    Returns:
        Filtered list of samples
    """
    # If all four sources are enabled, skip filtering
    if set(allowed_sources) == {'math', 'aime', 'openscience', 'opencoder'}:
        print(f"\nAll data sources enabled for {split_name} set, skipping filtering")
        return samples

    allowed_sources_set = set(allowed_sources)

    # Track statistics
    source_counts = {}  # Total count per source
    source_kept = {}    # Kept count per source

    filtered_samples = []

    print(f"\nFiltering {split_name} samples by data_source...")
    print(f"  Allowed sources: {', '.join(sorted(allowed_sources))}")

    for sample in tqdm(samples, desc=f"Filtering {split_name}"):
        data_source = sample.get('data_source', 'unknown')

        # Track counts
        if data_source not in source_counts:
            source_counts[data_source] = 0
            source_kept[data_source] = 0
        source_counts[data_source] += 1

        # Keep if data_source starts with any allowed prefix
        # e.g., "aime" matches "aime_1984", "aime_2006", etc.
        # Also handle exact match for sources without suffixes
        should_keep = False
        for allowed_source in allowed_sources_set:
            if data_source == allowed_source or data_source.startswith(allowed_source + '_'):
                should_keep = True
                break

        if should_keep:
            filtered_samples.append(sample)
            source_kept[data_source] += 1

    # Print statistics
    print(f"\n{split_name.capitalize()} set filtering results:")
    print(f"  Total samples: {len(samples)} → {len(filtered_samples)}")
    print(f"  Filtered out: {len(samples) - len(filtered_samples)}")
    print(f"\n  Per-source breakdown:")

    for source in sorted(source_counts.keys()):
        kept = source_kept[source]
        total = source_counts[source]
        status = "KEPT" if source in allowed_sources_set else "FILTERED OUT"
        print(f"    {source:12s}: {kept:6d} / {total:6d} ({status})")

    return filtered_samples


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss with class-specific weights.

    Loss = -[w_1 * y * log(p) + w_0 * (1-y) * log(1-p)]

    where:
    - y is the true label (0 or 1)
    - p is the predicted probability (sigmoid of logits)
    - w_0 is the weight for class 0
    - w_1 is the weight for class 1

    This implementation takes logits as input and uses binary_cross_entropy_with_logits
    for numerical stability (applies sigmoid internally using log-sum-exp trick).

    Args:
        weight_0: Weight for class 0 (before boundary). Default 1.0 (balanced).
        weight_1: Weight for class 1 (after boundary). Default 1.0 (balanced).
        reduction: 'none', 'mean', or 'sum'. Default 'none' to allow custom masking.
    """

    def __init__(self, weight_0: float = 1.0, weight_1: float = 1.0, reduction: str = 'none'):
        super().__init__()
        self.weight_0 = weight_0
        self.weight_1 = weight_1
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits (before sigmoid), shape [batch, seq_len]
            targets: Ground truth labels (0 or 1), shape [batch, seq_len]

        Returns:
            loss: Weighted BCE loss values, shape depends on reduction
        """
        # Binary cross entropy with logits (numerically stable)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Apply class-specific weights
        # For each position: weight = w_1 if label=1, w_0 if label=0
        weights = self.weight_1 * targets + self.weight_0 * (1 - targets)
        weighted_loss = bce_loss * weights

        if self.reduction == 'none':
            return weighted_loss
        elif self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class AnswerSpanDataset(Dataset):
    """Dataset for answer span prediction task (using preprocessed data)."""

    def __init__(self, samples: List[Dict[str, Any]]):
        """
        Initialize dataset with preprocessed samples.

        Args:
            samples: List of preprocessed samples with 'input_ids', 'labels', 'valid_mask' fields
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a preprocessed sample.

        The sample already contains tokenized input_ids, labels, and valid_mask
        from the preprocessing step (prepare_span_training_data.py).

        For BCE loss, we convert labels to float32 and change ignore index from -100 to -1.
        """
        sample = self.samples[idx]

        # Load preprocessed fields (already tokenized and labeled)
        input_ids = torch.tensor(sample['input_ids'], dtype=torch.long)
        labels = torch.tensor(sample['labels'], dtype=torch.long)

        # Convert labels for BCE loss: -100 (ignore) → -1.0, 0 → 0.0, 1 → 1.0
        labels_float = labels.float()
        labels_float[labels == -100] = -1.0

        # Create attention mask (all 1s for valid tokens, padding handled by collate_fn)
        attention_mask = torch.ones_like(input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels_float,
        }


def create_collate_fn(pad_token_id):
    """Create collate function with proper pad token."""
    def collate_fn(batch):
        """Collate batch with padding."""
        # Find max length in batch
        max_len = max(item['input_ids'].size(0) for item in batch)

        # Pad all sequences
        input_ids = []
        attention_mask = []
        labels = []

        for item in batch:
            seq_len = item['input_ids'].size(0)
            pad_len = max_len - seq_len

            # Pad input_ids with tokenizer.pad_token_id
            padded_input_ids = F.pad(item['input_ids'], (0, pad_len), value=pad_token_id)
            input_ids.append(padded_input_ids)

            # Pad attention_mask with 0
            padded_attention_mask = F.pad(item['attention_mask'], (0, pad_len), value=0)
            attention_mask.append(padded_attention_mask)

            # Pad labels with -1 (ignore index)
            padded_labels = F.pad(item['labels'], (0, pad_len), value=-1.0)
            labels.append(padded_labels)

        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(labels),
        }

    return collate_fn


def compute_metrics(predictions, labels, attention_mask):
    """Compute comprehensive metrics for binary classification."""
    # Convert to float32 first (numpy doesn't support bfloat16)
    predictions = predictions.float().cpu().numpy()
    labels = labels.float().cpu().numpy()
    attention_mask = attention_mask.float().cpu().numpy()

    # Only consider non-padded positions
    mask = attention_mask.astype(bool)

    predictions_flat = predictions[mask]
    labels_flat = labels[mask]

    # Convert probabilities to binary predictions
    preds_binary = (predictions_flat > 0.5).astype(int)
    labels_binary = labels_flat.astype(int)

    # Overall accuracy
    accuracy = (preds_binary == labels_binary).mean()

    # Per-class accuracy
    class_0_mask = labels_binary == 0
    class_1_mask = labels_binary == 1

    if class_0_mask.sum() > 0:
        class_0_accuracy = (preds_binary[class_0_mask] == labels_binary[class_0_mask]).mean()
    else:
        class_0_accuracy = 0.0

    if class_1_mask.sum() > 0:
        class_1_accuracy = (preds_binary[class_1_mask] == labels_binary[class_1_mask]).mean()
    else:
        class_1_accuracy = 0.0

    # F1 scores (handle edge cases)
    if len(np.unique(labels_binary)) > 1:
        # Macro F1: average of F1 for both classes
        f1_macro = f1_score(labels_binary, preds_binary, average='macro', zero_division=0)
        # F1 for class 0 (negative class)
        f1_class_0 = f1_score(labels_binary, preds_binary, pos_label=0, zero_division=0)
        # F1 for class 1 (positive class)
        f1_class_1 = f1_score(labels_binary, preds_binary, pos_label=1, zero_division=0)
    else:
        f1_macro = 0.0
        f1_class_0 = 0.0
        f1_class_1 = 0.0

    return {
        'accuracy': accuracy,
        'class_0_accuracy': class_0_accuracy,
        'class_1_accuracy': class_1_accuracy,
        'f1_macro': f1_macro,
        'f1_class_0': f1_class_0,
        'f1_class_1': f1_class_1,
    }


def compute_position_weights(
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    w_early: float,
    w_mid: float,
    w_late: float,
    decay_len: int
) -> torch.Tensor:
    """
    Compute position-based weights for loss (vectorized, no CPU sync).

    The weighting scheme applies different weights based on position relative to
    the boundary where labels transition from 0 to 1:
    - [0, boundary): w_early
    - [boundary, boundary+decay_len): linear interpolation from w_early to w_mid
    - [boundary+decay_len, end]: w_late

    Args:
        labels: [batch, seq_len] ground truth labels (0 or 1)
        attention_mask: [batch, seq_len] attention mask
        w_early: weight for positions before boundary (class 0 region)
        w_mid: weight at end of decay region
        w_late: weight after decay region
        decay_len: number of positions to linearly decay over

    Returns:
        weights: [batch, seq_len] position-based weights
    """
    batch_size, seq_len = labels.shape
    device = labels.device

    # Create position indices [batch, seq_len]
    position_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # Find boundary for each sample: first position where label is 1
    # For positions with label=1, use position index; for label=0, use seq_len (large value)
    label_ones = (labels > 0.5).float()
    # Set invalid positions to seq_len
    masked_positions = torch.where(
        (label_ones > 0) & (attention_mask > 0),
        position_indices,
        torch.full_like(position_indices, seq_len)
    )
    # Boundary is the minimum position with label=1 (or seq_len if no 1s)
    boundaries = masked_positions.min(dim=1, keepdim=True)[0]  # [batch, 1]

    # Compute relative position from boundary
    rel_pos = position_indices - boundaries  # [batch, seq_len]

    # Initialize weights to w_late
    weights = torch.full_like(labels, w_late)

    # Before boundary: w_early
    before_boundary = rel_pos < 0
    weights = torch.where(before_boundary, torch.full_like(weights, w_early), weights)

    # In decay region [boundary, boundary+decay_len): linear interpolation
    in_decay = (rel_pos >= 0) & (rel_pos < decay_len)
    progress = rel_pos.float() / max(decay_len, 1)  # Avoid division by zero
    decay_weight = w_early + progress * (w_mid - w_early)
    weights = torch.where(in_decay, decay_weight, weights)

    # After decay region: w_late (already set as default)

    # Apply attention mask (zero out invalid positions)
    weights = weights * attention_mask

    return weights


def compute_class_weights_from_dataset(samples: List[Dict]) -> Tuple[float, float]:
    """
    Compute inverse frequency weights for classes 0 and 1 from the dataset.

    Args:
        samples: List of preprocessed samples with 'labels' field

    Returns:
        Tuple of (weight_for_class_0, weight_for_class_1)
    """
    count_0 = 0
    count_1 = 0

    for sample in samples:
        labels = sample['labels']
        # Count labels (ignore -100)
        for label in labels:
            if label == 0:
                count_0 += 1
            elif label == 1:
                count_1 += 1

    if count_0 == 0 or count_1 == 0:
        print(f"Warning: Found {count_0} class-0 labels and {count_1} class-1 labels")
        return 1.0, 1.0

    # Inverse frequency weighting
    total = count_0 + count_1
    w_0 = total / (2.0 * count_0)
    w_1 = total / (2.0 * count_1)

    print(f"Class distribution: {count_0} class-0 labels ({count_0/total*100:.1f}%), "
          f"{count_1} class-1 labels ({count_1/total*100:.1f}%)")
    print(f"Computed class weights: w_0={w_0:.4f}, w_1={w_1:.4f}")

    return w_0, w_1


def compute_per_sample_class_weights(
    labels: torch.Tensor,
    valid_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute inverse frequency class weights per sample (vectorized).

    Args:
        labels: [batch, seq_len] ground truth labels (0 or 1, with -1 for ignore)
        valid_mask: [batch, seq_len] mask for valid positions (1 for valid, 0 for ignore)

    Returns:
        Tuple of (weight_0_per_sample, weight_1_per_sample), each [batch, 1]
    """
    # Count class 0 and 1 for each sample (only valid positions)
    # labels should already be 0/1 with -1 for ignore positions
    count_0 = ((labels == 0) & (valid_mask > 0)).sum(dim=1, keepdim=True).float()  # [batch, 1]
    count_1 = ((labels == 1) & (valid_mask > 0)).sum(dim=1, keepdim=True).float()  # [batch, 1]

    # Compute inverse frequency weights per sample
    total = count_0 + count_1
    # Avoid division by zero: if count is 0, set weight to 1.0
    w_0 = torch.where(count_0 > 0, total / (2.0 * count_0), torch.ones_like(count_0))
    w_1 = torch.where(count_1 > 0, total / (2.0 * count_1), torch.ones_like(count_1))

    return w_0, w_1


def train_epoch(
    model,
    layer_idx: int,
    ffn: LayerFFN,
    extra_layers: Optional[ExtraTransformerLayers],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    global_step: int,
    criterion: WeightedBCELoss,
    save_dir: Path,
    config: Dict[str, Any],
    args: argparse.Namespace,
    lag_size: int = 0,
    use_wandb: bool = False,
    log_frequency: int = 100,
    val_log_frequency: int = 1000
) -> Tuple[Dict[str, float], int]:
    """Train for one epoch."""
    model.eval()  # Base model stays in eval mode
    ffn.train()
    if extra_layers is not None:
        extra_layers.train()

    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    # Initialize gradients
    optimizer.zero_grad()

    # Track micro-batches that actually call backward() (for correct gradient accumulation)
    micro_step = 0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device=device, dtype=torch.float32)

        # Use forward hook to capture only the specific layer's output
        captured_hidden_states = {}

        def hook_fn(module, input, output):
            # For Qwen models, output is a tuple (hidden_states, ...)
            # or just hidden_states depending on return_dict and other settings
            if isinstance(output, tuple):
                captured_hidden_states['hidden'] = output[0].detach()
            else:
                captured_hidden_states['hidden'] = output.detach()

        # Register hook on the specific layer
        target_layer = model.model.layers[layer_idx]
        hook_handle = target_layer.register_forward_hook(hook_fn)

        # Get hidden states from the specified layer only (no gradients for base model)
        with torch.no_grad():
            # Forward pass - don't need output, just triggers the hook
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False
            )

        # Remove hook
        hook_handle.remove()

        # Get hidden states for the specified layer
        layer_hidden = captured_hidden_states['hidden']  # [batch, seq_len, hidden_size] (BF16)

        # Cast BF16 hidden states to FP32 for stable training
        layer_hidden_fp32 = layer_hidden.float()

        # Pass through extra transformer layers if present
        if extra_layers is not None:
            layer_hidden_fp32 = extra_layers(layer_hidden_fp32, attention_mask=attention_mask)

        # Apply lag if specified: use h_{t+lag_size} to predict at position t
        if lag_size > 0:
            # Check if thinking region is long enough
            valid_mask_check = (labels != -1)
            thinking_length = valid_mask_check.sum(dim=1)  # [batch]

            # Skip samples where thinking_length <= lag_size (need at least lag_size+1 to predict)
            # We only need lag_size tokens to start predicting, not 2*lag_size
            skip_mask = thinking_length <= lag_size
            if skip_mask.any():
                num_skipped = skip_mask.sum().item()
                print(f"\nWARNING: Skipping {num_skipped}/{len(skip_mask)} samples with thinking_length <= {lag_size} (batch {batch_idx})")

                # If all samples need to be skipped, continue to next batch
                if skip_mask.all():
                    continue

                # Otherwise, filter out the skipped samples
                keep_mask = ~skip_mask
                layer_hidden_fp32 = layer_hidden_fp32[keep_mask]
                labels = labels[keep_mask]
                attention_mask = attention_mask[keep_mask]

            # Shift hidden states: use h_{t+lag_size} to predict at position t
            layer_hidden_fp32 = layer_hidden_fp32[:, lag_size:, :]  # [batch, seq-lag, hidden]

            # Truncate labels and attention to match
            labels = labels[:, :-lag_size]  # [batch, seq-lag]
            attention_mask = attention_mask[:, :-lag_size]  # [batch, seq-lag]

        # Forward pass through FFN (outputs logits now, not probabilities)
        logits = ffn(layer_hidden_fp32)  # [batch, seq_len]

        # Create valid mask: only positions where labels != -1 (i.e., within <think>...</think>)
        valid_mask = (labels >= 0).float()

        # Replace -1 with 0 in labels for loss computation (BCE loss expects values in [0, 1])
        labels_for_loss = labels.clone()
        labels_for_loss[labels < 0] = 0.0

        # Compute loss (only on valid positions)
        if args.use_per_sample_class_weights:
            # Compute per-sample class weights
            w_0_per_sample, w_1_per_sample = compute_per_sample_class_weights(
                labels_for_loss, valid_mask
            )  # Each is [batch, 1]

            # Compute unweighted BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels_for_loss, reduction='none')

            # Apply per-sample class weights
            # For each position: weight = w_1 if label=1, w_0 if label=0
            # Broadcast per-sample weights [batch, 1] to [batch, seq_len]
            class_weights = w_1_per_sample * labels_for_loss + w_0_per_sample * (1 - labels_for_loss)
            loss_per_token = bce_loss * class_weights
        else:
            # Use dataset-level class weights via criterion
            loss_per_token = criterion(logits, labels_for_loss)

        # Apply position-based weights if enabled
        if args.use_position_weights:
            position_weights = compute_position_weights(
                labels_for_loss, valid_mask,
                args.w_early, args.w_mid, args.w_late, args.decay_len
            )
            loss_per_token = loss_per_token * position_weights

        # Apply valid mask (only compute loss within <think>...</think>)
        loss_per_token = loss_per_token * valid_mask
        loss = loss_per_token.sum() / (valid_mask.sum() + 1e-8)

        # Scale loss by gradient accumulation steps
        loss = loss / args.gradient_accumulation_steps

        # Backward pass (accumulate gradients)
        loss.backward()

        # Track metrics (use unscaled loss for logging)
        total_loss += loss.item() * args.gradient_accumulation_steps

        with torch.no_grad():
            # Convert logits to probabilities for metrics
            predictions = torch.sigmoid(logits)
            metrics = compute_metrics(predictions, labels_for_loss, valid_mask)
            total_accuracy += metrics['accuracy']

        num_batches += 1
        micro_step += 1  # Only increment after successful backward()

        # Optimizer step every N accumulation steps (using micro_step, not batch_idx)
        if micro_step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            micro_step = 0  # Reset micro-batch counter after optimizer step
            global_step += 1

            # Log training every N optimizer steps
            if use_wandb and WANDB_AVAILABLE and global_step % log_frequency == 0:
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({
                    'train_batch/loss': total_loss / num_batches,
                    'train_batch/accuracy': total_accuracy / num_batches,
                    'train_batch/lr': current_lr,
                }, step=global_step)

            # Run validation every N optimizer steps
            if global_step % val_log_frequency == 0:
                print(f"\n  Running validation at step {global_step}...")
                val_metrics = validate(
                    model, layer_idx, ffn, extra_layers, val_loader,
                    device, global_step, criterion, args, use_wandb
                )
                # Save checkpoint after validation
                save_checkpoint(
                    ffn, extra_layers, optimizer, scheduler, epoch, global_step,
                    layer_idx, save_dir, config, args
                )
                # Set FFN back to train mode
                ffn.train()
                if extra_layers is not None:
                    extra_layers.train()

    # Flush any remaining accumulated gradients at epoch end
    if micro_step > 0:
        print(f"\n  Flushing {micro_step} accumulated micro-batches at epoch end...")
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        global_step += 1

    # Average metrics
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    # Log epoch-level metrics to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            'train/loss': avg_loss,
            'train/accuracy': avg_accuracy,
        }, step=global_step)

    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy
    }, global_step


def validate(
    model,
    layer_idx: int,
    ffn: LayerFFN,
    extra_layers: Optional[ExtraTransformerLayers],
    val_loader: DataLoader,
    device: torch.device,
    global_step: int,
    criterion: WeightedBCELoss,
    args: argparse.Namespace,
    use_wandb: bool = False
) -> Dict[str, Any]:
    """Validate the FFN."""
    model.eval()
    ffn.eval()
    if extra_layers is not None:
        extra_layers.eval()

    total_loss = 0.0
    total_metrics = {}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device=device, dtype=torch.float32)

            # Use forward hook to capture only the specific layer's output
            captured_hidden_states = {}

            def hook_fn(module, input, output):
                # For Qwen models, output is a tuple (hidden_states, ...)
                # or just hidden_states depending on return_dict and other settings
                if isinstance(output, tuple):
                    captured_hidden_states['hidden'] = output[0].detach()
                else:
                    captured_hidden_states['hidden'] = output.detach()

            # Register hook on the specific layer
            target_layer = model.model.layers[layer_idx]
            hook_handle = target_layer.register_forward_hook(hook_fn)

            # Get hidden states from the specified layer only
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False
            )

            # Remove hook
            hook_handle.remove()

            # Get hidden states for the specified layer
            layer_hidden = captured_hidden_states['hidden']  # BF16

            # Cast to FP32 to match FFN parameters
            layer_hidden_fp32 = layer_hidden.float()

            # Pass through extra transformer layers if present
            if extra_layers is not None:
                layer_hidden_fp32 = extra_layers(layer_hidden_fp32, attention_mask=attention_mask)

            # Apply lag if specified: use h_{t+lag_size} to predict at position t
            if args.lag_size > 0:
                # Check if thinking region is long enough
                valid_mask_check = (labels != -1)
                thinking_length = valid_mask_check.sum(dim=1)  # [batch]

                # Skip samples where thinking_length <= lag_size (need at least lag_size+1 to predict)
                # We only need lag_size tokens to start predicting, not 2*lag_size
                skip_mask = thinking_length <= args.lag_size
                if skip_mask.any():
                    # Skip entire batch if any sample is too short
                    # (In validation, we typically skip rather than filter)
                    if skip_mask.all():
                        continue

                    # Filter out the skipped samples
                    keep_mask = ~skip_mask
                    layer_hidden_fp32 = layer_hidden_fp32[keep_mask]
                    labels = labels[keep_mask]
                    attention_mask = attention_mask[keep_mask]

                # Shift hidden states: use h_{t+lag_size} to predict at position t
                layer_hidden_fp32 = layer_hidden_fp32[:, args.lag_size:, :]  # [batch, seq-lag, hidden]

                # Truncate labels and attention to match
                labels = labels[:, :-args.lag_size]  # [batch, seq-lag]
                attention_mask = attention_mask[:, :-args.lag_size]  # [batch, seq-lag]

            # Forward pass (outputs logits)
            logits = ffn(layer_hidden_fp32)

            # Create valid mask: only positions where labels != -1
            valid_mask = (labels >= 0).float()

            # Replace -1 with 0 in labels for loss computation
            labels_for_loss = labels.clone()
            labels_for_loss[labels < 0] = 0.0

            # Compute loss
            if args.use_per_sample_class_weights:
                # Compute per-sample class weights
                w_0_per_sample, w_1_per_sample = compute_per_sample_class_weights(
                    labels_for_loss, valid_mask
                )  # Each is [batch, 1]

                # Compute unweighted BCE loss
                bce_loss = F.binary_cross_entropy_with_logits(logits, labels_for_loss, reduction='none')

                # Apply per-sample class weights
                class_weights = w_1_per_sample * labels_for_loss + w_0_per_sample * (1 - labels_for_loss)
                loss_per_token = bce_loss * class_weights
            else:
                # Use dataset-level class weights via criterion
                loss_per_token = criterion(logits, labels_for_loss)

            # Apply position-based weights if enabled
            if args.use_position_weights:
                position_weights = compute_position_weights(
                    labels_for_loss, valid_mask,
                    args.w_early, args.w_mid, args.w_late, args.decay_len
                )
                loss_per_token = loss_per_token * position_weights

            # Apply valid mask
            loss_per_token = loss_per_token * valid_mask
            loss = loss_per_token.sum() / (valid_mask.sum() + 1e-8)

            total_loss += loss.item()

            # Compute metrics (convert logits to probabilities)
            predictions = torch.sigmoid(logits)
            metrics = compute_metrics(predictions, labels_for_loss, valid_mask)

            # Accumulate metrics
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value

            num_batches += 1

    # Average metrics
    avg_metrics = {
        key: value / num_batches
        for key, value in total_metrics.items()
    }
    avg_metrics['loss'] = total_loss / num_batches

    # Log to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            'val/loss': avg_metrics['loss'],
            'val/accuracy': avg_metrics['accuracy'],
            'val/class_0_accuracy': avg_metrics['class_0_accuracy'],
            'val/class_1_accuracy': avg_metrics['class_1_accuracy'],
            'val/f1_macro': avg_metrics['f1_macro'],
            'val/f1_class_0': avg_metrics['f1_class_0'],
            'val/f1_class_1': avg_metrics['f1_class_1'],
        }, step=global_step)

    return avg_metrics


def save_checkpoint(
    ffn: LayerFFN,
    extra_layers: Optional[ExtraTransformerLayers],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    global_step: int,
    layer_idx: int,
    save_dir: Path,
    config: Dict[str, Any],
    args: argparse.Namespace
):
    """Save checkpoint with model, optimizer, and scheduler state."""
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'layer_idx': layer_idx,
        'ffn_state_dict': ffn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'config': config,
        'args': vars(args)
    }

    # Save extra layers if present
    if extra_layers is not None:
        checkpoint['extra_layers_state_dict'] = extra_layers.state_dict()

    checkpoint_path = save_dir / 'checkpoint.pt'
    torch.save(checkpoint, checkpoint_path)


def validate_and_reconcile_checkpoint_config(
    args: argparse.Namespace,
    checkpoint_config: Dict[str, Any],
    num_layers: int
) -> Tuple[argparse.Namespace, int]:
    """
    Validate checkpoint config against CLI args and reconcile differences.

    Architectural parameters (layer_idx, num_extra_layers, model_name) must match exactly.
    Training hyperparameters are loaded from checkpoint and override CLI args with warnings.

    Args:
        args: Command-line arguments
        checkpoint_config: Config dict from checkpoint
        num_layers: Number of layers in the loaded model

    Returns:
        Tuple of (updated_args, resolved_layer_idx)

    Raises:
        ValueError: If architectural parameters don't match
    """
    print("\n" + "="*60)
    print("Validating checkpoint configuration...")
    print("="*60)

    # 1. Validate architectural parameters (must match exactly)

    # Model name
    checkpoint_model = checkpoint_config.get('model_name')
    if checkpoint_model != args.model_name:
        raise ValueError(
            f"Model mismatch!\n"
            f"  Checkpoint was trained on: {checkpoint_model}\n"
            f"  Current CLI specifies: {args.model_name}\n"
            f"  These must match exactly. Use --model-name '{checkpoint_model}' to resume."
        )

    # Layer index - resolve checkpoint's layer_idx first
    checkpoint_layer_idx = checkpoint_config.get('layer_idx')
    if checkpoint_layer_idx is None:
        raise ValueError("Checkpoint does not contain 'layer_idx' - corrupt or old format")

    # Resolve current CLI layer_idx
    if args.layer_idx < 0:
        cli_layer_idx = num_layers + args.layer_idx
    else:
        cli_layer_idx = args.layer_idx

    if cli_layer_idx != checkpoint_layer_idx:
        raise ValueError(
            f"Layer index mismatch!\n"
            f"  Checkpoint was trained on layer: {checkpoint_layer_idx}\n"
            f"  Current CLI resolves to layer: {cli_layer_idx} (from --layer-idx {args.layer_idx})\n"
            f"  Resuming with different layer would corrupt training.\n"
            f"  Use --layer-idx {checkpoint_layer_idx} to resume."
        )

    # Extra transformer layers architecture
    checkpoint_num_extra = checkpoint_config.get('num_extra_layers', 0)
    if args.num_extra_layers != checkpoint_num_extra:
        raise ValueError(
            f"Extra layers mismatch!\n"
            f"  Checkpoint was trained with: {checkpoint_num_extra} extra layer(s)\n"
            f"  Current CLI specifies: {args.num_extra_layers} extra layer(s)\n"
            f"  Architecture must match. Use --num-extra-layers {checkpoint_num_extra} to resume."
        )

    if checkpoint_num_extra > 0:
        checkpoint_init = checkpoint_config.get('extra_layers_init', 'random')
        if args.extra_layers_init != checkpoint_init:
            print(f"WARNING: extra_layers_init differs (checkpoint: {checkpoint_init}, CLI: {args.extra_layers_init})")
            print(f"         This only affects initialization, weights will be loaded from checkpoint.")

    # 2. Override training hyperparameters from checkpoint (with warnings)
    print("\nLoading training hyperparameters from checkpoint:")

    overrides = []

    # Class weighting hyperparams
    if 'use_class_weights' in checkpoint_config and args.use_class_weights != checkpoint_config['use_class_weights']:
        overrides.append(f"  use_class_weights: {args.use_class_weights} → {checkpoint_config['use_class_weights']}")
        args.use_class_weights = checkpoint_config['use_class_weights']

    if 'use_per_sample_class_weights' in checkpoint_config and args.use_per_sample_class_weights != checkpoint_config['use_per_sample_class_weights']:
        overrides.append(f"  use_per_sample_class_weights: {args.use_per_sample_class_weights} → {checkpoint_config['use_per_sample_class_weights']}")
        args.use_per_sample_class_weights = checkpoint_config['use_per_sample_class_weights']

    if 'class_weight_0' in checkpoint_config and args.class_weight_0 != checkpoint_config['class_weight_0']:
        overrides.append(f"  class_weight_0: {args.class_weight_0} → {checkpoint_config['class_weight_0']}")
        args.class_weight_0 = checkpoint_config['class_weight_0']

    if 'class_weight_1' in checkpoint_config and args.class_weight_1 != checkpoint_config['class_weight_1']:
        overrides.append(f"  class_weight_1: {args.class_weight_1} → {checkpoint_config['class_weight_1']}")
        args.class_weight_1 = checkpoint_config['class_weight_1']

    # Position weighting hyperparams
    if 'use_position_weights' in checkpoint_config and args.use_position_weights != checkpoint_config['use_position_weights']:
        overrides.append(f"  use_position_weights: {args.use_position_weights} → {checkpoint_config['use_position_weights']}")
        args.use_position_weights = checkpoint_config['use_position_weights']

    if 'w_early' in checkpoint_config and args.w_early != checkpoint_config['w_early']:
        overrides.append(f"  w_early: {args.w_early} → {checkpoint_config['w_early']}")
        args.w_early = checkpoint_config['w_early']

    if 'w_mid' in checkpoint_config and args.w_mid != checkpoint_config['w_mid']:
        overrides.append(f"  w_mid: {args.w_mid} → {checkpoint_config['w_mid']}")
        args.w_mid = checkpoint_config['w_mid']

    if 'w_late' in checkpoint_config and args.w_late != checkpoint_config['w_late']:
        overrides.append(f"  w_late: {args.w_late} → {checkpoint_config['w_late']}")
        args.w_late = checkpoint_config['w_late']

    if 'decay_len' in checkpoint_config and args.decay_len != checkpoint_config['decay_len']:
        overrides.append(f"  decay_len: {args.decay_len} → {checkpoint_config['decay_len']}")
        args.decay_len = checkpoint_config['decay_len']

    # Lag size
    if 'lag_size' in checkpoint_config and args.lag_size != checkpoint_config['lag_size']:
        overrides.append(f"  lag_size: {args.lag_size} → {checkpoint_config['lag_size']}")
        args.lag_size = checkpoint_config['lag_size']

    # FFN architecture (should match but load from checkpoint to be safe)
    if 'ffn_layers' in checkpoint_config and args.ffn_layers != checkpoint_config['ffn_layers']:
        overrides.append(f"  ffn_layers: {args.ffn_layers} → {checkpoint_config['ffn_layers']}")
        args.ffn_layers = checkpoint_config['ffn_layers']

    if 'ffn_activation' in checkpoint_config and args.ffn_activation != checkpoint_config['ffn_activation']:
        overrides.append(f"  ffn_activation: {args.ffn_activation} → {checkpoint_config['ffn_activation']}")
        args.ffn_activation = checkpoint_config['ffn_activation']

    if 'ffn_intermediate_size' in checkpoint_config:
        checkpoint_ffn_size = checkpoint_config['ffn_intermediate_size']
        if args.ffn_intermediate_size != checkpoint_ffn_size:
            overrides.append(f"  ffn_intermediate_size: {args.ffn_intermediate_size} → {checkpoint_ffn_size}")
            args.ffn_intermediate_size = checkpoint_ffn_size

    if 'dropout' in checkpoint_config and args.dropout != checkpoint_config['dropout']:
        overrides.append(f"  dropout: {args.dropout} → {checkpoint_config['dropout']}")
        args.dropout = checkpoint_config['dropout']

    if overrides:
        print("  Overriding CLI arguments with checkpoint values:")
        for override in overrides:
            print(override)
    else:
        print("  All hyperparameters match checkpoint (no overrides needed)")

    print("\n✓ Checkpoint validation passed")
    print("="*60)

    return args, checkpoint_layer_idx


def load_checkpoint(
    checkpoint_path: Path,
    ffn: LayerFFN,
    extra_layers: Optional[ExtraTransformerLayers],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
) -> Tuple[int, int, Dict[str, Any]]:
    """Load checkpoint and restore training state.

    Returns:
        Tuple of (epoch, global_step, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    ffn.load_state_dict(checkpoint['ffn_state_dict'])

    # Load extra layers if present in checkpoint
    if extra_layers is not None and 'extra_layers_state_dict' in checkpoint:
        extra_layers.load_state_dict(checkpoint['extra_layers_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']
    config = checkpoint['config']

    print(f"✓ Loaded checkpoint from {checkpoint_path}")
    print(f"  Resuming from epoch {epoch}, step {global_step}")

    return epoch, global_step, config


def save_model(
    ffn: LayerFFN,
    extra_layers: Optional[ExtraTransformerLayers],
    layer_idx: int,
    save_dir: Path,
    config: Dict[str, Any],
    args: argparse.Namespace
):
    """Save the FFN, configuration, and training arguments."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = save_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Save training arguments
    args_path = save_dir / 'args.json'
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Save FFN
    model_path = save_dir / f'layer_{layer_idx}.pt'
    model_dict = {
        'layer_idx': layer_idx,
        'state_dict': ffn.state_dict(),
        'config': config
    }

    # Save extra layers if present
    if extra_layers is not None:
        model_dict['extra_layers_state_dict'] = extra_layers.state_dict()

    torch.save(model_dict, model_path)

    print(f"\n✓ Saved model to {save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a single FFN at a specific layer for answer span prediction using weighted BCE loss"
    )

    # Data arguments
    parser.add_argument('--train-path', type=str, required=True,
                        help='Path to training data JSONL file')
    parser.add_argument('--val-path', type=str, required=True,
                        help='Path to validation data JSONL file')
    parser.add_argument('--data-sources', type=str, nargs='+',
                        default=['math', 'aime', 'openscience', 'opencoder'],
                        choices=['math', 'aime', 'openscience', 'opencoder'],
                        help='Data sources to include (default: all four)')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed for reproducibility')

    # Model arguments
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen3-8B',
                        help='Base model name')
    parser.add_argument('--layer-idx', type=int, default=-1,
                        help='Layer index to attach FFN to (-1 for last layer)')
    parser.add_argument('--num-extra-layers', type=int, default=0,
                        help='Number of trainable transformer layers to add before FFN (default: 0)')
    parser.add_argument('--extra-layers-init', type=str, default='random',
                        choices=['random', 'copy_last'],
                        help='Initialization for extra transformer layers: random or copy from last LLM layer')
    parser.add_argument('--ffn-layers', type=int, default=1,
                        help='Number of layers in the FFN (1 = linear only)')
    parser.add_argument('--ffn-activation', type=str, default='gelu',
                        choices=['relu', 'gelu', 'tanh'],
                        help='Activation function for FFN')
    parser.add_argument('--ffn-intermediate-size', type=int, default=None,
                        help='Intermediate size for FFN (default: hidden_size * 2)')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Dropout rate for FFN')
    parser.add_argument('--lag-size', type=int, default=0,
                        help='Number of future tokens to use for context (default: 0, predictions use h_t at position t)')

    # Class weighting arguments
    parser.add_argument('--use-class-weights', action='store_true',
                        help='Enable class-based loss weighting (computes inverse frequency from data if manual weights not provided)')
    parser.add_argument('--class-weight-0', type=float, default=None,
                        help='Manual weight for class 0 (if not specified with --use-class-weights, computed from inverse frequency)')
    parser.add_argument('--class-weight-1', type=float, default=None,
                        help='Manual weight for class 1 (if not specified with --use-class-weights, computed from inverse frequency)')
    parser.add_argument('--use-per-sample-class-weights', action='store_true',
                        help='Enable per-sample class weighting (computes inverse frequency weights separately for each sample). '
                             'Overrides --use-class-weights and manual class weights if enabled.')

    # Position-based weighting arguments
    parser.add_argument('--use-position-weights', action='store_true',
                        help='Enable position-based weighting for loss (default: disabled)')
    parser.add_argument('--w-early', type=float, default=5.0,
                        help='Weight for positions before boundary (class 0 region). Default: 5.0')
    parser.add_argument('--w-mid', type=float, default=3.0,
                        help='Weight at end of decay region. Default: 3.0')
    parser.add_argument('--w-late', type=float, default=2.0,
                        help='Weight after decay region. Default: 2.0')
    parser.add_argument('--decay-len', type=int, default=500,
                        help='Number of positions to linearly decay from w-early to w-mid. Default: 500')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training (per GPU, before accumulation)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Number of gradient accumulation steps (effective_batch_size = batch_size * this)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--log-freq', type=int, default=100,
                        help='Log training to W&B every N optimizer steps (not mini-batches)')
    parser.add_argument('--val-log-freq', type=int, default=1000,
                        help='Run validation every N optimizer steps (not mini-batches)')

    # Output arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument('--save-dir', type=str,
                        default=f'models/terminator_{timestamp}',
                        help='Directory to save trained model')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb-project', type=str, default='terminator',
                        help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='W&B run name')

    # Resume training
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint file to resume training from')

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Initialize wandb
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not available, disabling logging")
            args.use_wandb = False
        else:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args)
            )

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading training data from {args.train_path}...")
    train_samples = load_samples(Path(args.train_path))
    print(f"Train samples loaded: {len(train_samples)}")

    print(f"\nLoading validation data from {args.val_path}...")
    val_samples = load_samples(Path(args.val_path))
    print(f"Val samples loaded: {len(val_samples)}")

    # Filter by data source
    train_samples = filter_samples_by_data_source(train_samples, args.data_sources, "train")
    val_samples = filter_samples_by_data_source(val_samples, args.data_sources, "val")

    # Load tokenizer (only needed for pad_token_id in collate_fn)
    print(f"\nLoading tokenizer for padding: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    # Get model config
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    print(f"Model has {num_layers} layers with hidden size {hidden_size}")

    # If resuming, load checkpoint config early and validate/reconcile with CLI args
    resume_checkpoint = None
    if args.resume_from:
        print(f"\nDetected --resume-from: {args.resume_from}")
        print("Loading checkpoint config for validation...")
        resume_checkpoint = torch.load(args.resume_from, map_location='cpu')
        checkpoint_config = resume_checkpoint['config']

        # Validate and reconcile configuration
        args, layer_idx = validate_and_reconcile_checkpoint_config(
            args, checkpoint_config, num_layers
        )
    else:
        # Resolve layer index from CLI
        if args.layer_idx < 0:
            layer_idx = num_layers + args.layer_idx
        else:
            layer_idx = args.layer_idx

        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"Invalid layer index {args.layer_idx}. Model has {num_layers} layers (0-{num_layers-1})")

    print(f"\nTraining FFN at layer {layer_idx}")

    # Compute class weights if enabled
    class_weights = (1.0, 1.0)  # Default: balanced
    if args.use_per_sample_class_weights:
        # Per-sample class weighting overrides dataset-level weights
        print(f"\n*** Using per-sample class weights (computed dynamically per batch) ***")
        print(f"  Class weights will be computed separately for each sample based on its label distribution")
        print(f"  This overrides any dataset-level class weights")
        # Still compute dataset-level weights for the criterion (won't be used in training, but needed for initialization)
        if args.use_class_weights:
            if args.class_weight_0 is not None and args.class_weight_1 is not None:
                class_weights = (args.class_weight_0, args.class_weight_1)
            else:
                class_weights = compute_class_weights_from_dataset(train_samples)
    elif args.use_class_weights:
        if args.class_weight_0 is not None and args.class_weight_1 is not None:
            # Use manual weights
            class_weights = (args.class_weight_0, args.class_weight_1)
            print(f"\n*** Using manual class weights ***")
            print(f"  Class 0 weight: {class_weights[0]:.4f}")
            print(f"  Class 1 weight: {class_weights[1]:.4f}")
        else:
            # Compute from inverse frequency
            print(f"\n*** Computing class weights from training data (inverse frequency) ***")
            class_weights = compute_class_weights_from_dataset(train_samples)
    else:
        print(f"\n*** No class weighting (balanced loss weighting) ***")

    # Create datasets (data is already preprocessed and tokenized)
    train_dataset = AnswerSpanDataset(train_samples)
    val_dataset = AnswerSpanDataset(val_samples)

    # Create collate function with proper pad token
    collate_fn = create_collate_fn(tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Create FFN in FP32 for optimizer stability
    # (We'll cast BF16 inputs from LLM to FP32 before passing to FFN)
    print(f"\nInitializing FFN for layer {layer_idx}...")

    # Get normalization layer from model (use same type as model for consistency)
    # Create a fresh instance for training (don't reuse model's final norm)
    norm_layer = None
    if hasattr(model.model, 'norm') and model.model.norm is not None:
        norm_class = model.model.norm.__class__
        # Get epsilon from config if available
        eps = getattr(model.config, 'rms_norm_eps', 1e-6) if hasattr(model.config, 'rms_norm_eps') else 1e-6
        norm_layer = norm_class(hidden_size, eps=eps)
        print(f"  Using {norm_class.__name__} for pre-FFN normalization (eps={eps})")
    else:
        norm_layer = nn.LayerNorm(hidden_size)
        print(f"  Using LayerNorm for pre-FFN normalization (fallback)")

    ffn = LayerFFN(
        hidden_size=hidden_size,
        num_hidden_layers=args.ffn_layers,
        activation=args.ffn_activation,
        intermediate_size=args.ffn_intermediate_size,
        dropout=args.dropout,
        norm_layer=norm_layer
    ).to(device=device)  # FP32 (default dtype)

    # Create extra transformer layers if requested
    extra_layers = None
    if args.num_extra_layers > 0:
        print(f"\nInitializing {args.num_extra_layers} extra transformer layer(s)...")
        print(f"  Initialization method: {args.extra_layers_init}")

        # Get the layer class from the base model
        base_layer_class = model.model.layers[0].__class__
        model_config = model.config

        # Get rotary embedding module if available (needed for Qwen3 and similar models)
        rotary_emb = getattr(model.model, 'rotary_emb', None)

        # Create extra layers
        extra_layers = ExtraTransformerLayers(
            base_layer_class,
            args.num_extra_layers,
            model_config,
            rotary_emb=rotary_emb
        ).to(device=device)  # FP32 (default dtype) for stability

        # Initialize weights
        if args.extra_layers_init == 'copy_last':
            print(f"  Copying weights from layer {layer_idx}...")
            with torch.no_grad():
                last_layer = model.model.layers[layer_idx]
                for i, extra_layer in enumerate(extra_layers.layers):
                    # Copy all parameters from the last layer
                    for name, param in last_layer.named_parameters():
                        if hasattr(extra_layer, name.split('.')[0]):
                            target_param = extra_layer
                            for attr in name.split('.'):
                                target_param = getattr(target_param, attr)
                            target_param.copy_(param.data.float())  # Cast BF16 to FP32
        else:
            print(f"  Using random initialization (default PyTorch init)")

        # Make extra layers trainable
        for param in extra_layers.parameters():
            param.requires_grad = True

        print(f"  Extra layers: {sum(p.numel() for p in extra_layers.parameters()):,} parameters")

    # Create weighted BCE loss
    criterion = WeightedBCELoss(weight_0=class_weights[0], weight_1=class_weights[1], reduction='none')

    # Create optimizer (include extra layers if present)
    trainable_params = list(ffn.parameters())
    if extra_layers is not None:
        trainable_params += list(extra_layers.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Print training configuration
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Batch size (per step): {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Class weights: w_0={class_weights[0]:.4f}, w_1={class_weights[1]:.4f}")
    print(f"{'='*60}")

    # Create linear warmup + cosine annealing scheduler
    # Note: total_steps is now in terms of optimizer steps, not mini-batches
    total_mini_batches = len(train_loader) * args.epochs
    total_steps = total_mini_batches // args.gradient_accumulation_steps
    warmup_steps = min(100, total_steps)  # Clamp to avoid crash on small datasets
    remaining_steps = max(0, total_steps - warmup_steps)  # Allow 0 if no remaining steps

    if warmup_steps > 0:
        # Warmup: 1e-6 to lr over warmup_steps
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6 / args.lr,  # Start from 1e-6
            end_factor=1.0,
            total_iters=warmup_steps
        )

    if remaining_steps > 0:
        # Cosine annealing: lr to 1e-6 over remaining steps
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=remaining_steps,
            eta_min=1e-6  # Minimum LR at the end
        )

    # Create sequential scheduler
    if warmup_steps > 0 and remaining_steps > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    elif warmup_steps > 0:
        scheduler = warmup_scheduler
    elif remaining_steps > 0:
        scheduler = cosine_scheduler
    else:
        # No scheduler (constant LR) for very small datasets
        scheduler = None

    # Prepare save directory and config - save ALL arguments plus computed values
    save_dir = Path(args.save_dir)

    # Start with all CLI arguments
    config = vars(args).copy()

    # Remove resume_from (only relevant for current run, not future resumes)
    config.pop('resume_from', None)

    # Add computed/derived values
    config['layer_idx'] = layer_idx  # Resolved layer index (may differ from args.layer_idx if negative)
    config['num_layers'] = num_layers
    config['hidden_size'] = hidden_size
    config['loss_type'] = 'bce'
    config['class_weight_0'] = class_weights[0]  # Store actual weights used
    config['class_weight_1'] = class_weights[1]

    # Convert Path objects to strings for JSON serialization
    if 'train_path' in config:
        config['train_path'] = str(config['train_path'])
    if 'val_path' in config:
        config['val_path'] = str(config['val_path'])
    if 'save_dir' in config:
        config['save_dir'] = str(config['save_dir'])

    # Load checkpoint state if resuming
    start_epoch = 0
    global_step = 0
    if resume_checkpoint is not None:
        print(f"\nLoading checkpoint state (weights, optimizer, scheduler)...")

        # Load FFN state
        ffn.load_state_dict(resume_checkpoint['ffn_state_dict'])

        # Load extra layers if present in checkpoint
        if extra_layers is not None and 'extra_layers_state_dict' in resume_checkpoint:
            extra_layers.load_state_dict(resume_checkpoint['extra_layers_state_dict'])

        # Load optimizer and scheduler
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        if scheduler is not None and resume_checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])

        # Get training state
        start_epoch = resume_checkpoint['epoch']
        global_step = resume_checkpoint['global_step']
        loaded_config = resume_checkpoint['config']

        # Update config with loaded values (already validated)
        config.update(loaded_config)

        print(f"✓ Checkpoint state loaded successfully")
        print(f"  Resuming from epoch {start_epoch}, step {global_step}")
    else:
        # Run initial validation only if not resuming
        print("\nRunning initial validation...")
        initial_val_metrics = validate(
            model, layer_idx, ffn, extra_layers, val_loader,
            device, global_step=0, criterion=criterion,
            args=args, use_wandb=args.use_wandb
        )

        print("\nInitial Validation Metrics:")
        print(f"  Loss: {initial_val_metrics['loss']:.4f}")
        print(f"  Accuracy: {initial_val_metrics['accuracy']:.4f}")
        print(f"  Class 0 Acc: {initial_val_metrics['class_0_accuracy']:.4f}")
        print(f"  Class 1 Acc: {initial_val_metrics['class_1_accuracy']:.4f}")
        print(f"  F1 Macro: {initial_val_metrics['f1_macro']:.4f}")
        print(f"  F1 Class 0: {initial_val_metrics['f1_class_0']:.4f}")
        print(f"  F1 Class 1: {initial_val_metrics['f1_class_1']:.4f}")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_metrics, global_step = train_epoch(
            model, layer_idx, ffn, extra_layers, optimizer, scheduler,
            train_loader, val_loader, device, epoch, global_step,
            criterion, save_dir, config, args, args.lag_size,
            args.use_wandb, args.log_freq, args.val_log_freq
        )

        # Print train metrics
        print(f"\nTraining Metrics:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")

        # Validate at end of epoch
        print(f"\nRunning end-of-epoch validation at step {global_step}...")
        val_metrics = validate(
            model, layer_idx, ffn, extra_layers, val_loader,
            device, global_step, criterion, args, args.use_wandb
        )

        # Save checkpoint after end-of-epoch validation
        save_checkpoint(
            ffn, extra_layers, optimizer, scheduler, epoch + 1, global_step,
            layer_idx, save_dir, config, args
        )

        # Print val metrics
        print(f"\nValidation Metrics:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Class 0 Acc: {val_metrics['class_0_accuracy']:.4f}")
        print(f"  Class 1 Acc: {val_metrics['class_1_accuracy']:.4f}")
        print(f"  F1 Macro: {val_metrics['f1_macro']:.4f}")
        print(f"  F1 Class 0: {val_metrics['f1_class_0']:.4f}")
        print(f"  F1 Class 1: {val_metrics['f1_class_1']:.4f}")

    # Save final model
    save_model(ffn, extra_layers, layer_idx, save_dir, config, args)

    if args.use_wandb:
        wandb.finish()

    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()
