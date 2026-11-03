#!/usr/bin/env python3
"""
Shared utilities for Terminator training and inference scripts.

This module contains common classes and functions used by train_terminator.py
and inference_terminator.py.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn


class LayerFFN(nn.Module):
    """Feed-forward network for binary classification at each position (outputs logits)."""

    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int = 1,
        activation: str = 'gelu',
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation

        # Pre-normalization layer (critical for numerical stability)
        self.norm = norm_layer if norm_layer is not None else nn.LayerNorm(hidden_size)

        if num_hidden_layers == 1:
            # Simple linear layer (outputs logits, no sigmoid)
            self.network = nn.Linear(hidden_size, 1)
        else:
            # Multi-layer FFN
            if intermediate_size is None:
                intermediate_size = hidden_size * 2

            layers = []
            layers.append(nn.Linear(hidden_size, intermediate_size))

            # Activation function
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            layers.append(nn.Dropout(dropout))

            # Hidden layers
            for _ in range(num_hidden_layers - 2):
                layers.append(nn.Linear(intermediate_size, intermediate_size))
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                layers.append(nn.Dropout(dropout))

            # Output layer (no sigmoid, outputs logits for numerical stability)
            layers.append(nn.Linear(intermediate_size, 1))

            self.network = nn.Sequential(*layers)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            logits: [batch_size, seq_len] raw logits (before sigmoid)
        """
        # Normalize hidden states before FFN (critical for numerical stability)
        hidden_states = self.norm(hidden_states)
        # Apply network to each position
        output = self.network(hidden_states)  # [batch, seq_len, 1]
        return output.squeeze(-1)  # [batch, seq_len]


class ExtraTransformerLayers(nn.Module):
    """Trainable transformer layers stacked before the FFN."""

    def __init__(self, base_layer_class, num_layers: int, config, rotary_emb=None):
        """
        Args:
            base_layer_class: The transformer layer class from the base model
            num_layers: Number of layers to create
            config: Model config for creating layers
            rotary_emb: Optional rotary embedding module for position embeddings
        """
        super().__init__()
        self.layers = nn.ModuleList([
            base_layer_class(config, layer_idx=i)
            for i in range(num_layers)
        ])
        self.rotary_emb = rotary_emb
        self.config = config

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        """
        Forward pass through extra transformer layers.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len] simple mask
            position_ids: [batch, seq_len] (optional)

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len = hidden_states.shape[:2]

        # Prepare attention mask: convert simple [batch, seq_len] to 4D causal mask
        if attention_mask is not None:
            # If already 4D, just ensure correct dtype
            if attention_mask.dim() == 4:
                if attention_mask.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    attention_mask = attention_mask.to(hidden_states.dtype)
            # If 2D, create 4D causal attention mask
            elif attention_mask.dim() == 2:
                # Create causal mask [seq_len, seq_len]
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
                    diagonal=1
                )
                # Expand attention_mask to [batch, 1, 1, seq_len]
                expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # Combine with causal mask: [batch, 1, seq_len, seq_len]
                attention_mask = expanded_mask.masked_fill(causal_mask.unsqueeze(0), 0.0)
                # Convert to proper dtype and invert (1 -> 0, 0 -> -inf for masking)
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)
                attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
            else:
                raise ValueError(f"Unexpected attention_mask shape: {attention_mask.shape}")

        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)

        # Compute position embeddings if rotary_emb is available
        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            # Different models have different forward signatures
            # Try with position_embeddings first (Qwen3 style)
            try:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
            except TypeError:
                # Try without position_embeddings (older models)
                try:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )
                except TypeError:
                    # Some models don't use position_ids at all
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                    )

            # Extract hidden states from output (might be tuple)
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

        return hidden_states


def load_samples(data_path: Path) -> List[Dict]:
    """Load samples from a JSONL file."""
    samples = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples
