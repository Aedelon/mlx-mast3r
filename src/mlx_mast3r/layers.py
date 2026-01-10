# Copyright (c) 2024 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""Shared neural network layers for MLX MASt3R.

This module contains reusable building blocks shared across encoders and decoders.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class MLP(nn.Module):
    """Two-layer MLP with GELU activation.

    Args:
        in_dim: Input dimension.
        hidden_dim: Hidden layer dimension.
        out_dim: Output dimension (defaults to in_dim if None).
        fast_gelu: Use gelu_fast_approx (faster but less precise).
                   Set to False for MASt3R encoder to maintain PyTorch correlation.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int | None = None,
        fast_gelu: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim or in_dim)
        self._gelu = nn.gelu_fast_approx if fast_gelu else nn.gelu_approx

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self._gelu(self.fc1(x)))
