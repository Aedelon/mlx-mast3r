# Copyright (c) 2024 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""Base encoder components shared across MASt3R and DUNE.

This module provides reusable building blocks for Vision Transformer encoders:
- Attention: Multi-head self-attention with optional RoPE
- EncoderBlock: Transformer block with optional layer scale
- PatchEmbed: Patch embedding with configurable patch size
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from mlx_mast3r.constants import LAYER_NORM_EPS
from mlx_mast3r.layers import MLP

if TYPE_CHECKING:
    from mlx_mast3r.encoders.mast3r import RoPE2D


class Attention(nn.Module):
    """Multi-head self-attention with optional RoPE and fused SDPA.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        rope: Optional RoPE2D instance for positional encoding.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        rope: "RoPE2D | None" = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.rope = rope

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def __call__(self, x: mx.array, positions: mx.array | None = None) -> mx.array:
        """Forward pass.

        Args:
            x: [B, N, D] input tokens.
            positions: [B, N, 2] position indices (y, x) for RoPE. Required if rope is set.

        Returns:
            [B, N, D] output tokens.
        """
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = (
            qkv[:, :, 0].transpose(0, 2, 1, 3),
            qkv[:, :, 1].transpose(0, 2, 1, 3),
            qkv[:, :, 2].transpose(0, 2, 1, 3),
        )

        # Apply RoPE if configured (use fused kernel for q,k together)
        if self.rope is not None and positions is not None:
            q, k = self.rope.apply_fused(q, k, positions)

        # Fused SDPA
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

        return self.proj(out.transpose(0, 2, 1, 3).reshape(B, N, D))


class EncoderBlock(nn.Module):
    """Transformer encoder block with optional layer scale and RoPE.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        mlp_dim: MLP hidden dimension.
        rope: Optional RoPE2D instance for positional encoding.
        use_layer_scale: Whether to use layer scale (DINOv2 style).
        fast_gelu: Whether to use fast GELU approximation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
        rope: "RoPE2D | None" = None,
        use_layer_scale: bool = False,
        fast_gelu: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_layer_scale = use_layer_scale

        # LayerNorm weights
        self.norm1_weight = mx.ones((embed_dim,))
        self.norm1_bias = mx.zeros((embed_dim,))
        self.norm2_weight = mx.ones((embed_dim,))
        self.norm2_bias = mx.zeros((embed_dim,))

        # Layer scale (DINOv2)
        if use_layer_scale:
            self.ls1_gamma = mx.ones((embed_dim,))
            self.ls2_gamma = mx.ones((embed_dim,))

        self.attn = Attention(embed_dim, num_heads, head_dim, rope=rope)
        self.mlp = MLP(embed_dim, mlp_dim, fast_gelu=fast_gelu)

    def __call__(self, x: mx.array, positions: mx.array | None = None) -> mx.array:
        """Forward pass.

        Args:
            x: [B, N, D] input tokens.
            positions: [B, N, 2] position indices for RoPE. None if no RoPE.

        Returns:
            [B, N, D] output tokens.
        """
        # Pre-norm attention
        normed1 = mx.fast.layer_norm(x, self.norm1_weight, self.norm1_bias, eps=LAYER_NORM_EPS)
        attn_out = self.attn(normed1, positions)

        if self.use_layer_scale:
            x = x + self.ls1_gamma * attn_out
        else:
            x = x + attn_out

        # Pre-norm MLP
        normed2 = mx.fast.layer_norm(x, self.norm2_weight, self.norm2_bias, eps=LAYER_NORM_EPS)
        mlp_out = self.mlp(normed2)

        if self.use_layer_scale:
            x = x + self.ls2_gamma * mlp_out
        else:
            x = x + mlp_out

        return x


class PatchEmbed(nn.Module):
    """Patch embedding with configurable patch size.

    Args:
        embed_dim: Output embedding dimension.
        patch_size: Size of each patch (assumes square patches).
    """

    def __init__(self, embed_dim: int, patch_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: [B, H, W, 3] input image (NHWC format).

        Returns:
            [B, N, D] patch embeddings where N = (H/patch_size) * (W/patch_size).
        """
        B = x.shape[0]
        x = self.proj(x)
        return x.reshape(B, -1, self.embed_dim)
