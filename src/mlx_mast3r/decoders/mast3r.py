"""MASt3R Decoder - Ultra-optimized MLX implementation.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.

MASt3R full pipeline = MASt3R Encoder (1024 dim) + MASt3R Decoder (this file)

Architecture:
- decoder_embed: Project encoder features (1024) to decoder space (768)
- dec_blocks: 12 transformer decoder blocks (view 1)
- dec_blocks2: 12 transformer decoder blocks (view 2)
- downstream_head1/2: DPT heads for 3D points + descriptors

Optimizations:
- mx.fast.scaled_dot_product_attention
- mx.fast.layer_norm
- mx.compile()
- FP16/BF16 precision
- Fused Metal kernel for 2D RoPE (2x speedup)
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_mast3r.constants import LAYER_NORM_EPS
from mlx_mast3r.layers import MLP


@dataclass
class Mast3rDecoderConfig:
    """MASt3R decoder configuration."""

    encoder_dim: int = 1024  # MASt3R ViT-Large output dim
    decoder_dim: int = 768
    num_heads: int = 12
    head_dim: int = 64
    mlp_ratio: float = 4.0
    decoder_depth: int = 12
    patch_size: int = 16  # MASt3R uses 16x16 patches
    output_pts_dim: int = 3  # 3D points
    output_desc_dim: int = 24  # Descriptor dimension
    precision: Literal["fp32", "fp16", "bf16"] = "fp16"

    @property
    def dtype(self) -> mx.Dtype:
        return {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}[self.precision]

    @property
    def mlp_dim(self) -> int:
        return int(self.decoder_dim * self.mlp_ratio)

    @classmethod
    def default(cls, precision: str = "fp16") -> "Mast3rDecoderConfig":
        """Default config for MASt3R ViT-Large."""
        return cls(precision=precision)


def precompute_rope_2d(
    height: int,
    width: int,
    head_dim: int,
    theta: float = 100.0,
    dtype: mx.Dtype = mx.float32,
) -> tuple[mx.array, mx.array, mx.array]:
    """Precompute 2D RoPE cos/sin tables (pure MLX GPU).

    PyTorch RoPE2D:
    - Splits head_dim in half: first half for y-coords, second half for x-coords
    - Each half uses RoPE 1D with dimension head_dim // 4

    Returns:
        (cos_table, sin_table, positions)
        - cos_table, sin_table: [max_pos, head_dim // 4]
        - positions: [N, 2] - (y, x) position indices for each token
    """
    # RoPE dimension for each spatial direction (y and x)
    # PyTorch: D = head_dim // 2, then freqs = 1/(base^(2i/D)) for i in [0, D/2)
    D = head_dim // 2  # 32 for head_dim=64

    # Pure MLX computation (no NumPy transfer)
    dim_range = mx.arange(0, D, 2, dtype=mx.float32)
    inv_freq = 1.0 / (theta ** (dim_range / D))

    # Max position is max(height, width)
    max_pos = max(height, width)
    t = mx.arange(max_pos, dtype=mx.float32)

    # Compute freqs: [max_pos, freq_dim] via outer product
    freqs = t[:, None] * inv_freq[None, :]
    # Double the freqs for the full dimension (PyTorch does cat((freqs, freqs)))
    freqs_full = mx.concatenate([freqs, freqs], axis=-1)  # [max_pos, D]

    cos_table = mx.cos(freqs_full).astype(dtype)
    sin_table = mx.sin(freqs_full).astype(dtype)

    # Compute position indices for each token in the grid (pure MLX)
    y_pos = mx.arange(height, dtype=mx.int32)
    x_pos = mx.arange(width, dtype=mx.int32)
    # Create meshgrid equivalent
    grid_y = mx.broadcast_to(y_pos[:, None], (height, width))
    grid_x = mx.broadcast_to(x_pos[None, :], (height, width))
    # Stack and reshape to [N, 2]
    positions = mx.stack([grid_y.flatten(), grid_x.flatten()], axis=-1)

    return cos_table, sin_table, positions


def apply_rope_2d(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
    positions: mx.array,
) -> tuple[mx.array, mx.array]:
    """Apply 2D RoPE to query and key tensors using fused Metal kernel.

    PyTorch RoPE2D splits head_dim in half and applies 1D RoPE on each half
    with y-positions and x-positions respectively.

    Args:
        q, k: [B, nheads, N, head_dim]
        cos, sin: [max_pos, head_dim // 2]
        positions: [N, 2] - (y, x) indices for each token

    Returns:
        (q_rotated, k_rotated) with same shapes as inputs
    """
    from mlx_mast3r.kernels.rope2d import apply_rope_2d_fused

    return apply_rope_2d_fused(q, k, cos, sin, positions)


class DecoderSelfAttention(nn.Module):
    """Multi-head self-attention with optional 2D RoPE."""

    def __init__(self, config: Mast3rDecoderConfig, use_rope: bool = True):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_rope = use_rope

        dim = config.decoder_dim
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

        self._rope_cos: mx.array | None = None
        self._rope_sin: mx.array | None = None
        self._rope_positions: mx.array | None = None

    def set_rope_tables(self, cos: mx.array, sin: mx.array, positions: mx.array) -> None:
        """Set precomputed RoPE tables and positions."""
        self._rope_cos = cos
        self._rope_sin = sin
        self._rope_positions = positions

    def __call__(self, x: mx.array) -> mx.array:
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = (
            qkv[:, :, 0].transpose(0, 2, 1, 3),
            qkv[:, :, 1].transpose(0, 2, 1, 3),
            qkv[:, :, 2].transpose(0, 2, 1, 3),
        )

        if self.use_rope and self._rope_cos is not None:
            q, k = apply_rope_2d(q, k, self._rope_cos, self._rope_sin, self._rope_positions)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.proj(out.transpose(0, 2, 1, 3).reshape(B, N, D))


class DecoderCrossAttention(nn.Module):
    """Cross-attention between two views with optional 2D RoPE."""

    def __init__(self, config: Mast3rDecoderConfig, use_rope: bool = True):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_rope = use_rope

        dim = config.decoder_dim
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

        # RoPE tables for cross-attention
        self._rope_cos: mx.array | None = None
        self._rope_sin: mx.array | None = None
        self._rope_positions: mx.array | None = None

    def set_rope_tables(self, cos: mx.array, sin: mx.array, positions: mx.array) -> None:
        """Set precomputed RoPE tables and positions."""
        self._rope_cos = cos
        self._rope_sin = sin
        self._rope_positions = positions

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        B, N, D = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0].transpose(0, 2, 1, 3), kv[:, :, 1].transpose(0, 2, 1, 3)

        # Apply RoPE to Q and K (same positions for both views with same resolution)
        if self.use_rope and self._rope_cos is not None:
            q, k = apply_rope_2d(q, k, self._rope_cos, self._rope_sin, self._rope_positions)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.proj(out.transpose(0, 2, 1, 3).reshape(B, N, D))


class DecoderBlock(nn.Module):
    """Decoder transformer block with self and cross attention.

    PyTorch MASt3R architecture:
    - norm1: normalizes x before self-attention
    - norm2: normalizes x (query) before cross-attention
    - norm_y: normalizes context (key/value) before cross-attention
    - norm3: normalizes x before MLP
    """

    def __init__(self, config: Mast3rDecoderConfig, use_rope: bool = True):
        super().__init__()
        dim = config.decoder_dim

        # Self-attention
        self.norm1_weight = mx.ones((dim,))
        self.norm1_bias = mx.zeros((dim,))
        self.self_attn = DecoderSelfAttention(config, use_rope=use_rope)

        # Cross-attention
        # norm2: normalizes query (x)
        self.norm2_weight = mx.ones((dim,))
        self.norm2_bias = mx.zeros((dim,))
        # norm_y: normalizes context (key/value)
        self.norm_y_weight = mx.ones((dim,))
        self.norm_y_bias = mx.zeros((dim,))
        self.cross_attn = DecoderCrossAttention(config, use_rope=use_rope)

        # MLP
        self.norm3_weight = mx.ones((dim,))
        self.norm3_bias = mx.zeros((dim,))
        # Use exact GELU (not fast approx) to match PyTorch decoder
        self.mlp = MLP(config.decoder_dim, config.mlp_dim, fast_gelu=False)

    def set_rope_tables(self, cos: mx.array, sin: mx.array, positions: mx.array) -> None:
        """Propagate RoPE tables to self-attention and cross-attention."""
        self.self_attn.set_rope_tables(cos, sin, positions)
        self.cross_attn.set_rope_tables(cos, sin, positions)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        # Self-attention
        normed = mx.fast.layer_norm(x, self.norm1_weight, self.norm1_bias, eps=LAYER_NORM_EPS)
        x = x + self.self_attn(normed)

        # Cross-attention (norm2 on query, norm_y on context)
        x_normed = mx.fast.layer_norm(x, self.norm2_weight, self.norm2_bias, eps=LAYER_NORM_EPS)
        context_normed = mx.fast.layer_norm(context, self.norm_y_weight, self.norm_y_bias, eps=LAYER_NORM_EPS)
        x = x + self.cross_attn(x_normed, context_normed)

        # MLP
        normed = mx.fast.layer_norm(x, self.norm3_weight, self.norm3_bias, eps=LAYER_NORM_EPS)
        x = x + self.mlp(normed)

        return x


# Type alias for bilinear cache result
BilinearParams = tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]


@functools.lru_cache(maxsize=8)
def _compute_bilinear_params(H: int, W: int, dtype_str: str) -> BilinearParams:
    """Compute bilinear upsample parameters (cached with LRU, max 8 resolutions).

    Args:
        H: Input height.
        W: Input width.
        dtype_str: String representation of dtype for cache key.

    Returns:
        (idx00, idx01, idx10, idx11, w00, w01, w10, w11) index and weight arrays.
    """
    dtype = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}[dtype_str]
    out_H, out_W = H * 2, W * 2

    # Pure MLX computation (no NumPy transfer)
    oh = mx.arange(out_H, dtype=mx.float32)
    ow = mx.arange(out_W, dtype=mx.float32)

    src_h = oh * (H - 1) / (out_H - 1) if out_H > 1 else mx.zeros_like(oh)
    src_w = ow * (W - 1) / (out_W - 1) if out_W > 1 else mx.zeros_like(ow)

    # Floor indices
    h0 = mx.floor(src_h).astype(mx.int32)
    w0 = mx.floor(src_w).astype(mx.int32)
    h1 = mx.minimum(h0 + 1, H - 1)
    w1 = mx.minimum(w0 + 1, W - 1)

    # Fractional parts
    fh = src_h - h0.astype(mx.float32)
    fw = src_w - w0.astype(mx.float32)

    # 2D weight grids via broadcasting [out_H, out_W, 1]
    fh_2d = fh[:, None]
    fw_2d = fw[None, :]

    w00 = ((1 - fh_2d) * (1 - fw_2d)).astype(dtype)[:, :, None]
    w01 = ((1 - fh_2d) * fw_2d).astype(dtype)[:, :, None]
    w10 = (fh_2d * (1 - fw_2d)).astype(dtype)[:, :, None]
    w11 = (fh_2d * fw_2d).astype(dtype)[:, :, None]

    # Index meshgrids via broadcasting
    h0_2d = mx.broadcast_to(h0[:, None], (out_H, out_W))
    w0_2d = mx.broadcast_to(w0[None, :], (out_H, out_W))
    h1_2d = mx.broadcast_to(h1[:, None], (out_H, out_W))
    w1_2d = mx.broadcast_to(w1[None, :], (out_H, out_W))

    # Linear indices (flattened for gather)
    idx00 = (h0_2d * W + w0_2d).flatten().astype(mx.int32)
    idx01 = (h0_2d * W + w1_2d).flatten().astype(mx.int32)
    idx10 = (h1_2d * W + w0_2d).flatten().astype(mx.int32)
    idx11 = (h1_2d * W + w1_2d).flatten().astype(mx.int32)

    return (idx00, idx01, idx10, idx11, w00, w01, w10, w11)


def _get_bilinear_params(H: int, W: int, dtype: mx.Dtype) -> BilinearParams:
    """Get cached bilinear upsample parameters.

    Args:
        H: Input height.
        W: Input width.
        dtype: MLX dtype.

    Returns:
        (idx00, idx01, idx10, idx11, w00, w01, w10, w11) index and weight arrays.
    """
    dtype_str = {mx.float32: "float32", mx.float16: "float16", mx.bfloat16: "bfloat16"}[dtype]
    return _compute_bilinear_params(H, W, dtype_str)


def bilinear_upsample_2x(x: mx.array, align_corners: bool = True) -> mx.array:
    """Bilinear upsampling by factor 2 using fused Metal kernel.

    Input: [B, H, W, C], Output: [B, 2H, 2W, C]

    Uses a fused Metal kernel for ~2x speedup over separate gather/reshape ops.
    """
    if not align_corners:
        return nearest_upsample_2x(x)

    from mlx_mast3r.kernels.bilinear import bilinear_upsample_2x_fused

    return bilinear_upsample_2x_fused(x)


def nearest_upsample_2x(x: mx.array) -> mx.array:
    """Nearest neighbor upsampling by factor 2. Input: [B, H, W, C]."""
    B, H, W, C = x.shape
    x = x[:, :, None, :, None, :]  # [B, H, 1, W, 1, C]
    x = mx.broadcast_to(x, (B, H, 2, W, 2, C))
    return x.reshape(B, H * 2, W * 2, C)


class ResidualConvUnit(nn.Module):
    """Residual convolution unit for DPT.

    Architecture: ReLU -> Conv3x3 -> ReLU -> Conv3x3 -> Add(residual)
    """

    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        out = nn.relu(x)
        out = self.conv1(out)
        out = nn.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block with upsampling for DPT.

    Fuses two feature maps: processes input with ResidualConvUnit,
    adds optional skip connection, then upsamples 2x.
    """

    def __init__(self, features: int):
        super().__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, bias=True)

    def __call__(self, x: mx.array, skip: mx.array | None = None) -> mx.array:
        if skip is not None:
            res = self.resConfUnit1(skip)
            # Resize if needed (use bilinear for DPT)
            if res.shape[1:3] != x.shape[1:3]:
                res = bilinear_upsample_2x(res, align_corners=True)
                res = res[:, : x.shape[1], : x.shape[2], :]
            x = x + res

        x = self.resConfUnit2(x)
        # Use bilinear upsampling with align_corners=True (PyTorch DPT default)
        x = bilinear_upsample_2x(x, align_corners=True)
        x = self.out_conv(x)
        return x


class DPTHead(nn.Module):
    """Dense Prediction Transformer head for 3D reconstruction.

    Full DPT architecture with multi-scale feature fusion and progressive upsampling.
    Outputs at full image resolution.

    Architecture:
    - act_postprocess[0-3]: Adapt token dimensions per layer
    - scratch.layer_rn[0-3]: Project to feature_dim (256)
    - scratch.refinenet[1-4]: Multi-scale fusion with 2x upsampling
    - head: Final conv layers to output channels
    """

    def __init__(
        self,
        encoder_dim: int = 1024,
        decoder_dim: int = 768,
        feature_dim: int = 256,
        num_channels: int = 4,  # pts3d (3) + conf (1)
        hooks: list[int] | None = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_channels = num_channels
        # Default hooks for 12-layer decoder: [0, 6, 9, 12]
        self.hooks = hooks or [0, 6, 9, 12]

        # Input dimensions from encoder/decoder tokens
        # hook 0 = encoder output (1024), hooks 1-3 = decoder layers (768)
        in_dims = [encoder_dim, decoder_dim, decoder_dim, decoder_dim]

        # Output dimensions after act_postprocess (from PyTorch model)
        # These are the layer_dims used by scratch.layer_rn
        layer_dims = [96, 192, 384, 768]

        # act_postprocess: adapt dimensions for each hooked layer
        # Layer 0: Conv1x1 (1024->96) + ConvTranspose 4x4 stride 4 (upsample 4x)
        self.act_postprocess_0_conv = nn.Conv2d(in_dims[0], layer_dims[0], kernel_size=1)
        self.act_postprocess_0_up = nn.ConvTranspose2d(
            layer_dims[0], layer_dims[0], kernel_size=4, stride=4, padding=0
        )

        # Layer 1: Conv1x1 (768->192) + ConvTranspose 2x2 stride 2 (upsample 2x)
        self.act_postprocess_1_conv = nn.Conv2d(in_dims[1], layer_dims[1], kernel_size=1)
        self.act_postprocess_1_up = nn.ConvTranspose2d(
            layer_dims[1], layer_dims[1], kernel_size=2, stride=2, padding=0
        )

        # Layer 2: Conv1x1 (768->384) only (no upsampling)
        self.act_postprocess_2_conv = nn.Conv2d(in_dims[2], layer_dims[2], kernel_size=1)

        # Layer 3: Conv1x1 (768->768) + Conv 3x3 stride 2 (downsample 2x)
        self.act_postprocess_3_conv = nn.Conv2d(in_dims[3], layer_dims[3], kernel_size=1)
        self.act_postprocess_3_down = nn.Conv2d(
            layer_dims[3], layer_dims[3], kernel_size=3, stride=2, padding=1
        )

        # scratch.layer_rn: project each layer to feature_dim (256)
        self.layer1_rn = nn.Conv2d(layer_dims[0], feature_dim, kernel_size=3, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(layer_dims[1], feature_dim, kernel_size=3, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(layer_dims[2], feature_dim, kernel_size=3, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(layer_dims[3], feature_dim, kernel_size=3, padding=1, bias=False)

        # scratch.refinenet: multi-scale feature fusion with 2x upsampling
        self.refinenet4 = FeatureFusionBlock(feature_dim)
        self.refinenet3 = FeatureFusionBlock(feature_dim)
        self.refinenet2 = FeatureFusionBlock(feature_dim)
        self.refinenet1 = FeatureFusionBlock(feature_dim)

        # Output head: Conv -> Upsample 2x -> Conv -> ReLU -> Conv
        # Note: In MASt3R, last_dim = feature_dim // 2 = 128 (not 32 as in original DPT)
        last_dim = feature_dim // 2  # 128
        self.head_conv1 = nn.Conv2d(feature_dim, last_dim, kernel_size=3, padding=1)
        self.head_conv2 = nn.Conv2d(last_dim, last_dim, kernel_size=3, padding=1)  # 128 -> 128
        self.head_conv3 = nn.Conv2d(last_dim, num_channels, kernel_size=1)  # 128 -> 4

    def __call__(self, features: list[mx.array], H: int, W: int) -> mx.array:
        """Process multi-scale features through DPT.

        Args:
            features: List of [B, N, C] features at hooked layers
                     features[0] = encoder output (1024 dim)
                     features[1-3] = decoder layers (768 dim)
            H, W: Patch grid dimensions

        Returns:
            [B, H*16, W*16, num_channels] - full resolution output
        """
        B = features[0].shape[0]

        # Reshape features to spatial: [B, N, C] -> [B, H, W, C]
        layers = [f.reshape(B, H, W, -1) for f in features]

        # Apply act_postprocess to adapt dimensions
        # Layer 0: upsample 4x (Conv + ConvTranspose)
        l0 = self.act_postprocess_0_conv(layers[0])
        l0 = self.act_postprocess_0_up(l0)

        # Layer 1: upsample 2x
        l1 = self.act_postprocess_1_conv(layers[1])
        l1 = self.act_postprocess_1_up(l1)

        # Layer 2: no change
        l2 = self.act_postprocess_2_conv(layers[2])

        # Layer 3: downsample 2x
        l3 = self.act_postprocess_3_conv(layers[3])
        l3 = self.act_postprocess_3_down(l3)

        # Project all layers to feature_dim
        l0 = self.layer1_rn(l0)
        l1 = self.layer2_rn(l1)
        l2 = self.layer3_rn(l2)
        l3 = self.layer4_rn(l3)

        # Multi-scale fusion with progressive upsampling
        # Each refinenet does: process + upsample 2x
        path_4 = self.refinenet4(l3)  # [B, H/2, W/2, 256] -> [B, H, W, 256]
        # Crop to match l2 size
        path_4 = path_4[:, : l2.shape[1], : l2.shape[2], :]

        path_3 = self.refinenet3(path_4, l2)  # -> [B, 2H, 2W, 256]
        path_2 = self.refinenet2(path_3, l1)  # -> [B, 4H, 4W, 256]
        path_1 = self.refinenet1(path_2, l0)  # -> [B, 8H, 8W, 256]

        # Output head: Conv -> Upsample 2x -> Conv -> ReLU -> Conv
        out = self.head_conv1(path_1)
        out = bilinear_upsample_2x(out, align_corners=True)  # -> [B, 16H, 16W, 128]
        out = self.head_conv2(out)
        out = nn.relu(out)
        out = self.head_conv3(out)

        return out  # [B, H*16, W*16, num_channels]


class Mast3rDecoder(nn.Module):
    """MASt3R Decoder - Asymmetric decoder for stereo 3D reconstruction.

    Takes MASt3R encoder features (1024 dim) from two views and outputs:
    - 3D points in camera space
    - Confidence maps
    - Dense descriptors for matching

    Architecture follows the original MASt3R with:
    - Cross-attention between views
    - 2D RoPE positional encoding
    - DPT heads for dense prediction

    Example:
        >>> config = Mast3rDecoderConfig.default(precision="fp16")
        >>> decoder = Mast3rDecoder(config)
        >>> # After loading weights...
        >>> pts3d_1, pts3d_2 = decoder(feat1, feat2, (H, W), (H, W))
    """

    def __init__(self, config: Mast3rDecoderConfig):
        super().__init__()
        self.config = config

        # Project encoder features (1024) to decoder dim (768)
        self.decoder_embed = nn.Linear(config.encoder_dim, config.decoder_dim)

        # Encoder norm (applied to MASt3R encoder output)
        self.enc_norm_weight = mx.ones((config.encoder_dim,))
        self.enc_norm_bias = mx.zeros((config.encoder_dim,))

        # Mask token for masked regions
        self.mask_token = mx.zeros((1, 1, config.decoder_dim))

        # Decoder blocks for view 1 (with RoPE)
        self.dec_blocks = [DecoderBlock(config, use_rope=True) for _ in range(config.decoder_depth)]

        # Decoder blocks for view 2 (with RoPE)
        self.dec_blocks2 = [
            DecoderBlock(config, use_rope=True) for _ in range(config.decoder_depth)
        ]

        # Final decoder norm
        self.dec_norm_weight = mx.ones((config.decoder_dim,))
        self.dec_norm_bias = mx.zeros((config.decoder_dim,))

        # Output heads: pts3d (3) + confidence (1) = 4 channels
        # Descriptors are handled separately via MLP
        self.head1 = DPTHead(
            encoder_dim=config.encoder_dim,
            decoder_dim=config.decoder_dim,
            feature_dim=256,
            num_channels=4,  # pts3d (3) + conf (1)
        )
        self.head2 = DPTHead(
            encoder_dim=config.encoder_dim,
            decoder_dim=config.decoder_dim,
            feature_dim=256,
            num_channels=4,
        )

        # Local features MLP (descriptors)
        # Input: enc_dim + dec_dim = 1024 + 768 = 1792
        # Output: (desc_dim + 1) * patch_size^2 = 25 * 256 for pixel shuffle
        idim = config.encoder_dim + config.decoder_dim
        out_dim = (config.output_desc_dim + 1) * config.patch_size**2
        # Use exact GELU (not fast approx) to match PyTorch
        self.head_local_features1 = MLP(idim, idim * 4, out_dim, fast_gelu=False)
        self.head_local_features2 = MLP(idim, idim * 4, out_dim, fast_gelu=False)

        # RoPE tables
        self._rope_cos: mx.array | None = None
        self._rope_sin: mx.array | None = None
        self._rope_positions: mx.array | None = None

    def _init_rope(self, H: int, W: int) -> None:
        """Initialize RoPE tables for patch grid."""
        cos, sin, positions = precompute_rope_2d(
            H, W, self.config.head_dim, theta=100.0, dtype=self.config.dtype
        )
        self._rope_cos = cos
        self._rope_sin = sin
        self._rope_positions = positions

        # Propagate to all decoder blocks
        for blk in self.dec_blocks:
            blk.set_rope_tables(cos, sin, positions)
        for blk in self.dec_blocks2:
            blk.set_rope_tables(cos, sin, positions)

    def __call__(
        self,
        feat1: mx.array,
        feat2: mx.array,
        shape1: tuple[int, int],
        shape2: tuple[int, int],
    ) -> tuple[dict, dict]:
        """Forward pass.

        Args:
            feat1: [B, N1, D] encoder features for view 1 (D=1024)
            feat2: [B, N2, D] encoder features for view 2 (D=1024)
            shape1: (H1, W1) patch grid shape for view 1
            shape2: (H2, W2) patch grid shape for view 2

        Returns:
            (output1, output2) dicts with keys: pts3d, conf, desc
        """
        B = feat1.shape[0]
        H1, W1 = shape1
        H2, W2 = shape2

        # Initialize RoPE if needed
        if self._rope_cos is None:
            self._init_rope(H1, W1)

        # Encoder outputs are already normalized (enc_norm applied in encoder)
        # Project to decoder dim
        x1 = self.decoder_embed(feat1)
        x2 = self.decoder_embed(feat2)

        # Hooks: [0, 6, 9, 12] - collect features at these indices
        # Hook 0 = encoder output (already normalized, 1024 dim)
        hooks = [0, 6, 9, 12]
        features1 = [feat1]  # Hook 0: encoder features (1024 dim)
        features2 = [feat2]

        # Decoder blocks with cross-attention, collecting hooked outputs
        # IMPORTANT: PyTorch uses OLD x1/x2 for BOTH blocks in each iteration
        # blk1 gets (x1_old, x2_old), blk2 gets (x2_old, x1_old) - NOT the new x1!
        for i, (blk1, blk2) in enumerate(zip(self.dec_blocks, self.dec_blocks2)):
            # Save old values before updating (PyTorch uses final_output[-1] for both)
            x1_old, x2_old = x1, x2
            x1 = blk1(x1_old, x2_old)  # View 1 attends to view 2 (old)
            x2 = blk2(x2_old, x1_old)  # View 2 attends to view 1 (old, NOT new x1!)

            # Collect at hooks (1-indexed: after block i means index i+1)
            layer_idx = i + 1
            if layer_idx in hooks[1:]:  # Skip hook 0 which is encoder
                features1.append(x1)
                features2.append(x2)

        # Final norm
        x1_norm = mx.fast.layer_norm(x1, self.dec_norm_weight, self.dec_norm_bias, eps=LAYER_NORM_EPS)
        x2_norm = mx.fast.layer_norm(x2, self.dec_norm_weight, self.dec_norm_bias, eps=LAYER_NORM_EPS)

        # If we collected 12 (final layer), update it with normed version
        if len(features1) == 4:
            features1[-1] = x1_norm
            features2[-1] = x2_norm

        # DPT heads for pts3d + conf
        dpt_out1 = self.head1(features1, H1, W1)  # [B, H1*16, W1*16, 4]
        dpt_out2 = self.head2(features2, H2, W2)

        # Local features via MLP (for descriptors)
        # Concatenate encoder and decoder outputs
        cat1 = mx.concatenate([feat1, x1_norm], axis=-1)  # [B, N, 1792]
        cat2 = mx.concatenate([feat2, x2_norm], axis=-1)

        # MLP
        local_feat1 = self.head_local_features1(cat1)  # [B, N, 25*256]
        local_feat2 = self.head_local_features2(cat2)

        # Pixel shuffle to get full resolution descriptors
        ps = self.config.patch_size  # 16
        desc_dim = self.config.output_desc_dim + 1  # 25 (24 desc + 1 desc_conf)

        # Reshape: [B, H*W, desc_dim*ps*ps] -> [B, H, W, desc_dim, ps, ps]
        local_feat1 = local_feat1.reshape(B, H1, W1, desc_dim, ps, ps)
        local_feat2 = local_feat2.reshape(B, H2, W2, desc_dim, ps, ps)

        # Transpose and reshape for pixel shuffle: [B, H, ps, W, ps, desc_dim] -> [B, H*ps, W*ps, desc_dim]
        local_feat1 = local_feat1.transpose(0, 1, 4, 2, 5, 3).reshape(B, H1 * ps, W1 * ps, desc_dim)
        local_feat2 = local_feat2.transpose(0, 1, 4, 2, 5, 3).reshape(B, H2 * ps, W2 * ps, desc_dim)

        # Split descriptors and desc_conf
        from mlx_mast3r.utils.postprocessing import build_output_dict

        out1 = build_output_dict(dpt_out1, local_feat1, self.config.output_desc_dim)
        out2 = build_output_dict(dpt_out2, local_feat2, self.config.output_desc_dim)

        return out1, out2


class Mast3rDecoderEngine:
    """High-level MASt3R pipeline: encoder + decoder."""

    def __init__(
        self,
        resolution: int = 512,
        precision: Literal["fp32", "fp16", "bf16"] = "fp16",
        compile: bool = True,
    ):
        from mlx_mast3r.encoders import Mast3rEncoder
        from mlx_mast3r.encoders.mast3r import Mast3rEncoderConfig

        # Encoder config
        self.encoder_config = Mast3rEncoderConfig(resolution=resolution, precision=precision)
        self.encoder = Mast3rEncoder(self.encoder_config)

        # Decoder config
        self.decoder_config = Mast3rDecoderConfig.default(precision)
        self.decoder = Mast3rDecoder(self.decoder_config)

        self._compile = compile
        self._compiled_encoder = None
        self._compiled_decoder = None
        self._loaded = False

    def load(self, path: str | Path) -> None:
        """Load encoder and decoder weights from unified safetensors.

        The MASt3R checkpoint contains both encoder and decoder weights.
        """
        # Load encoder
        self._load_encoder(path)

        # Load decoder
        self._load_decoder(path)

        # Compile (shapeless=True not compatible with dynamic ops)
        if self._compile:
            self._compiled_encoder = mx.compile(self.encoder.__call__)
            self._compiled_decoder = mx.compile(self.decoder.__call__)

        self._loaded = True

    def _load_encoder(self, path: str | Path) -> None:
        """Load encoder weights from safetensors."""
        from mlx_mast3r.encoders.mast3r import Mast3rEncoderEngine

        engine = Mast3rEncoderEngine(
            resolution=self.encoder_config.resolution,
            precision=self.encoder_config.precision,
            compile=False,
        )
        engine.load(path)
        self.encoder = engine.model

    def _load_decoder(self, path: str | Path) -> None:
        """Load decoder weights from model.safetensors.

        The model.safetensors contains full MASt3R weights with keys like:
        - dec_blocks.X.* (no 'decoder.' prefix)
        - cross_attn uses projq/projk/projv (not combined kv)
        """
        from safetensors import safe_open

        from mlx_mast3r.decoders.weight_loader import (
            load_all_decoder_blocks,
            load_basic_params,
            load_dpt_head,
            load_local_features,
        )

        # Use model.safetensors which has full decoder weights
        path = Path(path)
        if path.name == "unified.safetensors":
            model_path = path.parent / "model.safetensors"
            if model_path.exists():
                path = model_path

        weights: dict[str, mx.array] = {}
        with safe_open(str(path), framework="numpy") as f:
            keys = list(f.keys())

            # Load all weight groups
            load_basic_params(f, keys, weights)
            load_all_decoder_blocks(f, keys, weights, self.decoder_config.decoder_depth)
            load_dpt_head(f, keys, weights, "downstream_head1", "head1")
            load_dpt_head(f, keys, weights, "downstream_head2", "head2")
            load_local_features(f, keys, weights)

        # Cast to dtype
        if self.decoder_config.dtype != mx.float32:
            weights = {k: v.astype(self.decoder_config.dtype) for k, v in weights.items()}

        print(f"Loading {len(weights)} decoder weights...")
        self.decoder.load_weights(list(weights.items()), strict=False)
        mx.eval(self.decoder.parameters())

    def __call__(
        self,
        img1: mx.array,
        img2: mx.array,
    ) -> tuple[dict, dict]:
        """Run full pipeline.

        Args:
            img1: [B, H, W, 3] first view image (NHWC, normalized)
            img2: [B, H, W, 3] second view image (NHWC, normalized)

        Returns:
            (output1, output2) with pts3d, conf, desc for each view
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Encode both views
        if self._compiled_encoder:
            feat1 = self._compiled_encoder(img1)
            feat2 = self._compiled_encoder(img2)
        else:
            feat1 = self.encoder(img1)
            feat2 = self.encoder(img2)

        # Evaluate encoder outputs before decoder (prevents NaN from deep lazy graphs)
        mx.eval(feat1, feat2)

        # Compute patch dimensions from actual image size
        _, H_img, W_img, _ = img1.shape
        patch_size = self.encoder_config.patch_size
        H = H_img // patch_size
        W = W_img // patch_size

        if self._compiled_decoder:
            return self._compiled_decoder(feat1, feat2, (H, W), (H, W))
        return self.decoder(feat1, feat2, (H, W), (H, W))

    def infer(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
    ) -> tuple[dict, dict, float]:
        """Run inference on numpy images.

        Args:
            img1, img2: [H, W, 3] uint8 images

        Returns:
            (output1, output2, time_ms)
        """
        import time

        # MASt3R preprocessing
        x1 = img1.astype(np.float32) / 255.0
        x1 = (x1 - 0.5) / 0.5
        x2 = img2.astype(np.float32) / 255.0
        x2 = (x2 - 0.5) / 0.5

        x1 = mx.array(x1[None, :, :, :])
        x2 = mx.array(x2[None, :, :, :])

        t0 = time.perf_counter()
        out1, out2 = self(x1, x2)
        mx.eval(out1["pts3d"], out2["pts3d"])
        ms = (time.perf_counter() - t0) * 1000

        # Convert to numpy
        out1_np = {k: np.array(v[0]) for k, v in out1.items()}
        out2_np = {k: np.array(v[0]) for k, v in out2.items()}

        return out1_np, out2_np, ms

    def warmup(self, iterations: int = 5) -> None:
        """Warmup the model."""
        H, W = self.encoder_config.img_h, self.encoder_config.img_w
        dummy = mx.zeros((1, H, W, 3), dtype=mx.float32)

        for _ in range(iterations):
            out1, out2 = self(dummy, dummy)
            mx.eval(out1["pts3d"], out2["pts3d"])
