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
- 2D RoPE in decoder attention
"""

from __future__ import annotations

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
    """Precompute 2D RoPE cos/sin tables.

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
    freq_dim = D // 2  # 16

    # Compute inverse frequencies (same as PyTorch)
    inv_freq = 1.0 / (theta ** (np.arange(0, D, 2, dtype=np.float32) / D))

    # Max position is max(height, width)
    max_pos = max(height, width)
    t = np.arange(max_pos, dtype=np.float32)

    # Compute freqs: [max_pos, freq_dim]
    freqs = np.outer(t, inv_freq)
    # Double the freqs for the full dimension (PyTorch does cat((freqs, freqs)))
    freqs_full = np.concatenate([freqs, freqs], axis=-1)  # [max_pos, D]

    cos_table = mx.array(np.cos(freqs_full), dtype=dtype)
    sin_table = mx.array(np.sin(freqs_full), dtype=dtype)

    # Compute position indices for each token in the grid
    # PyTorch: torch.cartesian_prod(y, x) gives (y, x) pairs
    y_pos = np.arange(height)
    x_pos = np.arange(width)
    # cartesian product: [[0,0], [0,1], ..., [0,W-1], [1,0], ...]
    grid_y, grid_x = np.meshgrid(y_pos, x_pos, indexing='ij')
    positions = np.stack([grid_y.flatten(), grid_x.flatten()], axis=-1)  # [N, 2]
    positions = mx.array(positions, dtype=mx.int32)

    return cos_table, sin_table, positions


def apply_rope_2d(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
    positions: mx.array,
) -> tuple[mx.array, mx.array]:
    """Apply 2D RoPE to query and key tensors.

    PyTorch RoPE2D splits head_dim in half and applies 1D RoPE on each half
    with y-positions and x-positions respectively.

    Args:
        q, k: [B, nheads, N, head_dim]
        cos, sin: [max_pos, head_dim // 2]
        positions: [N, 2] - (y, x) indices for each token
    """

    def rotate_half(x: mx.array) -> mx.array:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)

    def apply_rope_1d(tokens: mx.array, pos1d: mx.array, cos_t: mx.array, sin_t: mx.array) -> mx.array:
        """Apply 1D RoPE using position indices.

        Args:
            tokens: [B, nheads, N, D]
            pos1d: [N] - position indices
            cos_t, sin_t: [max_pos, D]
        """
        # Gather cos/sin for each token position: [N, D]
        cos_gathered = cos_t[pos1d]  # [N, D]
        sin_gathered = sin_t[pos1d]  # [N, D]

        # Expand for broadcast: [1, 1, N, D]
        cos_gathered = cos_gathered[None, None, :, :]
        sin_gathered = sin_gathered[None, None, :, :]

        return tokens * cos_gathered + rotate_half(tokens) * sin_gathered

    # Split into y-features and x-features (first half and second half of head_dim)
    head_dim = q.shape[-1]
    q_y, q_x = q[..., :head_dim // 2], q[..., head_dim // 2:]
    k_y, k_x = k[..., :head_dim // 2], k[..., head_dim // 2:]

    # Get position indices for y and x
    pos_y = positions[:, 0]  # [N]
    pos_x = positions[:, 1]  # [N]

    # Apply 1D RoPE to each half with corresponding positions
    q_y = apply_rope_1d(q_y, pos_y, cos, sin)
    q_x = apply_rope_1d(q_x, pos_x, cos, sin)
    k_y = apply_rope_1d(k_y, pos_y, cos, sin)
    k_x = apply_rope_1d(k_x, pos_x, cos, sin)

    # Concatenate back
    q_rotated = mx.concatenate([q_y, q_x], axis=-1)
    k_rotated = mx.concatenate([k_y, k_x], axis=-1)

    return q_rotated, k_rotated


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
        self.mlp = MLP(config.decoder_dim, config.mlp_dim)

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


# Global cache for bilinear upsample indices/weights
_bilinear_cache: dict[tuple[int, int, str], tuple] = {}


def _get_bilinear_params(
    H: int, W: int, dtype: mx.Dtype
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Get or compute cached bilinear upsample parameters.

    Returns: (idx00, idx01, idx10, idx11, w00, w01, w10, w11)
    """
    cache_key = (H, W, str(dtype))
    if cache_key in _bilinear_cache:
        return _bilinear_cache[cache_key]

    out_H, out_W = H * 2, W * 2

    # Compute source coordinates
    oh = np.arange(out_H, dtype=np.float32)
    ow = np.arange(out_W, dtype=np.float32)

    src_h = oh * (H - 1) / (out_H - 1) if out_H > 1 else np.zeros_like(oh)
    src_w = ow * (W - 1) / (out_W - 1) if out_W > 1 else np.zeros_like(ow)

    # Floor indices
    h0 = np.floor(src_h).astype(np.int32)
    w0 = np.floor(src_w).astype(np.int32)
    h1 = np.minimum(h0 + 1, H - 1)
    w1 = np.minimum(w0 + 1, W - 1)

    # Fractional parts
    fh = src_h - h0
    fw = src_w - w0

    # 2D weight grids
    fh_2d = fh[:, None]
    fw_2d = fw[None, :]

    w00 = mx.array((1 - fh_2d) * (1 - fw_2d), dtype=dtype)[:, :, None]
    w01 = mx.array((1 - fh_2d) * fw_2d, dtype=dtype)[:, :, None]
    w10 = mx.array(fh_2d * (1 - fw_2d), dtype=dtype)[:, :, None]
    w11 = mx.array(fh_2d * fw_2d, dtype=dtype)[:, :, None]

    # Index meshgrids
    h0_2d, w0_2d = np.meshgrid(h0, w0, indexing="ij")
    h1_2d, w1_2d = np.meshgrid(h1, w1, indexing="ij")

    # Linear indices (flattened for gather)
    idx00 = mx.array((h0_2d * W + w0_2d).flatten(), dtype=mx.int32)
    idx01 = mx.array((h0_2d * W + w1_2d).flatten(), dtype=mx.int32)
    idx10 = mx.array((h1_2d * W + w0_2d).flatten(), dtype=mx.int32)
    idx11 = mx.array((h1_2d * W + w1_2d).flatten(), dtype=mx.int32)

    result = (idx00, idx01, idx10, idx11, w00, w01, w10, w11)
    _bilinear_cache[cache_key] = result
    return result


def bilinear_upsample_2x(x: mx.array, align_corners: bool = True) -> mx.array:
    """Bilinear upsampling by factor 2 with align_corners support.

    Input: [B, H, W, C], Output: [B, 2H, 2W, C]

    Uses cached indices/weights for repeated calls with same dimensions.
    """
    B, H, W, C = x.shape
    out_H, out_W = H * 2, W * 2

    if not align_corners:
        return nearest_upsample_2x(x)

    # Get cached parameters
    idx00, idx01, idx10, idx11, w00, w01, w10, w11 = _get_bilinear_params(H, W, x.dtype)

    # Flatten spatial dims for gather
    x_flat = x.reshape(B, H * W, C)

    # Gather and reshape
    p00 = x_flat[:, idx00, :].reshape(B, out_H, out_W, C)
    p01 = x_flat[:, idx01, :].reshape(B, out_H, out_W, C)
    p10 = x_flat[:, idx10, :].reshape(B, out_H, out_W, C)
    p11 = x_flat[:, idx11, :].reshape(B, out_H, out_W, C)

    return p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11


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
        self.head_local_features1 = nn.Sequential(
            nn.Linear(idim, idim * 4),
            nn.GELU(),
            nn.Linear(idim * 4, (config.output_desc_dim + 1) * config.patch_size**2),
        )
        self.head_local_features2 = nn.Sequential(
            nn.Linear(idim, idim * 4),
            nn.GELU(),
            nn.Linear(idim * 4, (config.output_desc_dim + 1) * config.patch_size**2),
        )

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
        desc1 = local_feat1[..., :self.config.output_desc_dim]
        desc_conf1 = local_feat1[..., self.config.output_desc_dim:]
        desc2 = local_feat2[..., :self.config.output_desc_dim]
        desc_conf2 = local_feat2[..., self.config.output_desc_dim:]

        # Normalize descriptors
        desc1 = desc1 / (mx.linalg.norm(desc1, axis=-1, keepdims=True) + 1e-8)
        desc2 = desc2 / (mx.linalg.norm(desc2, axis=-1, keepdims=True) + 1e-8)

        # Post-processing (matching MASt3R depth_mode='exp', conf_mode='exp')
        def postprocess_pts3d(xyz: mx.array) -> mx.array:
            """Apply depth_mode='exp': pts3d = xyz * expm1(norm(xyz))."""
            d = mx.linalg.norm(xyz, axis=-1, keepdims=True)
            xyz_normalized = xyz / mx.maximum(d, mx.array(1e-8))
            return xyz_normalized * mx.expm1(d)

        def postprocess_conf(x: mx.array, vmin: float = 1.0) -> mx.array:
            """Apply conf_mode='exp': conf = vmin + exp(x)."""
            return vmin + mx.exp(x)

        def postprocess_desc_conf(x: mx.array, vmin: float = 0.0) -> mx.array:
            """Apply desc_conf_mode='exp': desc_conf = vmin + exp(x)."""
            return vmin + mx.exp(x)

        # Apply post-processing
        pts3d_1 = postprocess_pts3d(dpt_out1[..., :3])
        pts3d_2 = postprocess_pts3d(dpt_out2[..., :3])
        conf_1 = postprocess_conf(dpt_out1[..., 3:4])
        conf_2 = postprocess_conf(dpt_out2[..., 3:4])
        desc_conf1 = postprocess_desc_conf(desc_conf1)
        desc_conf2 = postprocess_desc_conf(desc_conf2)

        # Build output dicts
        out1 = {
            "pts3d": pts3d_1,
            "conf": conf_1,
            "desc": desc1,
            "desc_conf": desc_conf1,
        }
        out2 = {
            "pts3d": pts3d_2,
            "conf": conf_2,
            "desc": desc2,
            "desc_conf": desc_conf2,
        }

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
        from safetensors import safe_open

        # Load encoder
        self._load_encoder(path)

        # Load decoder
        self._load_decoder(path)

        # Compile
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

        # Use model.safetensors which has full decoder weights
        path = Path(path)
        if path.name == "unified.safetensors":
            model_path = path.parent / "model.safetensors"
            if model_path.exists():
                path = model_path

        weights = {}
        with safe_open(str(path), framework="numpy") as f:
            keys = list(f.keys())

            # decoder_embed (no prefix in model.safetensors)
            if "decoder_embed.weight" in keys:
                weights["decoder_embed.weight"] = mx.array(f.get_tensor("decoder_embed.weight"))
                weights["decoder_embed.bias"] = mx.array(f.get_tensor("decoder_embed.bias"))

            # enc_norm
            if "enc_norm.weight" in keys:
                weights["enc_norm_weight"] = mx.array(f.get_tensor("enc_norm.weight"))
                weights["enc_norm_bias"] = mx.array(f.get_tensor("enc_norm.bias"))

            # dec_norm
            if "dec_norm.weight" in keys:
                weights["dec_norm_weight"] = mx.array(f.get_tensor("dec_norm.weight"))
                weights["dec_norm_bias"] = mx.array(f.get_tensor("dec_norm.bias"))

            # Load decoder blocks helper
            def load_decoder_block(block_name: str, dst_prefix: str) -> None:
                """Load a single decoder block."""
                src_prefix = f"{block_name}."

                if f"{src_prefix}norm1.weight" not in keys:
                    return

                # Self-attention norms
                weights[f"{dst_prefix}norm1_weight"] = mx.array(
                    f.get_tensor(f"{src_prefix}norm1.weight")
                )
                weights[f"{dst_prefix}norm1_bias"] = mx.array(
                    f.get_tensor(f"{src_prefix}norm1.bias")
                )

                # Self-attention
                weights[f"{dst_prefix}self_attn.qkv.weight"] = mx.array(
                    f.get_tensor(f"{src_prefix}attn.qkv.weight")
                )
                weights[f"{dst_prefix}self_attn.qkv.bias"] = mx.array(
                    f.get_tensor(f"{src_prefix}attn.qkv.bias")
                )
                weights[f"{dst_prefix}self_attn.proj.weight"] = mx.array(
                    f.get_tensor(f"{src_prefix}attn.proj.weight")
                )
                weights[f"{dst_prefix}self_attn.proj.bias"] = mx.array(
                    f.get_tensor(f"{src_prefix}attn.proj.bias")
                )

                # Cross-attention norms
                # norm2: normalizes query (x) before cross-attention
                if f"{src_prefix}norm2.weight" in keys:
                    weights[f"{dst_prefix}norm2_weight"] = mx.array(
                        f.get_tensor(f"{src_prefix}norm2.weight")
                    )
                    weights[f"{dst_prefix}norm2_bias"] = mx.array(
                        f.get_tensor(f"{src_prefix}norm2.bias")
                    )

                # norm_y: normalizes context (key/value) before cross-attention
                if f"{src_prefix}norm_y.weight" in keys:
                    weights[f"{dst_prefix}norm_y_weight"] = mx.array(
                        f.get_tensor(f"{src_prefix}norm_y.weight")
                    )
                    weights[f"{dst_prefix}norm_y_bias"] = mx.array(
                        f.get_tensor(f"{src_prefix}norm_y.bias")
                    )

                # Cross-attention - MASt3R uses separate projq/projk/projv
                cross_key = f"{src_prefix}cross_attn.projq.weight"
                if cross_key in keys:

                    # Q projection
                    weights[f"{dst_prefix}cross_attn.q.weight"] = mx.array(
                        f.get_tensor(f"{src_prefix}cross_attn.projq.weight")
                    )
                    weights[f"{dst_prefix}cross_attn.q.bias"] = mx.array(
                        f.get_tensor(f"{src_prefix}cross_attn.projq.bias")
                    )

                    # Combine K and V into KV
                    k_weight = f.get_tensor(f"{src_prefix}cross_attn.projk.weight")
                    v_weight = f.get_tensor(f"{src_prefix}cross_attn.projv.weight")
                    kv_weight = np.concatenate([k_weight, v_weight], axis=0)
                    weights[f"{dst_prefix}cross_attn.kv.weight"] = mx.array(kv_weight)

                    k_bias = f.get_tensor(f"{src_prefix}cross_attn.projk.bias")
                    v_bias = f.get_tensor(f"{src_prefix}cross_attn.projv.bias")
                    kv_bias = np.concatenate([k_bias, v_bias], axis=0)
                    weights[f"{dst_prefix}cross_attn.kv.bias"] = mx.array(kv_bias)

                    # Output projection
                    weights[f"{dst_prefix}cross_attn.proj.weight"] = mx.array(
                        f.get_tensor(f"{src_prefix}cross_attn.proj.weight")
                    )
                    weights[f"{dst_prefix}cross_attn.proj.bias"] = mx.array(
                        f.get_tensor(f"{src_prefix}cross_attn.proj.bias")
                    )

                # MLP - norm3 is norm2 in original
                if f"{src_prefix}norm3.weight" in keys:
                    weights[f"{dst_prefix}norm3_weight"] = mx.array(
                        f.get_tensor(f"{src_prefix}norm3.weight")
                    )
                    weights[f"{dst_prefix}norm3_bias"] = mx.array(
                        f.get_tensor(f"{src_prefix}norm3.bias")
                    )

                weights[f"{dst_prefix}mlp.fc1.weight"] = mx.array(
                    f.get_tensor(f"{src_prefix}mlp.fc1.weight")
                )
                weights[f"{dst_prefix}mlp.fc1.bias"] = mx.array(
                    f.get_tensor(f"{src_prefix}mlp.fc1.bias")
                )
                weights[f"{dst_prefix}mlp.fc2.weight"] = mx.array(
                    f.get_tensor(f"{src_prefix}mlp.fc2.weight")
                )
                weights[f"{dst_prefix}mlp.fc2.bias"] = mx.array(
                    f.get_tensor(f"{src_prefix}mlp.fc2.bias")
                )

            # Load all decoder blocks
            for i in range(self.decoder_config.decoder_depth):
                load_decoder_block(f"dec_blocks.{i}", f"dec_blocks.{i}.")
                load_decoder_block(f"dec_blocks2.{i}", f"dec_blocks2.{i}.")

            # Helper to transpose conv weights from PyTorch (O,I,H,W) to MLX (O,H,W,I)
            def transpose_conv_weight(w):
                """Transpose conv weight from PyTorch to MLX format."""
                return np.transpose(w, (0, 2, 3, 1))

            # Load DPT heads (downstream_head1/2)
            def load_dpt_head(src_head: str, dst_head: str) -> None:
                """Load DPT head weights."""
                # act_postprocess layers
                # Layer 0: Conv + ConvTranspose (upsample 4x)
                if f"{src_head}.dpt.act_postprocess.0.0.weight" in keys:
                    weights[f"{dst_head}.act_postprocess_0_conv.weight"] = mx.array(
                        transpose_conv_weight(f.get_tensor(f"{src_head}.dpt.act_postprocess.0.0.weight"))
                    )
                    weights[f"{dst_head}.act_postprocess_0_conv.bias"] = mx.array(
                        f.get_tensor(f"{src_head}.dpt.act_postprocess.0.0.bias")
                    )
                    # ConvTranspose: PyTorch is (I,O,H,W), MLX is (O,H,W,I)
                    ct_w = f.get_tensor(f"{src_head}.dpt.act_postprocess.0.1.weight")
                    weights[f"{dst_head}.act_postprocess_0_up.weight"] = mx.array(
                        np.transpose(ct_w, (1, 2, 3, 0))  # (I,O,H,W) -> (O,H,W,I)
                    )
                    weights[f"{dst_head}.act_postprocess_0_up.bias"] = mx.array(
                        f.get_tensor(f"{src_head}.dpt.act_postprocess.0.1.bias")
                    )

                # Layer 1: Conv + ConvTranspose (upsample 2x)
                if f"{src_head}.dpt.act_postprocess.1.0.weight" in keys:
                    weights[f"{dst_head}.act_postprocess_1_conv.weight"] = mx.array(
                        transpose_conv_weight(f.get_tensor(f"{src_head}.dpt.act_postprocess.1.0.weight"))
                    )
                    weights[f"{dst_head}.act_postprocess_1_conv.bias"] = mx.array(
                        f.get_tensor(f"{src_head}.dpt.act_postprocess.1.0.bias")
                    )
                    ct_w = f.get_tensor(f"{src_head}.dpt.act_postprocess.1.1.weight")
                    weights[f"{dst_head}.act_postprocess_1_up.weight"] = mx.array(
                        np.transpose(ct_w, (1, 2, 3, 0))
                    )
                    weights[f"{dst_head}.act_postprocess_1_up.bias"] = mx.array(
                        f.get_tensor(f"{src_head}.dpt.act_postprocess.1.1.bias")
                    )

                # Layer 2: Conv only
                if f"{src_head}.dpt.act_postprocess.2.0.weight" in keys:
                    weights[f"{dst_head}.act_postprocess_2_conv.weight"] = mx.array(
                        transpose_conv_weight(f.get_tensor(f"{src_head}.dpt.act_postprocess.2.0.weight"))
                    )
                    weights[f"{dst_head}.act_postprocess_2_conv.bias"] = mx.array(
                        f.get_tensor(f"{src_head}.dpt.act_postprocess.2.0.bias")
                    )

                # Layer 3: Conv + Conv (downsample 2x)
                if f"{src_head}.dpt.act_postprocess.3.0.weight" in keys:
                    weights[f"{dst_head}.act_postprocess_3_conv.weight"] = mx.array(
                        transpose_conv_weight(f.get_tensor(f"{src_head}.dpt.act_postprocess.3.0.weight"))
                    )
                    weights[f"{dst_head}.act_postprocess_3_conv.bias"] = mx.array(
                        f.get_tensor(f"{src_head}.dpt.act_postprocess.3.0.bias")
                    )
                    weights[f"{dst_head}.act_postprocess_3_down.weight"] = mx.array(
                        transpose_conv_weight(f.get_tensor(f"{src_head}.dpt.act_postprocess.3.1.weight"))
                    )
                    weights[f"{dst_head}.act_postprocess_3_down.bias"] = mx.array(
                        f.get_tensor(f"{src_head}.dpt.act_postprocess.3.1.bias")
                    )

                # scratch.layer_rn: projection to feature_dim
                for i in range(1, 5):
                    layer_key = f"{src_head}.dpt.scratch.layer{i}_rn.weight"
                    if layer_key in keys:
                        weights[f"{dst_head}.layer{i}_rn.weight"] = mx.array(
                            transpose_conv_weight(f.get_tensor(layer_key))
                        )

                # scratch.refinenet: feature fusion blocks
                for i in range(1, 5):
                    refine_prefix = f"{src_head}.dpt.scratch.refinenet{i}"
                    dst_refine = f"{dst_head}.refinenet{i}"

                    # out_conv
                    if f"{refine_prefix}.out_conv.weight" in keys:
                        weights[f"{dst_refine}.out_conv.weight"] = mx.array(
                            transpose_conv_weight(f.get_tensor(f"{refine_prefix}.out_conv.weight"))
                        )
                        weights[f"{dst_refine}.out_conv.bias"] = mx.array(
                            f.get_tensor(f"{refine_prefix}.out_conv.bias")
                        )

                    # ResidualConvUnits
                    for unit in ["resConfUnit1", "resConfUnit2"]:
                        for conv in ["conv1", "conv2"]:
                            w_key = f"{refine_prefix}.{unit}.{conv}.weight"
                            if w_key in keys:
                                weights[f"{dst_refine}.{unit}.{conv}.weight"] = mx.array(
                                    transpose_conv_weight(f.get_tensor(w_key))
                                )
                                weights[f"{dst_refine}.{unit}.{conv}.bias"] = mx.array(
                                    f.get_tensor(f"{refine_prefix}.{unit}.{conv}.bias")
                                )

                # head: final projection (Conv -> Interpolate -> Conv -> ReLU -> Conv)
                # head.0 -> head_conv1, head.2 -> head_conv2, head.4 -> head_conv3
                head_map = [("0", "head_conv1"), ("2", "head_conv2"), ("4", "head_conv3")]
                for src_idx, dst_name in head_map:
                    w_key = f"{src_head}.dpt.head.{src_idx}.weight"
                    if w_key in keys:
                        weights[f"{dst_head}.{dst_name}.weight"] = mx.array(
                            transpose_conv_weight(f.get_tensor(w_key))
                        )
                        weights[f"{dst_head}.{dst_name}.bias"] = mx.array(
                            f.get_tensor(f"{src_head}.dpt.head.{src_idx}.bias")
                        )

            # Load both heads
            load_dpt_head("downstream_head1", "head1")
            load_dpt_head("downstream_head2", "head2")

            # Load local features MLP (for descriptors)
            for head_idx in [1, 2]:
                src = f"downstream_head{head_idx}.head_local_features"
                dst = f"head_local_features{head_idx}"

                # fc1 -> layers.0, fc2 -> layers.2
                if f"{src}.fc1.weight" in keys:
                    weights[f"{dst}.layers.0.weight"] = mx.array(
                        f.get_tensor(f"{src}.fc1.weight")
                    )
                    weights[f"{dst}.layers.0.bias"] = mx.array(
                        f.get_tensor(f"{src}.fc1.bias")
                    )
                    weights[f"{dst}.layers.2.weight"] = mx.array(
                        f.get_tensor(f"{src}.fc2.weight")
                    )
                    weights[f"{dst}.layers.2.bias"] = mx.array(
                        f.get_tensor(f"{src}.fc2.bias")
                    )

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

        # Decode
        H = self.encoder_config.patch_h
        W = self.encoder_config.patch_w

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
