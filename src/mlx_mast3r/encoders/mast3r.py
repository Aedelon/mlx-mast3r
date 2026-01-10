"""MASt3R ViT-Large Encoder - Ultra-optimized MLX implementation.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.

Optimizations:
- mx.fast.scaled_dot_product_attention (fused SDPA)
- mx.fast.layer_norm (fused LayerNorm)
- mx.compile() for graph compilation
- FP16/BF16 precision support
- gelu_fast_approx for faster activation
- Custom RoPE 2D implementation
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
from mlx_mast3r.encoders.base import EncoderBlock, PatchEmbed


@dataclass
class Mast3rEncoderConfig:
    """MASt3R ViT-Large encoder configuration."""

    resolution: int = 512
    patch_size: int = 16
    embed_dim: int = 1024
    num_heads: int = 16
    head_dim: int = 64
    mlp_ratio: float = 4.0
    depth: int = 24
    rope_theta: float = 100.0
    precision: Literal["fp32", "fp16", "bf16"] = "fp16"

    @property
    def dtype(self) -> mx.Dtype:
        return {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}[self.precision]

    @property
    def mlp_dim(self) -> int:
        return int(self.embed_dim * self.mlp_ratio)

    @property
    def img_h(self) -> int:
        return self.resolution

    @property
    def img_w(self) -> int:
        return (int(self.resolution * 4 / 3) // self.patch_size) * self.patch_size

    @property
    def patch_h(self) -> int:
        return self.img_h // self.patch_size

    @property
    def patch_w(self) -> int:
        return self.img_w // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.patch_h * self.patch_w


class RoPE2D:
    """2D Rotary Position Embedding - matching PyTorch implementation exactly.

    The RoPE2D splits head_dim into two halves:
    - First half: rotated by Y position
    - Second half: rotated by X position

    This matches the CroCo/MASt3R RoPE2D implementation.
    """

    def __init__(self, freq: float = 100.0, F0: float = 1.0):
        self.base = freq
        self.F0 = F0
        self._cos_cache: dict[tuple, mx.array] = {}
        self._sin_cache: dict[tuple, mx.array] = {}

    def get_cos_sin(
        self, D: int, max_pos: int, dtype: mx.Dtype
    ) -> tuple[mx.array, mx.array]:
        """Get cached cos/sin tables.

        Args:
            D: Half of head_dim (each half gets D dimensions)
            max_pos: Maximum position value
            dtype: Output dtype

        Returns:
            (cos, sin) each of shape [max_pos, D]
        """
        cache_key = (D, max_pos, dtype)
        if cache_key not in self._cos_cache:
            # inv_freq: [D//2]
            inv_freq = self.F0 / (self.base ** (np.arange(0, D, 2, dtype=np.float32) / D))

            # positions: [max_pos]
            t = np.arange(max_pos, dtype=np.float32)

            # freqs: [max_pos, D//2]
            freqs = np.outer(t, inv_freq)

            # Double frequencies to match head_dim//2: [max_pos, D]
            freqs = np.concatenate([freqs, freqs], axis=-1)

            self._cos_cache[cache_key] = mx.array(np.cos(freqs), dtype=dtype)
            self._sin_cache[cache_key] = mx.array(np.sin(freqs), dtype=dtype)

        return self._cos_cache[cache_key], self._sin_cache[cache_key]

    @staticmethod
    def rotate_half(x: mx.array) -> mx.array:
        """Rotate half of the dimensions."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)

    def apply_rope1d(
        self,
        tokens: mx.array,
        pos1d: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        """Apply 1D RoPE using position indexing.

        Args:
            tokens: [B, heads, N, D] where D = head_dim // 2
            pos1d: [B, N] integer positions
            cos: [max_pos, D] cos table
            sin: [max_pos, D] sin table

        Returns:
            Rotated tokens [B, heads, N, D]
        """
        # Index cos/sin by positions: [B, N, D]
        cos_indexed = cos[pos1d]  # [B, N, D]
        sin_indexed = sin[pos1d]  # [B, N, D]

        # Expand for heads: [B, 1, N, D]
        cos_indexed = cos_indexed[:, None, :, :]
        sin_indexed = sin_indexed[:, None, :, :]

        return tokens * cos_indexed + self.rotate_half(tokens) * sin_indexed

    def __call__(
        self,
        tokens: mx.array,
        positions: mx.array,
    ) -> mx.array:
        """Apply 2D RoPE to tokens.

        Args:
            tokens: [B, heads, N, head_dim]
            positions: [B, N, 2] where positions[:,:,0] = y, positions[:,:,1] = x

        Returns:
            Rotated tokens [B, heads, N, head_dim]
        """
        head_dim = tokens.shape[-1]
        D = head_dim // 2  # Each half gets D dimensions

        # Get cos/sin tables
        max_pos = int(positions.max()) + 1
        cos, sin = self.get_cos_sin(D, max_pos, tokens.dtype)

        # Split tokens into y and x halves
        y_tokens = tokens[..., :D]
        x_tokens = tokens[..., D:]

        # Get y and x positions
        pos_y = positions[:, :, 0].astype(mx.int32)  # [B, N]
        pos_x = positions[:, :, 1].astype(mx.int32)  # [B, N]

        # Apply rope1d to each half
        y_rotated = self.apply_rope1d(y_tokens, pos_y, cos, sin)
        x_rotated = self.apply_rope1d(x_tokens, pos_x, cos, sin)

        # Concatenate back
        return mx.concatenate([y_rotated, x_rotated], axis=-1)


def get_positions_grid(height: int, width: int) -> mx.array:
    """Generate position grid for patches.

    Returns:
        positions: [1, H*W, 2] where each position is (y, x)
    """
    y = np.arange(height)
    x = np.arange(width)
    # cartesian_prod equivalent: all (y, x) pairs in row-major order
    grid_y, grid_x = np.meshgrid(y, x, indexing='ij')
    positions = np.stack([grid_y.flatten(), grid_x.flatten()], axis=-1)
    return mx.array(positions[None, :, :], dtype=mx.int32)  # [1, H*W, 2]


def _create_mast3r_block(config: Mast3rEncoderConfig, rope: RoPE2D) -> EncoderBlock:
    """Create a MASt3R encoder block with the correct configuration."""
    return EncoderBlock(
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        mlp_dim=config.mlp_dim,
        rope=rope,
        use_layer_scale=False,
        fast_gelu=False,  # MASt3R uses gelu_approx for PyTorch correlation
    )


class Mast3rEncoder(nn.Module):
    """MASt3R ViT-Large Encoder - Ultra-optimized.

    Features:
    - 2D RoPE (Rotary Position Embedding) matching PyTorch exactly
    - mx.fast.scaled_dot_product_attention
    - mx.fast.layer_norm
    - 24 transformer blocks

    Example:
        >>> config = Mast3rEncoderConfig(resolution=512, precision="fp16")
        >>> encoder = Mast3rEncoder(config)
        >>> encoder.load_weights("path/to/unified.safetensors")
        >>> features = encoder(image)  # [B, N, D]
    """

    def __init__(self, config: Mast3rEncoderConfig):
        super().__init__()
        self.config = config

        # RoPE2D - shared across all attention layers
        self.rope = RoPE2D(freq=config.rope_theta)

        # Patch embedding
        self.patch_embed = PatchEmbed(config.embed_dim, config.patch_size)

        # Encoder blocks with shared RoPE
        self.blocks = [_create_mast3r_block(config, self.rope) for _ in range(config.depth)]

        # Final norm
        self.norm_weight = mx.ones((config.embed_dim,))
        self.norm_bias = mx.zeros((config.embed_dim,))

        # Cached positions grid
        self._positions: mx.array | None = None

    def _get_positions(self, H: int, W: int, B: int) -> mx.array:
        """Get position grid, caching for efficiency."""
        if self._positions is None or self._positions.shape[1] != H * W:
            self._positions = get_positions_grid(H, W)
        # Expand for batch: [1, H*W, 2] -> [B, H*W, 2]
        if B > 1:
            return mx.broadcast_to(self._positions, (B, H * W, 2))
        return self._positions

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: [B, H, W, 3] input image (NHWC format)

        Returns:
            [B, N, D] encoder features
        """
        B = x.shape[0]
        H, W = self.config.patch_h, self.config.patch_w

        # Get positions grid
        positions = self._get_positions(H, W, B)

        # Patch embedding
        x = self.patch_embed(x)

        # Encoder blocks with positions
        for block in self.blocks:
            x = block(x, positions)

        # Apply enc_norm (matches PyTorch _encode_image output)
        x = mx.fast.layer_norm(x, self.norm_weight, self.norm_bias, eps=LAYER_NORM_EPS)

        return x


class Mast3rEncoderEngine:
    """High-level MASt3R encoder with loading and inference."""

    def __init__(
        self,
        resolution: int = 512,
        precision: Literal["fp32", "fp16", "bf16"] = "fp16",
        compile: bool = True,
    ):
        self.config = Mast3rEncoderConfig(resolution=resolution, precision=precision)
        self.model = Mast3rEncoder(self.config)
        self._compiled_forward = None
        self._compile = compile
        self._loaded = False

    def load(self, path: str | Path) -> None:
        """Load weights from safetensors file (unified.safetensors format)."""
        from safetensors import safe_open

        weights = {}
        with safe_open(str(path), framework="numpy") as f:
            # Patch embedding (keys: patch_embed.proj.*)
            weights["patch_embed.proj.weight"] = self._convert_conv(
                f.get_tensor("patch_embed.proj.weight")
            )
            weights["patch_embed.proj.bias"] = mx.array(
                f.get_tensor("patch_embed.proj.bias")
            )

            # Encoder blocks (keys: enc_blocks.X.*)
            for i in range(self.config.depth):
                prefix = f"enc_blocks.{i}."

                weights[f"blocks.{i}.attn.qkv.weight"] = mx.array(
                    f.get_tensor(prefix + "attn.qkv.weight")
                )
                weights[f"blocks.{i}.attn.qkv.bias"] = mx.array(
                    f.get_tensor(prefix + "attn.qkv.bias")
                )
                weights[f"blocks.{i}.attn.proj.weight"] = mx.array(
                    f.get_tensor(prefix + "attn.proj.weight")
                )
                weights[f"blocks.{i}.attn.proj.bias"] = mx.array(
                    f.get_tensor(prefix + "attn.proj.bias")
                )

                weights[f"blocks.{i}.mlp.fc1.weight"] = mx.array(
                    f.get_tensor(prefix + "mlp.fc1.weight")
                )
                weights[f"blocks.{i}.mlp.fc1.bias"] = mx.array(
                    f.get_tensor(prefix + "mlp.fc1.bias")
                )
                weights[f"blocks.{i}.mlp.fc2.weight"] = mx.array(
                    f.get_tensor(prefix + "mlp.fc2.weight")
                )
                weights[f"blocks.{i}.mlp.fc2.bias"] = mx.array(
                    f.get_tensor(prefix + "mlp.fc2.bias")
                )

                weights[f"blocks.{i}.norm1_weight"] = mx.array(
                    f.get_tensor(prefix + "norm1.weight")
                )
                weights[f"blocks.{i}.norm1_bias"] = mx.array(f.get_tensor(prefix + "norm1.bias"))
                weights[f"blocks.{i}.norm2_weight"] = mx.array(
                    f.get_tensor(prefix + "norm2.weight")
                )
                weights[f"blocks.{i}.norm2_bias"] = mx.array(f.get_tensor(prefix + "norm2.bias"))

            # Final norm (keys: enc_norm.*)
            weights["norm_weight"] = mx.array(f.get_tensor("enc_norm.weight"))
            weights["norm_bias"] = mx.array(f.get_tensor("enc_norm.bias"))

        # Cast to dtype
        if self.config.dtype != mx.float32:
            weights = {k: v.astype(self.config.dtype) for k, v in weights.items()}

        self.model.load_weights(list(weights.items()), strict=False)
        mx.eval(self.model.parameters())

        # Compile
        if self._compile:
            self._compiled_forward = mx.compile(self.model.__call__)

        self._loaded = True

    def _convert_conv(self, w: np.ndarray) -> mx.array:
        """Convert PyTorch conv weight [O,I,H,W] -> MLX [O,H,W,I]."""
        return mx.array(np.transpose(w, (0, 2, 3, 1)))

    def __call__(self, x: mx.array) -> mx.array:
        """Run inference."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if x.dtype != self.config.dtype:
            x = x.astype(self.config.dtype)

        if self._compiled_forward:
            return self._compiled_forward(x)
        return self.model(x)

    def infer(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """Run inference on numpy image [H,W,3] uint8."""
        import time

        # Preprocess (MASt3R normalization)
        x = img.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = mx.array(x[None, :, :, :])

        t0 = time.perf_counter()
        out = self(x)
        mx.eval(out)
        ms = (time.perf_counter() - t0) * 1000

        return np.array(out[0]), ms

    def warmup(self, iterations: int = 5) -> None:
        """Warmup the model."""
        dummy = mx.zeros((1, self.config.img_h, self.config.img_w, 3), dtype=mx.float32)
        for _ in range(iterations):
            out = self(dummy)
            mx.eval(out)
