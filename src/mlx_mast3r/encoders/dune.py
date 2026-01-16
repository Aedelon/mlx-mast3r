"""DUNE DINOv2 Encoder - Ultra-optimized MLX implementation.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.

Optimizations:
- mx.fast.scaled_dot_product_attention (fused SDPA)
- mx.fast.layer_norm (fused LayerNorm)
- mx.compile() for graph compilation
- FP16/BF16 precision support
- gelu_fast_approx for faster activation
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
class DuneConfig:
    """DUNE DINOv2 encoder configuration."""

    variant: Literal["small", "base"] = "base"
    resolution: int = 336
    patch_size: int = 14
    embed_dim: int = 768
    num_heads: int = 12
    head_dim: int = 64
    mlp_ratio: float = 4.0
    depth: int = 12
    # Note: DuneMASt3R uses backbone without register tokens (num_register_tokens=0)
    # Standard DINOv2-reg has 4 register tokens
    num_register_tokens: int = 0
    precision: Literal["fp32", "fp16", "bf16"] = "fp16"

    def __post_init__(self) -> None:
        if self.variant == "small":
            self.embed_dim = 384
            self.num_heads = 6
        elif self.variant == "base":
            self.embed_dim = 768
            self.num_heads = 12

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
        return self.resolution  # Square images for DUNE

    @property
    def patch_h(self) -> int:
        return self.img_h // self.patch_size

    @property
    def patch_w(self) -> int:
        return self.img_w // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.patch_h * self.patch_w


def _create_dune_block(config: DuneConfig) -> EncoderBlock:
    """Create a DUNE encoder block with the correct configuration."""
    return EncoderBlock(
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        mlp_dim=config.mlp_dim,
        rope=None,  # DUNE doesn't use RoPE
        use_layer_scale=True,  # DINOv2 uses layer scale
        fast_gelu=False,  # Use exact GELU to match PyTorch
    )


class DuneEncoder(nn.Module):
    """DUNE DINOv2 Encoder - Ultra-optimized.

    Features:
    - mx.fast.scaled_dot_product_attention
    - mx.fast.layer_norm
    - Layer Scale (DINOv2)
    - CLS + Register tokens
    - Position embedding interpolation

    Example:
        >>> config = DuneConfig(variant="base", resolution=336, precision="fp16")
        >>> encoder = DuneEncoder(config)
        >>> encoder.load_weights("path/to/encoder.safetensors")
        >>> features = encoder(image)  # [B, N, D]
    """

    def __init__(self, config: DuneConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbed(config.embed_dim, config.patch_size)

        # Learnable tokens
        self.cls_token = mx.zeros((1, 1, config.embed_dim))
        self.register_tokens = mx.zeros((1, config.num_register_tokens, config.embed_dim))

        # Position embeddings (placeholder, loaded from weights)
        self.pos_embed = mx.zeros((1, 577, config.embed_dim))
        self.pos_embed_h = 24
        self.pos_embed_w = 24

        # Encoder blocks
        self.blocks = [_create_dune_block(config) for _ in range(config.depth)]

        # Final norm
        self.norm_weight = mx.ones((config.embed_dim,))
        self.norm_bias = mx.zeros((config.embed_dim,))

    def _interpolate_pos_embed(self, H: int, W: int) -> mx.array:
        """Interpolate position embeddings for different resolutions.

        Uses fused grid_sample Metal kernel for ~3x speedup over naive gather.
        """
        from mlx_mast3r.kernels import grid_sample

        cls_embed = self.pos_embed[:, :1, :]
        patch_embed = self.pos_embed[:, 1:, :]

        orig_H, orig_W = self.pos_embed_h, self.pos_embed_w

        if H == orig_H and W == orig_W:
            return self.pos_embed

        D = patch_embed.shape[-1]
        patch_2d = patch_embed.reshape(1, orig_H, orig_W, D)

        # Create normalized sampling grid [-1, 1] for grid_sample
        gy = mx.linspace(-1, 1, H)
        gx = mx.linspace(-1, 1, W)
        grid_y = mx.broadcast_to(gy[:, None], (H, W))
        grid_x = mx.broadcast_to(gx[None, :], (H, W))
        grid = mx.stack([grid_x, grid_y], axis=-1)[None, :, :, :]  # [1, H, W, 2]

        # Fused bilinear interpolation via Metal kernel
        interpolated = grid_sample(patch_2d, grid)  # [1, H, W, D]

        interpolated = interpolated.reshape(1, H * W, D)
        return mx.concatenate([cls_embed, interpolated], axis=1)

    def __call__(self, x: mx.array, apply_norm: bool = True) -> mx.array:
        """Forward pass.

        Args:
            x: [B, H, W, 3] input image (NHWC format)
            apply_norm: If True, apply final LayerNorm. Set to False when
                using DuneMASt3R which has its own trained norm weights.

        Returns:
            [B, N, D] encoder features (patch tokens only, no CLS/register)
        """
        B = x.shape[0]
        # Compute patch grid from actual input size, not config
        H = x.shape[1] // self.config.patch_size
        W = x.shape[2] // self.config.patch_size

        # Patch embedding
        x = self.patch_embed(x)

        # Add CLS token
        cls_tokens = mx.broadcast_to(self.cls_token, (B, 1, self.config.embed_dim))
        x = mx.concatenate([cls_tokens, x], axis=1)

        # Add position embeddings
        pos_embed = self._interpolate_pos_embed(H, W)
        x = x + pos_embed.astype(x.dtype)

        # Add register tokens
        if self.config.num_register_tokens > 0:
            reg_tokens = mx.broadcast_to(
                self.register_tokens, (B, self.config.num_register_tokens, self.config.embed_dim)
            )
            x = mx.concatenate([x[:, :1], reg_tokens, x[:, 1:]], axis=1)

        # Encoder blocks
        for block in self.blocks:
            x = block(x)

        # Final norm (optional - skip when DuneMASt3R will apply its own trained norm)
        if apply_norm:
            x = mx.fast.layer_norm(x, self.norm_weight, self.norm_bias, eps=LAYER_NORM_EPS)

        # Return patch tokens only
        start_idx = 1 + self.config.num_register_tokens
        return x[:, start_idx:, :]


class DuneEncoderEngine:
    """High-level DUNE encoder with loading and inference."""

    def __init__(
        self,
        variant: Literal["small", "base"] = "base",
        resolution: int = 336,
        precision: Literal["fp32", "fp16", "bf16"] = "fp16",
        compile: bool = True,
    ):
        self.config = DuneConfig(variant=variant, resolution=resolution, precision=precision)
        self.model = DuneEncoder(self.config)
        self._compiled_forward = None
        self._compile = compile
        self._loaded = False

    def load(self, path: str | Path) -> None:
        """Load weights from safetensors file."""
        from safetensors import safe_open

        weights = {}
        with safe_open(str(path), framework="numpy") as f:
            # Patch embedding
            weights["patch_embed.proj.weight"] = self._convert_conv(
                f.get_tensor("encoder.patch_embed.proj.weight")
            )
            weights["patch_embed.proj.bias"] = mx.array(
                f.get_tensor("encoder.patch_embed.proj.bias")
            )

            # Tokens
            weights["cls_token"] = mx.array(f.get_tensor("encoder.cls_token"))

            # Register tokens (optional - DuneMASt3R backbone has none)
            if "encoder.register_tokens" in f.keys():
                weights["register_tokens"] = mx.array(f.get_tensor("encoder.register_tokens"))
            elif self.config.num_register_tokens > 0:
                # Initialize with zeros if expected but not in weights
                weights["register_tokens"] = mx.zeros(
                    (1, self.config.num_register_tokens, self.config.embed_dim)
                )

            # Position embeddings
            pos_embed = f.get_tensor("encoder.pos_embed")
            pos_embed_mx = mx.array(pos_embed)
            n_patches = pos_embed.shape[1] - 1

            # Detect grid size
            if n_patches == 1024:
                self.model.pos_embed_h, self.model.pos_embed_w = 32, 32
            elif n_patches == 768:
                self.model.pos_embed_h, self.model.pos_embed_w = 24, 32
            elif n_patches == 576:
                self.model.pos_embed_h, self.model.pos_embed_w = 24, 24
            else:
                side = int(math.sqrt(n_patches))
                self.model.pos_embed_h = side
                self.model.pos_embed_w = n_patches // side

            # Encoder blocks
            for i in range(self.config.depth):
                prefix = f"encoder.blocks.0.{i}."

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

                weights[f"blocks.{i}.ls1_gamma"] = mx.array(f.get_tensor(prefix + "ls1.gamma"))
                weights[f"blocks.{i}.ls2_gamma"] = mx.array(f.get_tensor(prefix + "ls2.gamma"))

            # Final norm
            weights["norm_weight"] = mx.array(f.get_tensor("encoder.norm.weight"))
            weights["norm_bias"] = mx.array(f.get_tensor("encoder.norm.bias"))

        # Cast to target dtype
        if self.config.dtype != mx.float32:
            weights = {k: v.astype(self.config.dtype) for k, v in weights.items()}

        # Load weights (pos_embed handled separately due to variable size)
        self.model.pos_embed = pos_embed_mx.astype(self.config.dtype)
        self.model.load_weights(list(weights.items()), strict=False)
        mx.eval(self.model.parameters())

        # Compile (shapeless=True not compatible with dynamic slicing)
        if self._compile:
            self._compiled_forward = mx.compile(self.model.__call__)

        self._loaded = True

    def _convert_conv(self, w) -> mx.array:
        """Convert PyTorch conv weight [O,I,H,W] -> MLX [O,H,W,I]."""
        return mx.array(w).transpose(0, 2, 3, 1)

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

        # Preprocess - pure MLX
        x = mx.array(img).astype(mx.float32) / 127.5 - 1.0
        x = x[None, :, :, :]

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
