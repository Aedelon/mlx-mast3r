"""DUNE DINOv2 Encoder - Ultra-optimized MLX implementation.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.

Optimizations:
- mx.fast.scaled_dot_product_attention (fused SDPA)
- mx.fast.layer_norm (fused LayerNorm)
- mx.compile() for graph compilation
- FP16/BF16 precision support
- gelu_approx for faster activation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np


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
    num_register_tokens: int = 4
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


class DuneAttention(nn.Module):
    """Multi-head self-attention with fused SDPA."""

    def __init__(self, config: DuneConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        dim = config.embed_dim
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = (
            qkv[:, :, 0].transpose(0, 2, 1, 3),
            qkv[:, :, 1].transpose(0, 2, 1, 3),
            qkv[:, :, 2].transpose(0, 2, 1, 3),
        )

        # Fused scaled dot-product attention
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

        return self.proj(out.transpose(0, 2, 1, 3).reshape(B, N, D))


class DuneMLP(nn.Module):
    """MLP with approximate GELU for speed."""

    def __init__(self, config: DuneConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))


class DuneBlock(nn.Module):
    """DINOv2 encoder block with Layer Scale."""

    def __init__(self, config: DuneConfig):
        super().__init__()
        self.embed_dim = config.embed_dim

        # LayerNorm weights (for mx.fast.layer_norm)
        self.norm1_weight = mx.ones((config.embed_dim,))
        self.norm1_bias = mx.zeros((config.embed_dim,))
        self.norm2_weight = mx.ones((config.embed_dim,))
        self.norm2_bias = mx.zeros((config.embed_dim,))

        # Layer Scale (DINOv2 specific)
        self.ls1_gamma = mx.ones((config.embed_dim,))
        self.ls2_gamma = mx.ones((config.embed_dim,))

        self.attn = DuneAttention(config)
        self.mlp = DuneMLP(config)

    def __call__(self, x: mx.array) -> mx.array:
        # Pre-norm attention with layer scale
        normed1 = mx.fast.layer_norm(x, self.norm1_weight, self.norm1_bias, eps=1e-6)
        x = x + self.ls1_gamma * self.attn(normed1)

        # Pre-norm MLP with layer scale
        normed2 = mx.fast.layer_norm(x, self.norm2_weight, self.norm2_bias, eps=1e-6)
        x = x + self.ls2_gamma * self.mlp(normed2)

        return x


class DunePatchEmbed(nn.Module):
    """Patch embedding with 14x14 patches."""

    def __init__(self, config: DuneConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Conv2d(
            3,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        B = x.shape[0]
        x = self.proj(x)  # [B, H/14, W/14, embed_dim]
        return x.reshape(B, -1, self.config.embed_dim)


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
        self.patch_embed = DunePatchEmbed(config)

        # Learnable tokens
        self.cls_token = mx.zeros((1, 1, config.embed_dim))
        self.register_tokens = mx.zeros((1, config.num_register_tokens, config.embed_dim))

        # Position embeddings (placeholder, loaded from weights)
        self.pos_embed = mx.zeros((1, 577, config.embed_dim))
        self.pos_embed_h = 24
        self.pos_embed_w = 24

        # Encoder blocks
        self.blocks = [DuneBlock(config) for _ in range(config.depth)]

        # Final norm
        self.norm_weight = mx.ones((config.embed_dim,))
        self.norm_bias = mx.zeros((config.embed_dim,))

    def _interpolate_pos_embed(self, H: int, W: int) -> mx.array:
        """Interpolate position embeddings for different resolutions."""
        cls_embed = self.pos_embed[:, :1, :]
        patch_embed = self.pos_embed[:, 1:, :]

        orig_H, orig_W = self.pos_embed_h, self.pos_embed_w

        if H == orig_H and W == orig_W:
            return self.pos_embed

        D = patch_embed.shape[-1]
        patch_embed_np = np.array(patch_embed[0]).reshape(orig_H, orig_W, D)

        # Bilinear interpolation
        y_coords = np.linspace(0, orig_H - 1, H)
        x_coords = np.linspace(0, orig_W - 1, W)

        interpolated = np.zeros((H, W, D), dtype=np.float32)
        for i, y in enumerate(y_coords):
            y0, y1 = int(np.floor(y)), min(int(np.floor(y)) + 1, orig_H - 1)
            wy = y - y0
            for j, x in enumerate(x_coords):
                x0, x1 = int(np.floor(x)), min(int(np.floor(x)) + 1, orig_W - 1)
                wx = x - x0
                interpolated[i, j] = (
                    (1 - wy) * (1 - wx) * patch_embed_np[y0, x0]
                    + (1 - wy) * wx * patch_embed_np[y0, x1]
                    + wy * (1 - wx) * patch_embed_np[y1, x0]
                    + wy * wx * patch_embed_np[y1, x1]
                )

        interpolated = mx.array(interpolated.reshape(1, H * W, D))
        return mx.concatenate([cls_embed, interpolated], axis=1)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: [B, H, W, 3] input image (NHWC format)

        Returns:
            [B, N, D] encoder features (patch tokens only, no CLS/register)
        """
        B = x.shape[0]
        H, W = self.config.patch_h, self.config.patch_w

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

        # Final norm
        x = mx.fast.layer_norm(x, self.norm_weight, self.norm_bias, eps=1e-6)

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
            weights["register_tokens"] = mx.array(f.get_tensor("encoder.register_tokens"))

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

        # Preprocess
        x = img.astype(np.float32) / 127.5 - 1.0
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
