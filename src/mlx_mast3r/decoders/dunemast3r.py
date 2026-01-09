"""DuneMASt3R Decoder - Ultra-optimized MLX implementation.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.

DuneMASt3R = DUNE Encoder (separate) + MASt3R Decoder (this file)

Architecture:
- decoder_embed: Project DUNE features to decoder space
- dec_blocks: 12 transformer decoder blocks (view 1)
- dec_blocks2: 12 transformer decoder blocks (view 2)
- downstream_head1/2: DPT heads for 3D points + descriptors

Optimizations:
- mx.fast.scaled_dot_product_attention
- mx.fast.layer_norm
- mx.compile()
- FP16/BF16 precision
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
class DuneMast3rConfig:
    """DuneMASt3R decoder configuration."""

    encoder_dim: int = 768  # DUNE Base output dim
    decoder_dim: int = 768
    num_heads: int = 12
    head_dim: int = 64
    mlp_ratio: float = 4.0
    decoder_depth: int = 12
    patch_size: int = 14  # DUNE uses 14x14 patches
    precision: Literal["fp32", "fp16", "bf16"] = "fp16"

    @property
    def dtype(self) -> mx.Dtype:
        return {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}[self.precision]

    @property
    def mlp_dim(self) -> int:
        return int(self.decoder_dim * self.mlp_ratio)

    @classmethod
    def for_dune_base(cls, precision: str = "fp16") -> "DuneMast3rConfig":
        """Config for DUNE Base encoder (768 dim)."""
        return cls(encoder_dim=768, decoder_dim=768, precision=precision)

    @classmethod
    def for_dune_small(cls, precision: str = "fp16") -> "DuneMast3rConfig":
        """Config for DUNE Small encoder (384 dim)."""
        return cls(encoder_dim=384, decoder_dim=768, precision=precision)


class DecoderAttention(nn.Module):
    """Multi-head self-attention for decoder."""

    def __init__(self, config: DuneMast3rConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        dim = config.decoder_dim
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

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.proj(out.transpose(0, 2, 1, 3).reshape(B, N, D))


class CrossAttention(nn.Module):
    """Cross-attention between two views."""

    def __init__(self, config: DuneMast3rConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        dim = config.decoder_dim
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        B, N, D = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0].transpose(0, 2, 1, 3), kv[:, :, 1].transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.proj(out.transpose(0, 2, 1, 3).reshape(B, N, D))


class DecoderMLP(nn.Module):
    """MLP with approximate GELU."""

    def __init__(self, config: DuneMast3rConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.decoder_dim, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.decoder_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))


class DecoderBlock(nn.Module):
    """Decoder transformer block with self and cross attention."""

    def __init__(self, config: DuneMast3rConfig):
        super().__init__()
        dim = config.decoder_dim

        # Self-attention
        self.norm1_weight = mx.ones((dim,))
        self.norm1_bias = mx.zeros((dim,))
        self.self_attn = DecoderAttention(config)

        # Cross-attention
        self.norm2_weight = mx.ones((dim,))
        self.norm2_bias = mx.zeros((dim,))
        self.cross_attn = CrossAttention(config)

        # MLP
        self.norm3_weight = mx.ones((dim,))
        self.norm3_bias = mx.zeros((dim,))
        self.mlp = DecoderMLP(config)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        # Self-attention
        normed = mx.fast.layer_norm(x, self.norm1_weight, self.norm1_bias, eps=1e-6)
        x = x + self.self_attn(normed)

        # Cross-attention
        normed = mx.fast.layer_norm(x, self.norm2_weight, self.norm2_bias, eps=1e-6)
        x = x + self.cross_attn(normed, context)

        # MLP
        normed = mx.fast.layer_norm(x, self.norm3_weight, self.norm3_bias, eps=1e-6)
        x = x + self.mlp(normed)

        return x


class DPTHead(nn.Module):
    """Dense Prediction Transformer head for 3D points."""

    def __init__(self, config: DuneMast3rConfig, output_dim: int = 3):
        super().__init__()
        dim = config.decoder_dim

        # Reassembly layers
        self.layer1_rn = nn.Conv2d(dim, 256, kernel_size=3, padding=1)
        self.layer2_rn = nn.Conv2d(dim, 256, kernel_size=3, padding=1)
        self.layer3_rn = nn.Conv2d(dim, 256, kernel_size=3, padding=1)
        self.layer4_rn = nn.Conv2d(dim, 256, kernel_size=3, padding=1)

        # Fusion layers
        self.refinenet1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.refinenet2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.refinenet3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.refinenet4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Output projection
        self.output_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.output_conv2 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.output_conv3 = nn.Conv2d(32, output_dim, kernel_size=1)

    def __call__(self, features: list[mx.array], H: int, W: int) -> mx.array:
        """
        Args:
            features: List of 4 feature maps at different scales
            H, W: Output spatial dimensions

        Returns:
            [B, H, W, output_dim] predictions
        """
        # This is a simplified DPT - full implementation would include
        # proper multi-scale fusion with upsampling
        B = features[0].shape[0]

        # Reshape to spatial
        x = features[-1].reshape(B, H, W, -1)

        # Simple conv processing
        x = nn.relu(self.layer4_rn(x))
        x = nn.relu(self.refinenet4(x))
        x = nn.relu(self.output_conv1(x))
        x = nn.relu(self.output_conv2(x))
        x = self.output_conv3(x)

        return x


class DuneMast3rDecoder(nn.Module):
    """DuneMASt3R Decoder - Asymmetric decoder for stereo 3D reconstruction.

    Takes DUNE encoder features from two views and outputs:
    - 3D points in camera space
    - Confidence maps
    - Dense descriptors for matching

    Example:
        >>> config = DuneMast3rConfig.for_dune_base(precision="fp16")
        >>> decoder = DuneMast3rDecoder(config)
        >>> decoder.load("path/to/dunemast3r_vitbase.pth")
        >>> pts3d, conf, desc = decoder(feat1, feat2, shape1, shape2)
    """

    def __init__(self, config: DuneMast3rConfig):
        super().__init__()
        self.config = config

        # Project encoder features to decoder dim
        self.decoder_embed = nn.Linear(config.encoder_dim, config.decoder_dim)

        # Encoder norm (applied to DUNE output)
        self.enc_norm_weight = mx.ones((config.encoder_dim,))
        self.enc_norm_bias = mx.zeros((config.encoder_dim,))

        # Mask token for missing regions
        self.mask_token = mx.zeros((1, 1, config.decoder_dim))

        # Decoder blocks for view 1
        self.dec_blocks = [DecoderBlock(config) for _ in range(config.decoder_depth)]

        # Decoder blocks for view 2
        self.dec_blocks2 = [DecoderBlock(config) for _ in range(config.decoder_depth)]

        # Decoder norm
        self.dec_norm_weight = mx.ones((config.decoder_dim,))
        self.dec_norm_bias = mx.zeros((config.decoder_dim,))

        # Output heads
        self.head1 = DPTHead(config, output_dim=3 + 1 + 24)  # pts3d + conf + desc
        self.head2 = DPTHead(config, output_dim=3 + 1 + 24)

    def __call__(
        self,
        feat1: mx.array,
        feat2: mx.array,
        shape1: tuple[int, int],
        shape2: tuple[int, int],
    ) -> tuple[dict, dict]:
        """Forward pass.

        Args:
            feat1: [B, N1, D] encoder features for view 1
            feat2: [B, N2, D] encoder features for view 2
            shape1: (H1, W1) spatial shape for view 1
            shape2: (H2, W2) spatial shape for view 2

        Returns:
            (output1, output2) dicts with keys: pts3d, conf, desc
        """
        B = feat1.shape[0]
        H1, W1 = shape1
        H2, W2 = shape2

        # Normalize encoder outputs
        feat1 = mx.fast.layer_norm(feat1, self.enc_norm_weight, self.enc_norm_bias, eps=1e-6)
        feat2 = mx.fast.layer_norm(feat2, self.enc_norm_weight, self.enc_norm_bias, eps=1e-6)

        # Project to decoder dim
        x1 = self.decoder_embed(feat1)
        x2 = self.decoder_embed(feat2)

        # Decoder blocks with cross-attention
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            x1 = blk1(x1, x2)  # View 1 attends to view 2
            x2 = blk2(x2, x1)  # View 2 attends to view 1

        # Final norm
        x1 = mx.fast.layer_norm(x1, self.dec_norm_weight, self.dec_norm_bias, eps=1e-6)
        x2 = mx.fast.layer_norm(x2, self.dec_norm_weight, self.dec_norm_bias, eps=1e-6)

        # DPT heads
        out1 = self.head1([x1], H1, W1)  # [B, H1, W1, 28]
        out2 = self.head2([x2], H2, W2)  # [B, H2, W2, 28]

        # Split outputs
        def split_output(x: mx.array) -> dict:
            return {
                "pts3d": x[..., :3],
                "conf": x[..., 3:4],
                "desc": x[..., 4:],
            }

        return split_output(out1), split_output(out2)


class DuneMast3rEngine:
    """High-level DuneMASt3R pipeline: DUNE encoder + decoder."""

    def __init__(
        self,
        encoder_variant: Literal["small", "base"] = "base",
        resolution: int = 336,
        precision: Literal["fp32", "fp16", "bf16"] = "fp16",
        compile: bool = True,
    ):
        from mlx_mast3r.encoders import DuneEncoder, DuneConfig

        # Encoder
        self.encoder_config = DuneConfig(
            variant=encoder_variant, resolution=resolution, precision=precision
        )
        self.encoder = DuneEncoder(self.encoder_config)

        # Decoder
        if encoder_variant == "base":
            self.decoder_config = DuneMast3rConfig.for_dune_base(precision)
        else:
            self.decoder_config = DuneMast3rConfig.for_dune_small(precision)

        self.decoder = DuneMast3rDecoder(self.decoder_config)

        self._compile = compile
        self._compiled_encoder = None
        self._compiled_decoder = None
        self._loaded = False

    def load(self, encoder_path: str | Path, decoder_path: str | Path) -> None:
        """Load encoder and decoder weights."""
        # Load encoder
        from mlx_mast3r.encoders.dune import DuneEncoderEngine

        enc_engine = DuneEncoderEngine(
            variant=self.encoder_config.variant,
            resolution=self.encoder_config.resolution,
            precision=self.encoder_config.precision,
        )
        enc_engine.load(encoder_path)
        self.encoder = enc_engine.model

        # Load decoder
        self._load_decoder(decoder_path)

        # Compile
        if self._compile:
            self._compiled_encoder = mx.compile(self.encoder.__call__)
            self._compiled_decoder = mx.compile(self.decoder.__call__)

        self._loaded = True

    def _load_decoder(self, path: str | Path) -> None:
        """Load decoder weights from PyTorch checkpoint."""
        import torch

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        state_dict = ckpt["model"]

        weights = {}

        # decoder_embed
        weights["decoder_embed.weight"] = mx.array(
            state_dict["mast3r.decoder_embed.weight"].numpy()
        )
        weights["decoder_embed.bias"] = mx.array(state_dict["mast3r.decoder_embed.bias"].numpy())

        # enc_norm
        weights["enc_norm_weight"] = mx.array(state_dict["mast3r.enc_norm.weight"].numpy())
        weights["enc_norm_bias"] = mx.array(state_dict["mast3r.enc_norm.bias"].numpy())

        # mask_token
        weights["mask_token"] = mx.array(state_dict["mast3r.mask_token"].numpy())

        # dec_norm
        weights["dec_norm_weight"] = mx.array(state_dict["mast3r.dec_norm.weight"].numpy())
        weights["dec_norm_bias"] = mx.array(state_dict["mast3r.dec_norm.bias"].numpy())

        # Decoder blocks
        for i in range(self.decoder_config.decoder_depth):
            prefix_src = f"mast3r.dec_blocks.{i}."
            prefix_dst = f"dec_blocks.{i}."

            # Self-attention
            weights[f"{prefix_dst}norm1_weight"] = mx.array(
                state_dict[f"{prefix_src}norm1.weight"].numpy()
            )
            weights[f"{prefix_dst}norm1_bias"] = mx.array(
                state_dict[f"{prefix_src}norm1.bias"].numpy()
            )
            weights[f"{prefix_dst}self_attn.qkv.weight"] = mx.array(
                state_dict[f"{prefix_src}attn.qkv.weight"].numpy()
            )
            weights[f"{prefix_dst}self_attn.qkv.bias"] = mx.array(
                state_dict[f"{prefix_src}attn.qkv.bias"].numpy()
            )
            weights[f"{prefix_dst}self_attn.proj.weight"] = mx.array(
                state_dict[f"{prefix_src}attn.proj.weight"].numpy()
            )
            weights[f"{prefix_dst}self_attn.proj.bias"] = mx.array(
                state_dict[f"{prefix_src}attn.proj.bias"].numpy()
            )

            # Cross-attention (if present in checkpoint)
            if f"{prefix_src}cross_attn.q.weight" in state_dict:
                weights[f"{prefix_dst}norm2_weight"] = mx.array(
                    state_dict[f"{prefix_src}norm2.weight"].numpy()
                )
                weights[f"{prefix_dst}norm2_bias"] = mx.array(
                    state_dict[f"{prefix_src}norm2.bias"].numpy()
                )
                weights[f"{prefix_dst}cross_attn.q.weight"] = mx.array(
                    state_dict[f"{prefix_src}cross_attn.q.weight"].numpy()
                )
                weights[f"{prefix_dst}cross_attn.q.bias"] = mx.array(
                    state_dict[f"{prefix_src}cross_attn.q.bias"].numpy()
                )
                weights[f"{prefix_dst}cross_attn.kv.weight"] = mx.array(
                    state_dict[f"{prefix_src}cross_attn.kv.weight"].numpy()
                )
                weights[f"{prefix_dst}cross_attn.kv.bias"] = mx.array(
                    state_dict[f"{prefix_src}cross_attn.kv.bias"].numpy()
                )
                weights[f"{prefix_dst}cross_attn.proj.weight"] = mx.array(
                    state_dict[f"{prefix_src}cross_attn.proj.weight"].numpy()
                )
                weights[f"{prefix_dst}cross_attn.proj.bias"] = mx.array(
                    state_dict[f"{prefix_src}cross_attn.proj.bias"].numpy()
                )

            # MLP
            weights[f"{prefix_dst}norm3_weight"] = mx.array(
                state_dict[f"{prefix_src}norm2.weight"].numpy()
            )
            weights[f"{prefix_dst}norm3_bias"] = mx.array(
                state_dict[f"{prefix_src}norm2.bias"].numpy()
            )
            weights[f"{prefix_dst}mlp.fc1.weight"] = mx.array(
                state_dict[f"{prefix_src}mlp.fc1.weight"].numpy()
            )
            weights[f"{prefix_dst}mlp.fc1.bias"] = mx.array(
                state_dict[f"{prefix_src}mlp.fc1.bias"].numpy()
            )
            weights[f"{prefix_dst}mlp.fc2.weight"] = mx.array(
                state_dict[f"{prefix_src}mlp.fc2.weight"].numpy()
            )
            weights[f"{prefix_dst}mlp.fc2.bias"] = mx.array(
                state_dict[f"{prefix_src}mlp.fc2.bias"].numpy()
            )

        # Cast to dtype
        if self.decoder_config.dtype != mx.float32:
            weights = {k: v.astype(self.decoder_config.dtype) for k, v in weights.items()}

        self.decoder.load_weights(list(weights.items()), strict=False)
        mx.eval(self.decoder.parameters())

    def __call__(
        self,
        img1: mx.array,
        img2: mx.array,
    ) -> tuple[dict, dict]:
        """Run full pipeline.

        Args:
            img1: [B, H, W, 3] first view image (NHWC)
            img2: [B, H, W, 3] second view image (NHWC)

        Returns:
            (output1, output2) with pts3d, conf, desc for each view
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Encode
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

        # Preprocess
        x1 = img1.astype(np.float32) / 127.5 - 1.0
        x2 = img2.astype(np.float32) / 127.5 - 1.0
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
