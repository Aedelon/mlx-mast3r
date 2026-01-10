"""DuneMASt3R Decoder - Ultra-optimized MLX implementation.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.

DuneMASt3R = DUNE Encoder (separate) + MASt3R Decoder (this file)

Architecture:
- decoder_embed: Project DUNE features (768/384) to decoder space (768)
- dec_blocks: 12 transformer decoder blocks (view 1)
- dec_blocks2: 12 transformer decoder blocks (view 2)
- downstream_head1/2: DPT heads for 3D points + descriptors

Key differences from MASt3R:
- encoder_dim = 768 (base) or 384 (small) instead of 1024
- patch_size = 14 instead of 16
- DPT act_postprocess[0] input = encoder_dim (768/384)

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

# Import shared components from mast3r decoder
from mlx_mast3r.decoders.mast3r import (
    DecoderBlock,
    DecoderCrossAttention,
    DecoderSelfAttention,
    DPTHead,
    FeatureFusionBlock,
    ResidualConvUnit,
    apply_rope_2d,
    bilinear_upsample_2x,
    nearest_upsample_2x,
    precompute_rope_2d,
)


@dataclass
class DuneMast3rDecoderConfig:
    """DuneMASt3R decoder configuration."""

    encoder_dim: int = 768  # DUNE Base output dim (768) or Small (384)
    decoder_dim: int = 768
    num_heads: int = 12
    head_dim: int = 64
    mlp_ratio: float = 4.0
    decoder_depth: int = 12
    patch_size: int = 14  # DUNE uses 14x14 patches
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
    def for_dune_base(cls, precision: str = "fp16") -> "DuneMast3rDecoderConfig":
        """Config for DUNE Base encoder (768 dim)."""
        return cls(encoder_dim=768, decoder_dim=768, precision=precision)

    @classmethod
    def for_dune_small(cls, precision: str = "fp16") -> "DuneMast3rDecoderConfig":
        """Config for DUNE Small encoder (384 dim)."""
        return cls(encoder_dim=384, decoder_dim=768, precision=precision)


class DuneDPTHead(nn.Module):
    """DPT Head adapted for DUNE encoder dimensions.

    Same architecture as MASt3R DPT but with encoder_dim=768/384 instead of 1024.
    """

    def __init__(
        self,
        encoder_dim: int = 768,
        decoder_dim: int = 768,
        feature_dim: int = 256,
        num_channels: int = 4,  # pts3d (3) + conf (1)
        hooks: list[int] | None = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_channels = num_channels
        self.hooks = hooks or [0, 6, 9, 12]

        # Input dimensions: hook 0 = encoder (enc_dim), hooks 1-3 = decoder (dec_dim)
        in_dims = [encoder_dim, decoder_dim, decoder_dim, decoder_dim]

        # Layer dims after act_postprocess (same as MASt3R)
        layer_dims = [96, 192, 384, 768]

        # act_postprocess: adapt dimensions for each hooked layer
        # Layer 0: Conv1x1 (encoder_dim->96) + ConvTranspose 4x4 stride 4
        self.act_postprocess_0_conv = nn.Conv2d(in_dims[0], layer_dims[0], kernel_size=1)
        self.act_postprocess_0_up = nn.ConvTranspose2d(
            layer_dims[0], layer_dims[0], kernel_size=4, stride=4, padding=0
        )

        # Layer 1: Conv1x1 (768->192) + ConvTranspose 2x2 stride 2
        self.act_postprocess_1_conv = nn.Conv2d(in_dims[1], layer_dims[1], kernel_size=1)
        self.act_postprocess_1_up = nn.ConvTranspose2d(
            layer_dims[1], layer_dims[1], kernel_size=2, stride=2, padding=0
        )

        # Layer 2: Conv1x1 (768->384) only
        self.act_postprocess_2_conv = nn.Conv2d(in_dims[2], layer_dims[2], kernel_size=1)

        # Layer 3: Conv1x1 (768->768) + Conv 3x3 stride 2 (downsample)
        self.act_postprocess_3_conv = nn.Conv2d(in_dims[3], layer_dims[3], kernel_size=1)
        self.act_postprocess_3_down = nn.Conv2d(
            layer_dims[3], layer_dims[3], kernel_size=3, stride=2, padding=1
        )

        # scratch.layer_rn: project to feature_dim (256)
        self.layer1_rn = nn.Conv2d(layer_dims[0], feature_dim, kernel_size=3, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(layer_dims[1], feature_dim, kernel_size=3, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(layer_dims[2], feature_dim, kernel_size=3, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(layer_dims[3], feature_dim, kernel_size=3, padding=1, bias=False)

        # scratch.refinenet: multi-scale fusion
        self.refinenet4 = FeatureFusionBlock(feature_dim)
        self.refinenet3 = FeatureFusionBlock(feature_dim)
        self.refinenet2 = FeatureFusionBlock(feature_dim)
        self.refinenet1 = FeatureFusionBlock(feature_dim)

        # Output head
        last_dim = feature_dim // 2  # 128
        self.head_conv1 = nn.Conv2d(feature_dim, last_dim, kernel_size=3, padding=1)
        self.head_conv2 = nn.Conv2d(last_dim, last_dim, kernel_size=3, padding=1)
        self.head_conv3 = nn.Conv2d(last_dim, num_channels, kernel_size=1)

    def __call__(
        self, features: list[mx.array], H: int, W: int, target_h: int = 0, target_w: int = 0
    ) -> mx.array:
        """Process multi-scale features through DPT.

        Args:
            features: List of [B, N, C] features at hooked layers
            H, W: Patch grid dimensions
            target_h, target_w: Target output size (for resize). If 0, use H*16, W*16.

        Returns:
            [B, target_h, target_w, num_channels] - full resolution output
        """
        B = features[0].shape[0]

        # Reshape features to spatial: [B, N, C] -> [B, H, W, C]
        layers = [f.reshape(B, H, W, -1) for f in features]

        # Apply act_postprocess
        l0 = self.act_postprocess_0_conv(layers[0])
        l0 = self.act_postprocess_0_up(l0)

        l1 = self.act_postprocess_1_conv(layers[1])
        l1 = self.act_postprocess_1_up(l1)

        l2 = self.act_postprocess_2_conv(layers[2])

        l3 = self.act_postprocess_3_conv(layers[3])
        l3 = self.act_postprocess_3_down(l3)

        # Project to feature_dim
        l0 = self.layer1_rn(l0)
        l1 = self.layer2_rn(l1)
        l2 = self.layer3_rn(l2)
        l3 = self.layer4_rn(l3)

        # Multi-scale fusion
        path_4 = self.refinenet4(l3)
        path_4 = path_4[:, : l2.shape[1], : l2.shape[2], :]

        path_3 = self.refinenet3(path_4, l2)
        path_2 = self.refinenet2(path_3, l1)
        path_1 = self.refinenet1(path_2, l0)

        # Output head
        out = self.head_conv1(path_1)
        out = bilinear_upsample_2x(out, align_corners=True)
        out = self.head_conv2(out)
        out = nn.relu(out)
        out = self.head_conv3(out)

        # Resize to target size if specified (for DUNE patch_size=14)
        # DPT produces H*16 x W*16, but DUNE images are H*14 x W*14
        if target_h > 0 and target_w > 0:
            current_h, current_w = out.shape[1], out.shape[2]
            if current_h != target_h or current_w != target_w:
                # Use bilinear interpolation via nn.Upsample
                scale_h = target_h / current_h
                scale_w = target_w / current_w
                upsample = nn.Upsample(scale_factor=(scale_h, scale_w), mode="linear", align_corners=True)
                out = upsample(out)

        return out


class DuneMast3rDecoder(nn.Module):
    """DuneMASt3R Decoder - Asymmetric decoder for stereo 3D reconstruction.

    Takes DUNE encoder features from two views and outputs:
    - 3D points in camera space
    - Confidence maps
    - Dense descriptors for matching

    Architecture follows MASt3R with:
    - Cross-attention between views
    - 2D RoPE positional encoding
    - DPT heads for dense prediction
    """

    def __init__(self, config: DuneMast3rDecoderConfig):
        super().__init__()
        self.config = config

        # Project encoder features to decoder dim
        self.decoder_embed = nn.Linear(config.encoder_dim, config.decoder_dim)

        # Encoder norm (applied to DUNE encoder output)
        self.enc_norm_weight = mx.ones((config.encoder_dim,))
        self.enc_norm_bias = mx.zeros((config.encoder_dim,))

        # Mask token for masked regions
        self.mask_token = mx.zeros((1, 1, config.decoder_dim))

        # Import config type for decoder blocks
        from mlx_mast3r.decoders.mast3r import Mast3rDecoderConfig

        # Create a compatible config for DecoderBlock
        block_config = Mast3rDecoderConfig(
            encoder_dim=config.encoder_dim,
            decoder_dim=config.decoder_dim,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            mlp_ratio=config.mlp_ratio,
            decoder_depth=config.decoder_depth,
            patch_size=config.patch_size,
            precision=config.precision,
        )

        # Decoder blocks for view 1 (with RoPE)
        self.dec_blocks = [
            DecoderBlock(block_config, use_rope=True) for _ in range(config.decoder_depth)
        ]

        # Decoder blocks for view 2 (with RoPE)
        self.dec_blocks2 = [
            DecoderBlock(block_config, use_rope=True) for _ in range(config.decoder_depth)
        ]

        # Final decoder norm
        self.dec_norm_weight = mx.ones((config.decoder_dim,))
        self.dec_norm_bias = mx.zeros((config.decoder_dim,))

        # Output heads: pts3d (3) + confidence (1) = 4 channels
        self.head1 = DuneDPTHead(
            encoder_dim=config.encoder_dim,
            decoder_dim=config.decoder_dim,
            feature_dim=256,
            num_channels=4,
        )
        self.head2 = DuneDPTHead(
            encoder_dim=config.encoder_dim,
            decoder_dim=config.decoder_dim,
            feature_dim=256,
            num_channels=4,
        )

        # Local features MLP (descriptors)
        # Input: enc_dim + dec_dim (768+768=1536 for base, 384+768=1152 for small)
        # Output: (desc_dim + 1) * patch_size^2 = 25 * 196 = 4900
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
            feat1: [B, N1, D] encoder features for view 1 (D=768 or 384)
            feat2: [B, N2, D] encoder features for view 2
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

        # Encoder outputs are already normalized (norm applied in encoder)
        # Project to decoder dim
        x1 = self.decoder_embed(feat1)
        x2 = self.decoder_embed(feat2)

        # Hooks: [0, 6, 9, 12] - collect features at these indices
        # Hook 0 = encoder output (already normalized)
        hooks = [0, 6, 9, 12]
        features1 = [feat1]  # Hook 0: encoder features
        features2 = [feat2]

        # Decoder blocks with cross-attention, collecting hooked outputs
        # IMPORTANT: PyTorch uses OLD x1/x2 for BOTH blocks in each iteration
        for i, (blk1, blk2) in enumerate(zip(self.dec_blocks, self.dec_blocks2)):
            x1_old, x2_old = x1, x2
            x1 = blk1(x1_old, x2_old)  # View 1 attends to view 2 (old)
            x2 = blk2(x2_old, x1_old)  # View 2 attends to view 1 (old)

            # Collect at hooks
            layer_idx = i + 1
            if layer_idx in hooks[1:]:
                features1.append(x1)
                features2.append(x2)

        # Final norm
        x1_norm = mx.fast.layer_norm(x1, self.dec_norm_weight, self.dec_norm_bias, eps=LAYER_NORM_EPS)
        x2_norm = mx.fast.layer_norm(x2, self.dec_norm_weight, self.dec_norm_bias, eps=LAYER_NORM_EPS)

        # Update final layer features with normed version
        if len(features1) == 4:
            features1[-1] = x1_norm
            features2[-1] = x2_norm

        # DPT heads for pts3d + conf
        # Pass target size = H*patch_size x W*patch_size (image resolution)
        ps = self.config.patch_size  # 14
        target_h1, target_w1 = H1 * ps, W1 * ps
        target_h2, target_w2 = H2 * ps, W2 * ps
        dpt_out1 = self.head1(features1, H1, W1, target_h1, target_w1)
        dpt_out2 = self.head2(features2, H2, W2, target_h2, target_w2)

        # Local features via MLP (for descriptors)
        cat1 = mx.concatenate([feat1, x1_norm], axis=-1)
        cat2 = mx.concatenate([feat2, x2_norm], axis=-1)

        # MLP
        local_feat1 = self.head_local_features1(cat1)
        local_feat2 = self.head_local_features2(cat2)

        # Pixel shuffle to get full resolution descriptors
        ps = self.config.patch_size  # 14
        desc_dim = self.config.output_desc_dim + 1  # 25

        # Reshape: [B, H*W, desc_dim*ps*ps] -> [B, H, W, desc_dim, ps, ps]
        local_feat1 = local_feat1.reshape(B, H1, W1, desc_dim, ps, ps)
        local_feat2 = local_feat2.reshape(B, H2, W2, desc_dim, ps, ps)

        # Transpose and reshape for pixel shuffle
        local_feat1 = local_feat1.transpose(0, 1, 4, 2, 5, 3).reshape(B, H1 * ps, W1 * ps, desc_dim)
        local_feat2 = local_feat2.transpose(0, 1, 4, 2, 5, 3).reshape(B, H2 * ps, W2 * ps, desc_dim)

        # Build output dicts with post-processing
        from mlx_mast3r.utils.postprocessing import build_output_dict

        out1 = build_output_dict(dpt_out1, local_feat1, self.config.output_desc_dim)
        out2 = build_output_dict(dpt_out2, local_feat2, self.config.output_desc_dim)

        return out1, out2


class DuneMast3rDecoderEngine:
    """High-level DuneMASt3R decoder engine with weight loading."""

    def __init__(
        self,
        encoder_variant: Literal["small", "base"] = "base",
        resolution: int = 336,
        precision: Literal["fp32", "fp16", "bf16"] = "fp16",
        compile: bool = True,
    ):
        from mlx_mast3r.encoders.dune import DuneConfig, DuneEncoder

        # Encoder config
        self.encoder_config = DuneConfig(
            variant=encoder_variant, resolution=resolution, precision=precision
        )
        self.encoder = DuneEncoder(self.encoder_config)

        # Decoder config
        if encoder_variant == "base":
            self.decoder_config = DuneMast3rDecoderConfig.for_dune_base(precision)
        else:
            self.decoder_config = DuneMast3rDecoderConfig.for_dune_small(precision)

        self.decoder = DuneMast3rDecoder(self.decoder_config)

        self._compile = compile
        self._compiled_encoder = None
        self._compiled_decoder = None
        self._loaded = False

    def load(self, encoder_path: str | Path, decoder_path: str | Path) -> None:
        """Load encoder and decoder weights from safetensors files."""
        # Load encoder
        from mlx_mast3r.encoders.dune import DuneEncoderEngine

        enc_engine = DuneEncoderEngine(
            variant=self.encoder_config.variant,
            resolution=self.encoder_config.resolution,
            precision=self.encoder_config.precision,
            compile=False,
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
        """Load decoder weights from safetensors.

        The decoder.safetensors has keys prefixed with 'mast3r.' like:
        - mast3r.dec_blocks.X.*
        - mast3r.downstream_head1.*
        - mast3r.decoder_embed.*
        """
        from safetensors import safe_open

        from mlx_mast3r.decoders.weight_loader import (
            load_all_decoder_blocks,
            load_basic_params,
            load_dpt_head,
            load_local_features,
        )

        path = Path(path)
        prefix = "mast3r."  # DUNE decoder keys are prefixed

        weights: dict[str, mx.array] = {}
        with safe_open(str(path), framework="numpy") as f:
            keys = list(f.keys())

            # Load all weight groups with prefix
            load_basic_params(f, keys, weights, prefix=prefix)
            load_all_decoder_blocks(f, keys, weights, self.decoder_config.decoder_depth, prefix=prefix)
            load_dpt_head(f, keys, weights, f"{prefix}downstream_head1", "head1")
            load_dpt_head(f, keys, weights, f"{prefix}downstream_head2", "head2")
            load_local_features(f, keys, weights, prefix=prefix)

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
            img2: [B, H, W, 3] second view image

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

        # DUNE preprocessing (same as MASt3R)
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


# Backward compatibility aliases
DuneMast3rConfig = DuneMast3rDecoderConfig
DuneMast3rEngine = DuneMast3rDecoderEngine
