# Copyright (c) 2024 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""Weight loading utilities for MASt3R decoder.

This module provides modular functions for loading PyTorch checkpoint weights
into MLX decoder models, handling the necessary tensor transpositions and
key remapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from safetensors import SafetensorError


def transpose_conv_weight(w: np.ndarray) -> np.ndarray:
    """Transpose conv weight from PyTorch (O,I,H,W) to MLX (O,H,W,I)."""
    return np.transpose(w, (0, 2, 3, 1))


def transpose_conv_transpose_weight(w: np.ndarray) -> np.ndarray:
    """Transpose ConvTranspose weight from PyTorch (I,O,H,W) to MLX (O,H,W,I)."""
    return np.transpose(w, (1, 2, 3, 0))


def load_basic_params(
    f: Any,
    keys: list[str],
    weights: dict[str, mx.array],
    prefix: str = "",
) -> None:
    """Load basic decoder parameters (embed, norms).

    Args:
        f: SafeTensors file handle.
        keys: List of available keys in the file.
        weights: Dictionary to populate with loaded weights.
        prefix: Optional prefix for source keys (e.g., "mast3r." for DUNE).
    """
    # decoder_embed
    if f"{prefix}decoder_embed.weight" in keys:
        weights["decoder_embed.weight"] = mx.array(f.get_tensor(f"{prefix}decoder_embed.weight"))
        weights["decoder_embed.bias"] = mx.array(f.get_tensor(f"{prefix}decoder_embed.bias"))

    # enc_norm
    if f"{prefix}enc_norm.weight" in keys:
        weights["enc_norm_weight"] = mx.array(f.get_tensor(f"{prefix}enc_norm.weight"))
        weights["enc_norm_bias"] = mx.array(f.get_tensor(f"{prefix}enc_norm.bias"))

    # dec_norm
    if f"{prefix}dec_norm.weight" in keys:
        weights["dec_norm_weight"] = mx.array(f.get_tensor(f"{prefix}dec_norm.weight"))
        weights["dec_norm_bias"] = mx.array(f.get_tensor(f"{prefix}dec_norm.bias"))

    # mask_token (DUNE specific)
    if f"{prefix}mask_token" in keys:
        weights["mask_token"] = mx.array(f.get_tensor(f"{prefix}mask_token"))


def load_decoder_block(
    f: Any,
    keys: list[str],
    weights: dict[str, mx.array],
    block_name: str,
    dst_prefix: str,
) -> None:
    """Load a single decoder block's weights.

    Args:
        f: SafeTensors file handle.
        keys: List of available keys in the file.
        weights: Dictionary to populate with loaded weights.
        block_name: Source block name (e.g., "dec_blocks.0").
        dst_prefix: Destination prefix (e.g., "dec_blocks.0.").
    """
    src_prefix = f"{block_name}."

    if f"{src_prefix}norm1.weight" not in keys:
        return

    # Self-attention norms
    weights[f"{dst_prefix}norm1_weight"] = mx.array(f.get_tensor(f"{src_prefix}norm1.weight"))
    weights[f"{dst_prefix}norm1_bias"] = mx.array(f.get_tensor(f"{src_prefix}norm1.bias"))

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
    if f"{src_prefix}norm2.weight" in keys:
        weights[f"{dst_prefix}norm2_weight"] = mx.array(f.get_tensor(f"{src_prefix}norm2.weight"))
        weights[f"{dst_prefix}norm2_bias"] = mx.array(f.get_tensor(f"{src_prefix}norm2.bias"))

    if f"{src_prefix}norm_y.weight" in keys:
        weights[f"{dst_prefix}norm_y_weight"] = mx.array(f.get_tensor(f"{src_prefix}norm_y.weight"))
        weights[f"{dst_prefix}norm_y_bias"] = mx.array(f.get_tensor(f"{src_prefix}norm_y.bias"))

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
        weights[f"{dst_prefix}cross_attn.kv.weight"] = mx.array(
            np.concatenate([k_weight, v_weight], axis=0)
        )

        k_bias = f.get_tensor(f"{src_prefix}cross_attn.projk.bias")
        v_bias = f.get_tensor(f"{src_prefix}cross_attn.projv.bias")
        weights[f"{dst_prefix}cross_attn.kv.bias"] = mx.array(
            np.concatenate([k_bias, v_bias], axis=0)
        )

        # Output projection
        weights[f"{dst_prefix}cross_attn.proj.weight"] = mx.array(
            f.get_tensor(f"{src_prefix}cross_attn.proj.weight")
        )
        weights[f"{dst_prefix}cross_attn.proj.bias"] = mx.array(
            f.get_tensor(f"{src_prefix}cross_attn.proj.bias")
        )

    # MLP
    if f"{src_prefix}norm3.weight" in keys:
        weights[f"{dst_prefix}norm3_weight"] = mx.array(f.get_tensor(f"{src_prefix}norm3.weight"))
        weights[f"{dst_prefix}norm3_bias"] = mx.array(f.get_tensor(f"{src_prefix}norm3.bias"))

    weights[f"{dst_prefix}mlp.fc1.weight"] = mx.array(f.get_tensor(f"{src_prefix}mlp.fc1.weight"))
    weights[f"{dst_prefix}mlp.fc1.bias"] = mx.array(f.get_tensor(f"{src_prefix}mlp.fc1.bias"))
    weights[f"{dst_prefix}mlp.fc2.weight"] = mx.array(f.get_tensor(f"{src_prefix}mlp.fc2.weight"))
    weights[f"{dst_prefix}mlp.fc2.bias"] = mx.array(f.get_tensor(f"{src_prefix}mlp.fc2.bias"))


def load_all_decoder_blocks(
    f: Any,
    keys: list[str],
    weights: dict[str, mx.array],
    decoder_depth: int,
    prefix: str = "",
) -> None:
    """Load all decoder blocks (dec_blocks and dec_blocks2).

    Args:
        f: SafeTensors file handle.
        keys: List of available keys in the file.
        weights: Dictionary to populate with loaded weights.
        decoder_depth: Number of decoder layers (typically 12).
        prefix: Optional prefix for source keys (e.g., "mast3r." for DUNE).
    """
    for i in range(decoder_depth):
        load_decoder_block(f, keys, weights, f"{prefix}dec_blocks.{i}", f"dec_blocks.{i}.")
        load_decoder_block(f, keys, weights, f"{prefix}dec_blocks2.{i}", f"dec_blocks2.{i}.")


def load_dpt_head(
    f: Any,
    keys: list[str],
    weights: dict[str, mx.array],
    src_head: str,
    dst_head: str,
) -> None:
    """Load DPT head weights.

    Args:
        f: SafeTensors file handle.
        keys: List of available keys in the file.
        weights: Dictionary to populate with loaded weights.
        src_head: Source head name (e.g., "downstream_head1").
        dst_head: Destination head name (e.g., "head1").
    """
    # act_postprocess layer 0: Conv + ConvTranspose (upsample 4x)
    if f"{src_head}.dpt.act_postprocess.0.0.weight" in keys:
        weights[f"{dst_head}.act_postprocess_0_conv.weight"] = mx.array(
            transpose_conv_weight(f.get_tensor(f"{src_head}.dpt.act_postprocess.0.0.weight"))
        )
        weights[f"{dst_head}.act_postprocess_0_conv.bias"] = mx.array(
            f.get_tensor(f"{src_head}.dpt.act_postprocess.0.0.bias")
        )
        weights[f"{dst_head}.act_postprocess_0_up.weight"] = mx.array(
            transpose_conv_transpose_weight(
                f.get_tensor(f"{src_head}.dpt.act_postprocess.0.1.weight")
            )
        )
        weights[f"{dst_head}.act_postprocess_0_up.bias"] = mx.array(
            f.get_tensor(f"{src_head}.dpt.act_postprocess.0.1.bias")
        )

    # act_postprocess layer 1: Conv + ConvTranspose (upsample 2x)
    if f"{src_head}.dpt.act_postprocess.1.0.weight" in keys:
        weights[f"{dst_head}.act_postprocess_1_conv.weight"] = mx.array(
            transpose_conv_weight(f.get_tensor(f"{src_head}.dpt.act_postprocess.1.0.weight"))
        )
        weights[f"{dst_head}.act_postprocess_1_conv.bias"] = mx.array(
            f.get_tensor(f"{src_head}.dpt.act_postprocess.1.0.bias")
        )
        weights[f"{dst_head}.act_postprocess_1_up.weight"] = mx.array(
            transpose_conv_transpose_weight(
                f.get_tensor(f"{src_head}.dpt.act_postprocess.1.1.weight")
            )
        )
        weights[f"{dst_head}.act_postprocess_1_up.bias"] = mx.array(
            f.get_tensor(f"{src_head}.dpt.act_postprocess.1.1.bias")
        )

    # act_postprocess layer 2: Conv only
    if f"{src_head}.dpt.act_postprocess.2.0.weight" in keys:
        weights[f"{dst_head}.act_postprocess_2_conv.weight"] = mx.array(
            transpose_conv_weight(f.get_tensor(f"{src_head}.dpt.act_postprocess.2.0.weight"))
        )
        weights[f"{dst_head}.act_postprocess_2_conv.bias"] = mx.array(
            f.get_tensor(f"{src_head}.dpt.act_postprocess.2.0.bias")
        )

    # act_postprocess layer 3: Conv + Conv (downsample 2x)
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


def load_local_features(
    f: Any,
    keys: list[str],
    weights: dict[str, mx.array],
    prefix: str = "",
) -> None:
    """Load local features MLP weights (for descriptors).

    Args:
        f: SafeTensors file handle.
        keys: List of available keys in the file.
        weights: Dictionary to populate with loaded weights.
        prefix: Optional prefix for source keys (e.g., "mast3r." for DUNE).
    """
    for head_idx in [1, 2]:
        src = f"{prefix}downstream_head{head_idx}.head_local_features"
        dst = f"head_local_features{head_idx}"

        # fc1 -> layers.0, fc2 -> layers.2
        if f"{src}.fc1.weight" in keys:
            weights[f"{dst}.layers.0.weight"] = mx.array(f.get_tensor(f"{src}.fc1.weight"))
            weights[f"{dst}.layers.0.bias"] = mx.array(f.get_tensor(f"{src}.fc1.bias"))
            weights[f"{dst}.layers.2.weight"] = mx.array(f.get_tensor(f"{src}.fc2.weight"))
            weights[f"{dst}.layers.2.bias"] = mx.array(f.get_tensor(f"{src}.fc2.bias"))
