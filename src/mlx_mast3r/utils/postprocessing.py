# Copyright (c) 2024 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""Post-processing utilities for MASt3R/DuneMASt3R decoder outputs.

This module provides functions for normalizing descriptors and applying
depth/confidence transformations matching the original PyTorch implementation.
"""

from __future__ import annotations

import mlx.core as mx


def normalize_descriptors(desc: mx.array, eps: float = 1e-8) -> mx.array:
    """Normalize descriptor vectors to unit length.

    Args:
        desc: [..., D] descriptor tensor.
        eps: Small epsilon for numerical stability.

    Returns:
        [..., D] L2-normalized descriptors.
    """
    return desc / (mx.linalg.norm(desc, axis=-1, keepdims=True) + eps)


def postprocess_pts3d(xyz: mx.array, eps: float = 1e-8) -> mx.array:
    """Apply depth_mode='exp' transformation to 3D points.

    Transform: pts3d = xyz_normalized * expm1(norm(xyz))

    Args:
        xyz: [..., 3] raw 3D point predictions.
        eps: Small epsilon for numerical stability.

    Returns:
        [..., 3] post-processed 3D points.
    """
    d = mx.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_normalized = xyz / mx.maximum(d, mx.array(eps))
    return xyz_normalized * mx.expm1(d)


def postprocess_conf(x: mx.array, vmin: float = 1.0) -> mx.array:
    """Apply conf_mode='exp' transformation to confidence scores.

    Transform: conf = vmin + exp(x)

    Args:
        x: [...] raw confidence predictions.
        vmin: Minimum confidence value.

    Returns:
        [...] post-processed confidence scores.
    """
    return vmin + mx.exp(x)


def postprocess_desc_conf(x: mx.array, vmin: float = 0.0) -> mx.array:
    """Apply desc_conf_mode='exp' transformation to descriptor confidence.

    Transform: desc_conf = vmin + exp(x)

    Args:
        x: [...] raw descriptor confidence predictions.
        vmin: Minimum confidence value.

    Returns:
        [...] post-processed descriptor confidence scores.
    """
    return vmin + mx.exp(x)


def build_output_dict(
    dpt_out: mx.array,
    local_feat: mx.array,
    output_desc_dim: int,
) -> dict[str, mx.array]:
    """Build output dictionary from decoder predictions.

    Args:
        dpt_out: [B, H, W, 4] DPT output (xyz + conf).
        local_feat: [B, H, W, desc_dim + 1] local features (desc + desc_conf).
        output_desc_dim: Dimension of descriptor vectors.

    Returns:
        Dictionary with pts3d, conf, desc, desc_conf.
    """
    # Split local features
    desc = local_feat[..., :output_desc_dim]
    desc_conf = local_feat[..., output_desc_dim:]

    # Apply transformations
    return {
        "pts3d": postprocess_pts3d(dpt_out[..., :3]),
        "conf": postprocess_conf(dpt_out[..., 3:4]),
        "desc": normalize_descriptors(desc),
        "desc_conf": postprocess_desc_conf(desc_conf),
    }
