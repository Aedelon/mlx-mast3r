# Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""Focal length estimation from depth maps.

Implements robust focal estimation using median voting.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def estimate_focal_from_depth(
    pts3d: mx.array,
    pp: mx.array,
    min_focal: float = 0.5,
    max_focal: float = 3.5,
) -> mx.array:
    """Estimate focal length from depth map.

    Matches PyTorch MASt3R/DUSt3R estimate_focal_knowing_depth function.
    Uses median voting for robust focal estimation.

    Args:
        pts3d: Pointmap [H, W, 3]
        pp: Principal point [2]
        min_focal: Minimum focal (relative to size)
        max_focal: Maximum focal (relative to size)

    Returns:
        Estimated focal length
    """
    H, W = pts3d.shape[:2]
    size = max(H, W)

    # Get valid points (positive depth, non-zero X and Y)
    z = pts3d[..., 2]
    x_3d = pts3d[..., 0]
    y_3d = pts3d[..., 1]

    # Create pixel coordinates centered at principal point
    # Match PyTorch: pixels = xy_grid(W, H) - pp
    yy, xx = mx.meshgrid(mx.arange(H), mx.arange(W), indexing="ij")
    u = xx.astype(mx.float32) - pp[0]  # Centered x pixel coord
    v = yy.astype(mx.float32) - pp[1]  # Centered y pixel coord

    # Compute focal votes like PyTorch: fx = (u * z) / x, fy = (v * z) / y
    # Avoid division by zero
    eps = 1e-8
    fx_votes = (u * z) / (x_3d + mx.sign(x_3d) * eps + (mx.abs(x_3d) < eps) * eps)
    fy_votes = (v * z) / (y_3d + mx.sign(y_3d) * eps + (mx.abs(y_3d) < eps) * eps)

    # Valid mask: positive depth and non-small X, Y
    valid = (z > 0.1) & (mx.abs(x_3d) > eps) & (mx.abs(y_3d) > eps)

    # Combine fx and fy votes, take median
    focal_np_fx = np.array(fx_votes).flatten()
    focal_np_fy = np.array(fy_votes).flatten()
    valid_np = np.array(valid).flatten()

    # Filter valid and finite values
    f_votes = np.concatenate([focal_np_fx[valid_np], focal_np_fy[valid_np]])
    f_votes = f_votes[np.isfinite(f_votes)]

    if len(f_votes) < 10:
        return mx.array(size * 1.0)

    # Median as robust estimate (matches PyTorch nanmedian)
    focal = mx.array(np.nanmedian(f_votes))

    # Clamp to reasonable range
    focal = mx.clip(focal, min_focal * size, max_focal * size)

    return focal
