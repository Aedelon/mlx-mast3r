# Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""Visualization utilities for Gradio demo."""

from __future__ import annotations

import numpy as np


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    """Convert depth map to turbo colormap with enhanced contrast.

    Uses histogram equalization for better visualization of depth variations.

    Args:
        depth: [H, W] depth map

    Returns:
        [H, W, 3] RGB colormap image (uint8)
    """
    import matplotlib

    d = depth.copy().astype(np.float32)
    valid = np.isfinite(d) & (d > 0.1)

    if valid.sum() == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    d_log = np.zeros_like(d)
    d_log[valid] = np.log1p(d[valid])

    valid_vals = d_log[valid]
    hist, bin_edges = np.histogram(valid_vals, bins=256)
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]

    d_norm = np.zeros_like(d)
    bin_indices = np.clip(np.digitize(d_log[valid], bin_edges[:-1]) - 1, 0, len(cdf_normalized) - 1)
    d_norm[valid] = cdf_normalized[bin_indices]

    cmap = matplotlib.colormaps.get_cmap("turbo")
    rgb = cmap(d_norm)[:, :, :3]
    rgb[~valid] = 0

    return (rgb * 255).astype(np.uint8)


def conf_to_colormap(conf: np.ndarray) -> np.ndarray:
    """Convert confidence map to colormap.

    Uses a smooth gradient from blue (low) to green (mid) to red (high).

    Args:
        conf: [H, W] or [H, W, 1] confidence map

    Returns:
        [H, W, 3] RGB colormap image (uint8)
    """
    c = conf.squeeze()
    c_min, c_max = c.min(), c.max()
    if c_max > c_min:
        c = (c - c_min) / (c_max - c_min)
    else:
        c = np.ones_like(c)

    r = np.clip(1.5 - np.abs(4 * c - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * c - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * c - 1), 0, 1)

    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
