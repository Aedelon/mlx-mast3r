"""Focal length estimation utilities.

Adapted from dust3r/post_process.py for NumPy.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
Original dust3r code: Copyright (C) 2024-present Naver Corporation. CC BY-NC-SA 4.0.
"""

from __future__ import annotations

import numpy as np

from .geometry import xy_grid


def estimate_focal(
    pts3d: np.ndarray,
    pp: tuple[float, float] | None = None,
    method: str = "weiszfeld",
    min_focal_ratio: float = 0.5,
    max_focal_ratio: float = 3.5,
) -> float:
    """Estimate focal length from 3D points.

    Uses reprojection method when absolute depth is known.

    Args:
        pts3d: 3D points of shape (H, W, 3).
        pp: Principal point (cx, cy). If None, use image center.
        method: Estimation method ('median' or 'weiszfeld').
        min_focal_ratio: Minimum focal as ratio of image size.
        max_focal_ratio: Maximum focal as ratio of image size.

    Returns:
        Estimated focal length.
    """
    H, W = pts3d.shape[:2]

    if pp is None:
        pp = (W / 2, H / 2)

    # Centered pixel grid
    grid = xy_grid(W, H)  # (H, W, 2)
    pixels = grid - np.array(pp)  # (H, W, 2)
    pixels = pixels.reshape(-1, 2)  # (HW, 2)
    pts = pts3d.reshape(-1, 3)  # (HW, 3)

    # Filter valid points (finite and non-zero depth)
    valid = np.isfinite(pts).all(axis=-1) & (pts[:, 2] > 1e-6)
    pixels = pixels[valid]
    pts = pts[valid]

    if len(pts) < 10:
        # Fallback to default focal
        return max(H, W)

    if method == "median":
        # Direct estimation of focal
        u, v = pixels[:, 0], pixels[:, 1]
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        fx_votes = (u * z) / (x + 1e-8)
        fy_votes = (v * z) / (y + 1e-8)

        # Filter outliers
        fx_votes = fx_votes[np.isfinite(fx_votes)]
        fy_votes = fy_votes[np.isfinite(fy_votes)]

        f_votes = np.concatenate([fx_votes, fy_votes])
        focal = np.nanmedian(f_votes)

    elif method == "weiszfeld":
        # Weiszfeld iterative estimation
        # f = argmin Sum | pixel - f * (x,y)/z |

        xy_over_z = pts[:, :2] / (pts[:, 2:3] + 1e-8)

        # Init with closed form L2 solution
        dot_xy_px = (xy_over_z * pixels).sum(axis=-1)
        dot_xy_xy = (xy_over_z**2).sum(axis=-1)

        focal = dot_xy_px.mean() / (dot_xy_xy.mean() + 1e-8)

        # Iterative re-weighted least squares
        for _ in range(10):
            dis = np.linalg.norm(pixels - focal * xy_over_z, axis=-1)
            w = 1.0 / np.clip(dis, 1e-8, None)
            focal = (w * dot_xy_px).mean() / ((w * dot_xy_xy).mean() + 1e-8)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Clamp to reasonable range
    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # ~= max(H,W) / 1.1547
    focal = np.clip(focal, min_focal_ratio * focal_base, max_focal_ratio * focal_base)

    return float(focal)


def make_intrinsics(focal: float, pp: tuple[float, float], H: int, W: int) -> np.ndarray:
    """Create intrinsic matrix.

    Args:
        focal: Focal length.
        pp: Principal point (cx, cy).
        H: Image height.
        W: Image width.

    Returns:
        3x3 intrinsic matrix.
    """
    K = np.array([[focal, 0, pp[0]], [0, focal, pp[1]], [0, 0, 1]], dtype=np.float32)
    return K
