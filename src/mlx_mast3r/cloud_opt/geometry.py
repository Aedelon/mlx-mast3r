"""Geometry utilities for cloud optimization.

Adapted from dust3r/utils/geometry.py for NumPy and MLX.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
Original dust3r code: Copyright (C) 2024-present Naver Corporation. CC BY-NC-SA 4.0.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def xy_grid(W: int, H: int) -> np.ndarray:
    """Create a grid of (x, y) pixel coordinates.

    Args:
        W: Width of the grid.
        H: Height of the grid.

    Returns:
        Array of shape (H, W, 2) containing (x, y) coordinates.
    """
    x = np.arange(W, dtype=np.float32)
    y = np.arange(H, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx, yy], axis=-1)


def inv(mat: np.ndarray) -> np.ndarray:
    """Invert a 4x4 transformation matrix.

    Args:
        mat: 4x4 transformation matrix or batch of matrices.

    Returns:
        Inverted matrix.
    """
    if mat.ndim == 2:
        return np.linalg.inv(mat)
    else:
        return np.linalg.inv(mat)


def geotrf(
    Trf: np.ndarray,
    pts: np.ndarray,
    ncol: int | None = None,
    norm: bool = False,
) -> np.ndarray:
    """Apply a geometric transformation to points.

    Args:
        Trf: Transformation matrix (4x4 or 3x3).
        pts: Points to transform (..., 3) or (..., 2).
        ncol: Number of output columns (2 or 3). If None, same as input.
        norm: If True, normalize by last coordinate.

    Returns:
        Transformed points.
    """
    assert Trf.ndim >= 2
    assert pts.ndim >= 2

    # Determine if we're doing 3D or 2D transform
    if Trf.shape[-1] == 4:
        # 4x4 matrix - 3D transform
        if pts.shape[-1] == 3:
            # Homogeneous coords
            pts_h = np.concatenate([pts, np.ones((*pts.shape[:-1], 1), dtype=pts.dtype)], axis=-1)
            res = pts_h @ Trf.T
            if ncol == 3 or ncol is None:
                res = res[..., :3]
            elif ncol == 2:
                if norm:
                    res = res[..., :2] / res[..., 2:3]
                else:
                    res = res[..., :2]
        else:
            raise ValueError(f"Unexpected pts shape {pts.shape}")
    elif Trf.shape[-1] == 3:
        # 3x3 matrix - 2D transform or projection
        if pts.shape[-1] == 3:
            res = pts @ Trf.T
            if norm:
                res = res[..., :2] / res[..., 2:3]
        elif pts.shape[-1] == 2:
            pts_h = np.concatenate([pts, np.ones((*pts.shape[:-1], 1), dtype=pts.dtype)], axis=-1)
            res = pts_h @ Trf.T
            if norm:
                res = res[..., :2] / res[..., 2:3]
            else:
                res = res[..., :2]
        else:
            raise ValueError(f"Unexpected pts shape {pts.shape}")
    else:
        raise ValueError(f"Unexpected Trf shape {Trf.shape}")

    return res


def depthmap_to_pts3d(
    depth: np.ndarray,
    K: np.ndarray,
    cam2world: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a depth map to 3D points.

    Args:
        depth: Depth map of shape (H, W).
        K: Intrinsic matrix (3x3).
        cam2world: Optional camera-to-world transformation (4x4).

    Returns:
        3D points of shape (H, W, 3).
    """
    H, W = depth.shape
    grid = xy_grid(W, H)  # (H, W, 2)

    # Unproject to camera coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (grid[..., 0] - cx) / fx * depth
    y = (grid[..., 1] - cy) / fy * depth
    z = depth

    pts3d = np.stack([x, y, z], axis=-1)

    # Transform to world coordinates if cam2world is provided
    if cam2world is not None:
        pts3d = geotrf(cam2world, pts3d)

    return pts3d


def rodrigues_to_rotation(rvec: np.ndarray) -> np.ndarray:
    """Convert Rodrigues vector to rotation matrix.

    Args:
        rvec: Rodrigues vector (3,).

    Returns:
        Rotation matrix (3, 3).
    """
    theta = np.linalg.norm(rvec)
    if theta < 1e-8:
        return np.eye(3, dtype=rvec.dtype)

    k = rvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=rvec.dtype)

    R = np.eye(3, dtype=rvec.dtype) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def rotation_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to Rodrigues vector.

    Args:
        R: Rotation matrix (3, 3).

    Returns:
        Rodrigues vector (3,).
    """
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-8:
        return np.zeros(3, dtype=R.dtype)

    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=R.dtype) / (
        2 * np.sin(theta)
    )

    return k * theta


# =============================================================================
# MLX Native Versions
# =============================================================================


def xy_grid_mlx(W: int, H: int) -> mx.array:
    """Create a grid of (x, y) pixel coordinates (MLX version).

    Args:
        W: Width of the grid.
        H: Height of the grid.

    Returns:
        Array of shape (H, W, 2) containing (x, y) coordinates.
    """
    x = mx.arange(W, dtype=mx.float32)
    y = mx.arange(H, dtype=mx.float32)
    # meshgrid with indexing='xy' means xx varies along columns, yy along rows
    yy, xx = mx.meshgrid(y, x, indexing="ij")
    return mx.stack([xx, yy], axis=-1)


def geotrf_mlx(
    Trf: mx.array,
    pts: mx.array,
    ncol: int | None = None,
    norm: bool = False,
) -> mx.array:
    """Apply a geometric transformation to points (MLX version).

    Args:
        Trf: Transformation matrix (4x4 or 3x3).
        pts: Points to transform (..., 3) or (..., 2).
        ncol: Number of output columns (2 or 3). If None, same as input.
        norm: If True, normalize by last coordinate.

    Returns:
        Transformed points.
    """
    assert Trf.ndim >= 2
    assert pts.ndim >= 2

    # Determine if we're doing 3D or 2D transform
    if Trf.shape[-1] == 4:
        # 4x4 matrix - 3D transform
        if pts.shape[-1] == 3:
            # Homogeneous coords
            ones = mx.ones((*pts.shape[:-1], 1), dtype=pts.dtype)
            pts_h = mx.concatenate([pts, ones], axis=-1)
            res = pts_h @ Trf.T
            if ncol == 3 or ncol is None:
                res = res[..., :3]
            elif ncol == 2:
                if norm:
                    res = res[..., :2] / res[..., 2:3]
                else:
                    res = res[..., :2]
        else:
            raise ValueError(f"Unexpected pts shape {pts.shape}")
    elif Trf.shape[-1] == 3:
        # 3x3 matrix - 2D transform or projection
        if pts.shape[-1] == 3:
            res = pts @ Trf.T
            if norm:
                res = res[..., :2] / res[..., 2:3]
        elif pts.shape[-1] == 2:
            ones = mx.ones((*pts.shape[:-1], 1), dtype=pts.dtype)
            pts_h = mx.concatenate([pts, ones], axis=-1)
            res = pts_h @ Trf.T
            if norm:
                res = res[..., :2] / res[..., 2:3]
            else:
                res = res[..., :2]
        else:
            raise ValueError(f"Unexpected pts shape {pts.shape}")
    else:
        raise ValueError(f"Unexpected Trf shape {Trf.shape}")

    return res


def depthmap_to_pts3d_mlx(
    depth: mx.array,
    K: mx.array,
    cam2world: mx.array | None = None,
) -> mx.array:
    """Convert a depth map to 3D points (MLX version).

    Args:
        depth: Depth map of shape (H, W).
        K: Intrinsic matrix (3x3).
        cam2world: Optional camera-to-world transformation (4x4).

    Returns:
        3D points of shape (H, W, 3).
    """
    H, W = depth.shape
    grid = xy_grid_mlx(W, H)  # (H, W, 2)

    # Unproject to camera coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (grid[..., 0] - cx) / fx * depth
    y = (grid[..., 1] - cy) / fy * depth
    z = depth

    pts3d = mx.stack([x, y, z], axis=-1)

    # Transform to world coordinates if cam2world is provided
    if cam2world is not None:
        pts3d = geotrf_mlx(cam2world, pts3d)

    return pts3d


def inv_mlx(mat: mx.array) -> mx.array:
    """Invert a 4x4 transformation matrix (MLX version).

    Args:
        mat: 4x4 transformation matrix or batch of matrices.

    Returns:
        Inverted matrix.
    """
    # mx.linalg.inv is not yet supported on GPU, use CPU stream
    return mx.linalg.inv(mat, stream=mx.cpu)
