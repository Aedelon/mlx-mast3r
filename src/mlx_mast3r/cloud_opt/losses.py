"""Loss functions for global alignment optimization.

Implements gamma loss and other loss functions used in sparse_ga.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx


def l1_loss(x: mx.array, y: mx.array, weight: mx.array | None = None) -> mx.array:
    """L1 loss (mean absolute error).

    Args:
        x: Predicted values
        y: Target values
        weight: Optional per-element weights

    Returns:
        Scalar loss value
    """
    diff = mx.abs(x - y)
    if weight is not None:
        diff = diff * weight
    return diff.mean()


def l2_loss(x: mx.array, y: mx.array, weight: mx.array | None = None) -> mx.array:
    """L2 loss (mean squared error).

    Args:
        x: Predicted values
        y: Target values
        weight: Optional per-element weights

    Returns:
        Scalar loss value
    """
    diff = (x - y) ** 2
    if weight is not None:
        diff = diff * weight
    return diff.mean()


def l05_loss(x: mx.array, y: mx.array, weight: mx.array | None = None) -> mx.array:
    """L0.5 loss (square root of L2).

    More robust to outliers than L2.

    Args:
        x: Predicted values
        y: Target values
        weight: Optional per-element weights

    Returns:
        Scalar loss value
    """
    diff = mx.sqrt((x - y) ** 2 + 1e-8)  # Small epsilon for numerical stability
    if weight is not None:
        diff = diff * weight
    return diff.mean()


def gamma_loss(gamma: float) -> Callable:
    """Create a generalized Huber loss function.

    The gamma loss interpolates between different loss behaviors:
    - gamma > 1: More L2-like (sensitive to large errors)
    - gamma = 1: L1 loss
    - gamma < 1: More L0.5-like (robust to outliers)

    Formula: ((||x-y|| + offset)^gamma - offset^gamma)
    where offset = (1/gamma)^(1/(gamma-1))

    Args:
        gamma: Loss shape parameter

    Returns:
        Loss function callable
    """
    # Compute offset for smooth transition at zero
    if abs(gamma - 1.0) < 1e-6:
        # gamma = 1 is L1 loss
        offset = 0.0
    else:
        offset = (1.0 / gamma) ** (1.0 / (gamma - 1.0))

    def loss_fn(
        x: mx.array,
        y: mx.array,
        weight: mx.array | None = None,
    ) -> mx.array:
        """Compute gamma loss between x and y.

        Matches PyTorch MASt3R pix_loss behavior:
        - Returns per-point loss if weight is None (for external weighting)
        - Returns weighted mean if weight is provided

        Args:
            x: Predicted values [N, D]
            y: Target values [N, D]
            weight: Optional per-element weights [N]
                   If None, returns per-point losses [N]
                   If provided, returns weighted mean (scalar)

        Returns:
            Per-point losses [N] if weight is None, else scalar
        """
        # L2 distance per point (no epsilon, matches PyTorch torch.linalg.norm)
        l2 = mx.sqrt(mx.sum((x - y) ** 2, axis=-1))

        # Apply gamma transformation
        loss = (l2 + offset) ** gamma - offset**gamma

        if weight is not None:
            # Weighted mean (original behavior)
            loss = loss * weight.reshape(loss.shape)
            return loss.mean()
        else:
            # Return per-point losses for external weighting
            # (Matches PyTorch pix_loss behavior)
            return loss

    return loss_fn


def meta_gamma_loss(alpha: float = 1.0) -> Callable:
    """Create gamma loss with a default alpha value.

    Convenience wrapper around gamma_loss.

    Args:
        alpha: Gamma parameter value

    Returns:
        Loss function callable
    """
    return gamma_loss(alpha)


# Pre-configured loss functions for common use cases
loss_coarse = gamma_loss(1.5)  # Phase 1: more L2-like, fast convergence
loss_fine = gamma_loss(0.5)  # Phase 2: more robust to outliers


def confidence_weighted_loss(
    loss_fn: Callable,
    pts3d_1: mx.array,
    pts3d_2: mx.array,
    conf_1: mx.array,
    conf_2: mx.array,
) -> mx.array:
    """Apply loss with confidence weighting.

    Args:
        loss_fn: Base loss function
        pts3d_1: Points from view 1 [N, 3]
        pts3d_2: Points from view 2 [N, 3]
        conf_1: Confidence for view 1 [N]
        conf_2: Confidence for view 2 [N]

    Returns:
        Weighted loss value
    """
    # Combine confidences (geometric mean)
    weight = mx.sqrt(conf_1 * conf_2)

    # Normalize weights
    weight = weight / (weight.sum() + 1e-8)

    return loss_fn(pts3d_1, pts3d_2, weight)


def reprojection_loss(
    pts3d: mx.array,
    pixels_gt: mx.array,
    K: mx.array,
    weight: mx.array | None = None,
) -> mx.array:
    """Compute 2D reprojection loss.

    Args:
        pts3d: 3D points in camera frame [N, 3]
        pixels_gt: Ground truth 2D pixels [N, 2]
        K: Camera intrinsics [3, 3]
        weight: Optional per-point weights [N]

    Returns:
        Scalar reprojection loss
    """
    # Project 3D points to 2D
    pts3d_h = pts3d  # [N, 3]
    z = pts3d_h[:, 2:3]  # [N, 1]

    # Avoid division by zero
    z = mx.maximum(z, mx.array(1e-6))

    # Project to normalized coordinates
    pts2d_norm = pts3d_h[:, :2] / z  # [N, 2]

    # Apply intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    pixels_pred = mx.stack(
        [
            pts2d_norm[:, 0] * fx + cx,
            pts2d_norm[:, 1] * fy + cy,
        ],
        axis=-1,
    )

    # Compute L2 distance
    diff = mx.sqrt(mx.sum((pixels_pred - pixels_gt) ** 2, axis=-1) + 1e-8)

    if weight is not None:
        diff = diff * weight

    return diff.mean()
