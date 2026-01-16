"""Scene optimizer for multi-view global alignment.

Implements the core optimization loop for sparse global alignment,
using MLX for automatic differentiation.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .losses import gamma_loss
from .schedules import cosine_schedule


@dataclass
class OptimConfig:
    """Configuration for scene optimization."""

    # Phase 1: Coarse alignment (3D loss)
    lr1: float = 0.07
    niter1: int = 300

    # Phase 2: Fine alignment (2D reprojection loss)
    lr2: float = 0.01
    niter2: int = 300

    # Loss parameters
    gamma_coarse: float = 1.5
    gamma_fine: float = 0.5

    # Optimization targets
    optimize_depth: bool = True
    shared_intrinsics: bool = False

    # Constraints
    min_focal_ratio: float = 0.5
    max_focal_ratio: float = 2.0


def quaternion_to_rotation_matrix(q: mx.array) -> mx.array:
    """Convert unit quaternion to rotation matrix.

    Args:
        q: Quaternion [4] as (x, y, z, w)

    Returns:
        Rotation matrix [3, 3]
    """
    x, y, z, w = q[0], q[1], q[2], q[3]

    # Normalize
    norm = mx.sqrt(x * x + y * y + z * z + w * w + 1e-8)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    # Build rotation matrix
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = mx.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )
    return R


def rotation_matrix_to_quaternion(R: mx.array) -> mx.array:
    """Convert rotation matrix to unit quaternion.

    Args:
        R: Rotation matrix [3, 3]

    Returns:
        Quaternion [4] as (x, y, z, w)
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / mx.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        # Find largest diagonal element
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * mx.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * mx.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * mx.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return mx.array([x, y, z, w])


class SceneOptimizer(nn.Module):
    """Optimizable scene parameters for global alignment."""

    def __init__(
        self,
        n_images: int,
        image_sizes: list[tuple[int, int]],
        init_focals: mx.array | None = None,
        init_pps: mx.array | None = None,
        init_depths: list[mx.array] | None = None,
        subsample: int = 8,
        shared_intrinsics: bool = False,
    ):
        """Initialize scene optimizer.

        Args:
            n_images: Number of images
            image_sizes: List of (H, W) per image
            init_focals: Initial focal lengths [N]
            init_pps: Initial principal points [N, 2]
            init_depths: Initial depth maps per image
            subsample: Depth subsampling factor
            shared_intrinsics: Use single intrinsics for all views
        """
        super().__init__()

        self.n_images = n_images
        self.image_sizes = image_sizes
        self.subsample = subsample
        self.shared_intrinsics = shared_intrinsics

        # Initialize rotations as quaternions (x, y, z, w)
        # First view is identity, others are optimized
        init_quats = mx.zeros((n_images, 4))
        init_quats = init_quats.at[:, 3].add(1.0)  # w = 1
        self.quats = init_quats

        # Initialize translations
        self.trans = mx.zeros((n_images, 3))

        # Initialize log-focal lengths
        if init_focals is None:
            # Default focal = max(H, W)
            init_focals = mx.array([max(h, w) for h, w in image_sizes])
        self.log_focals = mx.log(init_focals)

        # Initialize principal points (normalized to [0, 1])
        if init_pps is None:
            init_pps = mx.array([[0.5, 0.5]] * n_images)
        self.pps = init_pps

        # Initialize log-scales (relative depth scaling)
        self.log_scales = mx.zeros(n_images)

        # Initialize sparse depths (already subsampled)
        if init_depths is not None:
            self.core_depths = []
            for i, depth in enumerate(init_depths):
                # Depths are already subsampled, just flatten
                d_sub = depth.flatten()
                self.core_depths.append(d_sub)
        else:
            self.core_depths = None

    def get_poses(self) -> mx.array:
        """Get camera-to-world transformation matrices.

        Returns:
            Camera poses [N, 4, 4]
        """
        poses = []
        for i in range(self.n_images):
            R = quaternion_to_rotation_matrix(self.quats[i])
            t = self.trans[i]

            # Build pose matrix directly without using .at[].add()
            # This is more gradient-friendly
            row0 = mx.concatenate([R[0], t[0:1]])
            row1 = mx.concatenate([R[1], t[1:2]])
            row2 = mx.concatenate([R[2], t[2:3]])
            row3 = mx.array([0.0, 0.0, 0.0, 1.0])

            pose = mx.stack([row0, row1, row2, row3], axis=0)
            poses.append(pose)

        return mx.stack(poses, axis=0)

    def get_focals(self) -> mx.array:
        """Get focal lengths.

        Returns:
            Focal lengths [N]
        """
        if self.shared_intrinsics:
            return mx.exp(self.log_focals[0:1]).broadcast_to((self.n_images,))
        return mx.exp(self.log_focals)

    def get_principal_points(self) -> mx.array:
        """Get principal points in pixel coordinates.

        Returns:
            Principal points [N, 2]
        """
        pps_pixel = []
        for i in range(self.n_images):
            H, W = self.image_sizes[i]
            if self.shared_intrinsics:
                pp = self.pps[0]
            else:
                pp = self.pps[i]
            pps_pixel.append(mx.array([pp[0] * W, pp[1] * H]))
        return mx.stack(pps_pixel, axis=0)

    def get_intrinsics(self) -> mx.array:
        """Get camera intrinsic matrices.

        Returns:
            Intrinsics [N, 3, 3]
        """
        focals = self.get_focals()
        pps = self.get_principal_points()

        Ks = []
        for i in range(self.n_images):
            K = mx.array(
                [
                    [focals[i], 0, pps[i, 0]],
                    [0, focals[i], pps[i, 1]],
                    [0, 0, 1],
                ]
            )
            Ks.append(K)
        return mx.stack(Ks, axis=0)

    def get_depths(self) -> list[mx.array]:
        """Get depth maps (upsampled from core depths).

        Returns:
            List of depth maps per image
        """
        if self.core_depths is None:
            return None

        depths = []
        for i in range(self.n_images):
            H, W = self.image_sizes[i]
            H_sub = (H + self.subsample - 1) // self.subsample
            W_sub = (W + self.subsample - 1) // self.subsample

            # Reshape and upsample
            d_sub = self.core_depths[i].reshape(H_sub, W_sub)

            # Apply scale
            scale = mx.exp(self.log_scales[i])
            d_sub = d_sub * scale

            # Simple nearest-neighbor upsample
            # (Full implementation would use bilinear)
            d_full = mx.repeat(mx.repeat(d_sub, self.subsample, axis=0), self.subsample, axis=1)
            d_full = d_full[:H, :W]
            depths.append(d_full)

        return depths


def compute_3d_loss(
    optimizer: SceneOptimizer,
    corres: list[dict],
    canonical_pts3d: list[mx.array],
    loss_fn: Callable,
) -> mx.array:
    """Compute 3D point matching loss.

    Args:
        optimizer: Scene optimizer with current parameters
        corres: List of correspondences between image pairs
        canonical_pts3d: Canonical 3D points per image
        loss_fn: Loss function to use

    Returns:
        Scalar loss value
    """
    poses = optimizer.get_poses()
    scales = mx.exp(optimizer.log_scales)

    total_loss = mx.array(0.0)
    n_pairs = 0

    for c in corres:
        idx1, idx2 = c["idx1"], c["idx2"]
        pts1_idx = c["pts1_idx"]  # Indices into canonical points
        pts2_idx = c["pts2_idx"]
        weights = c.get("weights", None)

        # Get 3D points in world frame
        pts1_local = canonical_pts3d[idx1][pts1_idx] * scales[idx1]
        pts2_local = canonical_pts3d[idx2][pts2_idx] * scales[idx2]

        # Transform to world coordinates
        R1 = poses[idx1, :3, :3]
        t1 = poses[idx1, :3, 3]
        R2 = poses[idx2, :3, :3]
        t2 = poses[idx2, :3, 3]

        pts1_world = pts1_local @ R1.T + t1
        pts2_world = pts2_local @ R2.T + t2

        # Compute loss
        loss = loss_fn(pts1_world, pts2_world, weights)
        total_loss = total_loss + loss
        n_pairs += 1

    return total_loss / max(n_pairs, 1)


def compute_2d_loss(
    optimizer: SceneOptimizer,
    corres: list[dict],
    loss_fn: Callable,
) -> mx.array:
    """Compute 2D reprojection loss.

    Args:
        optimizer: Scene optimizer with current parameters
        corres: List of correspondences between image pairs
        loss_fn: Loss function to use

    Returns:
        Scalar loss value
    """
    poses = optimizer.get_poses()
    Ks = optimizer.get_intrinsics()
    depths = optimizer.get_depths()

    if depths is None:
        return mx.array(0.0)

    total_loss = mx.array(0.0)
    n_pairs = 0

    for c in corres:
        idx1, idx2 = c["idx1"], c["idx2"]
        pts1_px = c["pts1"]  # Pixel coordinates [N, 2]
        pts2_px = c["pts2"]
        weights = c.get("weights", None)

        # Get depths at correspondence locations
        # (Simplified - full impl would interpolate)
        d1 = depths[idx1]
        d2 = depths[idx2]

        # Unproject pts1 to 3D
        K1 = Ks[idx1]
        fx1, fy1 = K1[0, 0], K1[1, 1]
        cx1, cy1 = K1[0, 2], K1[1, 2]

        x1_norm = (pts1_px[:, 0] - cx1) / fx1
        y1_norm = (pts1_px[:, 1] - cy1) / fy1

        # Sample depths (nearest neighbor for simplicity)
        pts1_int = pts1_px.astype(mx.int32)
        pts1_int = mx.clip(pts1_int, 0, mx.array([d1.shape[1] - 1, d1.shape[0] - 1]))
        z1 = d1[pts1_int[:, 1], pts1_int[:, 0]]

        pts3d_cam1 = mx.stack([x1_norm * z1, y1_norm * z1, z1], axis=-1)

        # Transform to world then to camera 2
        T1 = poses[idx1]
        T2 = poses[idx2]

        # Compute inverse of T2 manually (avoid mx.linalg.inv which is CPU-only)
        # For [R | t; 0 | 1], inverse is [R^T | -R^T*t; 0 | 1]
        R2 = T2[:3, :3]
        t2 = T2[:3, 3]
        R2_T = R2.T
        t2_inv = -(R2_T @ t2)

        pts3d_world = pts3d_cam1 @ T1[:3, :3].T + T1[:3, 3]
        pts3d_cam2 = pts3d_world @ R2_T.T + t2_inv

        # Project to camera 2
        K2 = Ks[idx2]
        fx2, fy2 = K2[0, 0], K2[1, 1]
        cx2, cy2 = K2[0, 2], K2[1, 2]

        z2 = mx.maximum(pts3d_cam2[:, 2], mx.array(1e-6))
        x2_proj = pts3d_cam2[:, 0] / z2 * fx2 + cx2
        y2_proj = pts3d_cam2[:, 1] / z2 * fy2 + cy2

        pts2_proj = mx.stack([x2_proj, y2_proj], axis=-1)

        # Compute reprojection error
        loss = loss_fn(pts2_proj, pts2_px, weights)
        total_loss = total_loss + loss
        n_pairs += 1

    return total_loss / max(n_pairs, 1)


def optimize_scene(
    optimizer: SceneOptimizer,
    corres: list[dict],
    canonical_pts3d: list[mx.array],
    config: OptimConfig | None = None,
    verbose: bool = True,
) -> SceneOptimizer:
    """Run two-phase scene optimization.

    Phase 1: Optimize poses and scales using 3D loss
    Phase 2: Refine with 2D reprojection loss

    Args:
        optimizer: Initial scene optimizer
        corres: Correspondences between views
        canonical_pts3d: Canonical 3D points per view
        config: Optimization configuration
        verbose: Print progress

    Returns:
        Optimized scene
    """
    if config is None:
        config = OptimConfig()

    loss_3d = gamma_loss(config.gamma_coarse)
    loss_2d = gamma_loss(config.gamma_fine)

    # Phase 1: Coarse alignment
    if verbose:
        print(f"Phase 1: Coarse alignment ({config.niter1} iterations)")

    def loss_fn_phase1(params):
        # Temporarily update optimizer with params
        optimizer.quats = params["quats"]
        optimizer.trans = params["trans"]
        optimizer.log_scales = params["log_scales"]
        return compute_3d_loss(optimizer, corres, canonical_pts3d, loss_3d)

    for step in range(config.niter1):
        lr = cosine_schedule(step / max(config.niter1 - 1, 1), config.lr1, config.lr1 * 0.1)

        # Extract current params as dict
        params = {
            "quats": optimizer.quats,
            "trans": optimizer.trans,
            "log_scales": optimizer.log_scales,
        }

        # Compute loss and gradients
        loss, grads = mx.value_and_grad(loss_fn_phase1)(params)

        # Update parameters (simplified gradient descent)
        optimizer.quats = params["quats"] - lr * grads["quats"]
        optimizer.trans = params["trans"] - lr * grads["trans"]
        optimizer.log_scales = params["log_scales"] - lr * grads["log_scales"]

        # Normalize quaternions
        quat_norms = mx.sqrt(mx.sum(optimizer.quats**2, axis=-1, keepdims=True) + 1e-8)
        optimizer.quats = optimizer.quats / quat_norms

        if verbose and step % 50 == 0:
            mx.eval(loss)
            print(f"  Step {step}: loss = {float(loss):.6f}")

    # Phase 2: Fine alignment
    if config.niter2 > 0 and config.optimize_depth:
        if verbose:
            print(f"Phase 2: Fine alignment ({config.niter2} iterations)")

        def loss_fn_phase2(params):
            optimizer.quats = params["quats"]
            optimizer.trans = params["trans"]
            optimizer.log_focals = params["log_focals"]
            optimizer.pps = params["pps"]
            return compute_2d_loss(optimizer, corres, loss_2d)

        for step in range(config.niter2):
            lr = cosine_schedule(step / max(config.niter2 - 1, 1), config.lr2, config.lr2 * 0.1)

            params = {
                "quats": optimizer.quats,
                "trans": optimizer.trans,
                "log_focals": optimizer.log_focals,
                "pps": optimizer.pps,
            }

            loss, grads = mx.value_and_grad(loss_fn_phase2)(params)

            # Clip gradients to prevent explosion
            def clip_grad(g, max_norm=1.0):
                norm = mx.sqrt(mx.sum(g**2) + 1e-8)
                scale = mx.minimum(max_norm / norm, mx.array(1.0))
                return g * scale

            # Update all parameters with gradient clipping
            optimizer.quats = params["quats"] - lr * clip_grad(grads["quats"])
            optimizer.trans = params["trans"] - lr * clip_grad(grads["trans"])
            optimizer.log_focals = params["log_focals"] - lr * clip_grad(grads["log_focals"], 0.1)
            optimizer.pps = params["pps"] - lr * clip_grad(grads["pps"], 0.1)

            # Normalize quaternions
            quat_norms = mx.sqrt(mx.sum(optimizer.quats**2, axis=-1, keepdims=True) + 1e-8)
            optimizer.quats = optimizer.quats / quat_norms

            # Clamp principal points
            optimizer.pps = mx.clip(optimizer.pps, 0.25, 0.75)

            if verbose and step % 50 == 0:
                mx.eval(loss)
                print(f"  Step {step}: loss = {float(loss):.6f}")

    return optimizer
