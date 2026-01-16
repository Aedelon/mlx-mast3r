"""Scene optimizer for multi-view global alignment.

Implements the core optimization loop for sparse global alignment,
using MLX for automatic differentiation.

Based on the PyTorch MASt3R implementation with:
- Depth normalization by median for numerical stability
- Kinematic chain (MST) for pose composition
- z_cameras reparametrization with sizes
- depth_mode='add' reconstruction
- Global scaling protection

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


def build_mst_from_correspondences(corres: list[dict], n_images: int) -> tuple[dict[int, int], int]:
    """Build minimum spanning tree from correspondences.

    Uses number of correspondences as edge weights (more = better connection).
    Returns parent map and root node.

    Args:
        corres: List of correspondence dicts with idx1, idx2, and point counts
        n_images: Total number of images

    Returns:
        Tuple of (parent_map {child: parent}, root_idx)
    """
    # Build adjacency with weights = number of correspondences
    adj: dict[tuple[int, int], int] = {}
    for c in corres:
        i, j = c["idx1"], c["idx2"]
        if i > j:
            i, j = j, i
        key = (i, j)
        # Weight = number of correspondences (we want max, so use negative for MST)
        n_pts = len(c.get("pts1_idx", c.get("pts1", [])))
        adj[key] = adj.get(key, 0) + n_pts

    # Prim's algorithm for MST (maximize connections = minimize -weight)
    if not adj:
        # No correspondences, return identity (all connected to 0)
        return {i: 0 for i in range(1, n_images)}, 0

    # Start from node 0
    in_tree = {0}
    parent_map: dict[int, int] = {}

    while len(in_tree) < n_images:
        best_edge = None
        best_weight = -1

        for (i, j), weight in adj.items():
            if (i in in_tree) != (j in in_tree):  # XOR - one in, one out
                if weight > best_weight:
                    best_weight = weight
                    best_edge = (i, j)

        if best_edge is None:
            # Disconnected graph - connect remaining nodes to 0
            for i in range(n_images):
                if i not in in_tree:
                    parent_map[i] = 0
                    in_tree.add(i)
        else:
            i, j = best_edge
            if i in in_tree:
                parent_map[j] = i
                in_tree.add(j)
            else:
                parent_map[i] = j
                in_tree.add(i)

    return parent_map, 0


def compose_poses_kinematic(
    quats: mx.array,
    trans: mx.array,
    parent_map: dict[int, int],
    root: int = 0,
) -> mx.array:
    """Compose poses hierarchically along the MST.

    Each camera's world pose is computed relative to its parent.
    T_world_child = T_world_parent @ T_parent_child

    Args:
        quats: Quaternions [N, 4]
        trans: Translations [N, 3]
        parent_map: {child_idx: parent_idx}
        root: Root node index (identity pose)

    Returns:
        World poses [N, 4, 4]
    """
    n_images = quats.shape[0]

    # Compute local poses (relative to parent)
    local_poses = []
    for i in range(n_images):
        R = quaternion_to_rotation_matrix(quats[i])
        t = trans[i]
        row0 = mx.concatenate([R[0], t[0:1]])
        row1 = mx.concatenate([R[1], t[1:2]])
        row2 = mx.concatenate([R[2], t[2:3]])
        row3 = mx.array([0.0, 0.0, 0.0, 1.0])
        pose = mx.stack([row0, row1, row2, row3], axis=0)
        local_poses.append(pose)

    # Build world poses by composing along tree
    # Use topological order (BFS from root)
    world_poses = [None] * n_images

    # Root has identity as world pose (or its local pose IS world pose)
    world_poses[root] = local_poses[root]

    # BFS to process in order
    queue = [root]
    visited = {root}

    # Build children map
    children: dict[int, list[int]] = {i: [] for i in range(n_images)}
    for child, parent in parent_map.items():
        children[parent].append(child)

    while queue:
        parent_idx = queue.pop(0)
        for child_idx in children[parent_idx]:
            if child_idx not in visited:
                # T_world_child = T_world_parent @ T_parent_child
                world_poses[child_idx] = world_poses[parent_idx] @ local_poses[child_idx]
                visited.add(child_idx)
                queue.append(child_idx)

    # Handle any unvisited (disconnected) nodes
    for i in range(n_images):
        if world_poses[i] is None:
            world_poses[i] = local_poses[i]

    return mx.stack(world_poses, axis=0)


class SceneOptimizer(nn.Module):
    """Optimizable scene parameters for global alignment.

    Uses PyTorch MASt3R-compatible reparametrization:
    - Depths normalized by median
    - z_cameras = sizes * median_depths * focals / base_focals
    - depth = z_cameras + (core_depth_norm - 1) * median_depth * size
    - Kinematic chain (MST) for pose composition
    """

    def __init__(
        self,
        n_images: int,
        image_sizes: list[tuple[int, int]],
        init_focals: mx.array | None = None,
        init_pps: mx.array | None = None,
        init_depths: list[mx.array] | None = None,
        subsample: int = 8,
        shared_intrinsics: bool = False,
        mst_edges: dict[int, int] | None = None,
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
            mst_edges: MST parent map {child: parent} for kinematic chain
        """
        super().__init__()

        self.n_images = n_images
        self.image_sizes = image_sizes
        self.subsample = subsample
        self.shared_intrinsics = shared_intrinsics
        self.mst_edges = mst_edges if mst_edges else {}

        # Initialize rotations as quaternions (x, y, z, w)
        # First view is identity, others are optimized
        init_quats = mx.zeros((n_images, 4))
        init_quats = init_quats.at[:, 3].add(1.0)  # w = 1
        self.quats = init_quats

        # Initialize translations
        self.trans = mx.zeros((n_images, 3))

        # Initialize focal lengths
        if init_focals is None:
            init_focals = mx.array([float(max(h, w)) for h, w in image_sizes])
        else:
            init_focals = (
                mx.array(init_focals) if not isinstance(init_focals, mx.array) else init_focals
            )

        # Base focals (fixed, for reparametrization)
        self.base_focals = init_focals

        # Log-focal lengths (optimizable)
        self.log_focals = mx.log(init_focals)

        # Initialize principal points (normalized to [0, 1])
        if init_pps is None:
            init_pps = mx.array([[0.5, 0.5]] * n_images)
        self.pps = init_pps

        # Sizes (replaces log_scales) - optimizable scale per view
        self.sizes = mx.ones(n_images)

        # Initialize depths with median normalization
        self.median_depths: list[mx.array] = []
        self.core_depths: list[mx.array] | None = None

        if init_depths is not None:
            self.core_depths = []
            for depth in init_depths:
                d_flat = depth.flatten()
                # Compute median for normalization
                median = mx.median(d_flat)
                median = mx.maximum(median, mx.array(1e-6))  # Prevent div by zero
                self.median_depths.append(median)
                # Normalize depth by median
                d_normalized = d_flat / median
                self.core_depths.append(d_normalized)
        else:
            # Default median depths if no init
            self.median_depths = [mx.array(1.0) for _ in range(n_images)]

    def get_global_scaling(self) -> mx.array:
        """Get global scaling factor to prevent scale collapse.

        Returns:
            Scalar scaling factor = 1 / min(sizes)
        """
        min_size = mx.minimum(self.sizes.min(), mx.array(1.0))
        return 1.0 / mx.maximum(min_size, mx.array(1e-6))

    def get_z_cameras(self) -> mx.array:
        """Compute z_cameras for each view.

        z_cameras = sizes * median_depths * focals / base_focals

        Returns:
            z_cameras [N]
        """
        focals = self.get_focals()
        z_cameras = []
        for i in range(self.n_images):
            z = self.sizes[i] * self.median_depths[i] * focals[i] / self.base_focals[i]
            z_cameras.append(z)
        return mx.stack(z_cameras)

    def get_poses(self) -> mx.array:
        """Get camera-to-world transformation matrices.

        Uses kinematic chain (MST) if available, otherwise independent poses.
        Applies global scaling to translations.

        Returns:
            Camera poses [N, 4, 4]
        """
        if self.mst_edges:
            poses = compose_poses_kinematic(self.quats, self.trans, self.mst_edges, root=0)
        else:
            # Independent poses (fallback)
            poses = []
            for i in range(self.n_images):
                R = quaternion_to_rotation_matrix(self.quats[i])
                t = self.trans[i]
                row0 = mx.concatenate([R[0], t[0:1]])
                row1 = mx.concatenate([R[1], t[1:2]])
                row2 = mx.concatenate([R[2], t[2:3]])
                row3 = mx.array([0.0, 0.0, 0.0, 1.0])
                pose = mx.stack([row0, row1, row2, row3], axis=0)
                poses.append(pose)
            poses = mx.stack(poses, axis=0)

        # Apply global scaling to translations
        scaling = self.get_global_scaling()
        # Scale translations only
        scaled_poses = []
        for i in range(self.n_images):
            pose = poses[i]
            # Scale translation part
            t_scaled = pose[:3, 3] * scaling
            row0 = mx.concatenate([pose[0, :3], t_scaled[0:1]])
            row1 = mx.concatenate([pose[1, :3], t_scaled[1:2]])
            row2 = mx.concatenate([pose[2, :3], t_scaled[2:3]])
            row3 = pose[3]
            scaled_poses.append(mx.stack([row0, row1, row2, row3], axis=0))

        return mx.stack(scaled_poses, axis=0)

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

    def get_depths(self) -> list[mx.array] | None:
        """Get depth maps using depth_mode='add' reconstruction.

        depth = z_cameras + (core_depth_norm - 1) * median_depth * size

        Applies global scaling for consistency.

        Returns:
            List of depth maps per image, or None if no depths
        """
        if self.core_depths is None:
            return None

        z_cameras = self.get_z_cameras()
        global_scaling = self.get_global_scaling()

        depths = []
        for i in range(self.n_images):
            H, W = self.image_sizes[i]
            H_sub = (H + self.subsample - 1) // self.subsample
            W_sub = (W + self.subsample - 1) // self.subsample

            # Reshape normalized core depth
            core_norm = self.core_depths[i].reshape(H_sub, W_sub)

            # depth_mode='add': depth = z_cameras + (core_norm - 1) * median * size
            depth_sub = z_cameras[i] + (core_norm - 1.0) * self.median_depths[i] * self.sizes[i]

            # Apply global scaling
            depth_sub = depth_sub * global_scaling

            # Upsample (nearest neighbor)
            d_full = mx.repeat(mx.repeat(depth_sub, self.subsample, axis=0), self.subsample, axis=1)
            d_full = d_full[:H, :W]
            depths.append(d_full)

        return depths

    def get_original_depths(self) -> list[mx.array] | None:
        """Get original (unoptimized) depth maps.

        Reconstructs from normalized core depths using median.

        Returns:
            List of original depth maps per image
        """
        if self.core_depths is None:
            return None

        depths = []
        for i in range(self.n_images):
            H, W = self.image_sizes[i]
            H_sub = (H + self.subsample - 1) // self.subsample
            W_sub = (W + self.subsample - 1) // self.subsample

            # Denormalize: original = normalized * median
            core_norm = self.core_depths[i].reshape(H_sub, W_sub)
            depth_sub = core_norm * self.median_depths[i]

            # Upsample
            d_full = mx.repeat(mx.repeat(depth_sub, self.subsample, axis=0), self.subsample, axis=1)
            d_full = d_full[:H, :W]
            depths.append(d_full)

        return depths


def make_pts3d_from_depth(
    optimizer: SceneOptimizer,
    anchor_data: dict | None = None,
) -> list[mx.array]:
    """Reconstruct 3D points from optimized depth maps.

    Matches PyTorch MASt3R make_pts3d function:
    - Uses depth_mode='add' reconstruction
    - Reconstructs full grid of 3D points (for indexing with pts_idx)
    - Transforms to world coordinates

    Args:
        optimizer: Scene optimizer with current parameters
        anchor_data: Optional dict for anchor-based reconstruction (not used in grid mode)

    Returns:
        List of 3D points per image in world coordinates [N, H_sub * W_sub, 3]
    """
    poses = optimizer.get_poses()
    focals = optimizer.get_focals()
    pps = optimizer.get_principal_points()
    z_cameras = optimizer.get_z_cameras()
    global_scaling = optimizer.get_global_scaling()

    all_pts3d = []

    for i in range(optimizer.n_images):
        H, W = optimizer.image_sizes[i]
        H_sub = (H + optimizer.subsample - 1) // optimizer.subsample
        W_sub = (W + optimizer.subsample - 1) // optimizer.subsample

        # Core depth (normalized) - flatten to 1D
        core_depth = optimizer.core_depths[i].reshape(H_sub * W_sub)

        # depth_mode='add': depth = z_cameras + (core_depth - 1) * median * size
        depth = (
            z_cameras[i] + (core_depth - 1.0) * optimizer.median_depths[i] * optimizer.sizes[i]
        )

        # Apply global scaling
        depth = depth * global_scaling

        # Create pixel grid for subsampled positions
        # Match PyTorch: centers are at subsample//2, subsample//2 + subsample, etc.
        ys = mx.arange(H_sub) * optimizer.subsample + optimizer.subsample // 2
        xs = mx.arange(W_sub) * optimizer.subsample + optimizer.subsample // 2
        grid_y, grid_x = mx.meshgrid(ys, xs, indexing="ij")
        pixels = mx.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1).astype(mx.float32)

        # Unproject to 3D (in camera coordinates)
        fx = focals[i]
        fy = focals[i]
        cx = pps[i, 0]
        cy = pps[i, 1]

        x_norm = (pixels[:, 0] - cx) / fx
        y_norm = (pixels[:, 1] - cy) / fy

        pts3d_cam = mx.stack([x_norm * depth, y_norm * depth, depth], axis=-1)

        # Transform to world coordinates
        R = poses[i, :3, :3]
        t = poses[i, :3, 3]
        pts3d_world = pts3d_cam @ R.T + t

        all_pts3d.append(pts3d_world)

    return all_pts3d


def compute_3d_loss(
    optimizer: SceneOptimizer,
    corres: list[dict],
    canonical_pts3d: list[mx.array],
    loss_fn: Callable,
    anchor_data: dict | None = None,
) -> mx.array:
    """Compute 3D point matching loss.

    Matches PyTorch MASt3R loss_3d function:
    - Reconstructs 3D points from optimized depth maps
    - Compares corresponding points in world coordinates
    - Weights by correspondence confidence

    Args:
        optimizer: Scene optimizer with current parameters
        corres: List of correspondences between image pairs
        canonical_pts3d: Canonical 3D points per image (used if anchor_data is None)
        loss_fn: Loss function to use
        anchor_data: Optional anchor data for precise 3D reconstruction

    Returns:
        Scalar loss value
    """
    # Reconstruct 3D points from depth maps if anchor_data is provided
    if anchor_data is not None:
        pts3d = make_pts3d_from_depth(optimizer, anchor_data)
    else:
        # Fallback: use canonical pts3d with simple scaling (less accurate)
        poses = optimizer.get_poses()
        global_scaling = optimizer.get_global_scaling()

        pts3d = []
        for i in range(optimizer.n_images):
            pts_local = canonical_pts3d[i] * optimizer.sizes[i] * global_scaling
            R = poses[i, :3, :3]
            t = poses[i, :3, 3]
            pts_world = pts_local @ R.T + t
            pts3d.append(pts_world)

    # Compute loss over all correspondences
    all_pts1 = []
    all_pts2 = []
    all_weights = []

    for c in corres:
        idx1, idx2 = c["idx1"], c["idx2"]
        pts1_idx = c["pts1_idx"]
        pts2_idx = c["pts2_idx"]
        weights = c.get("weights", None)

        # Get corresponding 3D points
        pts1 = pts3d[idx1][pts1_idx]
        pts2 = pts3d[idx2][pts2_idx]

        all_pts1.append(pts1)
        all_pts2.append(pts2)
        if weights is not None:
            all_weights.append(weights)

    if not all_pts1:
        return mx.array(0.0)

    # Concatenate all correspondences
    pts1_cat = mx.concatenate(all_pts1, axis=0)
    pts2_cat = mx.concatenate(all_pts2, axis=0)

    if all_weights:
        weights_cat = mx.concatenate(all_weights, axis=0)
        # Weighted loss: sum(weights * loss) / sum(weights)
        point_losses = loss_fn(pts1_cat, pts2_cat, None)
        total_loss = mx.sum(weights_cat * point_losses)
        return total_loss / mx.sum(weights_cat)
    else:
        return loss_fn(pts1_cat, pts2_cat, None)


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
        d1 = depths[idx1]
        d2 = depths[idx2]

        # Unproject pts1 to 3D
        K1 = Ks[idx1]
        fx1, fy1 = K1[0, 0], K1[1, 1]
        cx1, cy1 = K1[0, 2], K1[1, 2]

        x1_norm = (pts1_px[:, 0] - cx1) / fx1
        y1_norm = (pts1_px[:, 1] - cy1) / fy1

        # Sample depths (nearest neighbor)
        pts1_int = pts1_px.astype(mx.int32)
        pts1_int = mx.clip(pts1_int, 0, mx.array([d1.shape[1] - 1, d1.shape[0] - 1]))
        z1 = d1[pts1_int[:, 1], pts1_int[:, 0]]

        pts3d_cam1 = mx.stack([x1_norm * z1, y1_norm * z1, z1], axis=-1)

        # Transform to world then to camera 2
        T1 = poses[idx1]
        T2 = poses[idx2]

        # Compute inverse of T2
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
    anchor_data: dict | None = None,
    verbose: bool = True,
) -> SceneOptimizer:
    """Run two-phase scene optimization.

    Phase 1: Optimize poses and sizes using 3D loss
    Phase 2: Refine with 2D reprojection loss

    Args:
        optimizer: Initial scene optimizer
        corres: Correspondences between views
        canonical_pts3d: Canonical 3D points per view
        config: Optimization configuration
        anchor_data: Optional anchor data for precise 3D reconstruction
                    Dict mapping img_idx to (pixels, idxs, offsets)
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
        optimizer.quats = params["quats"]
        optimizer.trans = params["trans"]
        optimizer.sizes = params["sizes"]
        return compute_3d_loss(optimizer, corres, canonical_pts3d, loss_3d, anchor_data)

    for step in range(config.niter1):
        lr = cosine_schedule(step / max(config.niter1 - 1, 1), config.lr1, config.lr1 * 0.1)

        params = {
            "quats": optimizer.quats,
            "trans": optimizer.trans,
            "sizes": optimizer.sizes,
        }

        loss, grads = mx.value_and_grad(loss_fn_phase1)(params)

        # Clip gradients for stability
        def clip_grad(g, max_norm=1.0):
            norm = mx.sqrt(mx.sum(g**2) + 1e-8)
            scale = mx.minimum(max_norm / norm, mx.array(1.0))
            return g * scale

        # Update parameters with clipped gradients
        optimizer.quats = params["quats"] - lr * clip_grad(grads["quats"])
        optimizer.trans = params["trans"] - lr * clip_grad(grads["trans"])
        # Use smaller learning rate for sizes (more sensitive)
        optimizer.sizes = params["sizes"] - lr * 0.1 * clip_grad(grads["sizes"], 0.5)

        # Clamp sizes to prevent collapse and explosion
        optimizer.sizes = mx.clip(optimizer.sizes, 0.1, 10.0)

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

            # Clip gradients
            def clip_grad(g, max_norm=1.0):
                norm = mx.sqrt(mx.sum(g**2) + 1e-8)
                scale = mx.minimum(max_norm / norm, mx.array(1.0))
                return g * scale

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
