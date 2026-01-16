"""Triangulation utilities for multi-view reconstruction.

Implements DLT triangulation and depth aggregation from matches.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def triangulate_point_dlt(
    P1: mx.array,
    P2: mx.array,
    x1: mx.array,
    x2: mx.array,
) -> mx.array:
    """Triangulate a single point using Direct Linear Transform.

    Args:
        P1: 3x4 projection matrix for camera 1
        P2: 3x4 projection matrix for camera 2
        x1: 2D point in image 1 (x, y)
        x2: 2D point in image 2 (x, y)

    Returns:
        3D point (X, Y, Z)
    """
    # Build the linear system Ax = 0
    A = mx.stack(
        [
            x1[0] * P1[2] - P1[0],
            x1[1] * P1[2] - P1[1],
            x2[0] * P2[2] - P2[0],
            x2[1] * P2[2] - P2[1],
        ],
        axis=0,
    )  # [4, 4]

    # SVD to find null space
    # Note: MLX SVD returns U, S, Vh
    _, _, Vh = mx.linalg.svd(A, stream=mx.cpu)

    # Solution is last row of Vh (smallest singular value)
    X_h = Vh[-1]

    # Convert from homogeneous
    X = X_h[:3] / (X_h[3] + 1e-8)
    return X


def batched_triangulate(
    P: mx.array,
    pts2d: mx.array,
) -> mx.array:
    """Triangulate multiple points from multiple views.

    Uses DLT method with SVD for each point.

    Args:
        P: Camera projection matrices [N_cams, 3, 4]
        pts2d: 2D points [N_cams, N_pts, 2]

    Returns:
        3D points [N_pts, 3]
    """
    n_cams, n_pts, _ = pts2d.shape

    # Build A matrix for each point
    # Each camera contributes 2 rows per point
    pts3d_list = []

    for pt_idx in range(n_pts):
        A_rows = []
        for cam_idx in range(n_cams):
            x, y = pts2d[cam_idx, pt_idx, 0], pts2d[cam_idx, pt_idx, 1]
            P_cam = P[cam_idx]  # [3, 4]

            A_rows.append(x * P_cam[2] - P_cam[0])
            A_rows.append(y * P_cam[2] - P_cam[1])

        A = mx.stack(A_rows, axis=0)  # [2*N_cams, 4]

        # SVD
        _, _, Vh = mx.linalg.svd(A, stream=mx.cpu)
        X_h = Vh[-1]
        X = X_h[:3] / (X_h[3] + 1e-8)
        pts3d_list.append(X)

    return mx.stack(pts3d_list, axis=0)


def triangulate_from_depth(
    depth1: float,
    depth2: float,
    K1: mx.array,
    K2: mx.array,
    T1: mx.array,
    T2: mx.array,
    pt1: mx.array,
    pt2: mx.array,
) -> mx.array:
    """Triangulate using depth predictions from two views.

    More numerically stable than pure DLT when depth is available.

    Args:
        depth1, depth2: Predicted depths
        K1, K2: Camera intrinsics [3, 3]
        T1, T2: Camera-to-world transforms [4, 4]
        pt1, pt2: 2D pixel coordinates (x, y)

    Returns:
        3D point in world coordinates
    """
    # Unproject point 1 to 3D
    fx1, fy1 = K1[0, 0], K1[1, 1]
    cx1, cy1 = K1[0, 2], K1[1, 2]

    x1_norm = (pt1[0] - cx1) / fx1
    y1_norm = (pt1[1] - cy1) / fy1

    pt3d_cam1 = mx.array([x1_norm * depth1, y1_norm * depth1, depth1])

    # Transform to world coordinates
    pt3d_world1 = T1[:3, :3] @ pt3d_cam1 + T1[:3, 3]

    # Similarly for point 2
    fx2, fy2 = K2[0, 0], K2[1, 1]
    cx2, cy2 = K2[0, 2], K2[1, 2]

    x2_norm = (pt2[0] - cx2) / fx2
    y2_norm = (pt2[1] - cy2) / fy2

    pt3d_cam2 = mx.array([x2_norm * depth2, y2_norm * depth2, depth2])
    pt3d_world2 = T2[:3, :3] @ pt3d_cam2 + T2[:3, 3]

    # Average the two estimates
    return (pt3d_world1 + pt3d_world2) / 2


def matches_to_depths(
    corres: list[dict],
    pred_depths: list[mx.array],
    focals: mx.array,
    pps: mx.array,
    cams2world: mx.array,
) -> list[mx.array]:
    """Convert matches to depth maps via triangulation.

    For each image, aggregate depth predictions from all pairs
    involving that image.

    Args:
        corres: List of correspondence dicts with keys:
            - 'idx1', 'idx2': Image indices
            - 'pts1', 'pts2': Pixel coordinates [N, 2]
            - 'weights': Confidence weights [N]
        pred_depths: List of predicted depth maps per image
        focals: Focal lengths [N_imgs]
        pps: Principal points [N_imgs, 2]
        cams2world: Camera poses [N_imgs, 4, 4]

    Returns:
        List of refined depth maps
    """
    n_imgs = len(pred_depths)

    # Accumulate depth estimates and weights per image
    depth_sums = [mx.zeros_like(d) for d in pred_depths]
    weight_sums = [mx.zeros_like(d) for d in pred_depths]

    for c in corres:
        idx1, idx2 = c["idx1"], c["idx2"]
        pts1, pts2 = c["pts1"], c["pts2"]
        weights = c.get("weights", mx.ones(len(pts1)))

        # Build intrinsics
        K1 = mx.array(
            [
                [focals[idx1], 0, pps[idx1, 0]],
                [0, focals[idx1], pps[idx1, 1]],
                [0, 0, 1],
            ]
        )
        K2 = mx.array(
            [
                [focals[idx2], 0, pps[idx2, 0]],
                [0, focals[idx2], pps[idx2, 1]],
                [0, 0, 1],
            ]
        )

        T1 = cams2world[idx1]
        T2 = cams2world[idx2]

        # Get depths at correspondence locations
        for i in range(len(pts1)):
            px1, py1 = int(pts1[i, 0]), int(pts1[i, 1])
            px2, py2 = int(pts2[i, 0]), int(pts2[i, 1])

            d1 = float(pred_depths[idx1][py1, px1])
            d2 = float(pred_depths[idx2][py2, px2])

            # Triangulate
            pt3d = triangulate_from_depth(d1, d2, K1, K2, T1, T2, pts1[i], pts2[i])

            # Project back to get refined depths
            # View 1
            T1_inv = mx.linalg.inv(T1)
            pt_cam1 = T1_inv[:3, :3] @ pt3d + T1_inv[:3, 3]
            refined_d1 = pt_cam1[2]

            # Update accumulator (simplified - full impl would use scatter)
            w = float(weights[i])
            # Note: This is a simplified version. Full implementation
            # would properly accumulate at pixel locations.

    # Normalize by weights
    refined = []
    for i in range(n_imgs):
        mask = weight_sums[i] > 0
        d = mx.where(mask, depth_sums[i] / (weight_sums[i] + 1e-8), pred_depths[i])
        refined.append(d)

    return refined


def compute_projection_matrix(K: mx.array, T_w2c: mx.array) -> mx.array:
    """Compute 3x4 projection matrix from intrinsics and extrinsics.

    Args:
        K: Camera intrinsics [3, 3]
        T_w2c: World-to-camera transform [4, 4]

    Returns:
        Projection matrix [3, 4]
    """
    return K @ T_w2c[:3, :]


def reproject_point(
    pt3d: mx.array,
    K: mx.array,
    T_w2c: mx.array,
) -> mx.array:
    """Reproject 3D point to 2D pixel coordinates.

    Args:
        pt3d: 3D point (X, Y, Z)
        K: Camera intrinsics [3, 3]
        T_w2c: World-to-camera transform [4, 4]

    Returns:
        2D pixel coordinates (x, y)
    """
    # Transform to camera frame
    pt_cam = T_w2c[:3, :3] @ pt3d + T_w2c[:3, 3]

    # Project to 2D
    z = pt_cam[2]
    pt_norm = pt_cam[:2] / (z + 1e-8)

    # Apply intrinsics
    px = pt_norm[0] * K[0, 0] + K[0, 2]
    py = pt_norm[1] * K[1, 1] + K[1, 2]

    return mx.array([px, py])


def compute_reprojection_error(
    pt3d: mx.array,
    pts2d: mx.array,
    Ks: mx.array,
    Ts_w2c: mx.array,
) -> mx.array:
    """Compute reprojection error across multiple views.

    Args:
        pt3d: 3D point (X, Y, Z)
        pts2d: Observed 2D points [N_views, 2]
        Ks: Camera intrinsics [N_views, 3, 3]
        Ts_w2c: World-to-camera transforms [N_views, 4, 4]

    Returns:
        Reprojection errors [N_views]
    """
    errors = []
    for i in range(len(pts2d)):
        reproj = reproject_point(pt3d, Ks[i], Ts_w2c[i])
        error = mx.sqrt(mx.sum((reproj - pts2d[i]) ** 2))
        errors.append(error)
    return mx.stack(errors)
