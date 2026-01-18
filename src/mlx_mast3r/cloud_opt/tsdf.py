"""TSDF-based depth cleaning and post-processing.

Implements depth refinement using multi-view consistency checks.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .geometry import depthmap_to_pts3d_mlx, geotrf_mlx, inv_mlx


def clean_pointcloud(
    confs: list[mx.array],
    intrinsics: list[mx.array],
    w2cams: mx.array,
    depthmaps: list[mx.array],
    pts3d: list[mx.array],
    tol: float = 0.01,
) -> list[mx.array]:
    """Clean pointcloud using multi-view consistency.

    For each 3D point, check if it reprojects consistently to other views.
    Points with large reprojection errors are masked out.

    Args:
        confs: Confidence maps [N, H, W]
        intrinsics: Camera intrinsics [N, 3, 3]
        w2cams: World-to-camera transforms [N, 4, 4]
        depthmaps: Depth maps [N, H, W]
        pts3d: 3D points per view [N, H, W, 3]
        tol: Tolerance for depth consistency (relative)

    Returns:
        Cleaned confidence maps
    """
    n_views = len(confs)
    cleaned_confs = [mx.array(c) for c in confs]

    for i in range(n_views):
        # Get H, W from pts3d shape (confs may be 1D or 2D)
        if pts3d[i].ndim == 3:
            H, W = pts3d[i].shape[:2]
        else:
            # Flat pts3d, try to infer from confs
            if confs[i].ndim == 2:
                H, W = confs[i].shape
            else:
                # Both flat, assume square-ish
                n = len(confs[i].reshape(-1))
                H = int(np.sqrt(n))
                W = n // H
        pts3d_i = pts3d[i].reshape(-1, 3)  # [HW, 3]
        conf_i = confs[i].reshape(-1)

        # Check consistency with other views
        for j in range(n_views):
            if i == j:
                continue

            K_j = intrinsics[j]
            w2c_j = w2cams[j]

            # Project points from view i to view j
            pts_cam_j = geotrf_mlx(w2c_j, pts3d_i)  # [HW, 3]

            # Project to 2D
            z_j = pts_cam_j[:, 2:3]
            valid_z = z_j > 0.1

            pts_proj = pts_cam_j[:, :2] / (z_j + 1e-8)
            pts_2d = pts_proj @ K_j[:2, :2].T + K_j[:2, 2]

            # Check if points are in image bounds
            H_j, W_j = depthmaps[j].shape
            in_bounds = (
                (pts_2d[:, 0] >= 0)
                & (pts_2d[:, 0] < W_j)
                & (pts_2d[:, 1] >= 0)
                & (pts_2d[:, 1] < H_j)
            )

            # Sample depths at projected locations
            valid_mask = valid_z.reshape(-1) & in_bounds

            if mx.sum(valid_mask) > 0:
                pts_2d_int = pts_2d.astype(mx.int32)
                pts_2d_int = mx.clip(pts_2d_int, 0, mx.array([W_j - 1, H_j - 1]))

                # Get depth at projected locations
                depth_j_flat = depthmaps[j].reshape(-1)
                indices = pts_2d_int[:, 1] * W_j + pts_2d_int[:, 0]
                indices = mx.clip(indices, 0, len(depth_j_flat) - 1)
                sampled_depth = depth_j_flat[indices]

                # Compare with projected depth
                projected_depth = z_j.reshape(-1)
                depth_ratio = sampled_depth / (projected_depth + 1e-8)

                # Points are inconsistent if depth ratio is far from 1
                consistent = (depth_ratio > 1 - tol) & (depth_ratio < 1 + tol)
                consistent = consistent | ~valid_mask

                # Update confidence
                conf_i = conf_i * mx.where(consistent, 1.0, 0.5)

        cleaned_confs[i] = conf_i.reshape(H, W)

    return cleaned_confs


class TSDFPostProcess:
    """TSDF-based depth cleaning.

    Uses truncated signed distance function to refine depth estimates
    using multi-view consistency.
    """

    def __init__(
        self,
        sparse_ga,
        tsdf_thresh: float = 0.01,
    ):
        """Initialize TSDF post-processor.

        Args:
            sparse_ga: SparseGAResult from sparse_global_alignment
            tsdf_thresh: TSDF threshold for depth cleaning
        """
        self.sparse_ga = sparse_ga
        self.tsdf_thresh = tsdf_thresh

        # Cache computed values
        self._dense_pts3d = None
        self._cleaned_depths = None
        self._cleaned_confs = None

    def get_dense_pts3d(
        self,
        clean_depth: bool = True,
        subsample: int = 1,
    ) -> tuple[list[mx.array], list[mx.array], list[mx.array]]:
        """Get dense 3D points with optional cleaning.

        Args:
            clean_depth: Apply depth cleaning
            subsample: Subsampling factor

        Returns:
            Tuple of (pts3d, depthmaps, confs) lists
        """
        if self._dense_pts3d is not None and not clean_depth:
            return self._dense_pts3d, self.sparse_ga.depthmaps, self.sparse_ga.confs

        pts3d_list = []
        depth_list = []
        conf_list = []

        focals = self.sparse_ga.focals
        pps = self.sparse_ga.principal_points
        cam2w = self.sparse_ga.cam2w
        depthmaps = self.sparse_ga.depthmaps
        confs = self.sparse_ga.confs

        n_imgs = len(depthmaps)

        for i in range(n_imgs):
            depth = depthmaps[i]
            H, W = depth.shape

            # Build intrinsics
            f = focals[i]
            pp = pps[i]
            K = mx.array(
                [
                    [f, 0, pp[0]],
                    [0, f, pp[1]],
                    [0, 0, 1],
                ]
            )

            # Unproject to 3D (use MLX native version)
            pts3d = depthmap_to_pts3d_mlx(depth, K)

            # Transform to world
            pts3d_world = geotrf_mlx(cam2w[i], pts3d.reshape(-1, 3)).reshape(H, W, 3)

            pts3d_list.append(pts3d_world)
            depth_list.append(depth)
            conf_list.append(confs[i])

        if clean_depth and self.tsdf_thresh > 0:
            # Build intrinsics list
            intrinsics = []
            for i in range(n_imgs):
                f = focals[i]
                pp = pps[i]
                K = mx.array(
                    [
                        [f, 0, pp[0]],
                        [0, f, pp[1]],
                        [0, 0, 1],
                    ]
                )
                intrinsics.append(K)

            # Get world-to-camera transforms
            w2cams = inv_mlx(cam2w)

            # Clean pointcloud
            conf_list = clean_pointcloud(
                conf_list,
                intrinsics,
                w2cams,
                depth_list,
                pts3d_list,
                tol=self.tsdf_thresh,
            )

            self._cleaned_confs = conf_list

        self._dense_pts3d = pts3d_list
        self._cleaned_depths = depth_list

        return pts3d_list, depth_list, conf_list

    def get_mesh(
        self,
        clean_depth: bool = True,
        min_conf: float = 1.5,
    ):
        """Get mesh from depth maps.

        Args:
            clean_depth: Apply depth cleaning
            min_conf: Minimum confidence threshold

        Returns:
            Combined mesh (vertices, faces, colors)
        """
        pts3d, depths, confs = self.get_dense_pts3d(clean_depth=clean_depth)

        all_vertices = []
        all_colors = []

        for i in range(len(pts3d)):
            pts = pts3d[i]
            conf = confs[i]
            img = self.sparse_ga.imgs[i]

            H, W = pts.shape[:2]

            # Flatten
            pts_flat = pts.reshape(-1, 3)
            conf_flat = conf.reshape(-1)

            # Get colors from image
            if img.ndim == 3 and img.shape[2] == 3:
                colors = img.reshape(-1, 3)
            else:
                colors = np.ones((H * W, 3)) * 0.5

            # Apply confidence threshold
            valid = np.array(conf_flat) > min_conf
            pts_valid = np.array(pts_flat)[valid]
            colors_valid = np.array(colors)[valid]

            all_vertices.append(pts_valid)
            all_colors.append(colors_valid)

        if all_vertices:
            vertices = np.concatenate(all_vertices, axis=0)
            colors = np.concatenate(all_colors, axis=0)
        else:
            vertices = np.zeros((0, 3))
            colors = np.zeros((0, 3))

        return vertices, colors


def apply_tsdf_cleaning(
    sparse_ga,
    tsdf_thresh: float = 0.01,
    clean_depth: bool = True,
) -> tuple[list[mx.array], list[mx.array], list[mx.array]]:
    """Convenience function for TSDF cleaning.

    Args:
        sparse_ga: SparseGAResult
        tsdf_thresh: TSDF threshold
        clean_depth: Whether to apply cleaning

    Returns:
        Tuple of (pts3d, depthmaps, confs)
    """
    processor = TSDFPostProcess(sparse_ga, tsdf_thresh=tsdf_thresh)
    return processor.get_dense_pts3d(clean_depth=clean_depth)


def depth_edges_mask(
    depth: mx.array,
    threshold: float = 0.1,
) -> mx.array:
    """Create mask for depth edges (discontinuities).

    Args:
        depth: Depth map [H, W]
        threshold: Relative depth difference threshold

    Returns:
        Boolean mask where True = edge
    """
    H, W = depth.shape

    # Compute gradients
    grad_x = mx.abs(depth[:, 1:] - depth[:, :-1])
    grad_y = mx.abs(depth[1:, :] - depth[:-1, :])

    # Pad to original size
    grad_x = mx.pad(grad_x, [(0, 0), (0, 1)])
    grad_y = mx.pad(grad_y, [(0, 1), (0, 0)])

    # Relative gradient
    rel_grad_x = grad_x / (depth + 1e-8)
    rel_grad_y = grad_y / (depth + 1e-8)

    # Edge mask
    edge_mask = (rel_grad_x > threshold) | (rel_grad_y > threshold)

    return edge_mask


def median_filter_depth(
    depth: mx.array,
    kernel_size: int = 3,
) -> mx.array:
    """Apply median filter to depth map.

    Args:
        depth: Depth map [H, W]
        kernel_size: Filter kernel size

    Returns:
        Filtered depth map
    """
    # Convert to numpy for median filter
    depth_np = np.array(depth)
    H, W = depth_np.shape

    # Pad
    pad = kernel_size // 2
    depth_padded = np.pad(depth_np, pad, mode="reflect")

    # Apply median filter
    result = np.zeros_like(depth_np)
    for i in range(H):
        for j in range(W):
            window = depth_padded[i : i + kernel_size, j : j + kernel_size]
            result[i, j] = np.median(window)

    return mx.array(result)
