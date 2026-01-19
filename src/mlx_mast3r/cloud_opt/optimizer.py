"""Scene optimizer for multi-view global alignment - PyTorch-faithful port.

Exact port of PyTorch MASt3R sparse_scene_optimizer.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Callable

import mlx.core as mx
import numpy as np

from .losses import gamma_loss
from .schedules import cosine_schedule


def build_mst_from_correspondences(corres: list[dict], n_images: int) -> tuple[dict[int, int], int]:
    """Build minimum spanning tree from correspondences.

    Uses sum of confidence weights as edge weights (more confident = better connection).
    This is more robust than just counting correspondences.

    Args:
        corres: List of correspondence dicts with idx1, idx2, and weights
        n_images: Total number of images

    Returns:
        Tuple of (parent_map {child: parent}, root_idx)
    """
    # Build adjacency with weights = sum of confidences (like PyTorch)
    adj: dict[tuple[int, int], float] = {}
    for c in corres:
        i, j = c["idx1"], c["idx2"]
        if i > j:
            i, j = j, i
        key = (i, j)
        # Weight = sum of confidence weights (more reliable than just count)
        weights = c.get("weights", None)
        if weights is not None:
            weight = float(mx.sum(weights)) if isinstance(weights, mx.array) else sum(weights)
        else:
            # Fallback to count if no weights
            n_pts = len(c.get("pts1_idx", c.get("pts1", [])))
            weight = float(n_pts)
        adj[key] = adj.get(key, 0.0) + weight

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


# Match PyTorch structure
PairOfSlices = namedtuple(
    "ImgPair",
    "img1, slice1, pix1, anchor_idxs1, img2, slice2, pix2, anchor_idxs2, confs, confs_sum",
)


def normalize_quat(q: mx.array) -> mx.array:
    """Normalize quaternion."""
    return q / mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-8)


def inv_pose(pose: mx.array) -> mx.array:
    """Invert a 4x4 pose matrix."""
    R = pose[:3, :3]
    t = pose[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    # Build 4x4 matrix directly (MLX doesn't have .at[].set())
    row0 = mx.concatenate([R_inv[0], t_inv[0:1]])
    row1 = mx.concatenate([R_inv[1], t_inv[1:2]])
    row2 = mx.concatenate([R_inv[2], t_inv[2:3]])
    row3 = mx.array([0.0, 0.0, 0.0, 1.0])
    return mx.stack([row0, row1, row2, row3])


def reproj2d(K: mx.array, w2cam: mx.array, pts3d: mx.array) -> mx.array:
    """Project 3D world points to 2D image coordinates.

    Args:
        K: [3, 3] intrinsic matrix
        w2cam: [4, 4] world-to-camera transform
        pts3d: [N, 3] 3D points in world coordinates

    Returns:
        [N, 2] projected 2D points
    """
    # Transform to camera coords
    pts_cam = pts3d @ w2cam[:3, :3].T + w2cam[:3, 3]

    # Project
    z = mx.maximum(pts_cam[:, 2:3], mx.array(1e-6))
    uv = pts_cam[:, :2] / z

    # Apply intrinsics
    proj = uv * mx.array([K[0, 0], K[1, 1]]) + mx.array([K[0, 2], K[1, 2]])

    return mx.clip(proj, -1e4, 1e4)


def proj3d(inv_K: mx.array, pixels: mx.array, z: mx.array) -> mx.array:
    """Unproject 2D pixels to 3D using inverse intrinsics.

    Args:
        inv_K: [3, 3] inverse intrinsics
        pixels: [N, 2] pixel coordinates
        z: [N] depth values

    Returns:
        [N, 3] 3D points in camera coordinates
    """
    # Normalized coordinates: (u - cx) / fx, (v - cy) / fy
    fx, fy = inv_K[0, 0], inv_K[1, 1]
    cx, cy = inv_K[0, 2], inv_K[1, 2]

    x_norm = pixels[:, 0] * fx + cx
    y_norm = pixels[:, 1] * fy + cy

    pts3d = mx.stack([x_norm * z, y_norm * z, z], axis=-1)
    return pts3d


def geotrf(T: mx.array, pts: mx.array) -> mx.array:
    """Apply geometric transform to points.

    Args:
        T: [4, 4] transformation matrix
        pts: [N, 3] points

    Returns:
        [N, 3] transformed points
    """
    return pts @ T[:3, :3].T + T[:3, 3]


def adam_step(
    params: dict,
    grads: dict,
    m: dict,
    v: dict,
    step: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.9,
    eps: float = 1e-8,
):
    """Adam optimizer step, modifies params in-place."""
    for key in params:
        if key not in grads or grads[key] is None:
            continue

        g = grads[key]

        # Update biased first moment estimate
        m[key] = beta1 * m[key] + (1 - beta1) * g
        # Update biased second moment estimate
        v[key] = beta2 * v[key] + (1 - beta2) * (g * g)

        # Bias correction
        m_hat = m[key] / (1 - beta1 ** (step + 1))
        v_hat = v[key] / (1 - beta2 ** (step + 1))

        # Update
        params[key] = params[key] - lr * m_hat / (mx.sqrt(v_hat) + eps)


def sparse_scene_optimizer(
    imgs: list[str],
    imsizes: list[tuple[int, int]],  # [(H, W), ...] image sizes
    pps: list[mx.array],  # Principal points [N, 2]
    base_focals: mx.array,  # [N] initial focal estimates
    core_depth: list[mx.array],  # Subsampled normalized depth maps
    median_depths: mx.array,  # [N] median depths for normalization
    img_anchors: dict,  # {idx: (uv, idxs, offsets)} anchor data
    corres: list[dict],  # [{idx1, idx2, pts1_idx, pts2_idx, weights}, ...]
    preds_21: dict,  # Cross-predictions for dust3r loss
    mst: tuple,  # (root, edges) minimum spanning tree
    subsample: int = 8,
    lr1: float = 0.07,
    niter1: int = 300,
    lr2: float = 0.01,  # PyTorch default
    niter2: int = 300,
    loss1_fn: Callable = None,
    loss2_fn: Callable = None,
    lossd_fn: Callable = None,
    exp_depth: bool = True,
    shared_intrinsics: bool = False,
    matching_conf_thr: float = 5.0,
    loss_dust3r_w: float = 0.01,
    verbose: bool = True,
):
    """PyTorch-faithful sparse scene optimizer.

    This is an exact port of PyTorch MASt3R sparse_scene_optimizer.
    """
    if loss1_fn is None:
        loss1_fn = gamma_loss(1.5)  # Phase 1: 3D loss (PyTorch default)
    if loss2_fn is None:
        loss2_fn = gamma_loss(0.5)  # Phase 2: 2D loss
    if lossd_fn is None:
        lossd_fn = gamma_loss(1.1)  # DUSt3R loss

    n_imgs = len(imgs)

    # Convert imsizes to (H, W) tuples if needed
    imsizes_hw = []
    for sz in imsizes:
        if isinstance(sz, tuple):
            imsizes_hw.append(sz)
        else:
            imsizes_hw.append((int(sz[0]), int(sz[1])))

    # Compute image diagonals for focal constraints
    diags = mx.array([np.sqrt(H**2 + W**2) for H, W in imsizes_hw])
    # Tight focal constraints to prevent divergence in multi-view
    # Allow focals to vary by at most 30% from base estimate
    # This matches typical real camera variation and prevents optimization instability
    min_focals = mx.maximum(0.7 * base_focals, 0.25 * diags)
    max_focals = mx.minimum(1.3 * base_focals, 2.0 * diags)

    # === Initialize parameters exactly like PyTorch ===

    # Quaternions (identity rotation: [0, 0, 0, 1])
    quats = mx.stack([mx.array([0.0, 0.0, 0.0, 1.0]) for _ in range(n_imgs)])

    # Translations (zero)
    trans = mx.stack([mx.zeros(3) for _ in range(n_imgs)])

    # Principal points normalized by image size
    pps_norm = mx.stack(
        [
            pp / mx.array([float(imsizes_hw[i][1]), float(imsizes_hw[i][0])])
            for i, pp in enumerate(pps)
        ]
    )

    # Log focals
    log_focals = mx.log(base_focals)

    # Log sizes (scale factors)
    log_sizes = mx.zeros(n_imgs)

    # Depth parameters
    if exp_depth:
        core_depth_params = mx.concatenate(
            [mx.log(mx.maximum(d.reshape(-1), mx.array(1e-4))) for d in core_depth]
        )
    else:
        core_depth_params = mx.concatenate([d.reshape(-1) for d in core_depth])

    # Track depth boundaries for each image
    depth_bounds = []
    offset = 0
    for i, d in enumerate(core_depth):
        n = d.size
        depth_bounds.append((offset, offset + n))
        offset += n

    # === Build helper functions ===

    def make_K_cam_depth(log_focals, pps_norm, trans, quats, log_sizes, core_depth_params):
        """Build intrinsics, poses, and depthmaps exactly like PyTorch."""
        # Clamp focals
        focals = mx.clip(mx.exp(log_focals), min_focals, max_focals)

        # Build intrinsic matrices
        K_list = []
        for i in range(n_imgs):
            H, W = imsizes_hw[i]
            Ki = mx.array(
                [
                    [focals[i], 0.0, pps_norm[i, 0] * W],
                    [0.0, focals[i], pps_norm[i, 1] * H],
                    [0.0, 0.0, 1.0],
                ]
            )
            K_list.append(Ki)
        K = mx.stack(K_list)

        # Sizes and global scaling (security against scale collapse)
        sizes = mx.exp(log_sizes)
        global_scaling = 1.0 / mx.minimum(mx.min(sizes), mx.array(1.0))

        # z_cameras: reparametrization like PyTorch
        z_cameras = sizes * median_depths * focals / base_focals

        # Build rotation matrices from quaternions
        quats_norm = quats / mx.sqrt(mx.sum(quats**2, axis=-1, keepdims=True) + 1e-8)

        def quat_to_pose(q, t):
            """Convert quaternion and translation to 4x4 pose matrix."""
            x, y, z, w = q[0], q[1], q[2], q[3]
            # Build 4x4 matrix directly
            row0 = mx.array([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y), t[0]])
            row1 = mx.array([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x), t[1]])
            row2 = mx.array([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y), t[2]])
            row3 = mx.array([0.0, 0.0, 0.0, 1.0])
            return mx.stack([row0, row1, row2, row3])

        # Build relative poses
        rel_cam2cam = []
        for i in range(n_imgs):
            T = quat_to_pose(quats_norm[i], trans[i])
            rel_cam2cam.append(T)

        # Compose poses along MST (kinematic chain)
        root, edges = mst
        tmp_cam2w = [None] * n_imgs
        tmp_cam2w[root] = rel_cam2cam[root]
        for parent, child in edges:
            tmp_cam2w[child] = tmp_cam2w[parent] @ rel_cam2cam[child]

        tmp_cam2w = mx.stack(tmp_cam2w)

        # Smart reparametrization of camera translations (like PyTorch)
        trans_offset_list = []
        for i in range(n_imgs):
            H, W = imsizes_hw[i]
            offset_xy = mx.array([W, H]) / focals[i] * (0.5 - pps_norm[i])
            offset = z_cameras[i] * mx.concatenate([offset_xy, mx.ones(1)])
            trans_offset_list.append(offset)
        trans_offset = mx.stack(trans_offset_list)

        new_trans = global_scaling * (
            tmp_cam2w[:, :3, 3] - mx.sum(tmp_cam2w[:, :3, :3] * trans_offset[:, None, :], axis=-1)
        )

        # Build final cam2w matrices
        bottom = mx.broadcast_to(mx.array([[0.0, 0.0, 0.0, 1.0]]), (n_imgs, 1, 4))
        cam2w = mx.concatenate(
            [mx.concatenate([tmp_cam2w[:, :3, :3], new_trans[:, :, None]], axis=2), bottom], axis=1
        )

        # Compute depthmaps with depth_mode='add'
        depthmaps = []
        for i in range(n_imgs):
            start, end = depth_bounds[i]
            cd = core_depth_params[start:end]
            if exp_depth:
                cd = mx.exp(cd)
            # depth_mode='add': depth = z_cameras + (cd - 1) * median * size
            depth = z_cameras[i] + (cd - 1.0) * (median_depths[i] * sizes[i])
            depthmaps.append(global_scaling * depth)

        return K, cam2w, depthmaps, focals

    def make_pts3d(K, cam2w, depthmaps, focals):
        """Build 3D points at anchor positions with focal-compensated depth offsets.

        PyTorch-faithful implementation:
            offsets = 1 + (offsets - 1) * (base_focals[img] / focals[img])
            depth_at_anchor = depthmaps[img][idxs] * offsets
            pts3d = unproject(pixels, depth_at_anchor)
        """
        all_pts3d = {}

        for img_idx in range(n_imgs):
            H, W = imsizes_hw[img_idx]
            H_sub = (H + subsample - 1) // subsample
            W_sub = (W + subsample - 1) // subsample

            # Get flat depth (already transformed with z_cameras + (cd-1)*median*size)
            depth_flat = depthmaps[img_idx]

            # Check if we have anchor data for this image
            if img_idx in img_anchors:
                pixels, idxs, offsets = img_anchors[img_idx]

                # Focal compensation (PyTorch formula)
                focal_ratio = base_focals[img_idx] / focals[img_idx]
                offsets_compensated = 1.0 + (offsets - 1.0) * focal_ratio

                # Get depth at anchor indices and apply offset (PyTorch formula)
                idxs_clipped = mx.clip(idxs.astype(mx.int32), 0, len(depth_flat) - 1)
                depth_at_anchors = depth_flat[idxs_clipped] * offsets_compensated

                # Unproject to 3D camera coordinates
                fx, fy = K[img_idx, 0, 0], K[img_idx, 1, 1]
                cx, cy = K[img_idx, 0, 2], K[img_idx, 1, 2]

                x_norm = (pixels[:, 0] - cx) / fx
                y_norm = (pixels[:, 1] - cy) / fy
                pts_cam = mx.stack([
                    x_norm * depth_at_anchors,
                    y_norm * depth_at_anchors,
                    depth_at_anchors,
                ], axis=-1)

                # Transform to world coordinates
                pts3d_world = geotrf(cam2w[img_idx], pts_cam)
                all_pts3d[img_idx] = pts3d_world
            else:
                # Fallback: reconstruct on regular grid (for visualization)
                ys = mx.arange(H_sub) * subsample + subsample // 2
                xs = mx.arange(W_sub) * subsample + subsample // 2
                yy, xx = mx.meshgrid(ys, xs, indexing="ij")
                pixels = mx.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(mx.float32)

                fx, fy = K[img_idx, 0, 0], K[img_idx, 1, 1]
                cx, cy = K[img_idx, 0, 2], K[img_idx, 1, 2]

                x_norm = (pixels[:, 0] - cx) / fx
                y_norm = (pixels[:, 1] - cy) / fy
                pts_cam = mx.stack([x_norm * depth_flat, y_norm * depth_flat, depth_flat], axis=-1)

                pts3d_world = geotrf(cam2w[img_idx], pts_cam)
                all_pts3d[img_idx] = pts3d_world

        return all_pts3d

    def compute_loss_3d(K, cam2w, pts3d, loss_fn):
        """3D matching loss between corresponding points (PyTorch-faithful).

        Uses slices into anchor-aggregated pts3d, like PyTorch MASt3R loss_3d.
        Only uses correspondences where max_conf > matching_conf_thr (like PyTorch).

        PyTorch concatenates ALL correspondences and normalizes GLOBALLY:
            confs = torch.cat(confs)
            loss = confs @ pix_loss(pts3d_1, pts3d_2) / confs.sum()
        """
        if not corres:
            return mx.array(0.0)

        all_pts1 = []
        all_pts2 = []
        all_confs = []

        for c in corres:
            # Filter by matching confidence (like PyTorch is_matching_ok)
            max_conf = c.get("max_conf", float("inf"))
            if max_conf <= matching_conf_thr:
                continue

            idx1, idx2 = c["idx1"], c["idx2"]
            slice1 = c["slice1"]  # Slice into pts3d[idx1]
            slice2 = c["slice2"]  # Slice into pts3d[idx2]
            weights = c["weights"]

            if idx1 not in pts3d or idx2 not in pts3d:
                continue

            # Get 3D points using slices (like PyTorch)
            p1 = pts3d[idx1][slice1]
            p2 = pts3d[idx2][slice2]

            # Ensure same length (correspondences should match)
            n = min(len(p1), len(p2), len(weights))
            if n == 0:
                continue

            all_pts1.append(p1[:n])
            all_pts2.append(p2[:n])
            all_confs.append(weights[:n])

        if not all_pts1:
            return mx.array(0.0)

        # Concatenate all (like PyTorch torch.cat)
        pts3d_1 = mx.concatenate(all_pts1, axis=0)
        pts3d_2 = mx.concatenate(all_pts2, axis=0)
        confs = mx.concatenate(all_confs, axis=0)

        # Global weighted loss (PyTorch: confs @ pix_loss / confs.sum())
        dist = loss_fn(pts3d_1, pts3d_2)
        cf_sum = mx.sum(confs)
        loss = mx.sum(confs * dist) / mx.maximum(cf_sum, mx.array(1e-8))

        return loss

    def compute_loss_2d(K, cam2w, pts3d, loss_fn):
        """2D reprojection loss (PyTorch-faithful).

        PyTorch iterates over target images and accumulates globally:
            for img1, pix1, confs, cf_sum, slices in cleaned_corres2d:
                pts3d_in_img1 = cat([pts3d[img2][slice2] for img2, slice2 in slices])
                loss += confs @ pix_loss(pix1, reproj2d(pts3d_in_img1))
                npix += confs.sum()
            return loss / npix

        We implement bidirectional for better symmetry in multi-view.
        """
        if not corres:
            return mx.array(0.0)

        total_loss = mx.array(0.0)
        total_npix = mx.array(0.0)

        for c in corres:
            # Filter by matching confidence (like PyTorch is_matching_ok)
            max_conf = c.get("max_conf", float("inf"))
            if max_conf <= matching_conf_thr:
                continue

            idx1, idx2 = c["idx1"], c["idx2"]
            slice1 = c["slice1"]  # Slice into pts3d[idx1]
            slice2 = c["slice2"]  # Slice into pts3d[idx2]
            pts1 = c["pts1"]  # 2D pixel coords in img1
            pts2 = c["pts2"]  # 2D pixel coords in img2
            weights = c["weights"]

            if idx1 not in pts3d or idx2 not in pts3d:
                continue

            # Get 3D points using slices
            p1_3d = pts3d[idx1][slice1]  # 3D points from img1
            p2_3d = pts3d[idx2][slice2]  # 3D points from img2

            # Ensure same length
            n = min(len(p1_3d), len(p2_3d), len(pts1), len(pts2), len(weights))
            if n == 0:
                continue

            p1_3d = p1_3d[:n]
            p2_3d = p2_3d[:n]
            pix1 = pts1[:n]
            pix2 = pts2[:n]
            w = weights[:n]

            # BIDIRECTIONAL 2D loss for better multi-view consistency
            # Direction 1: Project p2_3d to img1
            w2cam1 = inv_pose(cam2w[idx1])
            proj_to_1 = reproj2d(K[idx1], w2cam1, p2_3d)
            dist1 = loss_fn(pix1, proj_to_1)

            # Direction 2: Project p1_3d to img2
            w2cam2 = inv_pose(cam2w[idx2])
            proj_to_2 = reproj2d(K[idx2], w2cam2, p1_3d)
            dist2 = loss_fn(pix2, proj_to_2)

            # Accumulate globally (like PyTorch)
            total_loss = total_loss + mx.sum(w * dist1) + mx.sum(w * dist2)
            total_npix = total_npix + 2.0 * mx.sum(w)  # Count both directions

        if float(total_npix) < 1e-8:
            return mx.array(0.0)

        return total_loss / total_npix

    # Build set of low-confidence pairs (for dust3r loss filtering)
    # PyTorch only applies loss_dust3r to pairs where is_matching_ok is False
    low_conf_pairs = set()
    for c in corres:
        max_conf = c.get("max_conf", float("inf"))
        if max_conf <= matching_conf_thr:
            low_conf_pairs.add((c["idx1"], c["idx2"]))
            low_conf_pairs.add((c["idx2"], c["idx1"]))  # Both directions

    def compute_loss_dust3r(cam2w, pts3d, loss_fn):
        """DUSt3R loss using cross-predictions (only for low-confidence pairs).

        PyTorch-faithful: Only applies to pairs in dust3r_slices (low confidence).
        """
        if not preds_21:
            return mx.array(0.0)

        total_loss = mx.array(0.0)
        total_conf = mx.array(0.0)

        for img1_path, preds_from_img1 in preds_21.items():
            try:
                idx1 = imgs.index(img1_path)
            except ValueError:
                continue

            for img2_path, (tgt_pts, tgt_confs) in preds_from_img1.items():
                try:
                    idx2 = imgs.index(img2_path)
                except ValueError:
                    continue

                # PyTorch-faithful: Only apply to low-confidence pairs
                if (idx1, idx2) not in low_conf_pairs:
                    continue

                if idx1 not in pts3d:
                    continue

                # Get reconstructed pts3d for idx1
                pts3d_1 = pts3d[idx1]

                # Transform target predictions to world coords
                tgt_pts_world = geotrf(cam2w[idx2], tgt_pts)

                # Match sizes
                n_pts = min(pts3d_1.shape[0], tgt_pts_world.shape[0], tgt_confs.shape[0])
                if n_pts == 0:
                    continue

                p1 = pts3d_1[:n_pts]
                p2 = tgt_pts_world[:n_pts]
                cf = tgt_confs[:n_pts]

                dist = loss_fn(p1, p2)
                total_loss = total_loss + mx.sum(cf * dist)
                total_conf = total_conf + mx.sum(cf)

        return total_loss / mx.maximum(total_conf, mx.array(1e-8))

    # === Phase 1: Coarse alignment (3D loss) ===
    # IMPORTANT: PyTorch does NOT optimize focals or depths in Phase 1
    # Only quats, trans, and log_sizes are trainable

    if verbose:
        print(f"Phase 1: Coarse alignment ({niter1} iterations, lr={lr1})")

    # Parameters for Phase 1: ONLY poses and sizes (like PyTorch)
    # log_focals and core_depth are FROZEN
    params_p1 = {
        "trans": trans,
        "quats": quats,
        "log_sizes": log_sizes,
    }

    # Frozen parameters (not optimized)
    frozen_log_focals = log_focals
    frozen_core_depth = core_depth_params

    # Adam moments
    m_p1 = {k: mx.zeros_like(v) for k, v in params_p1.items()}
    v_p1 = {k: mx.zeros_like(v) for k, v in params_p1.items()}

    def loss_phase1(trans, quats, log_sizes):
        """Total loss for Phase 1 (focals and depths are frozen)."""
        K, cam2w, depthmaps, focals = make_K_cam_depth(
            frozen_log_focals, pps_norm, trans, quats, log_sizes, frozen_core_depth
        )
        pts3d = make_pts3d(K, cam2w, depthmaps, focals)

        loss_3d = compute_loss_3d(K, cam2w, pts3d, loss1_fn)
        loss_d = compute_loss_dust3r(cam2w, pts3d, lossd_fn)

        return loss_3d + loss_dust3r_w * loss_d

    # Optimization loop Phase 1
    for step in range(niter1):
        alpha = step / max(niter1 - 1, 1)
        lr = cosine_schedule(alpha, lr1, 0)  # PyTorch uses lr_end=0

        # Compute loss and gradients (only for trainable params)
        loss_val, grads = mx.value_and_grad(loss_phase1, argnums=(0, 1, 2))(
            params_p1["trans"],
            params_p1["quats"],
            params_p1["log_sizes"],
        )

        grad_dict = {
            "trans": grads[0],
            "quats": grads[1],
            "log_sizes": grads[2],
        }

        # Adam update
        adam_step(params_p1, grad_dict, m_p1, v_p1, step, lr, beta1=0.9, beta2=0.9)

        # Normalize quaternions after each step (like PyTorch)
        # This ensures quats remain valid rotation representations
        quats_normalized = params_p1["quats"] / mx.sqrt(
            mx.sum(params_p1["quats"] ** 2, axis=-1, keepdims=True) + 1e-8
        )
        params_p1["quats"] = quats_normalized

        # Force evaluation to prevent graph explosion
        mx.eval(*params_p1.values(), *m_p1.values(), *v_p1.values())

        if verbose and step % 50 == 0:
            focals_now = mx.clip(mx.exp(frozen_log_focals), min_focals, max_focals)
            print(
                f"  Step {step:3d}: loss={float(loss_val):.4f}, focals={[float(f) for f in focals_now]} (frozen)"
            )

    # === Phase 2: Fine alignment (2D loss) ===
    # Now we CAN optimize focals and depths (like PyTorch)

    if verbose:
        print(f"\nPhase 2: Fine alignment ({niter2} iterations, lr={lr2})")

    # Parameters for Phase 2: NOW include focals and depths
    # NOTE: pps_norm is NOT optimized (like PyTorch default opt_pp=False)
    # Optimizing pp can lead to divergence, especially for portrait images
    frozen_pps_norm = pps_norm
    params_p2 = {
        "log_focals": frozen_log_focals,  # Now trainable
        "trans": params_p1["trans"],
        "quats": params_p1["quats"],
        "log_sizes": params_p1["log_sizes"],
        "core_depth": frozen_core_depth,  # Now trainable
    }

    m_p2 = {k: mx.zeros_like(v) for k, v in params_p2.items()}
    v_p2 = {k: mx.zeros_like(v) for k, v in params_p2.items()}

    def loss_phase2(log_focals, trans, quats, log_sizes, core_depth):
        """Total loss for Phase 2."""
        K, cam2w, depthmaps, focals = make_K_cam_depth(
            log_focals, frozen_pps_norm, trans, quats, log_sizes, core_depth
        )
        pts3d = make_pts3d(K, cam2w, depthmaps, focals)

        loss_2d = compute_loss_2d(K, cam2w, pts3d, loss2_fn)
        loss_d = compute_loss_dust3r(cam2w, pts3d, lossd_fn)

        return loss_2d + loss_dust3r_w * loss_d

    # Optimization loop Phase 2
    for step in range(niter2):
        alpha = step / max(niter2 - 1, 1)
        lr = cosine_schedule(alpha, lr2, 0)  # PyTorch uses lr_end=0

        # Compute loss and gradients (pps_norm is frozen, only 5 params)
        loss_val, grads = mx.value_and_grad(loss_phase2, argnums=(0, 1, 2, 3, 4))(
            params_p2["log_focals"],
            params_p2["trans"],
            params_p2["quats"],
            params_p2["log_sizes"],
            params_p2["core_depth"],
        )

        grad_dict = {
            "log_focals": grads[0],
            "trans": grads[1],
            "quats": grads[2],
            "log_sizes": grads[3],
            "core_depth": grads[4],
        }

        # Adam update
        adam_step(params_p2, grad_dict, m_p2, v_p2, step, lr, beta1=0.9, beta2=0.9)

        # Force evaluation to prevent graph explosion
        mx.eval(*params_p2.values(), *m_p2.values(), *v_p2.values())

        if verbose and step % 50 == 0:
            print(f"  Step {step:3d}: loss={float(loss_val):.4f} px")

    # === Final results ===

    K, cam2w, depthmaps, focals = make_K_cam_depth(
        params_p2["log_focals"],
        frozen_pps_norm,  # Use frozen principal points
        params_p2["trans"],
        params_p2["quats"],
        params_p2["log_sizes"],
        params_p2["core_depth"],
    )
    pts3d = make_pts3d(K, cam2w, depthmaps, focals)

    # Convert depthmaps to 2D
    depthmaps_2d = []
    for i in range(n_imgs):
        H, W = imsizes_hw[i]
        H_sub = (H + subsample - 1) // subsample
        W_sub = (W + subsample - 1) // subsample
        d = depthmaps[i].reshape(H_sub, W_sub)
        depthmaps_2d.append(d)

    if verbose:
        print(f"\nFinal focals: {[float(f) for f in focals]}")
        print(f"Final sizes: {[float(mx.exp(s)) for s in params_p2['log_sizes']]}")

    return {
        "intrinsics": K,
        "cam2w": cam2w,
        "depthmaps": depthmaps_2d,
        "focals": focals,
        "pts3d": pts3d,
        "pps_norm": frozen_pps_norm,  # Use frozen principal points
        "log_sizes": params_p2["log_sizes"],
    }
