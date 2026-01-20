# Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""Correspondence extraction and matching for sparse global alignment.

Implements reciprocal nearest neighbor matching for descriptor-based correspondences.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def fast_reciprocal_nns(
    A: np.ndarray, B: np.ndarray, subsample: int
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative reciprocal nearest neighbor matching like PyTorch fast_reciprocal_NNs.

    Uses full-resolution features for matching but starts from subsampled points.

    Args:
        A: Features from image 1 [H1, W1, D]
        B: Features from image 2 [H2, W2, D]
        subsample: Initial subsampling factor

    Returns:
        Tuple of (idx1, idx2) flat indices into A and B
    """
    H1, W1, D = A.shape
    H2, W2, D2 = B.shape
    assert D == D2

    A_flat = A.reshape(-1, D)
    B_flat = B.reshape(-1, D)

    # Normalize for dot product
    A_norm = A_flat / (np.linalg.norm(A_flat, axis=-1, keepdims=True) + 1e-8)
    B_norm = B_flat / (np.linalg.norm(B_flat, axis=-1, keepdims=True) + 1e-8)

    # Start from subsampled points
    S = subsample
    y1, x1 = np.mgrid[S // 2 : H1 : S, S // 2 : W1 : S].reshape(2, -1)
    xy1 = np.int32(np.unique(x1 + W1 * y1))  # Flat indices into A
    xy2 = np.full_like(xy1, -1)  # Matching indices in B

    max_iter = 10
    old_xy1 = xy1.copy()
    notyet = np.ones(len(xy1), dtype=bool)

    for _ in range(max_iter):
        if not notyet.any():
            break

        # Find best match in B for each point in A
        sims = A_norm[xy1[notyet]] @ B_norm.T
        xy2[notyet] = np.argmax(sims, axis=1)

        # Find best match in A for each matched point in B
        sims_back = B_norm[xy2[notyet]] @ A_norm.T
        xy1[notyet] = np.argmax(sims_back, axis=1)

        # Check convergence
        notyet &= old_xy1 != xy1
        old_xy1[:] = xy1

    # Keep only converged (reciprocal) matches
    converged = ~notyet
    return xy1[converged], xy2[converged]


def merge_corres(
    idx1: np.ndarray, idx2: np.ndarray, shape1: tuple, shape2: tuple
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge correspondences and convert to xy coordinates.

    Like PyTorch merge_corres function.

    Args:
        idx1: Flat indices into image 1
        idx2: Flat indices into image 2
        shape1: (H1, W1) of image 1
        shape2: (H2, W2) of image 2

    Returns:
        Tuple of (xy1, xy2, indices) - indices map to original arrays
    """
    idx1 = idx1.astype(np.int32)
    idx2 = idx2.astype(np.int32)

    # Unique and sort along idx1, return indices
    combined = np.c_[idx2, idx1].view(np.int64)
    unique_combined, indices = np.unique(combined, return_index=True)
    xy2_flat, xy1_flat = unique_combined[:, None].view(np.int32).T

    # Convert to xy coordinates
    y1, x1 = np.unravel_index(xy1_flat, shape1)
    y2, x2 = np.unravel_index(xy2_flat, shape2)

    xy1 = np.stack([x1, y1], axis=-1).astype(np.float32)
    xy2 = np.stack([x2, y2], axis=-1).astype(np.float32)

    return xy1, xy2, indices


def extract_correspondences(
    feats: list[mx.array],
    qonfs: list[mx.array],
    subsample: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract correspondences from descriptor features.

    Uses iterative reciprocal nearest neighbor matching like PyTorch MASt3R.
    Works on full-resolution features for better matching quality.

    Args:
        feats: List of descriptor features [feat11, feat21, feat22, feat12]
        qonfs: List of confidence maps
        subsample: Initial subsampling factor

    Returns:
        Tuple of (xy1, xy2, confidences)
    """
    feat11, feat21, feat22, feat12 = feats
    qonf11, qonf21, qonf22, qonf12 = qonfs

    H1, W1 = feat11.shape[:2]
    H2, W2 = feat22.shape[:2]

    # Convert to numpy for matching
    feat11_np = np.array(feat11)
    feat21_np = np.array(feat21)
    feat22_np = np.array(feat22)
    feat12_np = np.array(feat12)

    qonf11_np = np.array(qonf11).ravel()
    qonf21_np = np.array(qonf21).ravel()
    qonf22_np = np.array(qonf22).ravel()
    qonf12_np = np.array(qonf12).ravel()

    all_idx1 = []
    all_idx2 = []
    all_qonf1 = []
    all_qonf2 = []

    # Match both pairs in both directions (like PyTorch)
    for A, B, QA, QB in [
        (feat11_np, feat21_np, qonf11_np, qonf21_np),
        (feat12_np, feat22_np, qonf12_np, qonf22_np),
    ]:
        # Forward matching: A → B
        nn1to2_idx1, nn1to2_idx2 = fast_reciprocal_nns(A, B, subsample)
        # Backward matching: B → A
        nn2to1_idx2, nn2to1_idx1 = fast_reciprocal_nns(B, A, subsample)

        # Concatenate both directions (like PyTorch)
        all_idx1.append(np.r_[nn1to2_idx1, nn2to1_idx1])
        all_idx2.append(np.r_[nn1to2_idx2, nn2to1_idx2])
        all_qonf1.append(QA[np.r_[nn1to2_idx1, nn2to1_idx1]])
        all_qonf2.append(QB[np.r_[nn1to2_idx2, nn2to1_idx2]])

    # Merge all correspondences
    idx1 = np.concatenate(all_idx1).astype(np.int32)
    idx2 = np.concatenate(all_idx2).astype(np.int32)
    qonf1 = np.concatenate(all_qonf1)
    qonf2 = np.concatenate(all_qonf2)

    # Merge and deduplicate (like PyTorch merge_corres)
    xy1, xy2, merge_indices = merge_corres(idx1, idx2, (H1, W1), (H2, W2))

    # Use the merged confidences exactly like PyTorch
    if len(xy1) > 0:
        # PyTorch: confs = np.sqrt(cat(qonf1)[idx] * cat(qonf2)[idx])
        confs = np.sqrt(qonf1[merge_indices] * qonf2[merge_indices])
    else:
        confs = np.zeros(0, dtype=np.float32)

    return xy1, xy2, confs


def anchor_depth_offsets(
    canon_depth: mx.array,
    pixels: dict[str, tuple[mx.array, mx.array, mx.array]],
    subsample: int = 8,
) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
    """Compute depth offsets for anchor points.

    Matches PyTorch MASt3R anchor_depth_offsets function.
    For each correspondence pixel, computes:
    - The index into the subsampled core depth
    - The ratio offset = pixel_depth / core_depth

    Args:
        canon_depth: Full canonical depth map [H, W]
        pixels: Dict mapping img2 -> (pts1, pts2, conf) for correspondences
        subsample: Subsampling factor

    Returns:
        Tuple of (core_idxs, core_offs) dicts, both keyed by img2
    """
    H1, W1 = canon_depth.shape
    H_sub = (H1 + subsample - 1) // subsample
    W_sub = (W1 + subsample - 1) // subsample

    # Get subsampled depth at anchor centers
    core_depth_sub = canon_depth[subsample // 2 :: subsample, subsample // 2 :: subsample]
    core_depth_flat = core_depth_sub.reshape(-1)

    # Ensure positive depth
    core_depth_flat = mx.maximum(core_depth_flat, mx.array(1e-6))

    core_idxs = {}
    core_offs = {}

    for img2, pixel_data in pixels.items():
        # Unpack (pts1, pts2, conf) or (pts1, conf)
        if len(pixel_data) == 3:
            xy1, xy2, confs = pixel_data
        else:
            xy1, confs = pixel_data
            xy2 = xy1

        # Get pixel coordinates as integers
        px = xy1[:, 0].astype(mx.int32)
        py = xy1[:, 1].astype(mx.int32)

        # Clip to valid range
        px = mx.clip(px, 0, W1 - 1)
        py = mx.clip(py, 0, H1 - 1)

        # Find nearest anchor (block quantization)
        core_idx = (py // subsample) * W_sub + (px // subsample)
        core_idx = mx.clip(core_idx, 0, H_sub * W_sub - 1)

        # Get reference depth at anchor
        ref_z = core_depth_flat[core_idx]

        # Get actual depth at pixel
        pts_z = canon_depth[py, px]

        # Compute offset ratio
        offset = pts_z / (ref_z + 1e-8)

        core_idxs[img2] = core_idx
        core_offs[img2] = offset

    return core_idxs, core_offs
