"""Sparse Global Alignment for MASt3R with MLX.

Implements the main pipeline for multi-view reconstruction using
sparse correspondences and global optimization.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .geometry import depthmap_to_pts3d, geotrf, inv
from .optimizer import build_mst_from_correspondences
from .optimizer import sparse_scene_optimizer as _sparse_optimizer_core


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
    # PyTorch: yx = np.mgrid[subsample//2:H1:subsample, subsample//2:W1:subsample]
    # cy, cx = yx.reshape(2, -1)
    # core_depth = canon_depth[cy, cx]
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
        # core_idx = (py // subsample) * W_sub + (px // subsample)
        core_idx = (py // subsample) * W_sub + (px // subsample)
        core_idx = mx.clip(core_idx, 0, H_sub * W_sub - 1)

        # Get reference depth at anchor
        ref_z = core_depth_flat[core_idx]

        # Get actual depth at pixel
        # Need to sample from full depth map
        pts_z = canon_depth[py, px]

        # Compute offset ratio
        offset = pts_z / (ref_z + 1e-8)

        core_idxs[img2] = core_idx
        core_offs[img2] = offset

    return core_idxs, core_offs


@dataclass
class SparseGAResult:
    """Result container for sparse global alignment."""

    imgs: list[np.ndarray]
    img_paths: list[str]
    focals: mx.array  # [N] optimized focals
    principal_points: mx.array  # [N, 2]
    cam2w: mx.array  # [N, 4, 4] camera-to-world (with global scaling applied)
    depthmaps: list[mx.array]  # [H, W] per image (with global scaling applied)
    pts3d: list[mx.array]  # Sparse 3D points per image
    pts3d_colors: list[np.ndarray]  # Colors for 3D points
    confs: list[mx.array]  # Confidence maps per image
    canonical_paths: list[str] | None  # Cache paths
    base_focals: mx.array | None = None  # [N] initial focal estimates
    sizes: mx.array | None = None  # [N] optimized scale factors per view

    @property
    def n_imgs(self) -> int:
        return len(self.imgs)

    def get_focals(self) -> mx.array:
        return self.focals

    def get_principal_points(self) -> mx.array:
        return self.principal_points

    def get_im_poses(self) -> mx.array:
        return self.cam2w

    def get_sparse_pts3d(self) -> list[mx.array]:
        return self.pts3d

    def get_pts3d_colors(self) -> list[np.ndarray]:
        return self.pts3d_colors

    def get_depthmaps(self) -> list[mx.array]:
        return self.depthmaps

    def get_masks(self) -> list[slice]:
        return [slice(None) for _ in range(self.n_imgs)]

    def get_dense_pts3d(
        self,
        clean_depth: bool = True,
        subsample: int = 8,
    ) -> tuple[list[mx.array], list[mx.array], list[mx.array]]:
        """Get dense 3D points from depthmaps.

        The optimizer already applies global_scaling to both poses and depths,
        ensuring consistent scale between translations and depth values.

        Args:
            clean_depth: Apply depth cleaning
            subsample: Subsampling factor

        Returns:
            Tuple of (pts3d, depthmaps, confs) lists
        """
        pts3d_list = []
        depth_list = []

        for i in range(self.n_imgs):
            H, W = self.depthmaps[i].shape
            depth = self.depthmaps[i]

            # Use optimized focals for unprojection
            f = self.focals[i]
            pp = self.principal_points[i]
            K = mx.array(
                [
                    [f, 0, pp[0]],
                    [0, f, pp[1]],
                    [0, 0, 1],
                ]
            )

            # Unproject to 3D
            pts3d = depthmap_to_pts3d(depth, K)

            # Transform to world coordinates using optimized poses
            # Poses already have global_scaling applied to translations
            cam2w = self.cam2w[i]

            pts3d_world = geotrf(cam2w, pts3d.reshape(-1, 3)).reshape(H, W, 3)

            pts3d_list.append(pts3d_world)
            depth_list.append(depth)

        return pts3d_list, depth_list, self.confs


def hash_md5(s: str) -> str:
    """Create MD5 hash of string."""
    return hashlib.md5(s.encode()).hexdigest()[:16]


def mkdir_for(path: str) -> str:
    """Create directory for file path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def sparse_global_alignment(
    imgs: list[str],
    pairs_in: list[tuple[dict, dict]],
    cache_path: str,
    model: AsymmetricMASt3R,
    subsample: int = 8,
    desc_conf: str = "desc_conf",
    shared_intrinsics: bool = False,
    lr1: float = 0.07,
    niter1: int = 300,
    lr2: float = 0.01,  # PyTorch default
    niter2: int = 300,
    matching_conf_thr: float = 5.0,
    loss_dust3r_w: float = 0.01,  # PyTorch default
    verbose: bool = True,
) -> SparseGAResult:
    """Sparse alignment with MASt3R MLX.

    Main entry point for multi-view reconstruction.

    Args:
        imgs: List of image paths
        pairs_in: List of (img1, img2) dicts from make_pairs
        cache_path: Directory for caching intermediate results
        model: MLX MASt3R model
        subsample: Subsampling factor for correspondences
        desc_conf: Descriptor confidence type
        shared_intrinsics: Use single intrinsics for all cameras
        lr1, niter1: Coarse alignment parameters
        lr2, niter2: Fine refinement parameters
        matching_conf_thr: Minimum matching confidence threshold
        loss_dust3r_w: Weight for DUSt3R loss regularization (default 0.01)
        verbose: Print progress

    Returns:
        SparseGAResult with optimized scene
    """
    # Convert pair naming convention
    pairs_in = convert_dust3r_pairs_naming(imgs, pairs_in)

    if verbose:
        print(f"Processing {len(imgs)} images with {len(pairs_in)} pairs")

    # Forward pass through model
    pairs_data, cache_path = forward_mast3r(
        pairs_in,
        model,
        cache_path=cache_path,
        subsample=subsample,
        desc_conf=desc_conf,
        verbose=verbose,
    )

    # Extract canonical pointmaps
    (
        tmp_pairs,
        pairwise_scores,
        canonical_views,
        canonical_paths,
        preds_21,
    ) = prepare_canonical_data(
        imgs,
        pairs_data,
        subsample=subsample,
        cache_path=cache_path,
        verbose=verbose,
    )

    # Condense all data
    (
        imsizes,
        pps,
        base_focals,
        core_depth,
        img_confs,
        anchors,
        anchor_data,  # NEW: anchor data for make_pts3d_from_depth
        corres,
        corres2d,
    ) = condense_data(
        imgs,
        tmp_pairs,
        canonical_views,
        preds_21,
    )

    # Run optimization
    # Note: MST is built inside sparse_scene_optimizer from correspondences
    result = sparse_scene_optimizer(
        imgs=imgs,
        pairs_in=pairs_in,
        subsample=subsample,
        imsizes=imsizes,
        pps=pps,
        base_focals=base_focals,
        core_depth=core_depth,
        img_confs=img_confs,
        anchors=anchors,
        anchor_data=anchor_data,
        corres=corres,
        canonical_paths=canonical_paths,
        preds_21=preds_21,
        lr1=lr1,
        niter1=niter1,
        lr2=lr2,
        niter2=niter2,
        shared_intrinsics=shared_intrinsics,
        matching_conf_thr=matching_conf_thr,
        loss_dust3r_w=loss_dust3r_w,
        verbose=verbose,
    )

    return result


def convert_dust3r_pairs_naming(
    imgs: list[str],
    pairs_in: list[tuple[dict, dict]],
) -> list[tuple[dict, dict]]:
    """Convert pair naming to use instance paths."""
    for pair in pairs_in:
        for i in range(2):
            pair[i]["instance"] = imgs[pair[i]["idx"]]
    return pairs_in


def forward_mast3r(
    pairs: list[tuple[dict, dict]],
    model: AsymmetricMASt3R,
    cache_path: str,
    desc_conf: str = "desc_conf",
    subsample: int = 8,
    verbose: bool = True,
) -> tuple[dict, str]:
    """Run MASt3R forward pass on all pairs.

    Args:
        pairs: List of image pairs
        model: MLX MASt3R model
        cache_path: Cache directory
        desc_conf: Descriptor confidence type
        subsample: Subsampling factor
        verbose: Print progress

    Returns:
        Tuple of (results dict, cache_path)
    """
    res_paths = {}

    for idx, (img1, img2) in enumerate(pairs):
        if verbose and (idx % 5 == 0 or idx == len(pairs) - 1):
            print(f"  Processing pair {idx + 1}/{len(pairs)}")

        idx1 = hash_md5(img1["instance"])
        idx2 = hash_md5(img2["instance"])

        path1 = f"{cache_path}/forward/{idx1}/{idx2}.npz"
        path2 = f"{cache_path}/forward/{idx2}/{idx1}.npz"
        path_corres = f"{cache_path}/corres_{desc_conf}_{subsample}/{idx1}-{idx2}.npz"

        # Check cache
        if all(os.path.isfile(p) for p in (path1, path2, path_corres)):
            res_paths[img1["instance"], img2["instance"]] = (path1, path2), path_corres
            continue

        if model is None:
            continue

        # Run symmetric inference
        res = symmetric_inference(model, img1, img2)

        X11, X21, X22, X12 = [r["pts3d"] for r in res]
        C11, C21, C22, C12 = [r["conf"] for r in res]
        descs = [r["desc"] for r in res]
        qonfs = [r[desc_conf] for r in res]

        # Save results - use uniform key names for both files
        np.savez(
            mkdir_for(path1),
            X1=np.array(X11),
            C1=np.array(C11),
            X2=np.array(X21),
            C2=np.array(C21),
        )
        np.savez(
            mkdir_for(path2),
            X1=np.array(X22),
            C1=np.array(C22),
            X2=np.array(X12),
            C2=np.array(C12),
        )

        # Extract correspondences
        corres = extract_correspondences(descs, qonfs, subsample=subsample)

        # Compute matching score
        conf_score = np.sqrt(
            np.sqrt(
                float(mx.mean(C11))
                * float(mx.mean(C12))
                * float(mx.mean(C21))
                * float(mx.mean(C22))
            )
        )
        matching_score = (conf_score, float(np.sum(corres[2])), len(corres[2]))

        np.savez(
            mkdir_for(path_corres),
            score=matching_score,
            xy1=corres[0],
            xy2=corres[1],
            confs=corres[2],
        )

        res_paths[img1["instance"], img2["instance"]] = (path1, path2), path_corres

    return res_paths, cache_path


def symmetric_inference(
    model,
    img1: dict,
    img2: dict,
) -> tuple[dict, dict, dict, dict]:
    """Run symmetric forward pass.

    Computes both (1→2) and (2→1) predictions.

    Args:
        model: MLX MASt3R model (DuneMast3r, Mast3rFull, etc.)
        img1, img2: Image dicts with 'img' and 'true_shape'

    Returns:
        Tuple of (res11, res21, res22, res12)
    """
    # Extract numpy images from dicts
    # img['img'] is [1, C, H, W] tensor, need to convert to [H, W, C] numpy
    def to_numpy_image(img_dict: dict) -> np.ndarray:
        img_tensor = img_dict["img"]
        if hasattr(img_tensor, "shape"):
            # Convert from [1, C, H, W] or [C, H, W] to [H, W, C]
            img_np = np.array(img_tensor)
            if img_np.ndim == 4:
                img_np = img_np[0]  # Remove batch dim
            if img_np.shape[0] == 3:  # CHW -> HWC
                img_np = img_np.transpose(1, 2, 0)
            # Denormalize if needed (from [-1, 1] to [0, 255])
            if img_np.min() < 0:
                img_np = ((img_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
            elif img_np.max() <= 1.0:
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            return img_np
        return img_tensor

    np_img1 = to_numpy_image(img1)
    np_img2 = to_numpy_image(img2)

    # Forward pass 1→2 using model.reconstruct()
    out1_12, out2_12 = model.reconstruct(np_img1, np_img2)

    # Forward pass 2→1
    out1_21, out2_21 = model.reconstruct(np_img2, np_img1)

    # Build result dicts
    # res11: view 1 in its own frame (from 1→2 pass)
    # res21: view 2 in view 1's frame (from 1→2 pass)
    # res22: view 2 in its own frame (from 2→1 pass)
    # res12: view 1 in view 2's frame (from 2→1 pass)

    def get_conf(out, key="conf"):
        """Get confidence map, with fallback to uniform confidence."""
        conf = out.get(key, None)
        if conf is None:
            # Fallback: try 'conf' if key was 'desc_conf'
            if key == "desc_conf":
                conf = out.get("conf", None)
            if conf is None:
                # Final fallback: uniform confidence
                pts3d = out["pts3d"]
                return np.ones(pts3d.shape[:2], dtype=np.float32)
        return np.array(conf).squeeze() if hasattr(conf, "__array__") else conf

    res11 = {
        "pts3d": mx.array(out1_12["pts3d"]),
        "conf": mx.array(get_conf(out1_12, "conf")),
        "desc": mx.array(out1_12.get("desc", np.zeros((*out1_12["pts3d"].shape[:2], 24)))),
        "desc_conf": mx.array(get_conf(out1_12, "desc_conf")),  # Use desc_conf!
    }
    res21 = {
        "pts3d": mx.array(out2_12["pts3d"]),
        "conf": mx.array(get_conf(out2_12, "conf")),
        "desc": mx.array(out2_12.get("desc", np.zeros((*out2_12["pts3d"].shape[:2], 24)))),
        "desc_conf": mx.array(get_conf(out2_12, "desc_conf")),  # Use desc_conf!
    }
    res22 = {
        "pts3d": mx.array(out1_21["pts3d"]),
        "conf": mx.array(get_conf(out1_21, "conf")),
        "desc": mx.array(out1_21.get("desc", np.zeros((*out1_21["pts3d"].shape[:2], 24)))),
        "desc_conf": mx.array(get_conf(out1_21, "desc_conf")),  # Use desc_conf!
    }
    res12 = {
        "pts3d": mx.array(out2_21["pts3d"]),
        "conf": mx.array(get_conf(out2_21, "conf")),
        "desc": mx.array(out2_21.get("desc", np.zeros((*out2_21["pts3d"].shape[:2], 24)))),
        "desc_conf": mx.array(get_conf(out2_21, "desc_conf")),  # Use desc_conf!
    }

    return res11, res21, res22, res12


def _fast_reciprocal_nns(A: np.ndarray, B: np.ndarray, subsample: int) -> tuple[np.ndarray, np.ndarray]:
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


def _merge_corres(idx1: np.ndarray, idx2: np.ndarray, shape1: tuple, shape2: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        nn1to2_idx1, nn1to2_idx2 = _fast_reciprocal_nns(A, B, subsample)
        # Backward matching: B → A
        nn2to1_idx2, nn2to1_idx1 = _fast_reciprocal_nns(B, A, subsample)

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
    xy1, xy2, merge_indices = _merge_corres(idx1, idx2, (H1, W1), (H2, W2))

    # Use the merged confidences exactly like PyTorch
    if len(xy1) > 0:
        # PyTorch: confs = np.sqrt(cat(qonf1)[idx] * cat(qonf2)[idx])
        confs = np.sqrt(qonf1[merge_indices] * qonf2[merge_indices])
    else:
        confs = np.zeros(0, dtype=np.float32)

    return xy1, xy2, confs


def prepare_canonical_data(
    imgs: list[str],
    pairs_data: dict,
    subsample: int = 8,
    cache_path: str | None = None,
    verbose: bool = True,
) -> tuple:
    """Prepare canonical view data for all images.

    Args:
        imgs: List of image paths
        pairs_data: Forward pass results
        subsample: Subsampling factor
        cache_path: Cache directory
        verbose: Print progress

    Returns:
        Tuple of (pairs, scores, canonical_views, paths, preds_21)
    """
    canonical_views = {}
    pairwise_scores = np.zeros((len(imgs), len(imgs)))
    canonical_paths = []
    preds_21 = {}

    if verbose:
        print("Preparing canonical data...")

    for img_idx, img in enumerate(imgs):
        if verbose and (img_idx % 5 == 0 or img_idx == len(imgs) - 1):
            print(f"  Processing image {img_idx + 1}/{len(imgs)}")

        if cache_path:
            cache = os.path.join(cache_path, "canon_views", hash_md5(img) + f"_{subsample}.npz")
            canonical_paths.append(cache)
        else:
            cache = None
            canonical_paths.append(None)

        # Try to load from cache
        canon = None
        focal = None
        if cache and os.path.isfile(cache):
            try:
                data = np.load(cache)
                canon = mx.array(data["canon"])
                focal = mx.array(data["focal"])
            except Exception:
                pass

        # Collect pointmaps for this image
        ptmaps = []
        confs = []
        pixels = {}

        for (img1, img2), ((path1, path2), path_corres) in pairs_data.items():
            if img == img1:
                if os.path.isfile(path1):
                    data = np.load(path1)
                    X = mx.array(data["X1"])
                    C = mx.array(data["C1"])
                    X2 = mx.array(data["X2"])
                    C2 = mx.array(data["C2"])

                    # Load correspondences
                    if os.path.isfile(path_corres):
                        corres_data = np.load(path_corres)
                        score = tuple(corres_data["score"])
                        xy1 = corres_data["xy1"]
                        xy2 = corres_data["xy2"]
                        conf = corres_data["confs"]
                        # Store (pts1, pts2, conf) - pts1 in current view, pts2 in other view
                        pixels[img2] = (mx.array(xy1), mx.array(xy2), mx.array(conf))

                        i, j = imgs.index(img1), imgs.index(img2)
                        pairwise_scores[i, j] = score[2]
                        pairwise_scores[j, i] = score[2]

                    # Store for preds_21
                    if img not in preds_21:
                        preds_21[img] = {}
                    preds_21[img][img2] = (
                        X2[::subsample, ::subsample].reshape(-1, 3),
                        C2[::subsample, ::subsample].reshape(-1),
                    )

                    ptmaps.append(X)
                    confs.append(C)

            if img == img2:
                if os.path.isfile(path2):
                    data = np.load(path2)
                    X = mx.array(data["X1"])
                    C = mx.array(data["C1"])
                    X2 = mx.array(data["X2"])
                    C2 = mx.array(data["C2"])

                    # Load correspondences
                    if os.path.isfile(path_corres):
                        corres_data = np.load(path_corres)
                        xy1 = corres_data["xy1"]
                        xy2 = corres_data["xy2"]
                        conf = corres_data["confs"]
                        # Store (pts1, pts2, conf) - pts1 in current view, pts2 in other view
                        # Here we're reading from img2's perspective, so xy2 is our pts1
                        pixels[img1] = (mx.array(xy2), mx.array(xy1), mx.array(conf))

                    if img not in preds_21:
                        preds_21[img] = {}
                    preds_21[img][img1] = (
                        X2[::subsample, ::subsample].reshape(-1, 3),
                        C2[::subsample, ::subsample].reshape(-1),
                    )

                    ptmaps.append(X)
                    confs.append(C)

        # Compute canonical view if not cached
        if canon is None and ptmaps:
            canon, cconf = compute_canonical_view(ptmaps, confs)
            if cache:
                # Estimate focal
                H, W = canon.shape[:2]
                pp = mx.array([W / 2, H / 2])
                focal = estimate_focal_from_depth(canon, pp)
                np.savez(
                    mkdir_for(cache),
                    canon=np.array(canon),
                    focal=np.array(focal),
                )

        if canon is None:
            # Fallback: use first pointmap
            if ptmaps:
                canon = ptmaps[0]
                cconf = confs[0]
            else:
                # Empty canonical view
                canon = mx.zeros((64, 64, 3))
                cconf = mx.ones((64, 64))

        H, W = canon.shape[:2]
        pp = mx.array([W / 2.0, H / 2.0])

        if focal is None:
            focal = estimate_focal_from_depth(canon, pp)

        # Extract core depth from canonical view
        core_depth = canon[subsample // 2 :: subsample, subsample // 2 :: subsample, 2]

        # Compute anchor depth offsets for precise 3D reconstruction
        # (Matches PyTorch MASt3R anchor_depth_offsets)
        canon_depth_full = canon[..., 2]  # Full depth map [H, W]
        idxs, offsets = anchor_depth_offsets(canon_depth_full, pixels, subsample=subsample)

        # Ensure cconf is defined
        if "cconf" not in locals() or cconf is None:
            cconf = mx.ones((H, W))

        canonical_views[img] = {
            "pp": pp,
            "shape": (H, W),
            "focal": focal,
            "core_depth": core_depth,
            "pixels": pixels,
            "anchor_idxs": idxs,  # NEW: indices into subsampled depth
            "anchor_offs": offsets,  # NEW: depth offset ratios
            "conf": cconf,
        }

    return pairs_data, pairwise_scores, canonical_views, canonical_paths, preds_21


def compute_canonical_view(
    ptmaps: list[mx.array],
    confs: list[mx.array],
) -> tuple[mx.array, mx.array]:
    """Compute canonical view from multiple pointmaps.

    Uses confidence-weighted averaging.

    Args:
        ptmaps: List of pointmaps [H, W, 3]
        confs: List of confidence maps [H, W]

    Returns:
        Tuple of (canonical pointmap, canonical confidence)
    """
    if len(ptmaps) == 1:
        return ptmaps[0], confs[0]

    # Stack for weighted averaging
    pts_stack = mx.stack(ptmaps, axis=0)  # [N, H, W, 3]
    conf_stack = mx.stack(confs, axis=0)  # [N, H, W]

    # Weighted average
    weights = conf_stack[..., None]  # [N, H, W, 1]
    weight_sum = mx.sum(weights, axis=0) + 1e-8  # [H, W, 1]

    canon = mx.sum(pts_stack * weights, axis=0) / weight_sum  # [H, W, 3]
    canon_conf = mx.mean(conf_stack, axis=0)  # [H, W]

    return canon, canon_conf


def estimate_focal_from_depth(
    pts3d: mx.array,
    pp: mx.array,
    min_focal: float = 0.5,
    max_focal: float = 3.5,
) -> mx.array:
    """Estimate focal length from depth map.

    Matches PyTorch MASt3R/DUSt3R estimate_focal_knowing_depth function.
    Uses median voting for robust focal estimation.

    Args:
        pts3d: Pointmap [H, W, 3]
        pp: Principal point [2]
        min_focal: Minimum focal (relative to size)
        max_focal: Maximum focal (relative to size)

    Returns:
        Estimated focal length
    """
    H, W = pts3d.shape[:2]
    size = max(H, W)

    # Get valid points (positive depth, non-zero X and Y)
    z = pts3d[..., 2]
    x_3d = pts3d[..., 0]
    y_3d = pts3d[..., 1]

    # Create pixel coordinates centered at principal point
    # Match PyTorch: pixels = xy_grid(W, H) - pp
    yy, xx = mx.meshgrid(mx.arange(H), mx.arange(W), indexing="ij")
    u = xx.astype(mx.float32) - pp[0]  # Centered x pixel coord
    v = yy.astype(mx.float32) - pp[1]  # Centered y pixel coord

    # Compute focal votes like PyTorch: fx = (u * z) / x, fy = (v * z) / y
    # Avoid division by zero
    eps = 1e-8
    fx_votes = (u * z) / (x_3d + mx.sign(x_3d) * eps + (mx.abs(x_3d) < eps) * eps)
    fy_votes = (v * z) / (y_3d + mx.sign(y_3d) * eps + (mx.abs(y_3d) < eps) * eps)

    # Valid mask: positive depth and non-small X, Y
    valid = (z > 0.1) & (mx.abs(x_3d) > eps) & (mx.abs(y_3d) > eps)

    # Combine fx and fy votes, take median
    focal_np_fx = np.array(fx_votes).flatten()
    focal_np_fy = np.array(fy_votes).flatten()
    valid_np = np.array(valid).flatten()

    # Filter valid and finite values
    f_votes = np.concatenate([focal_np_fx[valid_np], focal_np_fy[valid_np]])
    f_votes = f_votes[np.isfinite(f_votes)]

    if len(f_votes) < 10:
        return mx.array(size * 1.0)

    # Median as robust estimate (matches PyTorch nanmedian)
    focal = mx.array(np.nanmedian(f_votes))

    # Clamp to reasonable range
    focal = mx.clip(focal, min_focal * size, max_focal * size)

    return focal


def condense_data(
    imgs: list[str],
    pairs_data: dict,
    canonical_views: dict,
    preds_21: dict,
) -> tuple:
    """Condense all data for optimization (PyTorch-faithful version).

    This matches PyTorch MASt3R's condense_data structure:
    - anchor_data[idx] = (pixels, idxs, offsets) aggregated per source image
    - corres contains slices into anchor_data for each pair

    Args:
        imgs: List of image paths
        pairs_data: Forward pass results
        canonical_views: Canonical view data
        preds_21: Cross-predictions

    Returns:
        Tuple of condensed data including anchor_data for 3D reconstruction
    """
    n_imgs = len(imgs)

    imsizes = []
    pps = []
    base_focals = []
    core_depth = []
    confs = []  # Confidence maps for each image
    anchors = {}
    anchor_data = {}  # {idx: (pixels, idxs, offsets)} aggregated per image
    tmp_pixels = {}   # {(img1, img2): (pixels, confs, slice)} for slice tracking
    corres = []

    for idx, img in enumerate(imgs):
        cv = canonical_views[img]
        H, W = cv["shape"]

        imsizes.append((H, W))
        pps.append(cv["pp"])
        base_focals.append(cv["focal"])
        core_depth.append(cv["core_depth"])
        confs.append(cv.get("conf", mx.ones((H, W))))

        # Get anchor offsets computed in prepare_canonical_data
        anchor_idxs = cv.get("anchor_idxs", {})
        anchor_offs = cv.get("anchor_offs", {})

        # Build aggregated anchor data with slice tracking (like PyTorch)
        pixels_data = cv["pixels"]
        all_pixels = []
        all_idxs = []
        all_offs = []
        all_confs = []
        cur_n = [0]  # Track slice positions

        for other_img, pixel_data in pixels_data.items():
            # Unpack (pts1, pts2, conf)
            if len(pixel_data) == 3:
                pixels, pixels2, conf = pixel_data
            else:
                pixels, conf = pixel_data
                pixels2 = pixels

            all_pixels.append(pixels)
            all_confs.append(conf)

            # Get anchor indices and offsets for this pair
            if other_img in anchor_idxs:
                all_idxs.append(anchor_idxs[other_img])
                all_offs.append(anchor_offs[other_img])

            # Track slice position (like PyTorch tmp_pixels)
            cur_n.append(cur_n[-1] + len(pixels))
            tmp_pixels[img, other_img] = (pixels, pixels2, conf, slice(cur_n[-2], cur_n[-1]))

        if all_pixels:
            all_pixels = mx.concatenate(all_pixels, axis=0)
            all_confs = mx.concatenate(all_confs, axis=0)
        else:
            all_pixels = mx.zeros((0, 2))
            all_confs = mx.zeros(0)

        # Build anchor_data for make_pts3d (like PyTorch img_anchors)
        if all_idxs and all_offs:
            all_idxs_cat = mx.concatenate(all_idxs, axis=0)
            all_offs_cat = mx.concatenate(all_offs, axis=0)
            anchor_data[idx] = (all_pixels, all_idxs_cat, all_offs_cat)

        anchors[idx] = {
            "pixels": all_pixels,
            "confs": all_confs,
        }

    # Build correspondences with slices (like PyTorch imgs_slices)
    # Note: PyTorch filters by matching_conf_thr in loss calculation, not here
    seen_pairs = set()
    for (img1, img2), (pix1, pix2_fwd, conf1, slice1) in tmp_pixels.items():
        # Only process each pair once
        pair_key = tuple(sorted([img1, img2]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        # Get reverse mapping
        if (img2, img1) not in tmp_pixels:
            continue
        pix2_rev, pix1_rev, conf2, slice2 = tmp_pixels[img2, img1]

        idx1 = imgs.index(img1)
        idx2 = imgs.index(img2)

        # Confidences are geometric mean like PyTorch
        conf = mx.sqrt(conf1 * conf2)

        corres.append({
            "idx1": idx1,
            "idx2": idx2,
            "slice1": slice1,  # Slice into anchor_data[idx1]
            "slice2": slice2,  # Slice into anchor_data[idx2]
            "pts1": pix1,      # 2D pixel coords in img1
            "pts2": pix2_fwd,  # 2D pixel coords in img2
            "weights": conf,
            "max_conf": float(mx.max(conf)),  # For filtering
        })

    return (
        imsizes,
        pps,
        base_focals,
        core_depth,
        confs,
        anchors,
        anchor_data,
        corres,
        [],  # corres2d (unused)
    )


def sparse_scene_optimizer(
    imgs: list[str],
    pairs_in: list[tuple[dict, dict]],
    subsample: int,
    imsizes: list[tuple[int, int]],
    pps: list[mx.array],
    base_focals: list[mx.array],
    core_depth: list[mx.array],
    img_confs: list[mx.array],
    anchors: dict,
    anchor_data: dict,
    corres: list[dict],
    canonical_paths: list[str] | None,
    preds_21: dict | None = None,
    lr1: float = 0.07,
    niter1: int = 300,
    lr2: float = 0.01,  # PyTorch default
    niter2: int = 300,
    shared_intrinsics: bool = False,
    matching_conf_thr: float = 5.0,
    loss_dust3r_w: float = 0.01,
    verbose: bool = True,
) -> SparseGAResult:
    """Run sparse scene optimization using PyTorch-faithful v2 optimizer.

    Two-phase optimization:
    1. Coarse alignment with 3D point matching loss
    2. Fine refinement with 2D reprojection loss

    Args:
        imgs: Image paths
        pairs_in: Original image pairs
        subsample: Subsampling factor
        imsizes: Image sizes
        pps: Principal points
        base_focals: Initial focal estimates
        core_depth: Subsampled depth maps
        img_confs: Confidence maps per image
        anchors: Anchor point data
        anchor_data: Dict of (pixels, idxs, offsets) per image for make_pts3d
        corres: Correspondences
        canonical_paths: Cache paths
        preds_21: Cross-predictions for dust3r loss regularization
        lr1, niter1: Coarse phase parameters
        lr2, niter2: Fine phase parameters
        shared_intrinsics: Use single intrinsics
        matching_conf_thr: Confidence threshold
        loss_dust3r_w: Weight for dust3r loss (default 0.01 like PyTorch)
        verbose: Print progress

    Returns:
        SparseGAResult with optimized scene
    """
    n_imgs = len(imgs)

    # Convert to MLX arrays
    init_focals = mx.stack([mx.array(f).reshape(()) for f in base_focals])

    if verbose:
        print(f"init focals = {[float(f) for f in init_focals]}")

    # Prepare depths and compute medians (like PyTorch)
    init_depths = []
    median_depths = []
    for i, depth in enumerate(core_depth):
        H, W = imsizes[i]
        H_sub = (H + subsample - 1) // subsample
        W_sub = (W + subsample - 1) // subsample
        # Ensure correct size
        d = mx.array(depth).reshape(-1)
        expected_size = H_sub * W_sub
        if len(d) < expected_size:
            d = mx.pad(d, [(0, expected_size - len(d))])
        elif len(d) > expected_size:
            d = d[:expected_size]
        d_2d = d.reshape(H_sub, W_sub)
        init_depths.append(d_2d)

        # Compute median depth (like PyTorch core_depth /= median)
        median = float(mx.median(d))
        median = max(median, 1e-6)  # Avoid zero
        median_depths.append(median)

    median_depths = mx.array(median_depths)

    # Normalize core depths by median (like PyTorch)
    normalized_depths = []
    for i, d in enumerate(init_depths):
        normalized_depths.append(d / median_depths[i])

    # Build MST from correspondences
    mst_parent_map, mst_root = build_mst_from_correspondences(corres, n_imgs)

    # Convert MST format: {child: parent} -> (root, [(parent, child), ...])
    mst_edges = []
    for child, parent in mst_parent_map.items():
        mst_edges.append((parent, child))
    mst = (mst_root, mst_edges)

    if verbose:
        print(f"Built MST with root={mst_root}, edges={len(mst_edges)}")

    # Correspondences are already formatted with slices from condense_data
    # Just use them directly (like PyTorch imgs_slices)
    opt_corres = corres  # Contains: idx1, idx2, slice1, slice2, pts1, pts2, weights

    # Run PyTorch-faithful optimizer
    if verbose:
        print("Running scene optimization...")

    result = _sparse_optimizer_core(
        imgs=imgs,
        imsizes=imsizes,
        pps=pps,
        base_focals=init_focals,
        core_depth=normalized_depths,  # Normalized by median
        median_depths=median_depths,
        img_anchors=anchor_data,  # {idx: (pixels, idxs, offsets)}
        corres=opt_corres,
        preds_21=preds_21 or {},
        mst=mst,
        subsample=subsample,
        lr1=lr1,
        niter1=niter1,
        lr2=lr2,
        niter2=niter2,
        exp_depth=False,  # PyTorch default
        shared_intrinsics=shared_intrinsics,
        matching_conf_thr=matching_conf_thr,
        loss_dust3r_w=loss_dust3r_w,
        verbose=verbose,
    )

    # Extract results
    poses = result["cam2w"]
    focals = result["focals"]
    pps_norm = result["pps_norm"]
    depths = result["depthmaps"]
    log_sizes = result.get("log_sizes", mx.zeros(n_imgs))

    # Convert normalized pps back to pixel coordinates
    pps_out_list = []
    for i in range(n_imgs):
        H, W = imsizes[i]
        pps_out_list.append(pps_norm[i] * mx.array([float(W), float(H)]))
    pps_out = mx.stack(pps_out_list)

    # Compute final 3D points
    pts3d_list = []
    pts3d_colors = []
    confs_list = []

    for i in range(n_imgs):
        H, W = imsizes[i]
        depth = depths[i] if depths else mx.ones((H, W))

        # Build intrinsics
        K = mx.array([
            [focals[i], 0, pps_out[i, 0]],
            [0, focals[i], pps_out[i, 1]],
            [0, 0, 1],
        ])

        # Unproject and transform
        pts3d = depthmap_to_pts3d(depth, K)
        pts3d_world = geotrf(poses[i], pts3d.reshape(-1, 3))

        # depths is already subsampled by optimizer, so pts3d_world matches
        # depth shape is (H_sub, W_sub), pts3d_world is (H_sub * W_sub, 3)
        H_sub = depth.shape[0]
        W_sub = depth.shape[1]
        pts3d_list.append(pts3d_world)

        # Colors placeholder
        pts3d_colors.append(np.ones((len(pts3d_world), 3)) * 0.5)

        # Subsample confidence to match depth resolution
        conf_i = img_confs[i]
        if conf_i.shape[0] == H and conf_i.shape[1] == W:
            # Full resolution conf -> subsample to match depth
            conf_sparse = conf_i[::subsample, ::subsample].reshape(-1)
        elif conf_i.shape[0] == H_sub and conf_i.shape[1] == W_sub:
            # Already subsampled
            conf_sparse = conf_i.reshape(-1)
        else:
            # Unknown shape, just flatten
            conf_sparse = conf_i.reshape(-1)
        confs_list.append(conf_sparse)

    # Fetch actual images from pairs
    imgs_array = []

    def fetch_img(im: str) -> np.ndarray:
        for img1, img2 in pairs_in:
            if img1["instance"] == im:
                img_tensor = np.array(img1["img"]).astype(np.float32)
                if img_tensor.ndim == 4:
                    img_tensor = img_tensor[0]
                if img_tensor.shape[0] == 3:
                    img_tensor = img_tensor.transpose(1, 2, 0)
                if img_tensor.max() > 1.0:
                    return np.clip(img_tensor / 255.0, 0, 1)
                else:
                    return np.clip(img_tensor * 0.5 + 0.5, 0, 1)
            if img2["instance"] == im:
                img_tensor = np.array(img2["img"]).astype(np.float32)
                if img_tensor.ndim == 4:
                    img_tensor = img_tensor[0]
                if img_tensor.shape[0] == 3:
                    img_tensor = img_tensor.transpose(1, 2, 0)
                if img_tensor.max() > 1.0:
                    return np.clip(img_tensor / 255.0, 0, 1)
                else:
                    return np.clip(img_tensor * 0.5 + 0.5, 0, 1)
        return np.zeros((64, 64, 3))

    for img_path in imgs:
        imgs_array.append(fetch_img(img_path))

    return SparseGAResult(
        imgs=imgs_array,
        img_paths=imgs,
        focals=focals,
        principal_points=pps_out,
        cam2w=poses,
        depthmaps=depths if depths else [mx.ones(s) for s in imsizes],
        pts3d=pts3d_list,
        pts3d_colors=pts3d_colors,
        confs=confs_list,
        canonical_paths=canonical_paths,
        base_focals=init_focals,
        sizes=mx.exp(log_sizes),
    )
