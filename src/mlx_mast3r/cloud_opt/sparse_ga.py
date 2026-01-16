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
from .losses import gamma_loss
from .optimizer import (
    OptimConfig,
    SceneOptimizer,
    build_mst_from_correspondences,
    make_pts3d_from_depth,
    optimize_scene,
)
from .schedules import cosine_schedule


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
    lr2: float = 0.01,
    niter2: int = 300,
    matching_conf_thr: float = 5.0,
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
        lr1=lr1,
        niter1=niter1,
        lr2=lr2,
        niter2=niter2,
        shared_intrinsics=shared_intrinsics,
        matching_conf_thr=matching_conf_thr,
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

    def get_conf(out):
        conf = out.get("conf", None)
        if conf is None:
            # Fallback: create uniform confidence
            pts3d = out["pts3d"]
            return np.ones(pts3d.shape[:2], dtype=np.float32)
        return np.array(conf).squeeze() if hasattr(conf, "__array__") else conf

    res11 = {
        "pts3d": mx.array(out1_12["pts3d"]),
        "conf": mx.array(get_conf(out1_12)),
        "desc": mx.array(out1_12.get("desc", np.zeros((*out1_12["pts3d"].shape[:2], 24)))),
        "desc_conf": mx.array(get_conf(out1_12)),
    }
    res21 = {
        "pts3d": mx.array(out2_12["pts3d"]),
        "conf": mx.array(get_conf(out2_12)),
        "desc": mx.array(out2_12.get("desc", np.zeros((*out2_12["pts3d"].shape[:2], 24)))),
        "desc_conf": mx.array(get_conf(out2_12)),
    }
    res22 = {
        "pts3d": mx.array(out1_21["pts3d"]),
        "conf": mx.array(get_conf(out1_21)),
        "desc": mx.array(out1_21.get("desc", np.zeros((*out1_21["pts3d"].shape[:2], 24)))),
        "desc_conf": mx.array(get_conf(out1_21)),
    }
    res12 = {
        "pts3d": mx.array(out2_21["pts3d"]),
        "conf": mx.array(get_conf(out2_21)),
        "desc": mx.array(out2_21.get("desc", np.zeros((*out2_21["pts3d"].shape[:2], 24)))),
        "desc_conf": mx.array(get_conf(out2_21)),
    }

    return res11, res21, res22, res12


def extract_correspondences(
    feats: list[mx.array],
    qonfs: list[mx.array],
    subsample: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract correspondences from descriptor features.

    Uses reciprocal nearest neighbor matching.

    Args:
        feats: List of descriptor features [feat11, feat21, feat22, feat12]
        qonfs: List of confidence maps
        subsample: Subsampling factor

    Returns:
        Tuple of (xy1, xy2, confidences)
    """
    feat11, feat21, feat22, feat12 = feats
    qonf11, qonf21, qonf22, qonf12 = qonfs

    H1, W1 = feat11.shape[:2]
    H2, W2 = feat22.shape[:2]

    # Create pixel grids
    y1, x1 = np.mgrid[0:H1:subsample, 0:W1:subsample]
    y2, x2 = np.mgrid[0:H2:subsample, 0:W2:subsample]

    xy1_grid = np.stack([x1.ravel(), y1.ravel()], axis=-1).astype(np.float32)
    xy2_grid = np.stack([x2.ravel(), y2.ravel()], axis=-1).astype(np.float32)

    # Subsample features
    feat11_sub = np.array(feat11[::subsample, ::subsample]).reshape(-1, feat11.shape[-1])
    feat21_sub = np.array(feat21[::subsample, ::subsample]).reshape(-1, feat21.shape[-1])
    feat22_sub = np.array(feat22[::subsample, ::subsample]).reshape(-1, feat22.shape[-1])
    feat12_sub = np.array(feat12[::subsample, ::subsample]).reshape(-1, feat12.shape[-1])

    # Subsample confidences
    qonf11_sub = np.array(qonf11[::subsample, ::subsample]).ravel()
    qonf21_sub = np.array(qonf21[::subsample, ::subsample]).ravel()
    qonf22_sub = np.array(qonf22[::subsample, ::subsample]).ravel()
    qonf12_sub = np.array(qonf12[::subsample, ::subsample]).ravel()

    # Normalize features for dot product matching
    feat11_sub = feat11_sub / (np.linalg.norm(feat11_sub, axis=-1, keepdims=True) + 1e-8)
    feat21_sub = feat21_sub / (np.linalg.norm(feat21_sub, axis=-1, keepdims=True) + 1e-8)
    feat22_sub = feat22_sub / (np.linalg.norm(feat22_sub, axis=-1, keepdims=True) + 1e-8)
    feat12_sub = feat12_sub / (np.linalg.norm(feat12_sub, axis=-1, keepdims=True) + 1e-8)

    all_xy1 = []
    all_xy2 = []
    all_confs = []

    # Match 1→2 and 2→1 for both pairs
    for A, B, QA, QB, xyA, xyB in [
        (feat11_sub, feat21_sub, qonf11_sub, qonf21_sub, xy1_grid, xy2_grid),
        (feat12_sub, feat22_sub, qonf12_sub, qonf22_sub, xy1_grid, xy2_grid),
    ]:
        # Compute similarity matrix
        sim = A @ B.T

        # Find best matches A→B
        best_B = np.argmax(sim, axis=1)
        # Find best matches B→A
        best_A = np.argmax(sim, axis=0)

        # Reciprocal matches
        mutual_A = np.arange(len(A))
        mutual_B = best_B
        is_mutual = best_A[best_B] == mutual_A

        # Keep only mutual matches
        valid_A = mutual_A[is_mutual]
        valid_B = mutual_B[is_mutual]

        all_xy1.append(xyA[valid_A])
        all_xy2.append(xyB[valid_B])
        all_confs.append(np.sqrt(QA[valid_A] * QB[valid_B]))

    # Concatenate all matches
    xy1 = np.concatenate(all_xy1, axis=0)
    xy2 = np.concatenate(all_xy2, axis=0)
    confs = np.concatenate(all_confs, axis=0)

    # Remove duplicates
    unique_pairs = {}
    for i in range(len(xy1)):
        key = (int(xy1[i, 0]), int(xy1[i, 1]), int(xy2[i, 0]), int(xy2[i, 1]))
        if key not in unique_pairs or confs[i] > unique_pairs[key][2]:
            unique_pairs[key] = (xy1[i], xy2[i], confs[i])

    if unique_pairs:
        xy1 = np.array([v[0] for v in unique_pairs.values()])
        xy2 = np.array([v[1] for v in unique_pairs.values()])
        confs = np.array([v[2] for v in unique_pairs.values()])
    else:
        xy1 = np.zeros((0, 2), dtype=np.float32)
        xy2 = np.zeros((0, 2), dtype=np.float32)
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

    Uses Weiszfeld algorithm for robust estimation.

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

    # Get valid points
    z = pts3d[..., 2]
    valid = z > 0.1

    if mx.sum(valid) < 10:
        return mx.array(size * 1.0)

    # Create pixel coordinates
    y, x = mx.meshgrid(mx.arange(H), mx.arange(W), indexing="ij")
    xy = mx.stack([x, y], axis=-1).astype(mx.float32)

    # Centered coordinates
    xy_centered = xy - pp

    # Estimate focal from z/d relationship
    # f = z * sqrt((x-cx)^2 + (y-cy)^2) / sqrt(X^2 + Y^2)
    d_pixel = mx.sqrt(mx.sum(xy_centered**2, axis=-1) + 1e-8)
    d_3d = mx.sqrt(pts3d[..., 0] ** 2 + pts3d[..., 1] ** 2 + 1e-8)

    focal_estimates = z * d_pixel / (d_3d + 1e-8)

    # Robust median - use numpy for boolean indexing (not supported in MLX)
    focal_np = np.array(focal_estimates).flatten()
    valid_np = np.array(valid).flatten()
    focal_estimates_valid = focal_np[valid_np]
    focal = mx.array(np.median(focal_estimates_valid))

    # Clamp to reasonable range
    focal = mx.clip(focal, min_focal * size, max_focal * size)

    return focal


def condense_data(
    imgs: list[str],
    pairs_data: dict,
    canonical_views: dict,
    preds_21: dict,
) -> tuple:
    """Condense all data for optimization.

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
    anchor_data = {}  # NEW: For make_pts3d_from_depth (pixels, idxs, offsets)
    corres = []
    corres2d = []

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

        # Build anchor data
        pixels_data = cv["pixels"]
        all_pixels = []
        all_idxs = []
        all_offs = []
        all_confs = []

        for other_img, pixel_data in pixels_data.items():
            # Unpack (pts1, pts2, conf) or legacy (pts1, conf) format
            if len(pixel_data) == 3:
                pixels, pixels2, conf = pixel_data
            else:
                pixels, conf = pixel_data
                pixels2 = pixels  # Fallback for legacy format

            all_pixels.append(pixels)
            all_confs.append(conf)

            # Get anchor indices and offsets for this pair
            if other_img in anchor_idxs:
                all_idxs.append(anchor_idxs[other_img])
                all_offs.append(anchor_offs[other_img])

            # Build correspondence entry with both pts1 and pts2
            other_idx = imgs.index(other_img)
            corres.append(
                {
                    "idx1": idx,
                    "idx2": other_idx,
                    "pts1": pixels,
                    "pts2": pixels2,
                    "weights": conf,
                }
            )

        if all_pixels:
            all_pixels = mx.concatenate(all_pixels, axis=0)
            all_confs = mx.concatenate(all_confs, axis=0)
        else:
            all_pixels = mx.zeros((0, 2))
            all_confs = mx.zeros(0)

        # Build anchor_data for make_pts3d_from_depth
        # Format: (pixels, idxs, offsets)
        if all_idxs and all_offs:
            all_idxs_cat = mx.concatenate(all_idxs, axis=0)
            all_offs_cat = mx.concatenate(all_offs, axis=0)
            anchor_data[idx] = (all_pixels, all_idxs_cat, all_offs_cat)

        anchors[idx] = {
            "pixels": all_pixels,
            "confs": all_confs,
        }

    return (
        imsizes,
        pps,
        base_focals,
        core_depth,
        confs,
        anchors,
        anchor_data,  # NEW: Added anchor_data
        corres,
        corres2d,
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
    lr1: float = 0.07,
    niter1: int = 300,
    lr2: float = 0.01,
    niter2: int = 300,
    shared_intrinsics: bool = False,
    matching_conf_thr: float = 5.0,
    verbose: bool = True,
) -> SparseGAResult:
    """Run sparse scene optimization.

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
        lr1, niter1: Coarse phase parameters
        lr2, niter2: Fine phase parameters
        shared_intrinsics: Use single intrinsics
        matching_conf_thr: Confidence threshold
        verbose: Print progress

    Returns:
        SparseGAResult with optimized scene
    """
    n_imgs = len(imgs)

    # Convert to MLX arrays
    init_focals = mx.stack([mx.array(f).reshape(()) for f in base_focals])
    init_pps = mx.stack([pp / mx.array([w, h]) for pp, (h, w) in zip(pps, imsizes)])

    # Flatten and stack depths
    init_depths = []
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
        init_depths.append(d.reshape(H_sub, W_sub))

    # Build MST from correspondences for kinematic chain
    # Use input corres (already available) to build the MST
    mst_parent_map, mst_root = build_mst_from_correspondences(corres, n_imgs)

    if verbose:
        print(f"Built MST with root={mst_root}, edges={len(mst_parent_map)}")

    # Create optimizer with MST
    optimizer = SceneOptimizer(
        n_images=n_imgs,
        image_sizes=imsizes,
        init_focals=init_focals,
        init_pps=init_pps,
        init_depths=init_depths,
        subsample=subsample,
        shared_intrinsics=shared_intrinsics,
        mst_edges=mst_parent_map,
    )

    # Build canonical 3D points from depths
    canonical_pts3d = []
    for i in range(n_imgs):
        H, W = imsizes[i]
        depth = init_depths[i]

        # Create pixel grid
        H_sub, W_sub = depth.shape
        y, x = mx.meshgrid(
            mx.arange(H_sub) * subsample + subsample // 2,
            mx.arange(W_sub) * subsample + subsample // 2,
            indexing="ij",
        )
        xy = mx.stack([x, y], axis=-1).reshape(-1, 2).astype(mx.float32)

        # Unproject to 3D
        f = init_focals[i]
        pp = pps[i]
        z = depth.reshape(-1)

        x_norm = (xy[:, 0] - pp[0]) / f
        y_norm = (xy[:, 1] - pp[1]) / f

        pts3d = mx.stack([x_norm * z, y_norm * z, z], axis=-1)
        canonical_pts3d.append(pts3d)

    # Format correspondences for optimizer
    opt_corres = []
    for c in corres:
        idx1 = c["idx1"]
        idx2 = c["idx2"]
        pts1 = c["pts1"]
        pts2 = c.get("pts2", pts1)  # Use pts2 if available, fallback to pts1
        weights = c.get("weights", mx.ones(len(pts1)))

        # Find matching points in canonical data
        H1, W1 = imsizes[idx1]
        H_sub1 = (H1 + subsample - 1) // subsample
        W_sub1 = (W1 + subsample - 1) // subsample

        # Convert pts1 pixel coords to subsampled indices
        pts1_sub = (pts1 / subsample).astype(mx.int32)
        pts1_sub = mx.clip(pts1_sub, 0, mx.array([W_sub1 - 1, H_sub1 - 1]))
        pts1_idx = pts1_sub[:, 1] * W_sub1 + pts1_sub[:, 0]

        # Convert pts2 pixel coords to subsampled indices
        H2, W2 = imsizes[idx2]
        H_sub2 = (H2 + subsample - 1) // subsample
        W_sub2 = (W2 + subsample - 1) // subsample
        pts2_sub = (pts2 / subsample).astype(mx.int32)
        pts2_sub = mx.clip(pts2_sub, 0, mx.array([W_sub2 - 1, H_sub2 - 1]))
        pts2_idx = pts2_sub[:, 1] * W_sub2 + pts2_sub[:, 0]

        opt_corres.append(
            {
                "idx1": idx1,
                "idx2": idx2,
                "pts1_idx": pts1_idx,
                "pts2_idx": pts2_idx,
                "pts1": pts1,
                "pts2": pts2,
                "weights": weights,
            }
        )

    # Create optimization config
    config = OptimConfig(
        lr1=lr1,
        niter1=niter1,
        lr2=lr2,
        niter2=niter2,
        shared_intrinsics=shared_intrinsics,
    )

    # Run optimization
    if verbose:
        print("Running scene optimization...")

    optimizer = optimize_scene(
        optimizer=optimizer,
        corres=opt_corres,
        canonical_pts3d=canonical_pts3d,
        config=config,
        anchor_data=anchor_data,  # Pass anchor data for precise 3D reconstruction
        verbose=verbose,
    )

    # Extract results - poses and depths already have global_scaling applied
    poses = optimizer.get_poses()
    focals = optimizer.get_focals()
    pps_out = optimizer.get_principal_points()
    depths = optimizer.get_depths()
    sizes = optimizer.sizes  # Optimized scale factors per view

    # Compute final 3D points
    pts3d_list = []
    pts3d_colors = []
    confs_list = []

    for i in range(n_imgs):
        H, W = imsizes[i]
        depth = depths[i] if depths else mx.ones((H, W))

        # Build intrinsics
        K = mx.array(
            [
                [focals[i], 0, pps_out[i, 0]],
                [0, focals[i], pps_out[i, 1]],
                [0, 0, 1],
            ]
        )

        # Unproject and transform
        pts3d = depthmap_to_pts3d(depth, K)
        pts3d_world = geotrf(poses[i], pts3d.reshape(-1, 3))

        # Subsample for sparse output
        pts3d_sparse = pts3d_world[:: subsample * subsample]
        pts3d_list.append(pts3d_sparse)

        # Get colors (placeholder - would need actual images)
        pts3d_colors.append(np.ones((len(pts3d_sparse), 3)) * 0.5)

        # Use actual confidence from model
        confs_list.append(img_confs[i])

    # Fetch actual images from pairs
    imgs_array = []

    def fetch_img(im: str) -> np.ndarray:
        for img1, img2 in pairs_in:
            if img1["instance"] == im:
                img_tensor = np.array(img1["img"]).astype(np.float32)
                # Convert from CHW to HWC
                if img_tensor.ndim == 4:
                    img_tensor = img_tensor[0]
                if img_tensor.shape[0] == 3:
                    img_tensor = img_tensor.transpose(1, 2, 0)
                # Denormalize based on value range
                if img_tensor.max() > 1.0:
                    # uint8 [0-255] -> [0-1]
                    return np.clip(img_tensor / 255.0, 0, 1)
                else:
                    # normalized [-1, 1] -> [0-1]
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
        sizes=sizes,
    )
