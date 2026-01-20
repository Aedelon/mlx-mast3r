"""Sparse Global Alignment for MASt3R with MLX.

Implements the main pipeline for multi-view reconstruction using
sparse correspondences and global optimization.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from mlx_mast3r.types import (
        CanonicalViewDict,
        CorrespondenceDict,
        ImageDataDict,
        ImgAnchorsDict,
        PairsDataDict,
        Preds21Dict,
        Reconstructor,
    )

# Extracted modules
from .canonical import compute_canonical_view, condense_data, prepare_canonical_data
from .correspondences import extract_correspondences
from .focal_estimation import estimate_focal_from_depth
from .geometry import (
    clean_pointcloud_mlx,
    depthmap_to_pts3d,
    depthmap_to_pts3d_mlx,
    geotrf,
    geotrf_mlx,
    inv,
)
from .optimizer import build_mst_from_correspondences
from .optimizer import sparse_scene_optimizer as _sparse_optimizer_core
from .utils import hash_md5, mkdir_for


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
        """Get dense 3D points from depthmaps using anchor-based densification.

        Like PyTorch MASt3R, this uses anchor_depth_offsets to densify sparse
        depthmaps to full resolution. The optimizer applies a reparametrization
        to camera poses that shifts all cameras including the root. We compensate
        by transforming all poses relative to the root camera (cam 0).

        Args:
            clean_depth: Apply multi-view consistency cleaning (removes outliers)
            subsample: Subsampling factor

        Returns:
            Tuple of (pts3d, depthmaps, confs) lists at full resolution
        """
        pts3d_list = []
        depth_list = []
        confs_list = []
        K_list = []
        cam2w_list = []

        # Get inverse of root pose to recenter the scene
        root_pose = self.cam2w[0]
        root_pose_inv = mx.linalg.inv(root_pose, stream=mx.cpu)

        for i in range(self.n_imgs):
            H_sub, W_sub = self.depthmaps[i].shape
            depth_sub = self.depthmaps[i].flatten()  # Subsampled depth [H_sub * W_sub]

            # Full resolution intrinsics (like PyTorch)
            f = float(self.focals[i])
            pp = self.principal_points[i]

            # Load canonical view from cache for densification
            canon = None
            conf = None
            if self.canonical_paths and self.canonical_paths[i]:
                try:
                    data = np.load(self.canonical_paths[i])
                    canon = mx.array(data["canon"])
                    if "conf" in data:
                        conf = mx.array(data["conf"])
                except Exception:
                    pass

            if canon is not None:
                # Use anchor-based densification like PyTorch
                H, W = canon.shape[:2]
                canon_depth = canon[..., 2]  # Full resolution depth

                # Create dense pixel grid at full resolution
                pixels_x = mx.arange(W, dtype=mx.float32)
                pixels_y = mx.arange(H, dtype=mx.float32)
                py, px = mx.meshgrid(pixels_y, pixels_x, indexing="ij")
                pixels = mx.stack([px.flatten(), py.flatten()], axis=-1)  # [H*W, 2]

                # Compute anchor indices and offsets for ALL pixels
                W_sub_calc = (W + subsample - 1) // subsample
                H_sub_calc = (H + subsample - 1) // subsample

                # Anchor indices: which subsampled cell each pixel belongs to
                anchor_idx = (py.flatten().astype(mx.int32) // subsample) * W_sub_calc + \
                             (px.flatten().astype(mx.int32) // subsample)
                anchor_idx = mx.clip(anchor_idx, 0, H_sub_calc * W_sub_calc - 1)

                # Compute depth offsets: ratio of pixel depth to anchor depth
                # Get anchor center coordinates
                anchor_cy = (py.flatten().astype(mx.int32) // subsample) * subsample + subsample // 2
                anchor_cx = (px.flatten().astype(mx.int32) // subsample) * subsample + subsample // 2
                anchor_cy = mx.clip(anchor_cy, 0, H - 1)
                anchor_cx = mx.clip(anchor_cx, 0, W - 1)

                # Get depths
                ref_depth = canon_depth[anchor_cy.astype(mx.int32), anchor_cx.astype(mx.int32)]
                pixel_depth = canon_depth[py.flatten().astype(mx.int32), px.flatten().astype(mx.int32)]

                # Compute offset ratio (with safety for division)
                offset = pixel_depth / mx.maximum(ref_depth, mx.array(1e-6))

                # Focal compensation like PyTorch
                if self.base_focals is not None:
                    base_f = float(self.base_focals[i])
                    focal_ratio = base_f / f
                    offset = 1.0 + (offset - 1.0) * focal_ratio

                # Densify: depth = subsampled_depth[anchor_idx] * offset
                depth_dense = depth_sub[anchor_idx] * offset
                depth_dense = depth_dense.reshape(H, W)

                # Build full resolution intrinsics
                K = mx.array([
                    [f, 0, float(pp[0])],
                    [0, f, float(pp[1])],
                    [0, 0, 1],
                ])

                # Unproject to 3D at full resolution
                pts3d = depthmap_to_pts3d_mlx(depth_dense, K)

                # Use full resolution confidence if available
                if conf is None:
                    conf = mx.ones((H, W))

            else:
                # Fallback: use subsampled data (like before)
                H, W = H_sub, W_sub
                H_full = H * subsample
                W_full = W * subsample
                scale = H / H_full

                f_scaled = f * scale
                pp_scaled = pp * scale

                K = mx.array([
                    [f_scaled, 0, float(pp_scaled[0])],
                    [0, f_scaled, float(pp_scaled[1])],
                    [0, 0, 1],
                ])

                depth_dense = self.depthmaps[i]
                pts3d = depthmap_to_pts3d_mlx(depth_dense, K)

                # Reshape conf
                conf_raw = self.confs[i]
                if conf_raw.ndim == 1 and conf_raw.size == H * W:
                    conf = conf_raw.reshape(H, W)
                else:
                    conf = mx.ones((H, W))

            K_list.append(K)

            # Transform to world coordinates with recentered poses
            cam2w = self.cam2w[i]
            cam2w_recentered = root_pose_inv @ cam2w
            cam2w_list.append(cam2w_recentered)

            pts3d_world = geotrf_mlx(cam2w_recentered, pts3d.reshape(-1, 3)).reshape(H, W, 3)

            pts3d_list.append(pts3d_world)
            depth_list.append(depth_dense)
            confs_list.append(conf)

        # Apply multi-view consistency cleaning if requested
        if clean_depth and self.n_imgs > 1:
            world2cam_list = [mx.linalg.inv(c, stream=mx.cpu) for c in cam2w_list]

            confs_cleaned = clean_pointcloud_mlx(
                im_confs=confs_list,
                K_list=K_list,
                world2cams=world2cam_list,
                depthmaps=depth_list,
                all_pts3d=pts3d_list,
                tol=0.001,
                bad_conf=0.0,
            )
            return pts3d_list, depth_list, confs_cleaned

        return pts3d_list, depth_list, confs_list


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


# NOTE: prepare_canonical_data, compute_canonical_view, condense_data
# are now imported from .canonical module

# NOTE: extract_correspondences is now imported from .correspondences module

# NOTE: estimate_focal_from_depth is now imported from .focal_estimation module


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
    # Also filter outliers to prevent large depth values from causing scattered points
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

        # Ensure positive depths (like PyTorch: d.clip(min=1e-4))
        # Outlier removal is done later via clean_pointcloud_mlx
        d = mx.clip(d, a_min=mx.array(1e-4), a_max=None)

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

        # Unproject and transform (use MLX native versions)
        pts3d = depthmap_to_pts3d_mlx(depth, K)
        pts3d_world = geotrf_mlx(poses[i], pts3d.reshape(-1, 3))

        # depths is already subsampled by optimizer, so pts3d_world matches
        # depth shape is (H_sub, W_sub), pts3d_world is (H_sub * W_sub, 3)
        H_sub = depth.shape[0]
        W_sub = depth.shape[1]
        pts3d_list.append(pts3d_world)

        # Colors placeholder
        pts3d_colors.append(np.ones((len(pts3d_world), 3)) * 0.5)

        # Subsample confidence to match depth resolution
        conf_i = img_confs[i]
        n_pts = H_sub * W_sub

        # Try to subsample based on shape
        if conf_i.shape[0] == H and conf_i.shape[1] == W:
            # Full resolution conf (H, W) -> subsample to match depth
            conf_sparse = conf_i[::subsample, ::subsample].reshape(-1)
        elif conf_i.shape[0] == W and conf_i.shape[1] == H:
            # Transposed full resolution (W, H) -> transpose then subsample
            conf_sparse = conf_i.T[::subsample, ::subsample].reshape(-1)
        elif conf_i.shape[0] == H_sub and conf_i.shape[1] == W_sub:
            # Already subsampled with correct orientation
            conf_sparse = conf_i.reshape(-1)
        elif conf_i.shape[0] == W_sub and conf_i.shape[1] == H_sub:
            # Already subsampled but transposed
            conf_sparse = conf_i.T.reshape(-1)
        else:
            # Fallback: take first n_pts elements or pad with 1.0
            conf_flat = conf_i.reshape(-1)
            if len(conf_flat) >= n_pts:
                conf_sparse = conf_flat[:n_pts]
            else:
                conf_sparse = mx.concatenate([conf_flat, mx.ones(n_pts - len(conf_flat))])

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
