# Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""Canonical view computation for multi-view reconstruction.

Handles canonical view extraction and data preparation.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np

from .correspondences import anchor_depth_offsets
from .focal_estimation import estimate_focal_from_depth
from .utils import hash_md5, mkdir_for


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
        cconf_cached = None
        if cache and os.path.isfile(cache):
            try:
                data = np.load(cache)
                canon = mx.array(data["canon"])
                focal = mx.array(data["focal"])
                if "conf" in data:
                    cconf_cached = mx.array(data["conf"])
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
        cconf = None
        if canon is None and ptmaps:
            canon, cconf = compute_canonical_view(ptmaps, confs)
            if cache:
                # Estimate focal
                H, W = canon.shape[:2]
                pp = mx.array([W / 2, H / 2])
                focal = estimate_focal_from_depth(canon, pp)
                # Store canon pointmap, focal, and confidence for densification
                np.savez(
                    mkdir_for(cache),
                    canon=np.array(canon),
                    focal=np.array(focal),
                    conf=np.array(cconf) if cconf is not None else np.ones((H, W)),
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
        canon_depth_full = canon[..., 2]  # Full depth map [H, W]
        idxs, offsets = anchor_depth_offsets(canon_depth_full, pixels, subsample=subsample)

        # Ensure cconf is defined - use cached conf if available
        if cconf is None:
            if cconf_cached is not None:
                cconf = cconf_cached
            else:
                cconf = mx.ones((H, W))

        canonical_views[img] = {
            "pp": pp,
            "shape": (H, W),
            "focal": focal,
            "core_depth": core_depth,
            "pixels": pixels,
            "anchor_idxs": idxs,
            "anchor_offs": offsets,
            "conf": cconf,
        }

    return pairs_data, pairwise_scores, canonical_views, canonical_paths, preds_21


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
    confs = []
    anchors = {}
    anchor_data = {}
    tmp_pixels = {}
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

        corres.append(
            {
                "idx1": idx1,
                "idx2": idx2,
                "slice1": slice1,
                "slice2": slice2,
                "pts1": pix1,
                "pts2": pix2_fwd,
                "weights": conf,
                "max_conf": float(mx.max(conf)),
            }
        )

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
