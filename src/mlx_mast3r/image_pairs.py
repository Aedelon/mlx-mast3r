"""Image pairing utilities for multi-view reconstruction.

Implements various scene graph strategies for creating image pairs.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def make_pairs(
    imgs: list[dict],
    scene_graph: str = "complete",
    prefilter: str | None = None,
    symmetrize: bool = True,
    sim_mat: np.ndarray | None = None,
) -> list[tuple[dict, dict]]:
    """Create image pairs based on scene graph strategy.

    Args:
        imgs: List of image dicts with 'idx' key
        scene_graph: Pairing strategy:
            - 'complete': All possible pairs (N*(N-1)/2)
            - 'swin-K' or 'swin-K-noncyclic': Sliding window of size K
            - 'logwin-K' or 'logwin-K-noncyclic': Log-spaced window (powers of 2)
            - 'oneref-K': One reference image (idx K) paired with all others
            - 'retrieval-Na-k': Use similarity matrix for pairing
        prefilter: Optional filter ('seqN' or 'cycN' for sequential distance)
        symmetrize: If True, add reverse pairs (img2, img1)
        sim_mat: Similarity matrix for retrieval mode

    Returns:
        List of (img1, img2) tuples
    """
    pairs = []
    n = len(imgs)

    if scene_graph == "complete":
        # Complete graph: all pairs
        for i in range(n):
            for j in range(i):
                pairs.append((imgs[i], imgs[j]))

    elif scene_graph.startswith("swin"):
        # Sliding window
        iscyclic = not scene_graph.endswith("noncyclic")
        try:
            winsize = int(scene_graph.split("-")[1])
        except (IndexError, ValueError):
            winsize = 3

        pairsid = set()
        for i in range(n):
            for j in range(1, winsize + 1):
                idx = i + j
                if iscyclic:
                    idx = idx % n
                if idx >= n:
                    continue
                pairsid.add((i, idx) if i < idx else (idx, i))

        for i, j in pairsid:
            pairs.append((imgs[i], imgs[j]))

    elif scene_graph.startswith("logwin"):
        # Logarithmic window (powers of 2)
        iscyclic = not scene_graph.endswith("noncyclic")
        try:
            winsize = int(scene_graph.split("-")[1])
        except (IndexError, ValueError):
            winsize = 3

        offsets = [2**i for i in range(winsize)]
        pairsid = set()

        for i in range(n):
            ixs_l = [i - off for off in offsets]
            ixs_r = [i + off for off in offsets]
            for j in ixs_l + ixs_r:
                if iscyclic:
                    j = j % n
                if j < 0 or j >= n or j == i:
                    continue
                pairsid.add((i, j) if i < j else (j, i))

        for i, j in pairsid:
            pairs.append((imgs[i], imgs[j]))

    elif scene_graph.startswith("oneref"):
        # One reference image paired with all others
        try:
            refid = int(scene_graph.split("-")[1])
        except (IndexError, ValueError):
            refid = 0

        for j in range(n):
            if j != refid:
                pairs.append((imgs[refid], imgs[j]))

    elif scene_graph.startswith("retrieval"):
        # Retrieval-based pairing using similarity matrix
        parts = scene_graph.split("-")
        if len(parts) != 3:
            raise ValueError(f"retrieval mode requires format 'retrieval-Na-k', got {scene_graph}")

        Na = int(parts[1])
        k = int(parts[2])

        if sim_mat is None:
            raise ValueError("sim_mat is required for retrieval mode")

        fps_pairs, _ = make_pairs_fps(sim_mat, Na=Na, topK=k)
        for i, j in fps_pairs:
            pairs.append((imgs[i], imgs[j]))

    else:
        raise ValueError(f"Unrecognized scene graph: {scene_graph}")

    # Symmetrize: add reverse pairs
    if symmetrize:
        pairs += [(img2, img1) for img1, img2 in pairs]

    # Apply prefilter
    if isinstance(prefilter, str) and prefilter.startswith("seq"):
        pairs = filter_pairs_seq(pairs, int(prefilter[3:]))

    if isinstance(prefilter, str) and prefilter.startswith("cyc"):
        pairs = filter_pairs_seq(pairs, int(prefilter[3:]), cyclic=True)

    return pairs


def make_pairs_fps(
    sim_mat: np.ndarray,
    Na: int = 20,
    topK: int = 10,
    dist_thresh: float | None = None,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Create pairs using Farthest Point Sampling on similarity matrix.

    Args:
        sim_mat: NxN similarity matrix
        Na: Number of anchor images to select
        topK: Number of neighbors per anchor
        dist_thresh: Optional distance threshold

    Returns:
        Tuple of (pairs list, anchor indices)
    """
    n = len(sim_mat)
    Na = min(Na, n)

    # Convert similarity to distance
    dist_mat = 1.0 - sim_mat

    # FPS to select anchor points
    anchors = []
    min_dists = np.full(n, np.inf)

    # Start with image that has highest total similarity (most central)
    first = np.argmax(sim_mat.sum(axis=1))
    anchors.append(first)
    min_dists = np.minimum(min_dists, dist_mat[first])

    for _ in range(Na - 1):
        # Select point furthest from all anchors
        next_idx = np.argmax(min_dists)
        anchors.append(next_idx)
        min_dists = np.minimum(min_dists, dist_mat[next_idx])

    # For each anchor, get top-K most similar images
    pairs = set()
    for anchor in anchors:
        sims = sim_mat[anchor].copy()
        sims[anchor] = -np.inf  # Exclude self

        # Get top-K neighbors
        neighbors = np.argsort(sims)[-topK:]

        for neighbor in neighbors:
            if dist_thresh is not None and dist_mat[anchor, neighbor] > dist_thresh:
                continue
            pair = (anchor, neighbor) if anchor < neighbor else (neighbor, anchor)
            pairs.add(pair)

    return list(pairs), anchors


def sel(x: Any, kept: list[int]) -> Any:
    """Select elements from various container types.

    Args:
        x: Container (dict, array, list, tuple)
        kept: Indices to keep

    Returns:
        Filtered container of same type
    """
    if isinstance(x, dict):
        return {k: sel(v, kept) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x[kept]
    if isinstance(x, (tuple, list)):
        return type(x)([x[k] for k in kept])
    return x


def _filter_edges_seq(
    edges: list[tuple[int, int]],
    seq_dis_thr: int,
    cyclic: bool = False,
) -> list[int]:
    """Filter edges by sequential distance.

    Args:
        edges: List of (i, j) index pairs
        seq_dis_thr: Maximum allowed sequential distance
        cyclic: Treat sequence as cyclic

    Returns:
        List of indices of kept edges
    """
    if not edges:
        return []

    n = max(max(e) for e in edges) + 1

    kept = []
    for e, (i, j) in enumerate(edges):
        dis = abs(i - j)
        if cyclic:
            dis = min(dis, abs(i + n - j), abs(i - n - j))
        if dis <= seq_dis_thr:
            kept.append(e)
    return kept


def filter_pairs_seq(
    pairs: list[tuple[dict, dict]],
    seq_dis_thr: int,
    cyclic: bool = False,
) -> list[tuple[dict, dict]]:
    """Filter pairs by sequential distance between image indices.

    Args:
        pairs: List of (img1, img2) tuples with 'idx' keys
        seq_dis_thr: Maximum allowed sequential distance
        cyclic: Treat sequence as cyclic

    Returns:
        Filtered pairs
    """
    edges = [(img1["idx"], img2["idx"]) for img1, img2 in pairs]
    kept = _filter_edges_seq(edges, seq_dis_thr, cyclic=cyclic)
    return [pairs[i] for i in kept]


def filter_edges_seq(
    view1: dict,
    view2: dict,
    pred1: dict,
    pred2: dict,
    seq_dis_thr: int,
    cyclic: bool = False,
) -> tuple[dict, dict, dict, dict]:
    """Filter edge data by sequential distance.

    Args:
        view1, view2: View dicts with 'idx' arrays
        pred1, pred2: Prediction dicts
        seq_dis_thr: Maximum allowed sequential distance
        cyclic: Treat sequence as cyclic

    Returns:
        Filtered (view1, view2, pred1, pred2)
    """
    edges = [(int(i), int(j)) for i, j in zip(view1["idx"], view2["idx"])]
    kept = _filter_edges_seq(edges, seq_dis_thr, cyclic=cyclic)
    print(f">> Filtering edges > {seq_dis_thr} frames apart: kept {len(kept)}/{len(edges)}")
    return sel(view1, kept), sel(view2, kept), sel(pred1, kept), sel(pred2, kept)


def get_pairs_info(pairs: list[tuple[dict, dict]]) -> dict:
    """Get statistics about image pairs.

    Args:
        pairs: List of (img1, img2) tuples

    Returns:
        Dict with pair statistics
    """
    if not pairs:
        return {"n_pairs": 0, "n_images": 0}

    all_indices = set()
    for img1, img2 in pairs:
        all_indices.add(img1.get("idx", 0))
        all_indices.add(img2.get("idx", 0))

    return {
        "n_pairs": len(pairs),
        "n_images": len(all_indices),
        "indices": sorted(all_indices),
    }
