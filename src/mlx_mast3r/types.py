# Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""Type definitions for MLX-MASt3R.

This module provides TypedDict and Protocol definitions for better type safety
and IDE autocompletion throughout the codebase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict

import numpy as np

if TYPE_CHECKING:
    import mlx.core as mx


# =============================================================================
# Correspondence Types
# =============================================================================


class CorrespondenceDict(TypedDict):
    """Correspondence data between two images.

    Used in sparse global alignment for matching points between views.
    """

    idx1: int  # Index of first image
    idx2: int  # Index of second image
    slice1: slice  # Slice into pts3d[idx1] for anchor data
    slice2: slice  # Slice into pts3d[idx2] for anchor data
    pts1: "mx.array"  # [N, 2] 2D pixel coordinates in image 1
    pts2: "mx.array"  # [N, 2] 2D pixel coordinates in image 2
    weights: "mx.array"  # [N] confidence weights
    max_conf: float  # Maximum confidence (for filtering)


class CorrespondenceSlice(TypedDict):
    """Lightweight correspondence reference using slices."""

    idx1: int
    idx2: int
    slice1: slice
    slice2: slice
    max_conf: float


# =============================================================================
# Prediction Types
# =============================================================================


class Prediction21Entry(TypedDict):
    """Cross-prediction entry: pts3d and confidence from view 2 in view 1's frame."""

    pts3d: "mx.array"  # [N, 3] 3D points
    conf: "mx.array"  # [N] confidence values


# preds_21 structure: {img1_path: {img2_path: (pts3d, conf)}}
Preds21Dict = dict[str, dict[str, tuple["mx.array", "mx.array"]]]


# =============================================================================
# Anchor Types
# =============================================================================


class AnchorData(TypedDict):
    """Anchor data for depth interpolation.

    Used for anchor-based densification (PyTorch-faithful).
    """

    pixels: "mx.array"  # [N, 2] pixel coordinates
    confs: "mx.array"  # [N] confidence values


# img_anchors structure: {img_idx: (pixels, idxs, offsets)}
ImgAnchorsDict = dict[int, tuple["mx.array", "mx.array", "mx.array"]]


# =============================================================================
# Canonical View Types
# =============================================================================


class CanonicalViewDict(TypedDict):
    """Canonical view data for an image."""

    pp: "mx.array"  # [2] principal point
    shape: tuple[int, int]  # (H, W) image shape
    focal: "mx.array"  # Estimated focal length
    core_depth: "mx.array"  # Subsampled depth map
    pixels: dict[str, tuple["mx.array", "mx.array", "mx.array"]]  # {other_img: (pts1, pts2, conf)}
    anchor_idxs: dict[str, "mx.array"]  # {other_img: indices}
    anchor_offs: dict[str, "mx.array"]  # {other_img: offsets}
    conf: "mx.array"  # [H, W] confidence map


# =============================================================================
# Model Output Types
# =============================================================================


class ReconstructionOutput(TypedDict):
    """Output from stereo reconstruction (reconstruct method)."""

    pts3d: "mx.array"  # [H, W, 3] 3D points
    conf: "mx.array"  # [H, W] confidence map
    desc: "mx.array"  # [H, W, D] descriptors
    desc_conf: "mx.array"  # [H, W] descriptor confidence


class ForwardOutput(TypedDict):
    """Output from model forward pass."""

    pts3d: "mx.array"
    conf: "mx.array"
    desc: "mx.array"
    desc_conf: "mx.array"


# =============================================================================
# Optimizer Result Types
# =============================================================================


class OptimizerResult(TypedDict):
    """Result from sparse_scene_optimizer."""

    intrinsics: "mx.array"  # [N, 3, 3] intrinsic matrices
    cam2w: "mx.array"  # [N, 4, 4] camera-to-world poses
    depthmaps: list["mx.array"]  # [H_sub, W_sub] depth maps
    focals: "mx.array"  # [N] focal lengths
    pts3d: dict[int, "mx.array"]  # {img_idx: [M, 3] 3D points}
    pps_norm: "mx.array"  # [N, 2] normalized principal points
    log_sizes: "mx.array"  # [N] log scale factors


# =============================================================================
# Image Data Types
# =============================================================================


class ImageDataDict(TypedDict):
    """Image data dictionary used in pairs."""

    img: "mx.array"  # [1, C, H, W] image tensor
    true_shape: np.ndarray  # [2] original shape (H, W)
    idx: int  # Image index
    instance: str  # Image path/identifier


# =============================================================================
# Protocols (Structural Subtyping)
# =============================================================================


class Reconstructor(Protocol):
    """Protocol for models that can perform stereo reconstruction."""

    def reconstruct(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
    ) -> tuple[dict, dict]:
        """Reconstruct 3D from stereo pair.

        Args:
            img1: [H, W, 3] first image
            img2: [H, W, 3] second image

        Returns:
            Tuple of (output1, output2) dicts with pts3d, conf, desc
        """
        ...


class FeatureEncoder(Protocol):
    """Protocol for feature encoders (DUNE, MASt3R encoder)."""

    embed_dim: int
    num_patches: int

    def encode(self, img: np.ndarray) -> np.ndarray:
        """Encode image to features.

        Args:
            img: [H, W, 3] input image

        Returns:
            [N, D] feature vectors
        """
        ...


# =============================================================================
# Pair Types
# =============================================================================

# Image pair: (img1_dict, img2_dict)
ImagePair = tuple[ImageDataDict, ImageDataDict]

# Forward result paths: ((path1, path2), path_corres)
ForwardPaths = tuple[tuple[str, str], str]

# Pairs data: {(img1_path, img2_path): ForwardPaths}
PairsDataDict = dict[tuple[str, str], ForwardPaths]
