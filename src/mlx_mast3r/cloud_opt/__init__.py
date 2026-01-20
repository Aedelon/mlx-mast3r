"""Cloud optimization module for MLX-MASt3R.

This module provides scene reconstruction and global alignment utilities,
adapted from dust3r/mast3r for use with MLX.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
Original dust3r code: Copyright (C) 2024-present Naver Corporation. CC BY-NC-SA 4.0.
"""

from .geometry import (
    depthmap_to_pts3d,
    depthmap_to_pts3d_mlx,
    geotrf,
    geotrf_mlx,
    inv,
    inv_mlx,
    xy_grid,
    xy_grid_mlx,
)
from .focal import estimate_focal
from .pair_viewer import PairViewer
from .scene import Scene

# Optimization modules
from .losses import gamma_loss, l1_loss, l2_loss, reprojection_loss
from .schedules import LRScheduler, cosine_schedule, linear_schedule
from .sparse_ga import SparseGAResult, sparse_global_alignment
from .tsdf import TSDFPostProcess, apply_tsdf_cleaning, clean_pointcloud

# Extracted modules for sparse GA
from .canonical import compute_canonical_view, condense_data, prepare_canonical_data
from .correspondences import anchor_depth_offsets, extract_correspondences
from .focal_estimation import estimate_focal_from_depth
from .utils import hash_md5, mkdir_for

__all__ = [
    # Geometry (numpy)
    "inv",
    "geotrf",
    "xy_grid",
    "depthmap_to_pts3d",
    # Geometry (MLX native)
    "inv_mlx",
    "geotrf_mlx",
    "xy_grid_mlx",
    "depthmap_to_pts3d_mlx",
    # Focal estimation
    "estimate_focal",
    "estimate_focal_from_depth",
    # Viewers
    "PairViewer",
    "Scene",
    # Losses
    "gamma_loss",
    "l1_loss",
    "l2_loss",
    "reprojection_loss",
    # Schedules
    "LRScheduler",
    "cosine_schedule",
    "linear_schedule",
    # Sparse GA
    "SparseGAResult",
    "sparse_global_alignment",
    # Sparse GA utilities
    "compute_canonical_view",
    "condense_data",
    "prepare_canonical_data",
    "extract_correspondences",
    "anchor_depth_offsets",
    "hash_md5",
    "mkdir_for",
    # TSDF Post-processing
    "TSDFPostProcess",
    "apply_tsdf_cleaning",
    "clean_pointcloud",
]
