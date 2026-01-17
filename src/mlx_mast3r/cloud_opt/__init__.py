"""Cloud optimization module for MLX-MASt3R.

This module provides scene reconstruction and global alignment utilities,
adapted from dust3r/mast3r for use with MLX.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
Original dust3r code: Copyright (C) 2024-present Naver Corporation. CC BY-NC-SA 4.0.
"""

from .geometry import depthmap_to_pts3d, geotrf, inv, xy_grid
from .focal import estimate_focal
from .pair_viewer import PairViewer
from .scene import Scene

# Optimization modules
from .losses import gamma_loss, l1_loss, l2_loss, reprojection_loss
from .schedules import LRScheduler, cosine_schedule, linear_schedule
from .sparse_ga import SparseGAResult, sparse_global_alignment
from .tsdf import TSDFPostProcess, apply_tsdf_cleaning, clean_pointcloud

__all__ = [
    # Geometry
    "inv",
    "geotrf",
    "xy_grid",
    "depthmap_to_pts3d",
    # Focal estimation
    "estimate_focal",
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
    # TSDF Post-processing
    "TSDFPostProcess",
    "apply_tsdf_cleaning",
    "clean_pointcloud",
]
