# Copyright (c) 2024 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""Shared constants for MLX MASt3R."""

# =============================================================================
# Layer Norm
# =============================================================================
LAYER_NORM_EPS: float = 1e-6  # Matches PyTorch default

# =============================================================================
# Descriptor dimensions
# =============================================================================
DESC_DIM: int = 24
DESC_CONF_CHANNELS: int = 1

# =============================================================================
# Focal length constraints (optimizer)
# =============================================================================
# Allow focals to vary by at most ±50% from base estimate
FOCAL_MIN_RATIO: float = 0.5  # Min focal = 0.5 * base_focal
FOCAL_MAX_RATIO: float = 1.5  # Max focal = 1.5 * base_focal
# Absolute bounds relative to image diagonal
FOCAL_MIN_DIAG_RATIO: float = 0.25  # Min focal = 0.25 * diagonal
FOCAL_MAX_DIAG_RATIO: float = 3.0   # Max focal = 3.0 * diagonal

# =============================================================================
# Numeric stability
# =============================================================================
EPS: float = 1e-8  # General epsilon for division safety
MIN_DEPTH: float = 1e-4  # Minimum valid depth value
MIN_SIZE: float = 1e-6  # Minimum scale factor

# =============================================================================
# Optimization defaults (PyTorch MASt3R defaults)
# =============================================================================
DEFAULT_LR1: float = 0.07  # Phase 1 (coarse) learning rate
DEFAULT_LR2: float = 0.01  # Phase 2 (fine) learning rate
DEFAULT_NITER1: int = 300  # Phase 1 iterations
DEFAULT_NITER2: int = 300  # Phase 2 iterations
DEFAULT_SUBSAMPLE: int = 8  # Depth map subsampling factor
DEFAULT_MATCHING_CONF_THR: float = 5.0  # Minimum matching confidence
DEFAULT_LOSS_DUST3R_W: float = 0.01  # DUSt3R loss weight

# =============================================================================
# Loss function parameters
# =============================================================================
GAMMA_LOSS_3D: float = 1.5  # Phase 1: 3D point loss gamma
GAMMA_LOSS_2D: float = 0.5  # Phase 2: 2D reprojection loss gamma
GAMMA_LOSS_DUST3R: float = 1.1  # DUSt3R regularization loss gamma

# =============================================================================
# Adam optimizer
# =============================================================================
ADAM_BETA1: float = 0.9
ADAM_BETA2: float = 0.9
ADAM_EPS: float = 1e-8

# =============================================================================
# RoPE (Rotary Position Embedding)
# =============================================================================
ROPE_THETA: float = 100.0  # Base frequency for 2D RoPE
