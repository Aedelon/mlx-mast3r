# Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""Custom Metal kernels for MLX-MASt3R.

Note on kernel effectiveness:
- Kernels for COMPLEX memory access patterns provide speedups (bilinear, rope2d, grid_sample)
- Kernels for ELEMENT-WISE ops (bias+gelu) are slower because MLX fuses them automatically
"""

from mlx_mast3r.kernels.bilinear import benchmark_bilinear_upsample, bilinear_upsample_2x_fused
from mlx_mast3r.kernels.grid_sample import benchmark_grid_sample, grid_sample
from mlx_mast3r.kernels.rope2d import apply_rope_2d_fused, benchmark_rope2d

__all__ = [
    # RoPE 2D - 2x speedup (avoids reshape/concat)
    "apply_rope_2d_fused",
    "benchmark_rope2d",
    # Bilinear upsample - 2.7-3.6x speedup (complex gather pattern)
    "bilinear_upsample_2x_fused",
    "benchmark_bilinear_upsample",
    # Grid sample - useful for spatial transforms
    "grid_sample",
    "benchmark_grid_sample",
]
