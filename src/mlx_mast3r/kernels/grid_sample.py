# Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""Fused Grid Sample Metal Kernel for MLX.

Bilinear grid sampling matching PyTorch's F.grid_sample with align_corners=True.
Useful for spatial transformer networks and feature warping.
"""

from __future__ import annotations

import mlx.core as mx

# Metal kernel source for grid_sample (bilinear, align_corners=True) - scalar version
_GRID_SAMPLE_KERNEL_SCALAR = """
    uint elem = thread_position_in_grid.x;

    // Input dimensions [B, H, W, C]
    uint B = x_shape[0];
    uint H = x_shape[1];
    uint W = x_shape[2];
    uint C = x_shape[3];

    // Grid dimensions [B, gH, gW, 2]
    uint gH = grid_shape[1];
    uint gW = grid_shape[2];

    // Total output elements
    uint total = B * gH * gW * C;
    if (elem >= total) return;

    // Decode output indices
    uint c = elem % C;
    uint gw = (elem / C) % gW;
    uint gh = (elem / (C * gW)) % gH;
    uint b = elem / (C * gW * gH);

    // Get grid coordinates (normalized -1 to 1)
    uint grid_idx = b * (gH * gW * 2) + gh * (gW * 2) + gw * 2;
    float gx = grid[grid_idx];      // x coordinate
    float gy = grid[grid_idx + 1];  // y coordinate

    // Convert to pixel coordinates (align_corners=True)
    float ix = (gx + 1.0f) * (float(W) - 1.0f) / 2.0f;
    float iy = (gy + 1.0f) * (float(H) - 1.0f) / 2.0f;

    // Get corner indices
    int ix_nw = int(floor(ix));
    int iy_nw = int(floor(iy));
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;

    // Compute interpolation weights
    T nw = T((float(ix_se) - ix) * (float(iy_se) - iy));
    T ne = T((ix - float(ix_sw)) * (float(iy_sw) - iy));
    T sw = T((float(ix_ne) - ix) * (iy - float(iy_ne)));
    T se = T((ix - float(ix_nw)) * (iy - float(iy_nw)));

    // Input strides for [B, H, W, C]
    uint w_stride = C;
    uint h_stride = W * C;
    uint b_stride = H * W * C;

    uint base_idx = b * b_stride + c;

    // Sample corners with boundary check (zero padding)
    T val_nw = (iy_nw >= 0 && iy_nw < int(H) && ix_nw >= 0 && ix_nw < int(W))
               ? x[base_idx + iy_nw * h_stride + ix_nw * w_stride] : T(0);
    T val_ne = (iy_ne >= 0 && iy_ne < int(H) && ix_ne >= 0 && ix_ne < int(W))
               ? x[base_idx + iy_ne * h_stride + ix_ne * w_stride] : T(0);
    T val_sw = (iy_sw >= 0 && iy_sw < int(H) && ix_sw >= 0 && ix_sw < int(W))
               ? x[base_idx + iy_sw * h_stride + ix_sw * w_stride] : T(0);
    T val_se = (iy_se >= 0 && iy_se < int(H) && ix_se >= 0 && ix_se < int(W))
               ? x[base_idx + iy_se * h_stride + ix_se * w_stride] : T(0);

    // Bilinear interpolation
    out[elem] = nw * val_nw + ne * val_ne + sw * val_sw + se * val_se;
"""

# Vectorized version: 4 channels per thread
_GRID_SAMPLE_KERNEL_VEC4 = """
    uint tid = thread_position_in_grid.x;

    // Input dimensions [B, H, W, C]
    uint B = x_shape[0];
    uint H = x_shape[1];
    uint W = x_shape[2];
    uint C = x_shape[3];
    uint C4 = C / 4;

    // Grid dimensions [B, gH, gW, 2]
    uint gH = grid_shape[1];
    uint gW = grid_shape[2];

    // Total work items (each handles 4 channels)
    uint total = B * gH * gW * C4;
    if (tid >= total) return;

    // Decode output indices: [b, gh, gw, c4]
    uint c4 = tid % C4;
    uint gw = (tid / C4) % gW;
    uint gh = (tid / (C4 * gW)) % gH;
    uint b = tid / (C4 * gW * gH);

    // Get grid coordinates (same for all 4 channels)
    uint grid_idx = b * (gH * gW * 2) + gh * (gW * 2) + gw * 2;
    float gx = grid[grid_idx];
    float gy = grid[grid_idx + 1];

    // Convert to pixel coordinates (align_corners=True)
    float ix = (gx + 1.0f) * (float(W) - 1.0f) / 2.0f;
    float iy = (gy + 1.0f) * (float(H) - 1.0f) / 2.0f;

    // Get corner indices
    int ix_nw = int(floor(ix));
    int iy_nw = int(floor(iy));
    int ix_ne = ix_nw + 1;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;

    // Interpolation weights (compute once, reuse for 4 channels)
    T nw = T((float(ix_se) - ix) * (float(iy_se) - iy));
    T ne = T((ix - float(ix_nw)) * (float(iy_se) - iy));
    T sw = T((float(ix_se) - ix) * (iy - float(iy_nw)));
    T se = T((ix - float(ix_nw)) * (iy - float(iy_nw)));

    // Input strides
    uint w_stride = C;
    uint h_stride = W * C;
    uint b_stride = H * W * C;

    // Channel base
    uint c_base = c4 * 4;
    uint base_idx = b * b_stride + c_base;

    // Boundary checks (same for all channels at this position)
    bool valid_nw = (iy_nw >= 0 && iy_nw < int(H) && ix_nw >= 0 && ix_nw < int(W));
    bool valid_ne = (iy_nw >= 0 && iy_nw < int(H) && ix_ne >= 0 && ix_ne < int(W));
    bool valid_sw = (iy_sw >= 0 && iy_sw < int(H) && ix_nw >= 0 && ix_nw < int(W));
    bool valid_se = (iy_se >= 0 && iy_se < int(H) && ix_se >= 0 && ix_se < int(W));

    // Corner indices in input
    uint idx_nw = base_idx + iy_nw * h_stride + ix_nw * w_stride;
    uint idx_ne = base_idx + iy_nw * h_stride + ix_ne * w_stride;
    uint idx_sw = base_idx + iy_sw * h_stride + ix_nw * w_stride;
    uint idx_se = base_idx + iy_se * h_stride + ix_se * w_stride;

    // Output base index
    uint out_base = b * (gH * gW * C) + gh * (gW * C) + gw * C + c_base;

    // Process 4 channels
    for (uint i = 0; i < 4; i++) {
        T val_nw = valid_nw ? x[idx_nw + i] : T(0);
        T val_ne = valid_ne ? x[idx_ne + i] : T(0);
        T val_sw = valid_sw ? x[idx_sw + i] : T(0);
        T val_se = valid_se ? x[idx_se + i] : T(0);
        out[out_base + i] = nw * val_nw + ne * val_ne + sw * val_sw + se * val_se;
    }
"""

_grid_sample_kernel_scalar = None
_grid_sample_kernel_vec4 = None


def _get_grid_sample_kernel(vectorized: bool = False):
    """Get or create the grid sample kernel."""
    global _grid_sample_kernel_scalar, _grid_sample_kernel_vec4

    if vectorized:
        if _grid_sample_kernel_vec4 is None:
            _grid_sample_kernel_vec4 = mx.fast.metal_kernel(
                name="grid_sample_vec4",
                input_names=["x", "grid"],
                output_names=["out"],
                source=_GRID_SAMPLE_KERNEL_VEC4,
                header="",
                ensure_row_contiguous=True,
            )
        return _grid_sample_kernel_vec4
    else:
        if _grid_sample_kernel_scalar is None:
            _grid_sample_kernel_scalar = mx.fast.metal_kernel(
                name="grid_sample_scalar",
                input_names=["x", "grid"],
                output_names=["out"],
                source=_GRID_SAMPLE_KERNEL_SCALAR,
                header="",
                ensure_row_contiguous=True,
            )
        return _grid_sample_kernel_scalar


def grid_sample(
    x: mx.array,
    grid: mx.array,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> mx.array:
    """Sample from input using grid coordinates.

    Matches PyTorch's F.grid_sample behavior with align_corners=True.
    Uses vectorized kernel (4 channels/thread) when C % 4 == 0 for better performance.

    Args:
        x: Input tensor [B, H, W, C] (NHWC format)
        grid: Sampling grid [B, gH, gW, 2] with (x, y) coordinates in [-1, 1]
        mode: Interpolation mode ('bilinear' only for now)
        padding_mode: Padding mode ('zeros' only for now)
        align_corners: If True, corner pixels are aligned

    Returns:
        Sampled output [B, gH, gW, C]
    """
    assert x.ndim == 4, "Input must be 4D [B, H, W, C]"
    assert grid.ndim == 4, "Grid must be 4D [B, gH, gW, 2]"
    assert grid.shape[-1] == 2, "Grid last dim must be 2 (x, y)"
    assert mode == "bilinear", "Only bilinear mode supported"
    assert align_corners, "Only align_corners=True supported"

    B, H, W, C = x.shape
    _, gH, gW, _ = grid.shape
    out_shape = (B, gH, gW, C)

    # Use vectorized kernel if C is divisible by 4
    use_vec4 = (C % 4 == 0)
    kernel = _get_grid_sample_kernel(vectorized=use_vec4)

    # Ensure grid is float32 for coordinate calculations
    if grid.dtype != mx.float32:
        grid = grid.astype(mx.float32)

    # Template type
    if x.dtype == mx.float16:
        template_type = ("T", mx.float16)
    elif x.dtype == mx.bfloat16:
        template_type = ("T", mx.bfloat16)
    else:
        template_type = ("T", mx.float32)

    # Grid size depends on kernel type
    if use_vec4:
        total_work = B * gH * gW * (C // 4)
    else:
        total_work = B * gH * gW * C

    threads_per_group = 256

    outputs = kernel(
        inputs=[x, grid],
        template=[template_type],
        grid=(total_work, 1, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[x.dtype],
    )

    return outputs[0]


def benchmark_grid_sample(
    B: int = 1,
    H: int = 64,
    W: int = 84,
    C: int = 256,
    gH: int = 128,
    gW: int = 168,
    iterations: int = 100,
) -> None:
    """Benchmark grid_sample kernel."""
    import time

    # Setup
    x = mx.random.normal((B, H, W, C)).astype(mx.float16)

    # Create sampling grid (identity + small offset)
    gy = mx.linspace(-1, 1, gH)
    gx = mx.linspace(-1, 1, gW)
    grid_y = mx.broadcast_to(gy[:, None], (gH, gW))
    grid_x = mx.broadcast_to(gx[None, :], (gH, gW))
    grid = mx.stack([grid_x, grid_y], axis=-1)[None, :, :, :]  # [1, gH, gW, 2]
    grid = mx.broadcast_to(grid, (B, gH, gW, 2))

    mx.eval(x, grid)

    # Reference: naive Python implementation
    def grid_sample_reference(x, grid):
        B, H, W, C = x.shape
        _, gH, gW, _ = grid.shape

        # Convert grid to pixel coords
        ix = (grid[..., 0] + 1) * (W - 1) / 2
        iy = (grid[..., 1] + 1) * (H - 1) / 2

        # Floor indices
        ix0 = mx.floor(ix).astype(mx.int32)
        iy0 = mx.floor(iy).astype(mx.int32)
        ix1 = mx.minimum(ix0 + 1, W - 1)
        iy1 = mx.minimum(iy0 + 1, H - 1)

        # Clamp
        ix0 = mx.clip(ix0, 0, W - 1)
        iy0 = mx.clip(iy0, 0, H - 1)

        # Weights
        wx = ix - ix0.astype(mx.float32)
        wy = iy - iy0.astype(mx.float32)

        # Gather (simplified, not fully correct but for benchmark)
        # This is just to have something to compare timing
        out = mx.zeros((B, gH, gW, C), dtype=x.dtype)
        return out  # Placeholder

    # Warmup
    for _ in range(5):
        out = grid_sample(x, grid)
        mx.eval(out)

    # Benchmark fused kernel
    t0 = time.perf_counter()
    for _ in range(iterations):
        out = grid_sample(x, grid)
        mx.eval(out)
    fused_time = (time.perf_counter() - t0) / iterations * 1000

    print(f"Grid Sample Benchmark (B={B}, {H}x{W}x{C} -> {gH}x{gW}):")
    print(f"  Fused kernel: {fused_time:.3f}ms")
    print(f"  Output shape: {out.shape}")


if __name__ == "__main__":
    benchmark_grid_sample()
