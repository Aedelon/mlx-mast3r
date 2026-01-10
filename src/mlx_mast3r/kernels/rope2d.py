"""Fused RoPE 2D Metal Kernel for MLX.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.

This kernel fuses the 2D Rotary Position Embedding operation into a single
Metal kernel, avoiding intermediate allocations and memory transfers.

Vectorized version: processes 2 elements per thread (element + its rotate_half pair).
"""

from __future__ import annotations

import mlx.core as mx

# Vectorized kernel: 2 elements per thread (element and its rotate_half pair)
# This exploits the rotate_half structure: [x1, x2] -> [-x2, x1]
# Each thread handles d_local and d_local + D_quarter together
_ROPE2D_KERNEL_VEC2 = """
    uint tid = thread_position_in_grid.x;

    // Dimensions from shape
    uint B = q_shape[0];
    uint H = q_shape[1];      // num_heads
    uint N = q_shape[2];      // sequence length
    uint D = q_shape[3];      // head_dim
    uint D_half = D / 2;      // half of head_dim (y-half or x-half)
    uint D_quarter = D_half / 2;  // quarter for rotate_half pairs

    // Total work items: each thread handles 2 elements (a pair)
    // We have B * H * N * (D / 2) pairs total
    uint total_pairs = B * H * N * D_half;
    if (tid >= total_pairs) return;

    // Decode indices for pairs: [b, h, n, pair_idx]
    // pair_idx in [0, D_half) - each pair_idx covers 2 actual dimensions
    uint pair_idx = tid % D_half;
    uint n = (tid / D_half) % N;
    uint h = (tid / (D_half * N)) % H;
    uint b = tid / (D_half * N * H);

    // Get position for this token
    int pos_y = static_cast<int>(positions[n * 2]);
    int pos_x = static_cast<int>(positions[n * 2 + 1]);

    // Determine if we're in y-half or x-half
    bool is_x_half = (pair_idx >= D_quarter);
    uint d_local = is_x_half ? (pair_idx - D_quarter) : pair_idx;
    int pos = (pair_idx < D_quarter) ? pos_y : pos_x;

    // The two dimensions this thread handles
    // For y-half: d1 = d_local, d2 = d_local + D_quarter
    // For x-half: d1 = D_half + d_local, d2 = D_half + d_local + D_quarter
    uint base_d = is_x_half ? D_half : 0;
    uint d1 = base_d + d_local;
    uint d2 = base_d + d_local + D_quarter;

    // Compute indices into q/k tensors
    uint base_idx = b * (H * N * D) + h * (N * D) + n * D;
    uint idx1 = base_idx + d1;
    uint idx2 = base_idx + d2;

    // Load cos/sin - same for both elements in the pair
    uint cos_sin_idx = pos * D_half + d_local;
    T cos_val = cos_table[cos_sin_idx];
    T sin_val = sin_table[cos_sin_idx];

    // Also load cos/sin for the second element (at d_local + D_quarter)
    uint cos_sin_idx2 = pos * D_half + d_local + D_quarter;
    T cos_val2 = cos_table[cos_sin_idx2];
    T sin_val2 = sin_table[cos_sin_idx2];

    // Load values
    T q1 = q[idx1], q2 = q[idx2];
    T k1 = k[idx1], k2 = k[idx2];

    // Apply RoPE with rotate_half
    // rotate_half([x1, x2]) = [-x2, x1]
    // For d1 (first half of pair): out = val * cos + (-pair) * sin = val * cos - pair * sin
    // For d2 (second half of pair): out = val * cos + pair * sin
    q_out[idx1] = q1 * cos_val - q2 * sin_val;
    q_out[idx2] = q2 * cos_val2 + q1 * sin_val2;
    k_out[idx1] = k1 * cos_val - k2 * sin_val;
    k_out[idx2] = k2 * cos_val2 + k1 * sin_val2;
"""

# Scalar fallback kernel (1 element per thread)
_ROPE2D_KERNEL_SCALAR = """
    uint tid = thread_position_in_grid.x;

    uint B = q_shape[0];
    uint H = q_shape[1];
    uint N = q_shape[2];
    uint D = q_shape[3];
    uint D_half = D / 2;

    uint total_elements = B * H * N * D;
    if (tid >= total_elements) return;

    uint d = tid % D;
    uint n = (tid / D) % N;
    uint h = (tid / (D * N)) % H;
    uint b = tid / (D * N * H);

    int pos_y = static_cast<int>(positions[n * 2]);
    int pos_x = static_cast<int>(positions[n * 2 + 1]);

    bool is_x_half = (d >= D_half);
    uint d_local = is_x_half ? (d - D_half) : d;
    int pos = is_x_half ? pos_x : pos_y;

    uint cos_sin_idx = pos * D_half + d_local;
    T cos_val = cos_table[cos_sin_idx];
    T sin_val = sin_table[cos_sin_idx];

    T q_val = q[tid];
    T k_val = k[tid];

    uint D_quarter = D_half / 2;
    uint d_pair;
    T sign;

    if (d_local < D_quarter) {
        d_pair = d_local + D_quarter;
        sign = static_cast<T>(-1.0f);
    } else {
        d_pair = d_local - D_quarter;
        sign = static_cast<T>(1.0f);
    }

    uint paired_d = is_x_half ? (D_half + d_pair) : d_pair;
    uint paired_tid = b * (H * N * D) + h * (N * D) + n * D + paired_d;

    T q_pair = q[paired_tid];
    T k_pair = k[paired_tid];

    q_out[tid] = q_val * cos_val + sign * q_pair * sin_val;
    k_out[tid] = k_val * cos_val + sign * k_pair * sin_val;
"""

# Kernel instances (lazy initialization)
_rope2d_kernel_vec2 = None
_rope2d_kernel_scalar = None


def _get_rope2d_kernel(vectorized: bool = True):
    """Get or create the RoPE 2D kernel."""
    global _rope2d_kernel_vec2, _rope2d_kernel_scalar

    if vectorized:
        if _rope2d_kernel_vec2 is None:
            _rope2d_kernel_vec2 = mx.fast.metal_kernel(
                name="rope2d_fused_vec2",
                input_names=["q", "k", "cos_table", "sin_table", "positions"],
                output_names=["q_out", "k_out"],
                source=_ROPE2D_KERNEL_VEC2,
                header="",
                ensure_row_contiguous=True,
            )
        return _rope2d_kernel_vec2
    else:
        if _rope2d_kernel_scalar is None:
            _rope2d_kernel_scalar = mx.fast.metal_kernel(
                name="rope2d_fused_scalar",
                input_names=["q", "k", "cos_table", "sin_table", "positions"],
                output_names=["q_out", "k_out"],
                source=_ROPE2D_KERNEL_SCALAR,
                header="",
                ensure_row_contiguous=True,
            )
        return _rope2d_kernel_scalar


def apply_rope_2d_fused(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
    positions: mx.array,
) -> tuple[mx.array, mx.array]:
    """Apply 2D RoPE using fused Metal kernel.

    Uses vectorized kernel (2 elements/thread) for better performance.

    Args:
        q, k: [B, nheads, N, head_dim] query and key tensors
        cos, sin: [max_pos, head_dim // 2] precomputed tables
        positions: [N, 2] position indices (y, x) for each token

    Returns:
        (q_rotated, k_rotated) with same shapes as inputs
    """
    # Ensure positions is int32 for indexing
    if positions.dtype != mx.int32:
        positions = positions.astype(mx.int32)

    # Get dimensions
    B, H, N, D = q.shape
    D_half = D // 2

    # Use vectorized kernel (each thread handles 2 elements)
    kernel = _get_rope2d_kernel(vectorized=True)
    total_pairs = B * H * N * D_half

    # Grid and threadgroup sizes
    threads_per_group = 256

    # Determine template type
    if q.dtype == mx.float16:
        template_type = ("T", mx.float16)
    elif q.dtype == mx.bfloat16:
        template_type = ("T", mx.bfloat16)
    else:
        template_type = ("T", mx.float32)

    # Run kernel
    outputs = kernel(
        inputs=[q, k, cos, sin, positions],
        template=[template_type],
        grid=(total_pairs, 1, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[q.shape, k.shape],
        output_dtypes=[q.dtype, k.dtype],
    )

    return outputs[0], outputs[1]


# Benchmark helper
def benchmark_rope2d(B: int = 1, H: int = 16, N: int = 1344, D: int = 64, iterations: int = 100):
    """Benchmark fused vs non-fused RoPE 2D."""
    import time

    from mlx_mast3r.decoders.mast3r import precompute_rope_2d

    # Setup
    height, width = 32, 42  # N = 32 * 42 = 1344
    cos, sin, positions = precompute_rope_2d(height, width, D, dtype=mx.float16)

    q = mx.random.normal((B, H, N, D)).astype(mx.float16)
    k = mx.random.normal((B, H, N, D)).astype(mx.float16)
    mx.eval(q, k, cos, sin, positions)

    # Warmup
    for _ in range(5):
        q2, k2 = apply_rope_2d_fused(q, k, cos, sin, positions)
        mx.eval(q2, k2)

    # Benchmark fused
    t0 = time.perf_counter()
    for _ in range(iterations):
        q2, k2 = apply_rope_2d_fused(q, k, cos, sin, positions)
        mx.eval(q2, k2)
    fused_time = (time.perf_counter() - t0) / iterations * 1000

    print(f"RoPE 2D Benchmark (B={B}, H={H}, N={N}, D={D}):")
    print(f"  Fused vec2: {fused_time:.3f}ms")
    print(f"  Output shapes: q={q2.shape}, k={k2.shape}")


if __name__ == "__main__":
    benchmark_rope2d()
