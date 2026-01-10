# Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""Fused Bilinear Upsample 2x Metal Kernel for MLX.

This kernel fuses the bilinear upsampling operation into a single Metal kernel,
avoiding intermediate allocations for gather/reshape/multiply operations.

Speedup: ~2x compared to the non-fused MLX implementation.
"""

from __future__ import annotations

import mlx.core as mx

# Metal kernel source for fused bilinear upsample 2x (align_corners=True)
# Vectorized version: process 4 channels per thread when C % 4 == 0
_BILINEAR_UPSAMPLE_2X_KERNEL_SOURCE = """
    // Thread index = output pixel index (not element)
    // Each thread handles 4 consecutive channels
    uint tid = thread_position_in_grid.x;

    // Input dimensions from shape [B, H, W, C]
    uint B = x_shape[0];
    uint H = x_shape[1];
    uint W = x_shape[2];
    uint C = x_shape[3];

    // Output dimensions
    uint out_H = H * 2;
    uint out_W = W * 2;

    // Vectorized: 4 channels per thread
    uint C4 = C / 4;

    // Total work items (each handles 4 channels)
    uint total = B * out_H * out_W * C4;
    if (tid >= total) return;

    // Decode output indices: [b, oh, ow, c4]
    uint c4 = tid % C4;
    uint ow = (tid / C4) % out_W;
    uint oh = (tid / (C4 * out_W)) % out_H;
    uint b = tid / (C4 * out_W * out_H);

    // Compute source coordinates (align_corners=True)
    float src_h = (out_H > 1) ? (float(oh) * float(H - 1) / float(out_H - 1)) : 0.0f;
    float src_w = (out_W > 1) ? (float(ow) * float(W - 1) / float(out_W - 1)) : 0.0f;

    // Floor indices
    int h0 = int(floor(src_h));
    int w0 = int(floor(src_w));
    int h1 = min(h0 + 1, int(H) - 1);
    int w1 = min(w0 + 1, int(W) - 1);

    // Fractional parts for interpolation weights
    float fh = src_h - float(h0);
    float fw = src_w - float(w0);

    // Bilinear weights (compute once, reuse for all 4 channels)
    float w00 = (1.0f - fh) * (1.0f - fw);
    float w01 = (1.0f - fh) * fw;
    float w10 = fh * (1.0f - fw);
    float w11 = fh * fw;

    // Input strides for [B, H, W, C] layout
    uint w_stride = C;
    uint h_stride = W * C;
    uint b_stride = H * W * C;

    // Base index for channel group
    uint c_base = c4 * 4;
    uint base = b * b_stride + c_base;

    // Gather 4 corner values for 4 channels (16 loads -> 4 vec4 loads)
    uint idx00 = base + h0 * h_stride + w0 * w_stride;
    uint idx01 = base + h0 * h_stride + w1 * w_stride;
    uint idx10 = base + h1 * h_stride + w0 * w_stride;
    uint idx11 = base + h1 * h_stride + w1 * w_stride;

    // Load 4 channels at once using vec4
    // Note: channels are contiguous, so this is coalesced
    T p00_0 = x[idx00], p00_1 = x[idx00+1], p00_2 = x[idx00+2], p00_3 = x[idx00+3];
    T p01_0 = x[idx01], p01_1 = x[idx01+1], p01_2 = x[idx01+2], p01_3 = x[idx01+3];
    T p10_0 = x[idx10], p10_1 = x[idx10+1], p10_2 = x[idx10+2], p10_3 = x[idx10+3];
    T p11_0 = x[idx11], p11_1 = x[idx11+1], p11_2 = x[idx11+2], p11_3 = x[idx11+3];

    // Bilinear interpolation for all 4 channels
    T tw00 = T(w00), tw01 = T(w01), tw10 = T(w10), tw11 = T(w11);

    // Output indices (4 consecutive channels)
    uint out_base = b * (out_H * out_W * C) + oh * (out_W * C) + ow * C + c_base;

    out[out_base]     = tw00 * p00_0 + tw01 * p01_0 + tw10 * p10_0 + tw11 * p11_0;
    out[out_base + 1] = tw00 * p00_1 + tw01 * p01_1 + tw10 * p10_1 + tw11 * p11_1;
    out[out_base + 2] = tw00 * p00_2 + tw01 * p01_2 + tw10 * p10_2 + tw11 * p11_2;
    out[out_base + 3] = tw00 * p00_3 + tw01 * p01_3 + tw10 * p10_3 + tw11 * p11_3;
"""

# Fallback kernel for C not divisible by 4
_BILINEAR_UPSAMPLE_2X_KERNEL_SCALAR = """
    uint tid = thread_position_in_grid.x;
    uint B = x_shape[0], H = x_shape[1], W = x_shape[2], C = x_shape[3];
    uint out_H = H * 2, out_W = W * 2;
    uint total = B * out_H * out_W * C;
    if (tid >= total) return;

    uint c = tid % C;
    uint ow = (tid / C) % out_W;
    uint oh = (tid / (C * out_W)) % out_H;
    uint b = tid / (C * out_W * out_H);

    float src_h = (out_H > 1) ? (float(oh) * float(H - 1) / float(out_H - 1)) : 0.0f;
    float src_w = (out_W > 1) ? (float(ow) * float(W - 1) / float(out_W - 1)) : 0.0f;

    int h0 = int(floor(src_h)), w0 = int(floor(src_w));
    int h1 = min(h0 + 1, int(H) - 1), w1 = min(w0 + 1, int(W) - 1);
    float fh = src_h - float(h0), fw = src_w - float(w0);
    float w00 = (1.0f - fh) * (1.0f - fw), w01 = (1.0f - fh) * fw;
    float w10 = fh * (1.0f - fw), w11 = fh * fw;

    uint w_stride = C, h_stride = W * C, b_stride = H * W * C;
    uint base = b * b_stride + c;
    T p00 = x[base + h0 * h_stride + w0 * w_stride];
    T p01 = x[base + h0 * h_stride + w1 * w_stride];
    T p10 = x[base + h1 * h_stride + w0 * w_stride];
    T p11 = x[base + h1 * h_stride + w1 * w_stride];
    out[tid] = T(w00) * p00 + T(w01) * p01 + T(w10) * p10 + T(w11) * p11;
"""

# Kernel instances (created on first use)
_bilinear_upsample_2x_kernel_vec4 = None
_bilinear_upsample_2x_kernel_scalar = None


def _get_bilinear_upsample_2x_kernel(vectorized: bool = True):
    """Get or create the bilinear upsample 2x kernel."""
    global _bilinear_upsample_2x_kernel_vec4, _bilinear_upsample_2x_kernel_scalar

    if vectorized:
        if _bilinear_upsample_2x_kernel_vec4 is None:
            _bilinear_upsample_2x_kernel_vec4 = mx.fast.metal_kernel(
                name="bilinear_upsample_2x_vec4",
                input_names=["x"],
                output_names=["out"],
                source=_BILINEAR_UPSAMPLE_2X_KERNEL_SOURCE,
                header="",
                ensure_row_contiguous=True,
            )
        return _bilinear_upsample_2x_kernel_vec4
    else:
        if _bilinear_upsample_2x_kernel_scalar is None:
            _bilinear_upsample_2x_kernel_scalar = mx.fast.metal_kernel(
                name="bilinear_upsample_2x_scalar",
                input_names=["x"],
                output_names=["out"],
                source=_BILINEAR_UPSAMPLE_2X_KERNEL_SCALAR,
                header="",
                ensure_row_contiguous=True,
            )
        return _bilinear_upsample_2x_kernel_scalar


def bilinear_upsample_2x_fused(x: mx.array) -> mx.array:
    """Bilinear upsample 2x using fused Metal kernel.

    Uses vectorized kernel (4 channels/thread) when C % 4 == 0 for better performance.

    Args:
        x: Input tensor [B, H, W, C] in NHWC format.

    Returns:
        Output tensor [B, 2H, 2W, C] upsampled with bilinear interpolation.
        Uses align_corners=True to match PyTorch behavior.
    """
    B, H, W, C = x.shape
    out_H, out_W = H * 2, W * 2

    # Use vectorized kernel if C is divisible by 4
    use_vec4 = (C % 4 == 0)
    kernel = _get_bilinear_upsample_2x_kernel(vectorized=use_vec4)

    # Determine template type
    if x.dtype == mx.float16:
        template_type = ("T", mx.float16)
    elif x.dtype == mx.bfloat16:
        template_type = ("T", mx.bfloat16)
    else:
        template_type = ("T", mx.float32)

    # Grid size depends on kernel type
    if use_vec4:
        # Each thread handles 4 channels
        total_work = B * out_H * out_W * (C // 4)
    else:
        # Each thread handles 1 element
        total_work = B * out_H * out_W * C

    threads_per_group = 256

    # Run kernel
    outputs = kernel(
        inputs=[x],
        template=[template_type],
        grid=(total_work, 1, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[(B, out_H, out_W, C)],
        output_dtypes=[x.dtype],
    )

    return outputs[0]


def _bilinear_upsample_2x_reference(x: mx.array) -> mx.array:
    """Reference implementation for benchmarking (non-fused)."""
    import functools

    @functools.lru_cache(maxsize=4)
    def _compute_params(H: int, W: int, dtype_str: str):
        dtype = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}[dtype_str]
        out_H, out_W = H * 2, W * 2
        oh = mx.arange(out_H, dtype=mx.float32)
        ow = mx.arange(out_W, dtype=mx.float32)
        src_h = oh * (H - 1) / (out_H - 1) if out_H > 1 else mx.zeros_like(oh)
        src_w = ow * (W - 1) / (out_W - 1) if out_W > 1 else mx.zeros_like(ow)
        h0 = mx.floor(src_h).astype(mx.int32)
        w0 = mx.floor(src_w).astype(mx.int32)
        h1 = mx.minimum(h0 + 1, H - 1)
        w1 = mx.minimum(w0 + 1, W - 1)
        fh = src_h - h0.astype(mx.float32)
        fw = src_w - w0.astype(mx.float32)
        fh_2d, fw_2d = fh[:, None], fw[None, :]
        w00 = ((1 - fh_2d) * (1 - fw_2d)).astype(dtype)[:, :, None]
        w01 = ((1 - fh_2d) * fw_2d).astype(dtype)[:, :, None]
        w10 = (fh_2d * (1 - fw_2d)).astype(dtype)[:, :, None]
        w11 = (fh_2d * fw_2d).astype(dtype)[:, :, None]
        h0_2d = mx.broadcast_to(h0[:, None], (out_H, out_W))
        w0_2d = mx.broadcast_to(w0[None, :], (out_H, out_W))
        h1_2d = mx.broadcast_to(h1[:, None], (out_H, out_W))
        w1_2d = mx.broadcast_to(w1[None, :], (out_H, out_W))
        idx00 = (h0_2d * W + w0_2d).flatten().astype(mx.int32)
        idx01 = (h0_2d * W + w1_2d).flatten().astype(mx.int32)
        idx10 = (h1_2d * W + w0_2d).flatten().astype(mx.int32)
        idx11 = (h1_2d * W + w1_2d).flatten().astype(mx.int32)
        return (idx00, idx01, idx10, idx11, w00, w01, w10, w11)

    B, H, W, C = x.shape
    out_H, out_W = H * 2, W * 2
    dtype_str = {mx.float32: "float32", mx.float16: "float16", mx.bfloat16: "bfloat16"}[x.dtype]
    idx00, idx01, idx10, idx11, w00, w01, w10, w11 = _compute_params(H, W, dtype_str)
    x_flat = x.reshape(B, H * W, C)
    p00 = x_flat[:, idx00, :].reshape(B, out_H, out_W, C)
    p01 = x_flat[:, idx01, :].reshape(B, out_H, out_W, C)
    p10 = x_flat[:, idx10, :].reshape(B, out_H, out_W, C)
    p11 = x_flat[:, idx11, :].reshape(B, out_H, out_W, C)
    return p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11


def benchmark_bilinear_upsample(
    B: int = 1,
    H: int = 32,
    W: int = 42,
    C: int = 256,
    iterations: int = 100,
) -> None:
    """Benchmark fused vs non-fused bilinear upsample 2x."""
    import time

    # Setup
    x = mx.random.normal((B, H, W, C)).astype(mx.float16)
    mx.eval(x)

    # Warmup
    for _ in range(5):
        out1 = _bilinear_upsample_2x_reference(x)
        mx.eval(out1)
        out2 = bilinear_upsample_2x_fused(x)
        mx.eval(out2)

    # Benchmark non-fused (reference)
    t0 = time.perf_counter()
    for _ in range(iterations):
        out1 = _bilinear_upsample_2x_reference(x)
        mx.eval(out1)
    non_fused_time = (time.perf_counter() - t0) / iterations * 1000

    # Benchmark fused
    t0 = time.perf_counter()
    for _ in range(iterations):
        out2 = bilinear_upsample_2x_fused(x)
        mx.eval(out2)
    fused_time = (time.perf_counter() - t0) / iterations * 1000

    # Verify correctness
    out1 = _bilinear_upsample_2x_reference(x)
    out2 = bilinear_upsample_2x_fused(x)
    mx.eval(out1, out2)

    match = mx.allclose(out1, out2, atol=1e-2, rtol=1e-2)

    print(f"Bilinear Upsample 2x Benchmark (B={B}, H={H}, W={W}, C={C}):")
    print(f"  Non-fused: {non_fused_time:.3f}ms")
    print(f"  Fused:     {fused_time:.3f}ms")
    print(f"  Speedup:   {non_fused_time / fused_time:.2f}x")
    print(f"  Correct:   {match.item()}")


if __name__ == "__main__":
    benchmark_bilinear_upsample()
