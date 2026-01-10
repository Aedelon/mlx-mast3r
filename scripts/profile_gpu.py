#!/usr/bin/env python3
"""GPU Profiling for MLX-MASt3R components.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.

Profiles each component of the models to identify bottlenecks.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

SAFETENSORS_DIR = Path.home() / ".cache/mast3r_runtime/safetensors"
DUNE_SAFETENSORS_DIR = Path.home() / ".cache/mast3r_runtime/safetensors"

WARMUP = 3
ITERATIONS = 20


@dataclass
class ProfileResult:
    name: str
    mean_ms: float
    std_ms: float
    pct: float = 0.0

    def __str__(self):
        return f"{self.name:40s} {self.mean_ms:8.2f} ± {self.std_ms:5.2f} ms  ({self.pct:5.1f}%)"


def profile_op(name: str, fn, warmup: int = WARMUP, iterations: int = ITERATIONS) -> ProfileResult:
    """Profile a single operation."""
    # Warmup
    for _ in range(warmup):
        result = fn()
        if isinstance(result, tuple):
            mx.eval(*result)
        else:
            mx.eval(result)

    # Benchmark
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = fn()
        if isinstance(result, tuple):
            mx.eval(*result)
        else:
            mx.eval(result)
        times.append((time.perf_counter() - t0) * 1000)

    return ProfileResult(name, np.mean(times), np.std(times))


def profile_mast3r_encoder():
    """Profile MASt3R ViT-Large encoder components."""
    print("\n" + "=" * 70)
    print("PROFILING: MASt3R ViT-Large Encoder @ 512")
    print("=" * 70)

    from mlx_mast3r.encoders.mast3r import Mast3rEncoderConfig, Mast3rEncoder

    config = Mast3rEncoderConfig(resolution=512, precision="fp16")
    encoder = Mast3rEncoder(config)

    # Load weights
    path = SAFETENSORS_DIR / "mast3r_vit_large" / "unified.safetensors"
    from mlx_mast3r.encoders.mast3r import Mast3rEncoderEngine

    engine = Mast3rEncoderEngine(resolution=512, precision="fp16", compile=False)
    engine.load(path)
    encoder = engine.model

    # Create input
    x = mx.random.normal((1, 512, 672, 3)).astype(mx.float32)
    mx.eval(x)

    results = []

    # 1. Patch embedding
    results.append(profile_op("patch_embed", lambda: encoder.patch_embed(x)))

    # Get patch embedded output for next stages
    x_patched = encoder.patch_embed(x)
    B = x.shape[0]
    H, W = config.patch_h, config.patch_w
    positions = encoder._get_positions(H, W, B)
    mx.eval(x_patched, positions)

    # 2. Individual encoder blocks (sample first, middle, last)
    block_indices = [0, 11, 23]  # First, middle, last of 24 blocks

    # Run all blocks to get intermediate states
    x_blocks = [x_patched]
    temp = x_patched
    for block in encoder.blocks:
        temp = block(temp, positions)
        x_blocks.append(temp)
    mx.eval(*x_blocks)

    for idx in block_indices:
        block = encoder.blocks[idx]
        x_in = x_blocks[idx]
        results.append(
            profile_op(f"block[{idx}] (attention + MLP)", lambda b=block, xi=x_in: b(xi, positions))
        )

    # 3. All 24 blocks together
    def all_blocks():
        temp = x_patched
        for block in encoder.blocks:
            temp = block(temp, positions)
        return temp

    results.append(profile_op("all 24 blocks", all_blocks))

    # 4. Final norm
    x_final = x_blocks[-1]
    results.append(
        profile_op(
            "final norm",
            lambda: mx.fast.layer_norm(x_final, encoder.norm_weight, encoder.norm_bias, eps=1e-6),
        )
    )

    # 5. Full forward pass
    results.append(profile_op("TOTAL forward", lambda: encoder(x)))

    # Calculate percentages
    total_ms = results[-1].mean_ms
    for r in results[:-1]:
        r.pct = (r.mean_ms / total_ms) * 100

    print("\nComponent breakdown:")
    print("-" * 70)
    for r in results:
        print(r)

    return results


def profile_mast3r_decoder():
    """Profile MASt3R decoder components."""
    print("\n" + "=" * 70)
    print("PROFILING: MASt3R Decoder @ 512")
    print("=" * 70)

    from mlx_mast3r.decoders.mast3r import Mast3rDecoderEngine

    engine = Mast3rDecoderEngine(resolution=512, precision="fp16", compile=False)
    engine.load(SAFETENSORS_DIR / "mast3r_vit_large" / "unified.safetensors")

    decoder = engine.decoder

    # Create encoder features (simulated)
    B, N = 1, 1344  # 32 * 42 patches
    feat1 = mx.random.normal((B, N, 1024)).astype(mx.float16)
    feat2 = mx.random.normal((B, N, 1024)).astype(mx.float16)
    mx.eval(feat1, feat2)

    H, W = 32, 42
    shape = (H, W)

    results = []

    # 1. decoder_embed
    results.append(profile_op("decoder_embed", lambda: decoder.decoder_embed(feat1)))

    x1 = decoder.decoder_embed(feat1)
    x2 = decoder.decoder_embed(feat2)
    mx.eval(x1, x2)

    # Initialize RoPE
    decoder._init_rope(H, W)

    # 2. Single decoder block
    blk1 = decoder.dec_blocks[0]
    results.append(profile_op("dec_block[0] (self+cross+MLP)", lambda: blk1(x1, x2)))

    # 3. All 12 decoder blocks (both views)
    def all_decoder_blocks():
        t1, t2 = x1, x2
        for blk1, blk2 in zip(decoder.dec_blocks, decoder.dec_blocks2):
            t1_old, t2_old = t1, t2
            t1 = blk1(t1_old, t2_old)
            t2 = blk2(t2_old, t1_old)
        return t1, t2

    results.append(profile_op("all 12 decoder blocks (both views)", all_decoder_blocks))

    # Get final decoder output for DPT
    x1_out, x2_out = all_decoder_blocks()
    x1_norm = mx.fast.layer_norm(x1_out, decoder.dec_norm_weight, decoder.dec_norm_bias, eps=1e-6)
    mx.eval(x1_norm)

    # 4. DPT Head
    features = [feat1, x1_out, x1_out, x1_norm]  # Simulated hooks
    mx.eval(*features)

    results.append(profile_op("DPT head1", lambda: decoder.head1(features, H, W)))

    # 5. Local features MLP
    cat1 = mx.concatenate([feat1, x1_norm], axis=-1)
    mx.eval(cat1)
    results.append(
        profile_op("head_local_features1 MLP", lambda: decoder.head_local_features1(cat1))
    )

    # 6. Full decoder forward
    results.append(profile_op("TOTAL decoder forward", lambda: decoder(feat1, feat2, shape, shape)))

    # Calculate percentages
    total_ms = results[-1].mean_ms
    for r in results[:-1]:
        r.pct = (r.mean_ms / total_ms) * 100

    print("\nComponent breakdown:")
    print("-" * 70)
    for r in results:
        print(r)

    return results


def profile_mast3r_full():
    """Profile full MASt3R pipeline."""
    print("\n" + "=" * 70)
    print("PROFILING: MASt3R Full Pipeline @ 512")
    print("=" * 70)

    from mlx_mast3r.decoders.mast3r import Mast3rDecoderEngine

    engine = Mast3rDecoderEngine(resolution=512, precision="fp16", compile=False)
    engine.load(SAFETENSORS_DIR / "mast3r_vit_large" / "unified.safetensors")

    # Create input
    x1 = mx.random.normal((1, 512, 672, 3)).astype(mx.float32)
    x2 = mx.random.normal((1, 512, 672, 3)).astype(mx.float32)
    mx.eval(x1, x2)

    results = []

    # 1. Encoder (single image)
    results.append(profile_op("encoder (1 image)", lambda: engine.encoder(x1)))

    # 2. Encoder (both images)
    def encode_both():
        f1 = engine.encoder(x1)
        f2 = engine.encoder(x2)
        return f1, f2

    results.append(profile_op("encoder (2 images)", encode_both))

    # 3. Decoder only
    feat1 = engine.encoder(x1)
    feat2 = engine.encoder(x2)
    mx.eval(feat1, feat2)
    H, W = engine.encoder_config.patch_h, engine.encoder_config.patch_w

    results.append(profile_op("decoder only", lambda: engine.decoder(feat1, feat2, (H, W), (H, W))))

    # 4. Full pipeline
    results.append(profile_op("TOTAL full pipeline", lambda: engine(x1, x2)))

    # Calculate percentages
    total_ms = results[-1].mean_ms
    for r in results[:-1]:
        r.pct = (r.mean_ms / total_ms) * 100

    print("\nComponent breakdown:")
    print("-" * 70)
    for r in results:
        print(r)

    return results


def profile_dune_encoder(variant: str = "base", resolution: int = 336):
    """Profile DUNE encoder components."""
    print("\n" + "=" * 70)
    print(f"PROFILING: DUNE {variant.upper()} Encoder @ {resolution}")
    print("=" * 70)

    from mlx_mast3r.encoders.dune import DuneEncoderEngine

    engine = DuneEncoderEngine(
        variant=variant, resolution=resolution, precision="fp16", compile=False
    )
    path = DUNE_SAFETENSORS_DIR / f"dune_vit_{variant}_{resolution}" / "encoder.safetensors"
    engine.load(path)

    encoder = engine.model
    config = encoder.config

    # Create input
    x = mx.random.normal((1, resolution, resolution, 3)).astype(mx.float32)
    mx.eval(x)

    results = []

    # 1. Patch embedding
    results.append(profile_op("patch_embed", lambda: encoder.patch_embed(x)))

    x_patched = encoder.patch_embed(x)
    mx.eval(x_patched)

    # 2. Position embedding interpolation
    H, W = config.patch_h, config.patch_w
    results.append(
        profile_op("pos_embed interpolation", lambda: encoder._interpolate_pos_embed(H, W))
    )

    # 3. Single encoder block
    B = 1
    cls_tokens = mx.broadcast_to(encoder.cls_token, (B, 1, config.embed_dim))
    x_with_cls = mx.concatenate([cls_tokens, x_patched], axis=1)
    pos_embed = encoder._interpolate_pos_embed(H, W)
    x_pos = x_with_cls + pos_embed.astype(x_with_cls.dtype)

    if config.num_register_tokens > 0:
        reg_tokens = mx.broadcast_to(
            encoder.register_tokens, (B, config.num_register_tokens, config.embed_dim)
        )
        x_pos = mx.concatenate([x_pos[:, :1], reg_tokens, x_pos[:, 1:]], axis=1)
    mx.eval(x_pos)

    block = encoder.blocks[0]
    results.append(profile_op("block[0] (attention + MLP)", lambda: block(x_pos)))

    # 4. All blocks
    def all_blocks():
        temp = x_pos
        for block in encoder.blocks:
            temp = block(temp)
        return temp

    results.append(profile_op(f"all {config.depth} blocks", all_blocks))

    # 5. Full forward
    results.append(profile_op("TOTAL forward", lambda: encoder(x)))

    # Calculate percentages
    total_ms = results[-1].mean_ms
    for r in results[:-1]:
        r.pct = (r.mean_ms / total_ms) * 100

    print("\nComponent breakdown:")
    print("-" * 70)
    for r in results:
        print(r)

    return results


def profile_attention_mlp_breakdown():
    """Detailed breakdown of attention and MLP operations in transformer block."""
    print("\n" + "=" * 70)
    print("PROFILING: Transformer Block Detailed Breakdown")
    print("=" * 70)

    # Typical dimensions for MASt3R encoder
    B, N, D = 1, 1344, 1024
    num_heads, head_dim = 16, 64
    mlp_ratio = 4
    mlp_hidden = D * mlp_ratio  # 4096

    x = mx.random.normal((B, N, D)).astype(mx.float16)
    mx.eval(x)

    # Create weights
    norm_weight = mx.ones((D,)).astype(mx.float16)
    norm_bias = mx.zeros((D,)).astype(mx.float16)
    qkv_weight = mx.random.normal((3 * D, D)).astype(mx.float16)
    qkv_bias = mx.random.normal((3 * D,)).astype(mx.float16)
    proj_weight = mx.random.normal((D, D)).astype(mx.float16)
    proj_bias = mx.random.normal((D,)).astype(mx.float16)
    fc1_weight = mx.random.normal((mlp_hidden, D)).astype(mx.float16)
    fc1_bias = mx.random.normal((mlp_hidden,)).astype(mx.float16)
    fc2_weight = mx.random.normal((D, mlp_hidden)).astype(mx.float16)
    fc2_bias = mx.random.normal((D,)).astype(mx.float16)
    mx.eval(norm_weight, norm_bias, qkv_weight, qkv_bias, proj_weight, proj_bias)
    mx.eval(fc1_weight, fc1_bias, fc2_weight, fc2_bias)

    results = []

    # ===== ATTENTION BLOCK =====
    print("\n--- ATTENTION BLOCK ---")

    # 1. LayerNorm (pre-attention)
    results.append(
        profile_op(
            "LayerNorm (pre-attention)",
            lambda: mx.fast.layer_norm(x, norm_weight, norm_bias, eps=1e-6),
        )
    )

    x_norm = mx.fast.layer_norm(x, norm_weight, norm_bias, eps=1e-6)
    mx.eval(x_norm)

    # 2. QKV projection (D -> 3D)
    results.append(profile_op("QKV linear (D -> 3D)", lambda: x_norm @ qkv_weight.T + qkv_bias))

    qkv = x_norm @ qkv_weight.T + qkv_bias
    qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
    q = qkv[:, :, 0].transpose(0, 2, 1, 3)
    k = qkv[:, :, 1].transpose(0, 2, 1, 3)
    v = qkv[:, :, 2].transpose(0, 2, 1, 3)
    mx.eval(q, k, v)

    # 3. RoPE 2D
    from mlx_mast3r.decoders.mast3r import precompute_rope_2d
    from mlx_mast3r.kernels.rope2d import apply_rope_2d_fused

    cos, sin, positions = precompute_rope_2d(32, 42, head_dim, dtype=mx.float16)
    mx.eval(cos, sin, positions)

    results.append(
        profile_op("RoPE 2D fused kernel", lambda: apply_rope_2d_fused(q, k, cos, sin, positions))
    )

    q_rope, k_rope = apply_rope_2d_fused(q, k, cos, sin, positions)
    mx.eval(q_rope, k_rope)

    # 4. SDPA
    scale = 1.0 / (head_dim**0.5)
    results.append(
        profile_op(
            "SDPA (mx.fast)",
            lambda: mx.fast.scaled_dot_product_attention(q_rope, k_rope, v, scale=scale),
        )
    )

    attn_out = mx.fast.scaled_dot_product_attention(q_rope, k_rope, v, scale=scale)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, N, D)
    mx.eval(attn_out)

    # 5. Output projection (D -> D)
    results.append(profile_op("Output proj (D -> D)", lambda: attn_out @ proj_weight.T + proj_bias))

    proj_out = attn_out @ proj_weight.T + proj_bias
    mx.eval(proj_out)

    # 6. Residual add
    x_after_attn = x + proj_out
    mx.eval(x_after_attn)

    # ===== MLP BLOCK =====
    print("\n--- MLP BLOCK ---")

    # 7. LayerNorm (pre-MLP)
    results.append(
        profile_op(
            "LayerNorm (pre-MLP)",
            lambda: mx.fast.layer_norm(x_after_attn, norm_weight, norm_bias, eps=1e-6),
        )
    )

    x_norm2 = mx.fast.layer_norm(x_after_attn, norm_weight, norm_bias, eps=1e-6)
    mx.eval(x_norm2)

    # 8. FC1 (D -> 4D)
    results.append(profile_op("FC1 (D -> 4D)", lambda: x_norm2 @ fc1_weight.T + fc1_bias))

    fc1_out = x_norm2 @ fc1_weight.T + fc1_bias
    mx.eval(fc1_out)

    # 9. GELU activation (precise)
    results.append(profile_op("GELU precise", lambda: nn.gelu(fc1_out)))

    # 10. GELU fast approx (used in practice)
    results.append(profile_op("GELU fast approx", lambda: nn.gelu_fast_approx(fc1_out)))

    gelu_out = nn.gelu_fast_approx(fc1_out)
    mx.eval(gelu_out)

    # 11. FC2 (4D -> D)
    results.append(profile_op("FC2 (4D -> D)", lambda: gelu_out @ fc2_weight.T + fc2_bias))

    # ===== SUMMARY =====
    print("\n--- FULL BREAKDOWN (B=1, N=1344, D=1024) ---")
    print("-" * 70)

    # Calculate totals
    # Attention: LayerNorm, QKV, RoPE, SDPA, Output proj (indices 0-4)
    # MLP (with fast gelu): LayerNorm, FC1, GELU fast, FC2 (indices 5, 6, 8, 9)
    attn_total = sum(r.mean_ms for r in results[:5])
    mlp_total_fast = results[5].mean_ms + results[6].mean_ms + results[8].mean_ms + results[9].mean_ms
    block_total = attn_total + mlp_total_fast

    for r in results:
        r.pct = (r.mean_ms / block_total) * 100
        print(r)

    print("-" * 70)
    print(f"{'ATTENTION TOTAL':40s} {attn_total:8.2f} ms  ({attn_total/block_total*100:5.1f}%)")
    print(f"{'MLP TOTAL (fast gelu)':40s} {mlp_total_fast:8.2f} ms  ({mlp_total_fast/block_total*100:5.1f}%)")
    print(f"{'BLOCK TOTAL':40s} {block_total:8.2f} ms  (100.0%)")

    return results


def profile_kernels():
    """Profile custom Metal kernels."""
    print("\n" + "=" * 70)
    print("PROFILING: Custom Metal Kernels")
    print("=" * 70)

    results = []

    # 1. Bilinear upsample 2x
    from mlx_mast3r.kernels.bilinear import bilinear_upsample_2x_fused

    x_bilinear = mx.random.normal((1, 32, 42, 256)).astype(mx.float16)
    mx.eval(x_bilinear)
    results.append(
        profile_op(
            "bilinear_upsample_2x (32x42x256 -> 64x84)",
            lambda: bilinear_upsample_2x_fused(x_bilinear),
        )
    )

    # 2. Grid sample
    from mlx_mast3r.kernels.grid_sample import grid_sample

    x_grid = mx.random.normal((1, 24, 24, 768)).astype(mx.float16)
    gy = mx.linspace(-1, 1, 32)
    gx = mx.linspace(-1, 1, 42)
    grid_y = mx.broadcast_to(gy[:, None], (32, 42))
    grid_x = mx.broadcast_to(gx[None, :], (32, 42))
    grid = mx.stack([grid_x, grid_y], axis=-1)[None, :, :, :]
    mx.eval(x_grid, grid)

    results.append(
        profile_op("grid_sample (24x24x768 -> 32x42)", lambda: grid_sample(x_grid, grid))
    )

    # 3. RoPE 2D fused
    from mlx_mast3r.kernels.rope2d import apply_rope_2d_fused
    from mlx_mast3r.decoders.mast3r import precompute_rope_2d

    q = mx.random.normal((1, 16, 1344, 64)).astype(mx.float16)
    k = mx.random.normal((1, 16, 1344, 64)).astype(mx.float16)
    cos, sin, positions = precompute_rope_2d(32, 42, 64, dtype=mx.float16)
    mx.eval(q, k, cos, sin, positions)

    results.append(
        profile_op(
            "rope2d_fused (B=1, H=16, N=1344, D=64)",
            lambda: apply_rope_2d_fused(q, k, cos, sin, positions),
        )
    )

    print("\nKernel performance:")
    print("-" * 70)
    for r in results:
        print(r)

    return results


def main():
    print("=" * 70)
    print("MLX-MASt3R GPU PROFILING")
    print("=" * 70)
    print(f"Warmup: {WARMUP} | Iterations: {ITERATIONS}")

    # Profile custom kernels first
    profile_kernels()

    # Profile attention + MLP breakdown
    profile_attention_mlp_breakdown()

    # Profile DUNE encoder
    profile_dune_encoder("small", 336)
    profile_dune_encoder("base", 336)

    # Profile MASt3R encoder
    profile_mast3r_encoder()

    # Profile MASt3R decoder
    profile_mast3r_decoder()

    # Profile full pipeline
    profile_mast3r_full()

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
