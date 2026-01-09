#!/usr/bin/env python3
"""Test DuneMASt3R decoder by comparing intermediate outputs.

Since we can't easily load the full PyTorch DuneMASt3R model,
we test layer-by-layer against the saved weights.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path.cwd() / "src"))

SAFETENSORS_DIR = Path.home() / ".cache/mast3r_runtime/safetensors"


def compare(name: str, a: np.ndarray, b: np.ndarray) -> float:
    """Compare two arrays."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)

    if a_flat.shape != b_flat.shape:
        print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return 0.0

    # Check if arrays are identical (handles uniform weight case)
    if np.allclose(a_flat, b_flat, rtol=1e-5, atol=1e-5):
        print(f"  ✓ {name}: exact match")
        return 1.0

    # Handle uniform arrays that cause nan in corrcoef
    if np.std(a_flat) < 1e-10 or np.std(b_flat) < 1e-10:
        mse = np.mean((a_flat - b_flat) ** 2)
        status = "✓" if mse < 1e-10 else "✗"
        print(f"  {status} {name}: mse={mse:.2e} (uniform weights)")
        return 1.0 if mse < 1e-10 else 0.0

    corr = np.corrcoef(a_flat, b_flat)[0, 1]
    status = "✓" if corr > 0.99 else "✗" if corr < 0.9 else "~"
    print(f"  {status} {name}: corr={corr:.6f}")
    return corr


def test_weight_loading(variant: str = "base", resolution: int = 336):
    """Test that decoder weights are loaded correctly."""
    from safetensors import safe_open

    print("=" * 70)
    print(f"WEIGHT LOADING TEST: {variant} @ {resolution}")
    print("=" * 70)

    decoder_path = SAFETENSORS_DIR / f"dune_vit_{variant}_{resolution}" / "decoder.safetensors"

    print(f"\n[1] Loading decoder from {decoder_path}...")
    from mlx_mast3r.decoders.dunemast3r import (
        DuneMast3rDecoder,
        DuneMast3rDecoderConfig,
    )

    if variant == "base":
        config = DuneMast3rDecoderConfig.for_dune_base("fp32")
    else:
        config = DuneMast3rDecoderConfig.for_dune_small("fp32")

    decoder = DuneMast3rDecoder(config)

    # Load weights manually
    from mlx_mast3r.decoders.dunemast3r import DuneMast3rDecoderEngine

    engine = DuneMast3rDecoderEngine(
        encoder_variant=variant,
        resolution=resolution,
        precision="fp32",
        compile=False,
    )
    # Just load decoder weights
    engine._load_decoder(decoder_path)
    decoder = engine.decoder

    print("\n[2] Comparing loaded weights vs safetensors...")
    results = []

    with safe_open(str(decoder_path), framework="numpy") as f:
        # Test enc_norm
        enc_norm_w = f.get_tensor("mast3r.enc_norm.weight")
        mlx_enc_norm_w = np.array(decoder.enc_norm_weight)
        results.append(compare("enc_norm.weight", enc_norm_w, mlx_enc_norm_w))

        # Test decoder_embed
        dec_embed_w = f.get_tensor("mast3r.decoder_embed.weight")
        mlx_dec_embed_w = np.array(decoder.decoder_embed.weight)
        results.append(compare("decoder_embed.weight", dec_embed_w, mlx_dec_embed_w))

        # Test dec_blocks.0.norm1
        norm1_w = f.get_tensor("mast3r.dec_blocks.0.norm1.weight")
        mlx_norm1_w = np.array(decoder.dec_blocks[0].norm1_weight)
        results.append(compare("dec_blocks.0.norm1.weight", norm1_w, mlx_norm1_w))

        # Test dec_blocks.0.norm_y (cross-attn context norm)
        norm_y_w = f.get_tensor("mast3r.dec_blocks.0.norm_y.weight")
        mlx_norm_y_w = np.array(decoder.dec_blocks[0].norm_y_weight)
        results.append(compare("dec_blocks.0.norm_y.weight", norm_y_w, mlx_norm_y_w))

        # Test cross_attn Q weight
        projq_w = f.get_tensor("mast3r.dec_blocks.0.cross_attn.projq.weight")
        mlx_q_w = np.array(decoder.dec_blocks[0].cross_attn.q.weight)
        results.append(compare("dec_blocks.0.cross_attn.q.weight", projq_w, mlx_q_w))

        # Test DPT act_postprocess_0_conv
        act0_w = f.get_tensor("mast3r.downstream_head1.dpt.act_postprocess.0.0.weight")
        mlx_act0_w = np.array(decoder.head1.act_postprocess_0_conv.weight)
        # PyTorch: (O,I,H,W), MLX: (O,H,W,I)
        act0_w_transposed = np.transpose(act0_w, (0, 2, 3, 1))
        results.append(
            compare("head1.act_postprocess_0_conv.weight", act0_w_transposed, mlx_act0_w)
        )

        # Test head_local_features fc1
        fc1_w = f.get_tensor("mast3r.downstream_head1.head_local_features.fc1.weight")
        mlx_fc1_w = np.array(decoder.head_local_features1.layers[0].weight)
        results.append(compare("head_local_features1.fc1.weight", fc1_w, mlx_fc1_w))

    avg_corr = np.mean(results)
    print(f"\n  Average correlation: {avg_corr:.6f}")
    print(f"  All weights correct: {'✓' if all(r > 0.99 for r in results) else '✗'}")

    return avg_corr


def test_forward_pass(variant: str = "base", resolution: int = 336):
    """Test forward pass produces reasonable outputs."""
    print("\n" + "=" * 70)
    print(f"FORWARD PASS TEST: {variant} @ {resolution}")
    print("=" * 70)

    encoder_path = SAFETENSORS_DIR / f"dune_vit_{variant}_{resolution}" / "encoder.safetensors"
    decoder_path = SAFETENSORS_DIR / f"dune_vit_{variant}_{resolution}" / "decoder.safetensors"

    print("\n[1] Loading model...")
    from mlx_mast3r.decoders.dunemast3r import DuneMast3rDecoderEngine

    engine = DuneMast3rDecoderEngine(
        encoder_variant=variant,
        resolution=resolution,
        precision="fp32",
        compile=False,
    )
    engine.load(encoder_path, decoder_path)

    print("\n[2] Creating test input...")
    np.random.seed(42)
    img = np.random.rand(resolution, resolution, 3).astype(np.float32)
    img = (img - 0.5) / 0.5
    x = mx.array(img[None])
    print(f"  Input shape: {x.shape}")

    print("\n[3] Running inference...")
    out1, out2 = engine(x, x)
    mx.eval(out1["pts3d"], out2["pts3d"], out1["desc"], out1["conf"])

    print("\n[4] Output statistics:")
    pts3d = np.array(out1["pts3d"][0])
    conf = np.array(out1["conf"][0])
    desc = np.array(out1["desc"][0])

    print(f"  pts3d shape: {pts3d.shape}")
    print(
        f"  pts3d stats: mean={pts3d.mean():.4f}, std={pts3d.std():.4f}, "
        f"min={pts3d.min():.4f}, max={pts3d.max():.4f}"
    )

    print(f"  conf shape: {conf.shape}")
    print(f"  conf stats: mean={conf.mean():.4f}, std={conf.std():.4f}")

    print(f"  desc shape: {desc.shape}")
    print(f"  desc stats: mean={desc.mean():.4f}, std={desc.std():.4f}")

    # Verify desc is normalized
    desc_norms = np.linalg.norm(desc, axis=-1)
    print(f"  desc norms: mean={desc_norms.mean():.4f}, std={desc_norms.std():.6f}")

    # Check if outputs are reasonable
    checks = [
        ("pts3d finite", np.isfinite(pts3d).all()),
        ("conf positive", (conf > 0).all()),
        ("conf reasonable", conf.mean() > 1.0),  # exp(0) + 1 = 2
        ("desc normalized", abs(desc_norms.mean() - 1.0) < 0.01),
    ]

    print("\n[5] Sanity checks:")
    all_pass = True
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        if not passed:
            all_pass = False

    return all_pass


def test_all_variants():
    """Test all DUNE variants."""
    variants = [
        ("base", 336),
        ("base", 448),
        ("small", 336),
        ("small", 448),
    ]

    results = {}
    for variant, resolution in variants:
        key = f"{variant}_{resolution}"
        encoder_path = SAFETENSORS_DIR / f"dune_vit_{variant}_{resolution}" / "encoder.safetensors"
        decoder_path = SAFETENSORS_DIR / f"dune_vit_{variant}_{resolution}" / "decoder.safetensors"

        if not encoder_path.exists() or not decoder_path.exists():
            print(f"\nSkipping {key}: missing safetensors")
            continue

        try:
            weight_corr = test_weight_loading(variant, resolution)
            forward_pass = test_forward_pass(variant, resolution)
            results[key] = {
                "weight_corr": weight_corr,
                "forward_pass": forward_pass,
            }
        except Exception as e:
            print(f"\nError testing {key}: {e}")
            results[key] = {"error": str(e)}

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for key, result in results.items():
        if "error" in result:
            print(f"  ✗ {key}: {result['error']}")
        else:
            status = "✓" if result["weight_corr"] > 0.99 and result["forward_pass"] else "✗"
            print(
                f"  {status} {key}: weights={result['weight_corr']:.4f}, forward={result['forward_pass']}"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="base", choices=["base", "small"])
    parser.add_argument("--resolution", type=int, default=336, choices=[336, 448])
    parser.add_argument("--all", action="store_true", help="Test all variants")
    args = parser.parse_args()

    if args.all:
        test_all_variants()
    else:
        test_weight_loading(args.variant, args.resolution)
        test_forward_pass(args.variant, args.resolution)
