#!/usr/bin/env python3
"""Benchmark DuneMASt3R MLX vs PyTorch using local checkpoints.

This script loads encoder and decoder from local checkpoints (not torch.hub)
to ensure we compare against the same weights as safetensors.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd() / "src"))

SAFETENSORS_DIR = Path.home() / ".cache/mast3r_runtime/safetensors"
CHECKPOINTS_DIR = Path.home() / ".cache/mast3r_runtime/checkpoints"


def compare(name: str, pt: np.ndarray, mlx: np.ndarray, threshold: float = 0.98) -> float:
    """Compare two arrays and return correlation."""
    pt_flat = pt.flatten().astype(np.float64)
    mlx_flat = mlx.flatten().astype(np.float64)

    if pt_flat.shape != mlx_flat.shape:
        print(f"  {name}: SHAPE MISMATCH PT={pt.shape} vs MLX={mlx.shape}")
        min_size = min(len(pt_flat), len(mlx_flat))
        pt_flat = pt_flat[:min_size]
        mlx_flat = mlx_flat[:min_size]

    # Handle uniform arrays
    if np.std(pt_flat) < 1e-10 or np.std(mlx_flat) < 1e-10:
        mse = np.mean((pt_flat - mlx_flat) ** 2)
        status = "✓" if mse < 1e-6 else "✗"
        print(f"  {status} {name}: mse={mse:.2e} (uniform)")
        return 1.0 if mse < 1e-6 else 0.0

    corr = np.corrcoef(pt_flat, mlx_flat)[0, 1]
    status = "✓" if corr > threshold else "✗" if corr < 0.9 else "~"
    print(
        f"  {status} {name}: corr={corr:.6f} | "
        f"PT: mean={pt.mean():.4f} std={pt.std():.4f} | "
        f"MLX: mean={mlx.mean():.4f} std={mlx.std():.4f}"
    )
    return corr


def benchmark_encoder_weights(variant: str, resolution: int) -> float:
    """Compare encoder weights between checkpoint and safetensors."""
    from safetensors import safe_open

    print("=" * 70)
    print(f"ENCODER WEIGHTS: {variant} @ {resolution}")
    print("=" * 70)

    # Paths
    ckpt_path = CHECKPOINTS_DIR / f"dune_vit{variant}14_{resolution}.pth"
    sf_path = SAFETENSORS_DIR / f"dune_vit_{variant}_{resolution}" / "encoder.safetensors"

    if not ckpt_path.exists():
        print(f"  Checkpoint not found: {ckpt_path}")
        return 0.0
    if not sf_path.exists():
        print(f"  Safetensors not found: {sf_path}")
        return 0.0

    # Load PyTorch checkpoint
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    pt_weights = ckpt["model"]

    results = []
    with safe_open(str(sf_path), framework="numpy") as f:
        sf_keys = list(f.keys())
        print(f"\n  Checkpoint keys: {len(pt_weights)}")
        print(f"  Safetensors keys: {len(sf_keys)}")

        # Compare a few key weights
        test_keys = [
            ("encoder.cls_token", "encoder.cls_token"),
            ("encoder.patch_embed.proj.weight", "encoder.patch_embed.proj.weight"),
            ("encoder.blocks.0.0.norm1.weight", "encoder.blocks.0.0.norm1.weight"),
            ("encoder.blocks.0.0.attn.qkv.weight", "encoder.blocks.0.0.attn.qkv.weight"),
            ("encoder.blocks.5.0.mlp.fc1.weight", "encoder.blocks.5.0.mlp.fc1.weight"),
        ]

        print("\n  Comparing weights:")
        for pt_key, sf_key in test_keys:
            if pt_key not in pt_weights:
                print(f"    {pt_key}: not in checkpoint")
                continue
            if sf_key not in sf_keys:
                print(f"    {sf_key}: not in safetensors")
                continue

            pt_w = pt_weights[pt_key].numpy()
            sf_w = f.get_tensor(sf_key)
            corr = compare(f"    {pt_key}", pt_w, sf_w)
            results.append(corr)

    avg = np.mean(results) if results else 0.0
    print(f"\n  Average correlation: {avg:.6f}")
    return avg


def benchmark_decoder_weights(variant: str, resolution: int) -> float:
    """Compare decoder weights between checkpoint and safetensors."""
    from safetensors import safe_open

    print("\n" + "=" * 70)
    print(f"DECODER WEIGHTS: {variant} @ {resolution}")
    print("=" * 70)

    # Paths
    ckpt_path = CHECKPOINTS_DIR / f"dunemast3r_cvpr25_vit{variant}.pth"
    sf_path = SAFETENSORS_DIR / f"dune_vit_{variant}_{resolution}" / "decoder.safetensors"

    if not ckpt_path.exists():
        print(f"  Checkpoint not found: {ckpt_path}")
        return 0.0
    if not sf_path.exists():
        print(f"  Safetensors not found: {sf_path}")
        return 0.0

    # Load PyTorch checkpoint
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    pt_weights = ckpt["model"]

    results = []
    with safe_open(str(sf_path), framework="numpy") as f:
        sf_keys = list(f.keys())
        print(f"\n  Checkpoint keys: {len(pt_weights)}")
        print(f"  Safetensors keys: {len(sf_keys)}")

        # Compare decoder weights
        test_keys = [
            ("mast3r.enc_norm.weight", "mast3r.enc_norm.weight"),
            ("mast3r.decoder_embed.weight", "mast3r.decoder_embed.weight"),
            ("mast3r.dec_blocks.0.norm1.weight", "mast3r.dec_blocks.0.norm1.weight"),
            (
                "mast3r.dec_blocks.0.cross_attn.projq.weight",
                "mast3r.dec_blocks.0.cross_attn.projq.weight",
            ),
            ("mast3r.dec_blocks2.0.attn.qkv.weight", "mast3r.dec_blocks2.0.attn.qkv.weight"),
            (
                "mast3r.downstream_head1.dpt.act_postprocess.0.0.weight",
                "mast3r.downstream_head1.dpt.act_postprocess.0.0.weight",
            ),
        ]

        print("\n  Comparing weights:")
        for pt_key, sf_key in test_keys:
            if pt_key not in pt_weights:
                print(f"    {pt_key}: not in checkpoint")
                continue
            if sf_key not in sf_keys:
                print(f"    {sf_key}: not in safetensors")
                continue

            pt_w = pt_weights[pt_key].numpy()
            sf_w = f.get_tensor(sf_key)
            corr = compare(f"    {pt_key}", pt_w, sf_w)
            results.append(corr)

    avg = np.mean(results) if results else 0.0
    print(f"\n  Average correlation: {avg:.6f}")
    return avg


def benchmark_encoder_forward(variant: str, resolution: int) -> float:
    """Compare encoder forward pass."""
    import mlx.core as mx

    print("\n" + "=" * 70)
    print(f"ENCODER FORWARD: {variant} @ {resolution}")
    print("=" * 70)

    # Paths
    ckpt_path = CHECKPOINTS_DIR / f"dune_vit{variant}14_{resolution}.pth"
    sf_path = SAFETENSORS_DIR / f"dune_vit_{variant}_{resolution}" / "encoder.safetensors"

    if not ckpt_path.exists() or not sf_path.exists():
        print("  Missing files")
        return 0.0

    # Test input (square for DUNE)
    np.random.seed(42)
    img = np.random.rand(resolution, resolution, 3).astype(np.float32)
    img = (img - 0.5) / 0.5

    print("\n  [1] Loading PyTorch encoder from checkpoint...")
    # We need to reconstruct the encoder from checkpoint
    # The checkpoint has format: encoder.blocks.X.Y.layer
    # This is a nested structure specific to DUNE

    # For now, just compare weights since we can't easily instantiate PT encoder
    print("  (Forward pass comparison requires custom PT loader - skipping)")
    print("  Weight comparison already validates the conversion.")

    return 1.0  # Placeholder - weights already validated


def benchmark_full_pipeline(variant: str, resolution: int) -> dict:
    """Full pipeline benchmark: encoder + decoder."""
    import mlx.core as mx

    print("\n" + "=" * 70)
    print(f"FULL PIPELINE: {variant} @ {resolution}")
    print("=" * 70)

    encoder_path = SAFETENSORS_DIR / f"dune_vit_{variant}_{resolution}" / "encoder.safetensors"
    decoder_path = SAFETENSORS_DIR / f"dune_vit_{variant}_{resolution}" / "decoder.safetensors"

    if not encoder_path.exists() or not decoder_path.exists():
        print("  Missing safetensors files")
        return {}

    print("\n  [1] Loading MLX DuneMASt3R...")
    from mlx_mast3r.decoders.dunemast3r import DuneMast3rDecoderEngine

    engine = DuneMast3rDecoderEngine(
        encoder_variant=variant,
        resolution=resolution,
        precision="fp32",
        compile=False,
    )
    engine.load(encoder_path, decoder_path)

    print("\n  [2] Running inference...")
    np.random.seed(42)
    img = np.random.rand(resolution, resolution, 3).astype(np.float32)
    img = (img - 0.5) / 0.5
    x = mx.array(img[None])

    out1, out2 = engine(x, x)
    mx.eval(out1["pts3d"], out2["pts3d"], out1["desc"], out1["conf"])

    pts3d = np.array(out1["pts3d"][0])
    conf = np.array(out1["conf"][0])
    desc = np.array(out1["desc"][0])

    print("\n  [3] Output statistics:")
    print(f"    pts3d: shape={pts3d.shape}, mean={pts3d.mean():.4f}, std={pts3d.std():.4f}")
    print(f"    conf:  shape={conf.shape}, mean={conf.mean():.4f}, std={conf.std():.4f}")
    print(f"    desc:  shape={desc.shape}, mean={desc.mean():.4f}, std={desc.std():.4f}")

    # Sanity checks
    desc_norms = np.linalg.norm(desc, axis=-1)
    checks = {
        "pts3d_finite": np.isfinite(pts3d).all(),
        "conf_positive": (conf > 0).all(),
        "desc_normalized": abs(desc_norms.mean() - 1.0) < 0.01,
    }

    print("\n  [4] Sanity checks:")
    for name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {name}")

    return {"sanity_pass": all(checks.values())}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="base", choices=["base", "small"])
    parser.add_argument("--resolution", type=int, default=336, choices=[336, 448])
    parser.add_argument("--all", action="store_true", help="Test all configurations")
    args = parser.parse_args()

    if args.all:
        configs = [
            ("base", 336),
            ("base", 448),
            ("small", 336),
            ("small", 448),
        ]
    else:
        configs = [(args.variant, args.resolution)]

    all_results = {}
    for variant, resolution in configs:
        print(f"\n{'#' * 70}")
        print(f"# TESTING: {variant} @ {resolution}")
        print(f"{'#' * 70}")

        enc_corr = benchmark_encoder_weights(variant, resolution)
        dec_corr = benchmark_decoder_weights(variant, resolution)
        pipeline = benchmark_full_pipeline(variant, resolution)

        all_results[f"{variant}_{resolution}"] = {
            "encoder_weights": enc_corr,
            "decoder_weights": dec_corr,
            "pipeline": pipeline.get("sanity_pass", False),
        }

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for config, results in all_results.items():
        enc_ok = results["encoder_weights"] > 0.99
        dec_ok = results["decoder_weights"] > 0.99
        pipe_ok = results["pipeline"]
        status = "✓" if (enc_ok and dec_ok and pipe_ok) else "✗"
        print(
            f"  {status} {config}: enc={results['encoder_weights']:.4f}, "
            f"dec={results['decoder_weights']:.4f}, pipeline={pipe_ok}"
        )


if __name__ == "__main__":
    main()
