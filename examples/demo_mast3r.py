#!/usr/bin/env python3
"""Demo: Full MASt3R pipeline for highest quality 3D reconstruction.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.

Usage:
    uv run python examples/demo_mast3r.py
"""

import time
from pathlib import Path

from mlx_mast3r import Mast3r, Mast3rFull
from mlx_mast3r.utils import load_image


def main():
    # Paths
    assets_dir = Path(__file__).parent.parent / "assets"
    tower_dir = assets_dir / "NLE_tower"

    # Get stereo pair
    images = sorted(tower_dir.glob("*.jpg"))
    if len(images) < 2:
        print(f"Need at least 2 images in {tower_dir}")
        return

    img1_path, img2_path = images[0], images[1]

    # MASt3R uses 4:3 aspect ratio: 512x672 for resolution=512
    resolution = (512, 672)

    # Load and resize images
    print(f"Image 1: {img1_path.name}")
    print(f"Image 2: {img2_path.name}")
    img1 = load_image(img1_path, resolution=resolution)
    img2 = load_image(img2_path, resolution=resolution)
    print(f"Image shapes: {img1.shape}, {img2.shape}")

    # --- MASt3R Encoder only ---
    print("\n--- MASt3R Encoder (ViT-Large @ 512) ---")
    print("Loading model (downloading weights if needed)...")
    encoder = Mast3r.from_pretrained(resolution=512)

    # Warmup
    _ = encoder.encode(img1)

    # Benchmark
    n_runs = 5
    start = time.perf_counter()
    for _ in range(n_runs):
        features = encoder.encode(img1)
    elapsed = (time.perf_counter() - start) / n_runs * 1000

    print(f"Features shape: {features.shape}")
    print(f"Embed dim: {encoder.embed_dim}")
    print(f"Num patches: {encoder.num_patches}")
    print(f"Inference time: {elapsed:.1f}ms ({1000 / elapsed:.1f} FPS)")

    # --- Full MASt3R Pipeline ---
    print("\n--- MASt3rFull (Encoder + Decoder @ 512) ---")
    print("Loading model...")
    model = Mast3rFull.from_pretrained(resolution=512)

    # Warmup
    print("Warmup...")
    _ = model.reconstruct(img1, img2)

    # Benchmark
    print("Running inference...")
    n_runs = 3
    start = time.perf_counter()
    for _ in range(n_runs):
        out1, out2 = model.reconstruct(img1, img2)
    elapsed = (time.perf_counter() - start) / n_runs * 1000

    # Display results
    print("\nResults for image 1:")
    print(f"  pts3d shape: {out1['pts3d'].shape}")
    print(f"  conf shape:  {out1['conf'].shape}")
    print(f"  desc shape:  {out1['desc'].shape}")

    # Statistics
    pts3d = out1["pts3d"]
    conf = out1["conf"]

    print("\n3D Points statistics:")
    print(f"  X range: [{pts3d[:, :, 0].min():.2f}, {pts3d[:, :, 0].max():.2f}]")
    print(f"  Y range: [{pts3d[:, :, 1].min():.2f}, {pts3d[:, :, 1].max():.2f}]")
    print(f"  Z range: [{pts3d[:, :, 2].min():.2f}, {pts3d[:, :, 2].max():.2f}]")
    print(f"  Confidence: min={conf.min():.3f}, max={conf.max():.3f}, mean={conf.mean():.3f}")

    print(f"\nInference time: {elapsed:.1f}ms ({1000 / elapsed:.1f} FPS)")

    print("\nDone!")


if __name__ == "__main__":
    main()
