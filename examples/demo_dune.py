#!/usr/bin/env python3
"""Demo: DUNE encoder for fast feature extraction.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.

Usage:
    uv run python examples/demo_dune.py
"""

import time
from pathlib import Path

import numpy as np
from PIL import Image

from mlx_mast3r import DUNE


def load_image(path: Path) -> np.ndarray:
    """Load image as numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def main():
    # Paths
    assets_dir = Path(__file__).parent.parent / "assets"
    image_path = assets_dir / "demo.jpg"

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    # Load image
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    print(f"Image shape: {image.shape}")

    # Load DUNE model (weights auto-downloaded from HuggingFace)
    print("\n--- DUNE Small @ 336 ---")
    model_small = DUNE.from_pretrained(variant="small", resolution=336)

    # Warmup
    _ = model_small.encode(image)

    # Benchmark
    n_runs = 10
    start = time.perf_counter()
    for _ in range(n_runs):
        features = model_small.encode(image)
    elapsed = (time.perf_counter() - start) / n_runs * 1000

    print(f"Features shape: {features.shape}")
    print(f"Embed dim: {model_small.embed_dim}")
    print(f"Num patches: {model_small.num_patches}")
    print(f"Inference time: {elapsed:.1f}ms ({1000 / elapsed:.1f} FPS)")

    # DUNE Base
    print("\n--- DUNE Base @ 336 ---")
    model_base = DUNE.from_pretrained(variant="base", resolution=336)

    # Warmup
    _ = model_base.encode(image)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        features = model_base.encode(image)
    elapsed = (time.perf_counter() - start) / n_runs * 1000

    print(f"Features shape: {features.shape}")
    print(f"Embed dim: {model_base.embed_dim}")
    print(f"Num patches: {model_base.num_patches}")
    print(f"Inference time: {elapsed:.1f}ms ({1000 / elapsed:.1f} FPS)")

    print("\nDone!")


if __name__ == "__main__":
    main()
