#!/usr/bin/env python3
"""Convert safetensors to MLX-optimized format (pre-transposed conv weights).

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.

Usage:
    uv run python scripts/convert_to_mlx_optimized.py input.safetensors output_mlx.safetensors
    uv run python scripts/convert_to_mlx_optimized.py --all  # Convert all cached weights

This script pre-transposes conv weights from PyTorch format (O,I,H,W) to MLX format (O,H,W,I),
eliminating transposition overhead at load time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def is_conv_weight(name: str, shape: tuple) -> bool:
    """Check if tensor is a conv weight that needs transposition.

    Conv weights have 4 dims and names containing 'conv' or 'proj' (for patch_embed).
    """
    if len(shape) != 4:
        return False

    conv_patterns = [
        "conv",
        "patch_embed.proj.weight",
        "act_postprocess",
        "layer_rn",
        "refinenet",
        "head_conv",
    ]
    return any(p in name.lower() for p in conv_patterns)


def is_conv_transpose_weight(name: str, shape: tuple) -> bool:
    """Check if tensor is a ConvTranspose weight.

    ConvTranspose weights: PyTorch (I,O,H,W) -> MLX (O,H,W,I)
    In DPT head, they are act_postprocess.X.1.weight (upsample layers).
    """
    if len(shape) != 4:
        return False
    # DPT upsample layers: act_postprocess.0.1 and act_postprocess.1.1
    if "act_postprocess.0.1.weight" in name or "act_postprocess.1.1.weight" in name:
        return True
    return "_up.weight" in name or "upsample" in name.lower()


def transpose_conv(w: np.ndarray) -> np.ndarray:
    """Transpose conv weight: PyTorch (O,I,H,W) -> MLX (O,H,W,I)."""
    return np.transpose(w, (0, 2, 3, 1))


def transpose_conv_transpose(w: np.ndarray) -> np.ndarray:
    """Transpose ConvTranspose weight: PyTorch (I,O,H,W) -> MLX (O,H,W,I)."""
    return np.transpose(w, (1, 2, 3, 0))


def convert_safetensors(input_path: Path, output_path: Path) -> dict:
    """Convert safetensors to MLX-optimized format.

    Returns:
        Statistics dict with conversion info.
    """
    stats = {"total": 0, "conv_transposed": 0, "conv_t_transposed": 0, "unchanged": 0}
    tensors = {}

    with safe_open(str(input_path), framework="numpy") as f:
        keys = list(f.keys())
        stats["total"] = len(keys)

        for key in keys:
            tensor = f.get_tensor(key)
            shape = tensor.shape

            if is_conv_transpose_weight(key, shape):
                tensors[key] = transpose_conv_transpose(tensor)
                stats["conv_t_transposed"] += 1
            elif is_conv_weight(key, shape):
                tensors[key] = transpose_conv(tensor)
                stats["conv_transposed"] += 1
            else:
                tensors[key] = tensor
                stats["unchanged"] += 1

    # Add metadata to indicate MLX format
    metadata = {"format": "mlx_optimized", "version": "1.0"}

    save_file(tensors, str(output_path), metadata=metadata)
    return stats


def convert_all_cached() -> None:
    """Convert all safetensors in the cache directory."""
    cache_dir = Path.home() / ".cache" / "mast3r_runtime" / "safetensors"

    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        return

    for subdir in cache_dir.iterdir():
        if not subdir.is_dir():
            continue

        # Find safetensors files
        for st_file in subdir.glob("*.safetensors"):
            if "_mlx" in st_file.name:
                continue  # Skip already converted

            output_name = st_file.stem + "_mlx.safetensors"
            output_path = st_file.parent / output_name

            if output_path.exists():
                print(f"Skipping {st_file.name} (already converted)")
                continue

            print(f"Converting {st_file}...")
            stats = convert_safetensors(st_file, output_path)
            print(
                f"  Total: {stats['total']}, Conv: {stats['conv_transposed']}, "
                f"ConvT: {stats['conv_t_transposed']}, Unchanged: {stats['unchanged']}"
            )
            print(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert safetensors to MLX-optimized format")
    parser.add_argument(
        "input",
        nargs="?",
        help="Input safetensors file",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output safetensors file",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all cached safetensors",
    )

    args = parser.parse_args()

    if args.all:
        convert_all_cached()
    elif args.input and args.output:
        input_path = Path(args.input)
        output_path = Path(args.output)

        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return 1

        print(f"Converting {input_path}...")
        stats = convert_safetensors(input_path, output_path)
        print(f"Total: {stats['total']}")
        print(f"Conv transposed: {stats['conv_transposed']}")
        print(f"ConvTranspose transposed: {stats['conv_t_transposed']}")
        print(f"Unchanged: {stats['unchanged']}")
        print(f"Output: {output_path}")
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
