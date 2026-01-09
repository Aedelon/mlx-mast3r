"""Convert DUNE .pth checkpoints to safetensors fp16.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.

Usage:
    uv run python scripts/convert_dune_to_safetensors.py

This script converts DUNE encoder weights from PyTorch .pth format
to safetensors fp16 format for use with mlx-mast3r.
"""

from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import save_file


def convert_dune_checkpoint(
    pth_path: Path,
    output_dir: Path,
    variant: str,
    resolution: int,
) -> None:
    """Convert a DUNE .pth checkpoint to safetensors fp16.

    Args:
        pth_path: Path to input .pth file
        output_dir: Directory to save safetensors
        variant: "small" or "base"
        resolution: Image resolution (336 or 448)
    """
    print(f"Loading {pth_path}...")
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]

    # Extract encoder weights only
    tensors = {}

    # Patch embedding - keep PyTorch format [O,I,H,W], loader will transpose
    patch_weight = state_dict["encoder.patch_embed.proj.weight"].numpy()
    tensors["encoder.patch_embed.proj.weight"] = patch_weight.astype(np.float16)
    tensors["encoder.patch_embed.proj.bias"] = (
        state_dict["encoder.patch_embed.proj.bias"].numpy().astype(np.float16)
    )

    # Tokens
    tensors["encoder.cls_token"] = state_dict["encoder.cls_token"].numpy().astype(np.float16)
    tensors["encoder.register_tokens"] = (
        state_dict["encoder.register_tokens"].numpy().astype(np.float16)
    )

    # Position embeddings
    tensors["encoder.pos_embed"] = state_dict["encoder.pos_embed"].numpy().astype(np.float16)

    # Encoder blocks (12 blocks)
    depth = 12 if variant == "base" else 12  # Both small and base have 12 blocks
    for i in range(depth):
        src_prefix = f"encoder.blocks.0.{i}."

        # Attention
        tensors[f"encoder.blocks.0.{i}.attn.qkv.weight"] = (
            state_dict[src_prefix + "attn.qkv.weight"].numpy().astype(np.float16)
        )
        tensors[f"encoder.blocks.0.{i}.attn.qkv.bias"] = (
            state_dict[src_prefix + "attn.qkv.bias"].numpy().astype(np.float16)
        )
        tensors[f"encoder.blocks.0.{i}.attn.proj.weight"] = (
            state_dict[src_prefix + "attn.proj.weight"].numpy().astype(np.float16)
        )
        tensors[f"encoder.blocks.0.{i}.attn.proj.bias"] = (
            state_dict[src_prefix + "attn.proj.bias"].numpy().astype(np.float16)
        )

        # MLP
        tensors[f"encoder.blocks.0.{i}.mlp.fc1.weight"] = (
            state_dict[src_prefix + "mlp.fc1.weight"].numpy().astype(np.float16)
        )
        tensors[f"encoder.blocks.0.{i}.mlp.fc1.bias"] = (
            state_dict[src_prefix + "mlp.fc1.bias"].numpy().astype(np.float16)
        )
        tensors[f"encoder.blocks.0.{i}.mlp.fc2.weight"] = (
            state_dict[src_prefix + "mlp.fc2.weight"].numpy().astype(np.float16)
        )
        tensors[f"encoder.blocks.0.{i}.mlp.fc2.bias"] = (
            state_dict[src_prefix + "mlp.fc2.bias"].numpy().astype(np.float16)
        )

        # LayerNorm
        tensors[f"encoder.blocks.0.{i}.norm1.weight"] = (
            state_dict[src_prefix + "norm1.weight"].numpy().astype(np.float16)
        )
        tensors[f"encoder.blocks.0.{i}.norm1.bias"] = (
            state_dict[src_prefix + "norm1.bias"].numpy().astype(np.float16)
        )
        tensors[f"encoder.blocks.0.{i}.norm2.weight"] = (
            state_dict[src_prefix + "norm2.weight"].numpy().astype(np.float16)
        )
        tensors[f"encoder.blocks.0.{i}.norm2.bias"] = (
            state_dict[src_prefix + "norm2.bias"].numpy().astype(np.float16)
        )

        # Layer Scale
        tensors[f"encoder.blocks.0.{i}.ls1.gamma"] = (
            state_dict[src_prefix + "ls1.gamma"].numpy().astype(np.float16)
        )
        tensors[f"encoder.blocks.0.{i}.ls2.gamma"] = (
            state_dict[src_prefix + "ls2.gamma"].numpy().astype(np.float16)
        )

    # Final norm
    tensors["encoder.norm.weight"] = state_dict["encoder.norm.weight"].numpy().astype(np.float16)
    tensors["encoder.norm.bias"] = state_dict["encoder.norm.bias"].numpy().astype(np.float16)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "encoder.safetensors"

    print(f"Saving {len(tensors)} tensors to {output_path}...")
    save_file(tensors, str(output_path))

    # Compute size
    total_bytes = sum(t.nbytes for t in tensors.values())
    print(f"  Total size: {total_bytes / 1024 / 1024:.1f} MB")

    print(f"Done: {output_path}")


def main():
    checkpoints_dir = Path.home() / ".cache/mast3r_runtime/checkpoints"
    output_base = Path.home() / ".cache/mlx-mast3r"

    # Define conversions: (pth_name, variant, resolution)
    conversions = [
        ("dune_vitsmall14_336.pth", "small", 336),
        ("dune_vitsmall14_448.pth", "small", 448),
        ("dune_vitbase14_336.pth", "base", 336),
        ("dune_vitbase14_448.pth", "base", 448),
    ]

    for pth_name, variant, resolution in conversions:
        pth_path = checkpoints_dir / pth_name
        if not pth_path.exists():
            print(f"Skipping {pth_name}: not found")
            continue

        output_dir = output_base / f"dune_vit_{variant}_{resolution}"
        convert_dune_checkpoint(pth_path, output_dir, variant, resolution)
        print()


if __name__ == "__main__":
    main()
