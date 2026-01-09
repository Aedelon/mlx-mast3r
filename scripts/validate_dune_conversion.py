"""Validate DUNE safetensors conversion by comparing MLX vs PyTorch outputs.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""

import sys
from pathlib import Path

import numpy as np
import timm
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mlx_mast3r.encoders.dune import DuneEncoderEngine


def load_pytorch_dune(variant: str, resolution: int) -> torch.nn.Module:
    """Load PyTorch DUNE model using timm DINOv2."""
    model_name = f"vit_{variant}_patch14_reg4_dinov2.lvd142m"
    model = timm.create_model(model_name, pretrained=False, img_size=resolution)

    # Load DUNE checkpoint
    ckpt_path = (
        Path.home() / f".cache/mast3r_runtime/checkpoints/dune_vit{variant}14_{resolution}.pth"
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]

    # Map keys: DUNE -> timm
    new_state = {}
    for k, v in state_dict.items():
        if not k.startswith("encoder."):
            continue
        new_key = k.replace("encoder.", "")

        # register_tokens -> reg_token
        if new_key == "register_tokens":
            new_key = "reg_token"
        # blocks.0.X -> blocks.X
        elif new_key.startswith("blocks.0."):
            new_key = new_key.replace("blocks.0.", "blocks.")

        if new_key in model.state_dict():
            target_shape = model.state_dict()[new_key].shape
            if v.shape != target_shape:
                if new_key == "pos_embed":
                    # Truncate pos_embed to match timm's expected size
                    n_patches_needed = target_shape[1]
                    v = v[:, :n_patches_needed, :]
                else:
                    continue
            new_state[new_key] = v

    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model


def run_pytorch(model: torch.nn.Module, img: np.ndarray) -> np.ndarray:
    """Run PyTorch inference."""
    # Normalize
    x = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (x - mean) / std

    # NHWC -> NCHW
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        out = model.forward_features(x)
        # Remove CLS and register tokens
        n_reg = model.num_reg_tokens if hasattr(model, "num_reg_tokens") else 4
        out = out[:, 1 + n_reg :, :]

    return out.numpy()


def run_mlx(engine: DuneEncoderEngine, img: np.ndarray) -> np.ndarray:
    """Run MLX inference."""
    # Preprocess: ImageNet normalization (same as PyTorch)
    x = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (x - mean) / std

    import mlx.core as mx

    x = mx.array(x[None, :, :, :])
    out = engine(x)
    mx.eval(out)
    return np.array(out[0])


def main():
    variants = ["small", "base"]
    resolutions = [336, 448]

    print("=" * 60)
    print("DUNE Conversion Validation: MLX vs PyTorch")
    print("=" * 60)

    for variant in variants:
        for resolution in resolutions:
            print(f"\n--- DUNE {variant} @ {resolution}x{resolution} ---")

            # Check if safetensors exists
            safetensors_path = (
                Path.home()
                / f".cache/mlx-mast3r/dune_vit_{variant}_{resolution}/encoder.safetensors"
            )
            if not safetensors_path.exists():
                print(f"  SKIP: {safetensors_path} not found")
                continue

            # Load models
            print("  Loading PyTorch model...")
            pt_model = load_pytorch_dune(variant, resolution)

            print("  Loading MLX model...")
            mlx_engine = DuneEncoderEngine(
                variant=variant,
                resolution=resolution,
                precision="fp32",  # Use fp32 for fair comparison
                compile=False,
            )
            mlx_engine.load(safetensors_path)

            # Create test image (square)
            np.random.seed(42)
            img = np.random.randint(0, 256, (resolution, resolution, 3), dtype=np.uint8)

            # Run inference
            print("  Running inference...")
            pt_out = run_pytorch(pt_model, img)
            mlx_out = run_mlx(mlx_engine, img)

            # Remove batch dimension if present
            if pt_out.ndim == 3:
                pt_out = pt_out[0]

            print(f"  PyTorch output: {pt_out.shape}")
            print(f"  MLX output:     {mlx_out.shape}")

            # Compare
            if pt_out.shape != mlx_out.shape:
                print(f"  ERROR: Shape mismatch!")
                continue

            # Normalize both to compare
            pt_flat = pt_out.flatten()
            mlx_flat = mlx_out.flatten()

            correlation = np.corrcoef(pt_flat, mlx_flat)[0, 1]
            mse = np.mean((pt_flat - mlx_flat) ** 2)
            max_diff = np.max(np.abs(pt_flat - mlx_flat))

            print(f"  Correlation: {correlation:.6f}")
            print(f"  MSE: {mse:.6e}")
            print(f"  Max diff: {max_diff:.6e}")

            if correlation >= 0.98:
                print("  ✓ PASS")
            else:
                print("  ✗ FAIL (correlation < 0.98)")

    print("\n" + "=" * 60)
    print("Validation complete")


if __name__ == "__main__":
    main()
