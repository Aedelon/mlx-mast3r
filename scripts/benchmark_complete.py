#!/usr/bin/env python3
"""Complete benchmark: MLX vs PyTorch MPS for all models.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.

PTH files for PyTorch MPS: ~/.cache/mast3r_runtime/checkpoints/
Safetensors for MLX: ~/.cache/mast3r_runtime/safetensors/
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add paths - use thirdparty repos
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "thirdparty/dune"))
sys.path.insert(0, str(REPO_ROOT / "thirdparty/mast3r"))
sys.path.insert(0, str(REPO_ROOT / "thirdparty/mast3r/dust3r"))

import mlx.core as mx  # noqa: E402

# ============================================================================
# Configuration
# ============================================================================

WARMUP_ITERATIONS = 10  # MLX needs more warmup for graph compilation
BENCHMARK_ITERATIONS = 10

PTH_DIR = Path.home() / ".cache/mast3r_runtime/checkpoints"
SAFETENSORS_DIR = Path.home() / ".cache/mlx-mast3r"
DUNE_SAFETENSORS_DIR = Path.home() / ".cache/mast3r_runtime/safetensors"

# Image sizes for benchmarks - all square for DUNE, 4:3 for MASt3R
DUNE_IMG_SIZE_336 = (336, 336, 3)
DUNE_IMG_SIZE_448 = (448, 448, 3)
MAST3R_IMG_SIZE_512 = (512, 672, 3)  # MASt3R uses 4:3


def create_test_image(shape: tuple[int, int, int], seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.randint(0, 256, shape, dtype=np.uint8)


# ============================================================================
# PyTorch MPS Models (from .pth)
# ============================================================================


def load_pytorch_dune(variant: str, resolution: int):
    """Load DUNE model with PyTorch MPS from .pth using timm DINOv2."""
    import timm

    # DUNE checkpoints are trained on square images
    img_size = resolution

    # DUNE uses DINOv2 with register tokens
    model_name = f"vit_{variant}_patch14_reg4_dinov2.lvd142m"
    model = timm.create_model(model_name, pretrained=False, img_size=img_size)

    ckpt_path = PTH_DIR / f"dune_vit{variant}14_{resolution}.pth"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    # Map DUNE checkpoint keys to timm model keys
    encoder_state = {}
    for k, v in state_dict.items():
        if not k.startswith("encoder."):
            continue

        new_key = k.replace("encoder.", "")

        # Handle special keys
        if new_key == "register_tokens":
            new_key = "reg_token"
        elif new_key == "mask_token":
            continue  # Skip mask token (not used in inference)
        elif new_key == "pos_embed":
            # DUNE pos_embed has extra token, truncate to match timm
            v = v[:, :model.pos_embed.shape[1], :]

        # Handle nested block keys: blocks.0.X.* -> blocks.X.*
        if new_key.startswith("blocks.0."):
            new_key = "blocks." + new_key[9:]

        encoder_state[new_key] = v

    model.load_state_dict(encoder_state, strict=False)
    model = model.to("mps").eval()

    return model


def load_pytorch_mast3r():
    """Load MASt3R model with PyTorch MPS from .pth."""
    from mast3r.model import AsymmetricMASt3R

    model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    )
    model = model.to("mps").eval()
    return model


# ============================================================================
# MLX Models (from .safetensors)
# ============================================================================


def load_mlx_dune(variant: str, resolution: int):
    """Load DUNE encoder MLX from safetensors."""
    from mlx_mast3r.encoders.dune import DuneEncoderEngine

    engine = DuneEncoderEngine(
        variant=variant,
        resolution=resolution,
        precision="fp16",
        compile=True,
    )

    safetensors_path = DUNE_SAFETENSORS_DIR / f"dune_vit_{variant}_{resolution}" / "encoder.safetensors"
    engine.load(safetensors_path)

    return engine


def load_mlx_mast3r():
    """Load MASt3R encoder MLX from safetensors."""
    from mlx_mast3r.encoders.mast3r import Mast3rEncoderEngine

    engine = Mast3rEncoderEngine(
        resolution=512,
        precision="fp16",
        compile=True,
    )

    safetensors_path = SAFETENSORS_DIR / "mast3r_vit_large" / "unified.safetensors"
    engine.load(safetensors_path)

    return engine


# ============================================================================
# Benchmark Functions
# ============================================================================


def benchmark_dune_encoder(variant: str, resolution: int) -> dict:
    """Benchmark DUNE encoder: MLX vs PyTorch MPS with same square images."""
    print(f"\n{'=' * 60}")
    print(f"DUNE {variant.upper()} @ {resolution}x{resolution}")
    print("=" * 60)

    img_size = DUNE_IMG_SIZE_336 if resolution == 336 else DUNE_IMG_SIZE_448
    img = create_test_image(img_size)

    pt_features = None
    pt_mean_ms = None
    mlx_features = None
    mlx_mean_ms = None

    # --- PyTorch MPS ---
    print("\nPyTorch MPS...")
    try:
        pt_model = load_pytorch_dune(variant, resolution)

        # ImageNet normalization
        img_pt = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_pt = (img_pt - mean) / std
        img_pt = img_pt.to("mps")

        # Warmup
        with torch.no_grad():
            for _ in range(WARMUP_ITERATIONS):
                _ = pt_model.forward_features(img_pt)
                torch.mps.synchronize()

        # Benchmark
        pt_times = []
        with torch.no_grad():
            for _ in range(BENCHMARK_ITERATIONS):
                t0 = time.perf_counter()
                out = pt_model.forward_features(img_pt)
                torch.mps.synchronize()
                pt_times.append((time.perf_counter() - t0) * 1000)

        # Remove CLS and register tokens
        n_reg = pt_model.num_reg_tokens if hasattr(pt_model, "num_reg_tokens") else 4
        pt_features = out[:, 1 + n_reg:, :].cpu().numpy()[0]

        pt_mean_ms = np.mean(pt_times)
        pt_std_ms = np.std(pt_times)

        print(f"  {pt_mean_ms:.2f} ± {pt_std_ms:.2f} ms ({1000 / pt_mean_ms:.1f} FPS)")

    except Exception as e:
        print(f"  FAILED - {e}")
        import traceback
        traceback.print_exc()

    # --- MLX ---
    print("\nMLX FP16...")
    try:
        mlx_engine = load_mlx_dune(variant, resolution)

        # Preprocess with ImageNet normalization
        img_mlx = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_mlx = (img_mlx - mean) / std

        # Warmup
        mlx_engine.warmup(WARMUP_ITERATIONS)

        # Benchmark
        mlx_times = []
        for _ in range(BENCHMARK_ITERATIONS):
            img_mx = mx.array(img_mlx[None, :, :, :])
            t0 = time.perf_counter()
            out = mlx_engine(img_mx)
            mx.eval(out)
            mlx_times.append((time.perf_counter() - t0) * 1000)

        mlx_features = np.array(out[0])
        mlx_mean_ms = np.mean(mlx_times)
        mlx_std_ms = np.std(mlx_times)

        print(f"  {mlx_mean_ms:.2f} ± {mlx_std_ms:.2f} ms ({1000 / mlx_mean_ms:.1f} FPS)")

    except Exception as e:
        print(f"  FAILED - {e}")
        import traceback
        traceback.print_exc()

    # --- Correlation & Speedup ---
    correlation = None
    speedup = None
    if pt_features is not None and mlx_features is not None:
        correlation = np.corrcoef(pt_features.flatten(), mlx_features.flatten())[0, 1]
        speedup = pt_mean_ms / mlx_mean_ms if mlx_mean_ms else 0
        print(f"\n  Correlation: {correlation:.6f}")
        print(f"  Speedup:     {speedup:.2f}x MLX faster")

    return {
        "model": f"DUNE {variant} @ {resolution}",
        "pt_ms": pt_mean_ms,
        "mlx_ms": mlx_mean_ms,
        "correlation": correlation,
        "speedup": speedup,
    }


def load_mlx_mast3r_full():
    """Load full MASt3R pipeline (encoder + decoder) MLX."""
    from mlx_mast3r.decoders.mast3r import Mast3rDecoderEngine

    engine = Mast3rDecoderEngine(
        resolution=512,
        precision="fp16",
        compile=True,
    )

    safetensors_path = SAFETENSORS_DIR / "mast3r_vit_large" / "unified.safetensors"
    engine.load(safetensors_path)

    return engine


def benchmark_mast3r_encoder() -> dict:
    """Benchmark MASt3R encoder: MLX vs PyTorch MPS."""
    print(f"\n{'=' * 60}")
    print("MASt3R ViT-Large Encoder @ 512")
    print("=" * 60)

    img = create_test_image(MAST3R_IMG_SIZE_512)

    pt_features = None
    pt_mean_ms = None
    mlx_features = None
    mlx_mean_ms = None

    # --- PyTorch MPS ---
    print("\nPyTorch MPS...")
    try:
        pt_model = load_pytorch_mast3r()

        img_pt = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_pt = (img_pt - 0.5) / 0.5
        img_pt = img_pt.to("mps")

        # Warmup
        with torch.no_grad():
            for _ in range(WARMUP_ITERATIONS):
                _ = pt_model._encode_image(img_pt, True)
                torch.mps.synchronize()

        # Benchmark
        pt_times = []
        with torch.no_grad():
            for _ in range(BENCHMARK_ITERATIONS):
                t0 = time.perf_counter()
                out_pt = pt_model._encode_image(img_pt, True)
                torch.mps.synchronize()
                pt_times.append((time.perf_counter() - t0) * 1000)

        # out_pt is tuple: (features, pos, shape)
        if isinstance(out_pt, tuple):
            pt_features = out_pt[0].cpu().numpy()[0]
        else:
            pt_features = out_pt.cpu().numpy()[0]

        pt_mean_ms = np.mean(pt_times)
        pt_std_ms = np.std(pt_times)

        print(f"  {pt_mean_ms:.2f} ± {pt_std_ms:.2f} ms ({1000 / pt_mean_ms:.1f} FPS)")

    except Exception as e:
        print(f"  FAILED - {e}")
        import traceback

        traceback.print_exc()

    # --- MLX ---
    print("\nMLX FP16...")
    try:
        mlx_engine = load_mlx_mast3r()

        # Warmup
        mlx_engine.warmup(WARMUP_ITERATIONS)

        # Benchmark
        mlx_times = []
        for _ in range(BENCHMARK_ITERATIONS):
            features, ms = mlx_engine.infer(img)
            mlx_times.append(ms)

        # MLX encoder now applies enc_norm internally (line 294 in mast3r.py)
        # so features are already normalized - no need to apply again
        mlx_features = features

        mlx_mean_ms = np.mean(mlx_times)
        mlx_std_ms = np.std(mlx_times)

        print(f"  {mlx_mean_ms:.2f} ± {mlx_std_ms:.2f} ms ({1000 / mlx_mean_ms:.1f} FPS)")

    except Exception as e:
        print(f"  FAILED - {e}")
        import traceback

        traceback.print_exc()

    # --- Correlation ---
    correlation = None
    speedup = None
    if pt_features is not None and mlx_features is not None:
        correlation = np.corrcoef(pt_features.flatten(), mlx_features.flatten())[0, 1]
        speedup = pt_mean_ms / mlx_mean_ms if mlx_mean_ms else 0
        print(f"\n  Correlation: {correlation:.6f}")
        print(f"  Speedup:     {speedup:.2f}x MLX faster")

    return {
        "model": "MASt3R ViT-L @ 512",
        "pt_ms": pt_mean_ms,
        "mlx_ms": mlx_mean_ms,
        "correlation": correlation,
        "speedup": speedup,
    }


def benchmark_mast3r_full() -> dict:
    """Benchmark full MASt3R pipeline (encoder + decoder): MLX vs PyTorch MPS."""
    print(f"\n{'=' * 60}")
    print("MASt3R Full Pipeline (Encoder + Decoder) @ 512")
    print("=" * 60)

    img1 = create_test_image(MAST3R_IMG_SIZE_512, seed=42)
    img2 = create_test_image(MAST3R_IMG_SIZE_512, seed=43)

    pt_pts3d = None
    pt_mean_ms = None
    mlx_pts3d = None
    mlx_mean_ms = None

    # --- PyTorch MPS ---
    print("\nPyTorch MPS...")
    try:
        pt_model = load_pytorch_mast3r()

        # Preprocess
        img1_pt = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img1_pt = (img1_pt - 0.5) / 0.5
        img1_pt = img1_pt.to("mps")

        img2_pt = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img2_pt = (img2_pt - 0.5) / 0.5
        img2_pt = img2_pt.to("mps")

        # Create view dictionaries as expected by MASt3R
        view1 = {
            "img": img1_pt,
            "true_shape": torch.tensor([[512, 672]]),
            "idx": 0,
            "instance": "0",
        }
        view2 = {
            "img": img2_pt,
            "true_shape": torch.tensor([[512, 672]]),
            "idx": 1,
            "instance": "1",
        }

        # Warmup
        with torch.no_grad():
            for _ in range(WARMUP_ITERATIONS):
                _ = pt_model(view1, view2)
                torch.mps.synchronize()

        # Benchmark
        pt_times = []
        with torch.no_grad():
            for _ in range(BENCHMARK_ITERATIONS):
                t0 = time.perf_counter()
                out = pt_model(view1, view2)
                torch.mps.synchronize()
                pt_times.append((time.perf_counter() - t0) * 1000)

        # Extract pts3d from output (MASt3R returns tuple or dict depending on version)
        if isinstance(out, dict):
            pt_pts3d = out["pred1"]["pts3d"].cpu().numpy()[0]
        else:
            # Tuple format: (pred1, pred2) where pred1 has pts3d
            pred1, pred2 = out
            if isinstance(pred1, dict):
                pt_pts3d = pred1["pts3d"].cpu().numpy()[0]
            else:
                # pred1 is a tuple (pts3d, conf, desc)
                pt_pts3d = pred1[0].cpu().numpy()[0]

        pt_mean_ms = np.mean(pt_times)
        pt_std_ms = np.std(pt_times)

        print(f"  {pt_mean_ms:.2f} ± {pt_std_ms:.2f} ms ({1000 / pt_mean_ms:.1f} FPS)")

    except Exception as e:
        print(f"  FAILED - {e}")
        import traceback
        traceback.print_exc()

    # --- MLX ---
    print("\nMLX FP16...")
    try:
        mlx_engine = load_mlx_mast3r_full()

        # Warmup
        mlx_engine.warmup(WARMUP_ITERATIONS)

        # Benchmark
        mlx_times = []
        for _ in range(BENCHMARK_ITERATIONS):
            out1, out2, ms = mlx_engine.infer(img1, img2)
            mlx_times.append(ms)

        mlx_pts3d = out1["pts3d"]
        mlx_mean_ms = np.mean(mlx_times)
        mlx_std_ms = np.std(mlx_times)

        print(f"  {mlx_mean_ms:.2f} ± {mlx_std_ms:.2f} ms ({1000 / mlx_mean_ms:.1f} FPS)")

    except Exception as e:
        print(f"  FAILED - {e}")
        import traceback
        traceback.print_exc()

    # --- Correlation & Speedup ---
    correlation = None
    speedup = None
    if pt_pts3d is not None and mlx_pts3d is not None:
        # Flatten and compute correlation
        pt_flat = pt_pts3d.flatten()
        mlx_flat = mlx_pts3d.flatten()
        # Match shapes (MASt3R outputs may have different shapes due to DPT)
        min_len = min(len(pt_flat), len(mlx_flat))
        correlation = np.corrcoef(pt_flat[:min_len], mlx_flat[:min_len])[0, 1]
        speedup = pt_mean_ms / mlx_mean_ms if mlx_mean_ms else 0
        print(f"\n  Correlation: {correlation:.6f}")
        print(f"  Speedup:     {speedup:.2f}x MLX faster")

    return {
        "model": "MASt3R Full @ 512",
        "pt_ms": pt_mean_ms,
        "mlx_ms": mlx_mean_ms,
        "correlation": correlation,
        "speedup": speedup,
    }


def load_mlx_dunemast3r_full(variant: str, resolution: int):
    """Load full DuneMASt3R pipeline (encoder + decoder) MLX."""
    from mlx_mast3r.decoders.dunemast3r import DuneMast3rDecoderEngine

    engine = DuneMast3rDecoderEngine(
        encoder_variant=variant,
        resolution=resolution,
        precision="fp16",
        compile=True,
    )

    encoder_path = DUNE_SAFETENSORS_DIR / f"dune_vit_{variant}_{resolution}" / "encoder.safetensors"
    decoder_path = DUNE_SAFETENSORS_DIR / f"dune_vit_{variant}_{resolution}" / "decoder.safetensors"
    engine.load(encoder_path, decoder_path)

    return engine


def load_pytorch_dunemast3r_local(variant: str, resolution: int):
    """Load DuneMASt3R from local checkpoints using thirdparty/dune."""
    inf = float("inf")  # noqa: F841  # Required for eval(mast3r_model_str)
    from model.dune import load_dune_encoder_from_checkpoint
    from mast3r.model import AsymmetricMASt3R, AsymmetricMASt3RWithDUNEBackbone  # noqa: F401

    # Load encoder from local .pth using DUNE's own loader
    encoder_path = PTH_DIR / f"dune_vit{variant}14_{resolution}.pth"
    encoder, _ = load_dune_encoder_from_checkpoint(str(encoder_path))
    patch_size = encoder.patch_size

    # Load decoder checkpoint
    decoder_ckpt_path = PTH_DIR / f"dunemast3r_cvpr25_vit{variant}.pth"
    decoder_ckpt = torch.load(str(decoder_ckpt_path), map_location="cpu", weights_only=False)

    # Build MASt3R model string
    mast3r_model_str = decoder_ckpt["mast3r_model_str"]

    # Add landscape_only=False
    if "landscape_only" not in mast3r_model_str:
        mast3r_model_str = mast3r_model_str[:-1] + ", landscape_only=False)"

    # Add patch_size
    if "patch_size" not in mast3r_model_str:
        mast3r_model_str = mast3r_model_str[:-1] + f", patch_size={patch_size})"

    # Create MASt3R decoder
    mast3r = eval(mast3r_model_str)

    # Modify DPT heads for patch_size=14
    if patch_size != 16:
        for head in [mast3r.downstream_head1, mast3r.downstream_head2]:
            head.dpt.patch_size = (patch_size, patch_size)
            head.dpt.P_H = max(1, patch_size // head.dpt.stride_level)
            head.dpt.P_W = max(1, patch_size // head.dpt.stride_level)
            head.dpt.head[1].scale_factor *= 14 / 16

    # Create combined model
    model = AsymmetricMASt3RWithDUNEBackbone.__new__(AsymmetricMASt3RWithDUNEBackbone)
    torch.nn.Module.__init__(model)

    model.dune_backbone = encoder
    model.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1))
    model.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1))
    model.norm = torch.nn.LayerNorm(encoder.num_features)
    model.landscape_only = False
    model.patch_size = patch_size
    model.square_ok = True
    model.mast3r = mast3r

    # Load decoder weights
    model.load_state_dict(decoder_ckpt["model"], strict=False)

    # Freeze encoder
    model.dune_backbone.eval()
    for p in model.dune_backbone.parameters():
        p.requires_grad = False

    return model.to("mps").eval()


def benchmark_dunemast3r_full(variant: str, resolution: int) -> dict:
    """Benchmark full DuneMASt3R pipeline: MLX vs PyTorch MPS.

    Uses local checkpoints for both encoder and decoder so that weights
    match between PyTorch and MLX (safetensors converted from same checkpoints).
    """
    print(f"\n{'=' * 60}")
    print(f"DuneMASt3R {variant.upper()} @ {resolution} (Full Pipeline)")
    print("=" * 60)

    img_size = (resolution, resolution, 3)
    img1 = create_test_image(img_size, seed=42)
    img2 = create_test_image(img_size, seed=43)

    pt_pts3d = None
    pt_mean_ms = None
    mlx_pts3d = None
    mlx_mean_ms = None

    # --- PyTorch MPS ---
    print("\nPyTorch MPS (local checkpoints)...")
    try:
        pt_model = load_pytorch_dunemast3r_local(variant, resolution)

        # Prepare images - MASt3R expects [-1, 1] normalized NCHW
        img1_pt = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img1_pt = (img1_pt - 0.5) / 0.5
        img1_pt = img1_pt.to("mps")
        img2_pt = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img2_pt = (img2_pt - 0.5) / 0.5
        img2_pt = img2_pt.to("mps")

        H, W = resolution, resolution
        view1 = {
            "img": img1_pt,
            "true_shape": torch.tensor([[H, W]]),
            "idx": 0,
            "instance": "0",
        }
        view2 = {
            "img": img2_pt,
            "true_shape": torch.tensor([[H, W]]),
            "idx": 1,
            "instance": "1",
        }

        # Warmup
        with torch.no_grad():
            for _ in range(WARMUP_ITERATIONS):
                _ = pt_model(view1, view2)
                torch.mps.synchronize()

        # Benchmark
        pt_times = []
        with torch.no_grad():
            for _ in range(BENCHMARK_ITERATIONS):
                t0 = time.perf_counter()
                out = pt_model(view1, view2)
                torch.mps.synchronize()
                pt_times.append((time.perf_counter() - t0) * 1000)

        pt_pts3d = out[0]["pts3d"].cpu().numpy()[0]
        pt_mean_ms = np.mean(pt_times)
        pt_std_ms = np.std(pt_times)

        print(f"  {pt_mean_ms:.2f} ± {pt_std_ms:.2f} ms ({1000 / pt_mean_ms:.1f} FPS)")

    except Exception as e:
        print(f"  FAILED - {e}")
        import traceback

        traceback.print_exc()

    # --- MLX ---
    print("\nMLX FP16...")
    try:
        mlx_engine = load_mlx_dunemast3r_full(variant, resolution)

        # Warmup
        mlx_engine.warmup(WARMUP_ITERATIONS)

        # Benchmark
        mlx_times = []
        for _ in range(BENCHMARK_ITERATIONS):
            out1, out2, ms = mlx_engine.infer(img1, img2)
            mlx_times.append(ms)

        mlx_pts3d = out1["pts3d"]
        mlx_mean_ms = np.mean(mlx_times)
        mlx_std_ms = np.std(mlx_times)

        print(f"  {mlx_mean_ms:.2f} ± {mlx_std_ms:.2f} ms ({1000 / mlx_mean_ms:.1f} FPS)")

    except Exception as e:
        print(f"  FAILED - {e}")
        import traceback

        traceback.print_exc()

    # --- Correlation & Speedup ---
    correlation = None
    speedup = None
    if pt_pts3d is not None and mlx_pts3d is not None:
        # Flatten and compute correlation
        pt_flat = pt_pts3d.flatten()
        mlx_flat = mlx_pts3d.flatten()
        min_len = min(len(pt_flat), len(mlx_flat))
        correlation = np.corrcoef(pt_flat[:min_len], mlx_flat[:min_len])[0, 1]
        speedup = pt_mean_ms / mlx_mean_ms if mlx_mean_ms else 0
        print(f"\n  Correlation: {correlation:.6f}")
        print(f"  Speedup:     {speedup:.2f}x MLX faster")

    return {
        "model": f"DuneMASt3R {variant} @ {resolution}",
        "pt_ms": pt_mean_ms,
        "mlx_ms": mlx_mean_ms,
        "correlation": correlation,
        "speedup": speedup,
    }


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 70)
    print("MLX-MASt3R Benchmark: MLX (safetensors) vs PyTorch MPS (pth)")
    print("=" * 70)
    print(f"Warmup: {WARMUP_ITERATIONS} | Iterations: {BENCHMARK_ITERATIONS}")

    results = []

    # 1. DUNE Encoders
    print("\n" + "=" * 70)
    print("PART 1: DUNE ENCODERS")
    print("=" * 70)

    for variant in ["small", "base"]:
        for resolution in [336, 448]:
            try:
                result = benchmark_dune_encoder(variant, resolution)
                results.append(result)
            except Exception as e:
                print(f"FAILED: DUNE {variant} @ {resolution}: {e}")
                results.append(
                    {
                        "model": f"DUNE {variant} @ {resolution}",
                        "pt_ms": None,
                        "mlx_ms": None,
                        "correlation": None,
                        "speedup": None,
                    }
                )

    # 2. MASt3R Encoder
    print("\n" + "=" * 70)
    print("PART 2: MASt3R ENCODER")
    print("=" * 70)

    try:
        result = benchmark_mast3r_encoder()
        results.append(result)
    except Exception as e:
        print(f"FAILED: MASt3R Encoder: {e}")
        results.append(
            {
                "model": "MASt3R ViT-L @ 512",
                "pt_ms": None,
                "mlx_ms": None,
                "correlation": None,
                "speedup": None,
            }
        )

    # 3. MASt3R Full Pipeline (Encoder + Decoder)
    print("\n" + "=" * 70)
    print("PART 3: MASt3R FULL PIPELINE (ENCODER + DECODER)")
    print("=" * 70)

    try:
        result = benchmark_mast3r_full()
        results.append(result)
    except Exception as e:
        print(f"FAILED: MASt3R Full Pipeline: {e}")
        results.append(
            {
                "model": "MASt3R Full @ 512",
                "pt_ms": None,
                "mlx_ms": None,
                "correlation": None,
                "speedup": None,
            }
        )

    # 4. DuneMASt3R Full Pipeline (Encoder + Decoder)
    print("\n" + "=" * 70)
    print("PART 4: DuneMASt3R FULL PIPELINE (ENCODER + DECODER)")
    print("=" * 70)

    for variant in ["small", "base"]:
        for resolution in [336, 448]:
            try:
                result = benchmark_dunemast3r_full(variant, resolution)
                results.append(result)
            except Exception as e:
                print(f"FAILED: DuneMASt3R {variant} @ {resolution}: {e}")
                results.append(
                    {
                        "model": f"DuneMASt3R {variant} @ {resolution}",
                        "pt_ms": None,
                        "mlx_ms": None,
                        "correlation": None,
                        "speedup": None,
                    }
                )

    # Summary Table
    print("\n")
    print("=" * 90)
    print("SUMMARY: MLX vs PyTorch MPS")
    print("=" * 90)
    print(
        f"{'Model':<25} {'PyTorch (ms)':<15} {'MLX (ms)':<15} {'Speedup':<10} {'Correlation':<12}"
    )
    print("-" * 90)

    for r in results:
        pt_str = f"{r['pt_ms']:.2f}" if r["pt_ms"] else "N/A"
        mlx_str = f"{r['mlx_ms']:.2f}" if r["mlx_ms"] else "N/A"
        speedup_str = f"{r['speedup']:.2f}x" if r["speedup"] else "N/A"
        corr_str = f"{r['correlation']:.6f}" if r["correlation"] else "N/A"

        print(f"{r['model']:<25} {pt_str:<15} {mlx_str:<15} {speedup_str:<10} {corr_str:<12}")

    print("-" * 90)

    # Calculate averages
    valid_results = [r for r in results if r["speedup"] is not None]
    if valid_results:
        avg_speedup = np.mean([r["speedup"] for r in valid_results])
        valid_corrs = [r["correlation"] for r in valid_results if r["correlation"] is not None]
        avg_corr_str = f"{np.mean(valid_corrs):.6f}" if valid_corrs else "N/A"
        print(f"{'AVERAGE':<25} {'':<15} {'':<15} {avg_speedup:.2f}x      {avg_corr_str}")

    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
