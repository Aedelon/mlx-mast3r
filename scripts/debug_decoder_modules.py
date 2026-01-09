#!/usr/bin/env python3
"""Debug script: rigorous module-by-module comparison of decoder.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path.home() / "Workspace/mast3r"))

import mlx.core as mx

SAFETENSORS_DIR = Path.home() / ".cache/mlx-mast3r"


def create_test_image(shape: tuple[int, int, int], seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.randint(0, 256, shape, dtype=np.uint8)


def compare(name: str, pt: np.ndarray, mlx: np.ndarray, verbose: bool = True) -> float:
    """Compare arrays and return correlation."""
    pt_flat = pt.flatten().astype(np.float64)
    mlx_flat = mlx.flatten().astype(np.float64)

    min_len = min(len(pt_flat), len(mlx_flat))
    corr = np.corrcoef(pt_flat[:min_len], mlx_flat[:min_len])[0, 1]

    status = "✓" if corr > 0.99 else "✗" if corr < 0.9 else "~"

    if verbose:
        print(
            f"  {status} {name}: corr={corr:.6f} | PT shape={pt.shape} mean={pt.mean():.4f} | MLX shape={mlx.shape} mean={mlx.mean():.4f}"
        )

    return corr


def main():
    print("=" * 80)
    print("RIGOROUS MODULE-BY-MODULE DECODER COMPARISON")
    print("=" * 80)

    img_shape = (512, 672, 3)
    img1 = create_test_image(img_shape, seed=42)
    img2 = create_test_image(img_shape, seed=43)

    # =========================================================================
    # Load PyTorch model
    # =========================================================================
    print("\n[1] Loading PyTorch MASt3R...")
    from mast3r.model import AsymmetricMASt3R

    pt_model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    )
    pt_model = pt_model.to("mps").eval()

    # =========================================================================
    # Load MLX model
    # =========================================================================
    print("[2] Loading MLX MASt3R...")
    from mlx_mast3r.decoders.mast3r import Mast3rDecoderEngine

    mlx_engine = Mast3rDecoderEngine(resolution=512, precision="fp32", compile=False)
    safetensors_path = SAFETENSORS_DIR / "mast3r_vit_large" / "unified.safetensors"
    mlx_engine.load(safetensors_path)

    # =========================================================================
    # Prepare inputs
    # =========================================================================
    print("[3] Preparing inputs...")

    # PyTorch
    img1_pt = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img1_pt = (img1_pt - 0.5) / 0.5
    img1_pt = img1_pt.to("mps")

    img2_pt = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2_pt = (img2_pt - 0.5) / 0.5
    img2_pt = img2_pt.to("mps")

    # MLX
    x1 = img1.astype(np.float32) / 255.0
    x1 = (x1 - 0.5) / 0.5
    x2 = img2.astype(np.float32) / 255.0
    x2 = (x2 - 0.5) / 0.5

    x1_mx = mx.array(x1[None, :, :, :])
    x2_mx = mx.array(x2[None, :, :, :])

    # =========================================================================
    # Step 1: Compare encoder outputs
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: ENCODER OUTPUT")
    print("=" * 80)

    with torch.no_grad():
        pt_enc1 = pt_model._encode_image(img1_pt, True)
        pt_enc2 = pt_model._encode_image(img2_pt, True)
        if isinstance(pt_enc1, tuple):
            pt_enc1, pt_pos1, _ = pt_enc1
            pt_enc2, pt_pos2, _ = pt_enc2
        pt_enc1_np = pt_enc1.cpu().numpy()[0]
        pt_enc2_np = pt_enc2.cpu().numpy()[0]

    mlx_enc1 = mlx_engine.encoder(x1_mx)
    mlx_enc2 = mlx_engine.encoder(x2_mx)
    mx.eval(mlx_enc1, mlx_enc2)

    compare("encoder_feat1", pt_enc1_np, np.array(mlx_enc1[0]))
    compare("encoder_feat2", pt_enc2_np, np.array(mlx_enc2[0]))

    # =========================================================================
    # Step 2: Compare enc_norm
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: ENC_NORM (encoder normalization)")
    print("=" * 80)

    # PyTorch enc_norm
    with torch.no_grad():
        pt_enc1_normed = pt_model.enc_norm(pt_enc1)
        pt_enc2_normed = pt_model.enc_norm(pt_enc2)

    pt_enc1_normed_np = pt_enc1_normed.cpu().numpy()[0]
    pt_enc2_normed_np = pt_enc2_normed.cpu().numpy()[0]

    # MLX enc_norm
    decoder = mlx_engine.decoder
    mlx_enc1_normed = mx.fast.layer_norm(
        mlx_enc1, decoder.enc_norm_weight, decoder.enc_norm_bias, eps=1e-6
    )
    mlx_enc2_normed = mx.fast.layer_norm(
        mlx_enc2, decoder.enc_norm_weight, decoder.enc_norm_bias, eps=1e-6
    )
    mx.eval(mlx_enc1_normed, mlx_enc2_normed)

    compare("enc_norm_feat1", pt_enc1_normed_np, np.array(mlx_enc1_normed[0]))
    compare("enc_norm_feat2", pt_enc2_normed_np, np.array(mlx_enc2_normed[0]))

    # =========================================================================
    # Step 3: Compare decoder_embed
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: DECODER_EMBED (projection 1024 -> 768)")
    print("=" * 80)

    # PyTorch decoder_embed
    with torch.no_grad():
        pt_dec1 = pt_model.decoder_embed(pt_enc1_normed)
        pt_dec2 = pt_model.decoder_embed(pt_enc2_normed)

    pt_dec1_np = pt_dec1.cpu().numpy()[0]
    pt_dec2_np = pt_dec2.cpu().numpy()[0]

    # MLX decoder_embed
    mlx_dec1 = decoder.decoder_embed(mlx_enc1_normed)
    mlx_dec2 = decoder.decoder_embed(mlx_enc2_normed)
    mx.eval(mlx_dec1, mlx_dec2)

    compare("decoder_embed_feat1", pt_dec1_np, np.array(mlx_dec1[0]))
    compare("decoder_embed_feat2", pt_dec2_np, np.array(mlx_dec2[0]))

    # =========================================================================
    # Step 4: Compare decoder blocks one by one
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: DECODER BLOCKS (one by one)")
    print("=" * 80)

    # Initialize MLX decoder state
    H, W = mlx_engine.encoder_config.patch_h, mlx_engine.encoder_config.patch_w
    if decoder._rope_cos is None:
        decoder._init_rope(H, W)

    # Start with decoder_embed outputs
    pt_x1, pt_x2 = pt_dec1, pt_dec2
    mlx_x1, mlx_x2 = mlx_dec1, mlx_dec2

    # Process each decoder block
    for i in range(min(3, len(decoder.dec_blocks))):  # Test first 3 blocks
        print(f"\n--- Block {i} ---")

        # PyTorch block
        with torch.no_grad():
            pt_blk1 = pt_model.dec_blocks[i]
            pt_blk2 = pt_model.dec_blocks2[i]

            # Get intermediate outputs
            # Self-attention
            pt_x1_norm1 = pt_blk1.norm1(pt_x1)
            pt_x2_norm1 = pt_blk2.norm1(pt_x2)

        pt_x1_norm1_np = pt_x1_norm1.cpu().numpy()[0]

        # MLX block
        mlx_blk1 = decoder.dec_blocks[i]
        mlx_blk2 = decoder.dec_blocks2[i]

        # MLX norm1
        mlx_x1_norm1 = mx.fast.layer_norm(
            mlx_x1, mlx_blk1.norm1_weight, mlx_blk1.norm1_bias, eps=1e-6
        )
        mx.eval(mlx_x1_norm1)

        compare(f"block{i}_norm1", pt_x1_norm1_np, np.array(mlx_x1_norm1[0]))

        # Self-attention (PyTorch needs xpos for RoPE)
        with torch.no_grad():
            # Get position info from encoder
            pt_self_attn_out = pt_blk1.attn(pt_x1_norm1, pt_pos1)

        pt_self_attn_np = pt_self_attn_out.cpu().numpy()[0]

        mlx_self_attn_out = mlx_blk1.self_attn(mlx_x1_norm1)
        mx.eval(mlx_self_attn_out)

        compare(f"block{i}_self_attn", pt_self_attn_np, np.array(mlx_self_attn_out[0]))

        # Residual after self-attention
        with torch.no_grad():
            pt_x1_after_sa = pt_x1 + pt_self_attn_out

        mlx_x1_after_sa = mlx_x1 + mlx_self_attn_out
        mx.eval(mlx_x1_after_sa)

        compare(
            f"block{i}_after_self_attn",
            pt_x1_after_sa.cpu().numpy()[0],
            np.array(mlx_x1_after_sa[0]),
        )

        # Cross-attention norm (norm_y in PyTorch, norm2 in MLX)
        with torch.no_grad():
            pt_x1_norm_y = pt_blk1.norm_y(pt_x1_after_sa)

        mlx_x1_norm2 = mx.fast.layer_norm(
            mlx_x1_after_sa, mlx_blk1.norm2_weight, mlx_blk1.norm2_bias, eps=1e-6
        )
        mx.eval(mlx_x1_norm2)

        compare(f"block{i}_norm_y/norm2", pt_x1_norm_y.cpu().numpy()[0], np.array(mlx_x1_norm2[0]))

        # Cross-attention (skip detailed comparison, needs qpos/kpos)
        # Just compare full block output instead

        # Full block output
        with torch.no_grad():
            pt_out1 = pt_blk1(pt_x1, pt_x2, pt_pos1, pt_pos2)
            pt_out2 = pt_blk2(pt_x2, pt_x1, pt_pos2, pt_pos1)
            # Handle tuple output
            pt_x1_new = pt_out1[0] if isinstance(pt_out1, tuple) else pt_out1
            pt_x2_new = pt_out2[0] if isinstance(pt_out2, tuple) else pt_out2

        mlx_x1_new = mlx_blk1(mlx_x1, mlx_x2)
        mlx_x2_new = mlx_blk2(mlx_x2, mlx_x1)
        mx.eval(mlx_x1_new, mlx_x2_new)

        compare(f"block{i}_output_x1", pt_x1_new.cpu().numpy()[0], np.array(mlx_x1_new[0]))
        compare(f"block{i}_output_x2", pt_x2_new.cpu().numpy()[0], np.array(mlx_x2_new[0]))

        # Update for next iteration
        pt_x1, pt_x2 = pt_x1_new, pt_x2_new
        mlx_x1, mlx_x2 = mlx_x1_new, mlx_x2_new

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Check the correlations above to identify where divergence starts.")
    print("✓ = corr > 0.99 (good)")
    print("~ = 0.9 < corr < 0.99 (warning)")
    print("✗ = corr < 0.9 (problem)")


if __name__ == "__main__":
    main()
