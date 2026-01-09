#!/usr/bin/env python3
"""Debug script to compare DPT input features and intermediate outputs.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path.home() / "Workspace/mast3r"))

import mlx.core as mx

SAFETENSORS_DIR = Path.home() / ".cache/mlx-mast3r"


def create_test_image(shape: tuple[int, int, int], seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.randint(0, 256, shape, dtype=np.uint8)


def compare_arrays(name: str, pt: np.ndarray, mlx: np.ndarray) -> float:
    """Compare arrays and return correlation."""
    pt_flat = pt.flatten()
    mlx_flat = mlx.flatten()

    min_len = min(len(pt_flat), len(mlx_flat))
    if len(pt_flat) != len(mlx_flat):
        print(f"  {name}: SHAPE MISMATCH - PT {pt.shape} vs MLX {mlx.shape}")

    corr = np.corrcoef(pt_flat[:min_len], mlx_flat[:min_len])[0, 1]

    pt_mean = np.mean(pt_flat[:min_len])
    mlx_mean = np.mean(mlx_flat[:min_len])
    pt_std = np.std(pt_flat[:min_len])
    mlx_std = np.std(mlx_flat[:min_len])

    print(
        f"  {name}: corr={corr:.6f} | PT mean={pt_mean:.4f} std={pt_std:.4f} | MLX mean={mlx_mean:.4f} std={mlx_std:.4f}"
    )

    return corr


def main():
    print("=" * 70)
    print("DPT Features Debug: Compare input features")
    print("=" * 70)

    img_shape = (512, 672, 3)
    img1 = create_test_image(img_shape, seed=42)
    img2 = create_test_image(img_shape, seed=43)

    # =========================================================================
    # PyTorch model - hook to capture DPT input features
    # =========================================================================
    print("\n[1] Loading PyTorch MASt3R...")
    from mast3r.model import AsymmetricMASt3R

    pt_model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    )
    pt_model = pt_model.to("mps").eval()

    # Preprocess
    img1_pt = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img1_pt = (img1_pt - 0.5) / 0.5
    img1_pt = img1_pt.to("mps")

    img2_pt = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2_pt = (img2_pt - 0.5) / 0.5
    img2_pt = img2_pt.to("mps")

    view1 = {"img": img1_pt, "true_shape": torch.tensor([[512, 672]]), "idx": 0, "instance": "0"}
    view2 = {"img": img2_pt, "true_shape": torch.tensor([[512, 672]]), "idx": 1, "instance": "1"}

    # Hook to capture DPT input features
    pt_dpt_inputs = {}

    def make_dpt_input_hook(name):
        def hook(module, args):
            # DPT forward receives encoder_tokens as first positional arg
            if len(args) > 0:
                encoder_tokens = args[0]
                if isinstance(encoder_tokens, list):
                    pt_dpt_inputs[name] = [t.detach().cpu().numpy() for t in encoder_tokens]

        return hook

    # Register hook on DPT forward
    head1 = pt_model.downstream_head1
    head1.dpt.register_forward_pre_hook(make_dpt_input_hook("dpt1_input"))

    print("\n[2] Running PyTorch forward pass...")
    with torch.no_grad():
        pt_out = pt_model(view1, view2)

    # =========================================================================
    # Analyze PyTorch DPT inputs
    # =========================================================================
    print("\n[3] PyTorch DPT input features:")
    if "dpt1_input" in pt_dpt_inputs:
        pt_features = pt_dpt_inputs["dpt1_input"]
        for i, feat in enumerate(pt_features):
            print(
                f"  features[{i}]: shape={feat.shape}, mean={feat.mean():.4f}, std={feat.std():.4f}, range=[{feat.min():.4f}, {feat.max():.4f}]"
            )
    else:
        print("  ERROR: Could not capture DPT input features")

    # =========================================================================
    # MLX model
    # =========================================================================
    print("\n[4] Loading MLX MASt3R...")
    from mlx_mast3r.decoders.mast3r import Mast3rDecoderEngine

    mlx_engine = Mast3rDecoderEngine(resolution=512, precision="fp32", compile=False)
    safetensors_path = SAFETENSORS_DIR / "mast3r_vit_large" / "unified.safetensors"
    mlx_engine.load(safetensors_path)

    # Get MLX features by running encoder and decoder separately
    print("\n[5] Running MLX encoder...")

    # Preprocess for MLX
    x1 = img1.astype(np.float32) / 255.0
    x1 = (x1 - 0.5) / 0.5
    x2 = img2.astype(np.float32) / 255.0
    x2 = (x2 - 0.5) / 0.5

    x1_mx = mx.array(x1[None, :, :, :])
    x2_mx = mx.array(x2[None, :, :, :])

    # Run encoder
    feat1_mlx = mlx_engine.encoder(x1_mx)
    feat2_mlx = mlx_engine.encoder(x2_mx)
    mx.eval(feat1_mlx, feat2_mlx)

    print(f"  Encoder output: {feat1_mlx.shape}")

    # =========================================================================
    # Compare encoder outputs
    # =========================================================================
    print("\n[6] Comparing encoder outputs:")

    # Get PyTorch encoder output
    with torch.no_grad():
        pt_enc1 = pt_model._encode_image(img1_pt, True)
        if isinstance(pt_enc1, tuple):
            pt_enc1 = pt_enc1[0]
        pt_enc1_np = pt_enc1.cpu().numpy()[0]

    mlx_enc1_np = np.array(feat1_mlx[0])

    compare_arrays("encoder_output", pt_enc1_np, mlx_enc1_np)

    # =========================================================================
    # Compare DPT input features directly
    # =========================================================================
    print("\n[7] Comparing DPT input features:")

    if "dpt1_input" in pt_dpt_inputs:
        pt_features = pt_dpt_inputs["dpt1_input"]

        # Run MLX decoder to get features
        # We need to manually collect features from MLX decoder
        decoder = mlx_engine.decoder
        config = decoder.config

        H, W = mlx_engine.encoder_config.patch_h, mlx_engine.encoder_config.patch_w

        # Initialize RoPE if needed
        if decoder._rope_cos is None:
            decoder._init_rope(H, W)

        # Normalize encoder outputs
        enc_feat1 = mx.fast.layer_norm(
            feat1_mlx, decoder.enc_norm_weight, decoder.enc_norm_bias, eps=1e-6
        )
        enc_feat2 = mx.fast.layer_norm(
            feat2_mlx, decoder.enc_norm_weight, decoder.enc_norm_bias, eps=1e-6
        )

        # Project to decoder dim
        x1 = decoder.decoder_embed(enc_feat1)
        x2 = decoder.decoder_embed(enc_feat2)

        # Hooks: [0, 6, 9, 12]
        hooks = [0, 6, 9, 12]
        mlx_features = [feat1_mlx]  # Hook 0: original encoder features (1024 dim)

        # Decoder blocks with cross-attention
        for i, (blk1, blk2) in enumerate(zip(decoder.dec_blocks, decoder.dec_blocks2)):
            x1 = blk1(x1, x2)
            x2 = blk2(x2, x1)

            layer_idx = i + 1
            if layer_idx in hooks[1:]:
                mlx_features.append(x1)

        # Final norm
        x1_norm = mx.fast.layer_norm(x1, decoder.dec_norm_weight, decoder.dec_norm_bias, eps=1e-6)

        # Update last feature with normed version
        if len(mlx_features) == 4:
            mlx_features[-1] = x1_norm

        mx.eval(*mlx_features)

        print("\n  MLX collected features:")
        for i, feat in enumerate(mlx_features):
            feat_np = np.array(feat[0])
            print(
                f"    features[{i}]: shape={feat_np.shape}, mean={feat_np.mean():.4f}, std={feat_np.std():.4f}, range=[{feat_np.min():.4f}, {feat_np.max():.4f}]"
            )

        print("\n  Comparing each feature:")
        for i, (pt_feat, mlx_feat) in enumerate(zip(pt_features, mlx_features)):
            mlx_feat_np = np.array(mlx_feat[0])
            compare_arrays(f"feature[{i}]", pt_feat[0], mlx_feat_np)

    print("\n✓ Debug complete!")


if __name__ == "__main__":
    main()
