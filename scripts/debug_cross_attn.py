#!/usr/bin/env python3
"""Debug cross-attention: compare weights and intermediate outputs.

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


def compare(name: str, pt: np.ndarray, mlx: np.ndarray) -> float:
    pt_flat = pt.flatten().astype(np.float64)
    mlx_flat = mlx.flatten().astype(np.float64)
    min_len = min(len(pt_flat), len(mlx_flat))
    corr = np.corrcoef(pt_flat[:min_len], mlx_flat[:min_len])[0, 1]
    status = "✓" if corr > 0.99 else "✗" if corr < 0.9 else "~"
    print(f"  {status} {name}: corr={corr:.6f} | PT shape={pt.shape} | MLX shape={mlx.shape}")
    return corr


def main():
    print("=" * 80)
    print("DEBUG CROSS-ATTENTION WEIGHTS AND OUTPUTS")
    print("=" * 80)

    # Load models
    print("\n[1] Loading models...")
    from mast3r.model import AsymmetricMASt3R
    from mlx_mast3r.decoders.mast3r import Mast3rDecoderEngine

    pt_model = (
        AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
        .to("mps")
        .eval()
    )

    mlx_engine = Mast3rDecoderEngine(resolution=512, precision="fp32", compile=False)
    mlx_engine.load(SAFETENSORS_DIR / "mast3r_vit_large" / "unified.safetensors")
    decoder = mlx_engine.decoder

    # =========================================================================
    # Compare Cross-Attention Weights
    # =========================================================================
    print("\n" + "=" * 80)
    print("CROSS-ATTENTION WEIGHTS (Block 0)")
    print("=" * 80)

    pt_blk = pt_model.dec_blocks[0]
    mlx_blk = decoder.dec_blocks[0]

    # PyTorch cross_attn has: projq, projk, projv, proj
    pt_cross = pt_blk.cross_attn

    # Q weights
    pt_q_w = pt_cross.projq.weight.detach().cpu().numpy()
    pt_q_b = pt_cross.projq.bias.detach().cpu().numpy()
    mlx_q_w = np.array(mlx_blk.cross_attn.q.weight)
    mlx_q_b = np.array(mlx_blk.cross_attn.q.bias)

    print("\n[Q projection]")
    compare("q.weight", pt_q_w, mlx_q_w)
    compare("q.bias", pt_q_b, mlx_q_b)

    # K and V weights
    pt_k_w = pt_cross.projk.weight.detach().cpu().numpy()
    pt_k_b = pt_cross.projk.bias.detach().cpu().numpy()
    pt_v_w = pt_cross.projv.weight.detach().cpu().numpy()
    pt_v_b = pt_cross.projv.bias.detach().cpu().numpy()

    # MLX has combined KV
    mlx_kv_w = np.array(mlx_blk.cross_attn.kv.weight)
    mlx_kv_b = np.array(mlx_blk.cross_attn.kv.bias)

    print(f"\n[K/V weights]")
    print(f"  PT projk.weight: {pt_k_w.shape}")
    print(f"  PT projv.weight: {pt_v_w.shape}")
    print(f"  MLX kv.weight: {mlx_kv_w.shape}")

    # Compare K part (first half of KV)
    mlx_k_w = mlx_kv_w[:768, :]  # First 768 rows are K
    mlx_k_b = mlx_kv_b[:768]
    mlx_v_w = mlx_kv_w[768:, :]  # Second 768 rows are V
    mlx_v_b = mlx_kv_b[768:]

    compare("k.weight", pt_k_w, mlx_k_w)
    compare("k.bias", pt_k_b, mlx_k_b)
    compare("v.weight", pt_v_w, mlx_v_w)
    compare("v.bias", pt_v_b, mlx_v_b)

    # Output projection
    pt_proj_w = pt_cross.proj.weight.detach().cpu().numpy()
    pt_proj_b = pt_cross.proj.bias.detach().cpu().numpy()
    mlx_proj_w = np.array(mlx_blk.cross_attn.proj.weight)
    mlx_proj_b = np.array(mlx_blk.cross_attn.proj.bias)

    print("\n[Output projection]")
    compare("proj.weight", pt_proj_w, mlx_proj_w)
    compare("proj.bias", pt_proj_b, mlx_proj_b)

    # =========================================================================
    # Compare MLP Weights
    # =========================================================================
    print("\n" + "=" * 80)
    print("MLP WEIGHTS (Block 0)")
    print("=" * 80)

    # norm3 weights
    pt_norm3_w = pt_blk.norm3.weight.detach().cpu().numpy()
    pt_norm3_b = pt_blk.norm3.bias.detach().cpu().numpy()
    mlx_norm3_w = np.array(mlx_blk.norm3_weight)
    mlx_norm3_b = np.array(mlx_blk.norm3_bias)

    print("\n[norm3]")
    compare("norm3.weight", pt_norm3_w, mlx_norm3_w)
    compare("norm3.bias", pt_norm3_b, mlx_norm3_b)

    # MLP fc1, fc2
    pt_fc1_w = pt_blk.mlp.fc1.weight.detach().cpu().numpy()
    pt_fc1_b = pt_blk.mlp.fc1.bias.detach().cpu().numpy()
    pt_fc2_w = pt_blk.mlp.fc2.weight.detach().cpu().numpy()
    pt_fc2_b = pt_blk.mlp.fc2.bias.detach().cpu().numpy()

    mlx_fc1_w = np.array(mlx_blk.mlp.fc1.weight)
    mlx_fc1_b = np.array(mlx_blk.mlp.fc1.bias)
    mlx_fc2_w = np.array(mlx_blk.mlp.fc2.weight)
    mlx_fc2_b = np.array(mlx_blk.mlp.fc2.bias)

    print("\n[MLP]")
    compare("mlp.fc1.weight", pt_fc1_w, mlx_fc1_w)
    compare("mlp.fc1.bias", pt_fc1_b, mlx_fc1_b)
    compare("mlp.fc2.weight", pt_fc2_w, mlx_fc2_w)
    compare("mlp.fc2.bias", pt_fc2_b, mlx_fc2_b)

    # =========================================================================
    # Test Cross-Attention Operation
    # =========================================================================
    print("\n" + "=" * 80)
    print("CROSS-ATTENTION OPERATION TEST")
    print("=" * 80)

    # Create random test inputs
    np.random.seed(42)
    B, N, D = 1, 1344, 768

    test_query = np.random.randn(B, N, D).astype(np.float32) * 0.1
    test_context = np.random.randn(B, N, D).astype(np.float32) * 0.1

    # PyTorch
    pt_query = torch.from_numpy(test_query).to("mps")
    pt_context = torch.from_numpy(test_context).to("mps")

    # MLX
    mlx_query = mx.array(test_query)
    mlx_context = mx.array(test_context)

    print("\n[Testing with random input]")

    # Q projection
    with torch.no_grad():
        pt_q_out = pt_cross.projq(pt_query)

    mlx_q_out = mlx_blk.cross_attn.q(mlx_query)
    mx.eval(mlx_q_out)

    compare("Q projection output", pt_q_out.cpu().numpy()[0], np.array(mlx_q_out[0]))

    # K projection
    with torch.no_grad():
        pt_k_out = pt_cross.projk(pt_context)
        pt_v_out = pt_cross.projv(pt_context)

    mlx_kv_out = mlx_blk.cross_attn.kv(mlx_context)
    mx.eval(mlx_kv_out)

    # MLX KV is [B, N, 2*D], split into K and V
    mlx_k_out = mlx_kv_out[:, :, :D]
    mlx_v_out = mlx_kv_out[:, :, D:]

    compare("K projection output", pt_k_out.cpu().numpy()[0], np.array(mlx_k_out[0]))
    compare("V projection output", pt_v_out.cpu().numpy()[0], np.array(mlx_v_out[0]))

    # =========================================================================
    # Test with real decoder features
    # =========================================================================
    print("\n" + "=" * 80)
    print("CROSS-ATTENTION WITH REAL FEATURES")
    print("=" * 80)

    # Get real features from encoder
    img_shape = (512, 672, 3)
    np.random.seed(42)
    img1 = np.random.randint(0, 256, img_shape, dtype=np.uint8)
    img2 = np.random.randint(0, 256, img_shape, dtype=np.uint8)

    # PyTorch preprocessing
    img1_pt = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img1_pt = (img1_pt - 0.5) / 0.5
    img1_pt = img1_pt.to("mps")
    img2_pt = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2_pt = (img2_pt - 0.5) / 0.5
    img2_pt = img2_pt.to("mps")

    # Get PyTorch encoder output
    with torch.no_grad():
        pt_enc1, pt_pos1, _ = pt_model._encode_image(img1_pt, True)
        pt_enc2, pt_pos2, _ = pt_model._encode_image(img2_pt, True)
        pt_enc1_norm = pt_model.enc_norm(pt_enc1)
        pt_enc2_norm = pt_model.enc_norm(pt_enc2)
        pt_x1 = pt_model.decoder_embed(pt_enc1_norm)
        pt_x2 = pt_model.decoder_embed(pt_enc2_norm)

    # MLX preprocessing
    x1 = img1.astype(np.float32) / 255.0
    x1 = (x1 - 0.5) / 0.5
    x2 = img2.astype(np.float32) / 255.0
    x2 = (x2 - 0.5) / 0.5

    mlx_enc1 = mlx_engine.encoder(mx.array(x1[None]))
    mlx_enc2 = mlx_engine.encoder(mx.array(x2[None]))
    mlx_enc1_norm = mx.fast.layer_norm(mlx_enc1, decoder.enc_norm_weight, decoder.enc_norm_bias, eps=1e-6)
    mlx_enc2_norm = mx.fast.layer_norm(mlx_enc2, decoder.enc_norm_weight, decoder.enc_norm_bias, eps=1e-6)
    mlx_x1 = decoder.decoder_embed(mlx_enc1_norm)
    mlx_x2 = decoder.decoder_embed(mlx_enc2_norm)
    mx.eval(mlx_x1, mlx_x2)

    # Init RoPE
    H, W = mlx_engine.encoder_config.patch_h, mlx_engine.encoder_config.patch_w
    if decoder._rope_cos is None:
        decoder._init_rope(H, W)

    pt_blk = pt_model.dec_blocks[0]
    mlx_blk = decoder.dec_blocks[0]

    # Step through block 0
    print("\n[Block 0 step-by-step]")

    # Self-attention
    with torch.no_grad():
        pt_norm1 = pt_blk.norm1(pt_x1)
        pt_self_out = pt_blk.attn(pt_norm1, pt_pos1)
        pt_after_sa = pt_x1 + pt_self_out

    mlx_norm1 = mx.fast.layer_norm(mlx_x1, mlx_blk.norm1_weight, mlx_blk.norm1_bias, eps=1e-6)
    mlx_self_out = mlx_blk.self_attn(mlx_norm1)
    mlx_after_sa = mlx_x1 + mlx_self_out
    mx.eval(mlx_after_sa)

    compare("after_self_attn", pt_after_sa.cpu().numpy()[0], np.array(mlx_after_sa[0]))

    # Cross-attention norm (norm_y)
    with torch.no_grad():
        pt_norm_y = pt_blk.norm_y(pt_after_sa)

    mlx_norm_y = mx.fast.layer_norm(mlx_after_sa, mlx_blk.norm2_weight, mlx_blk.norm2_bias, eps=1e-6)
    mx.eval(mlx_norm_y)

    compare("norm_y", pt_norm_y.cpu().numpy()[0], np.array(mlx_norm_y[0]))

    # Cross-attention detailed
    print("\n[Cross-attention detailed]")

    # Q, K, V projections
    with torch.no_grad():
        pt_q = pt_blk.cross_attn.projq(pt_norm_y)
        pt_k = pt_blk.cross_attn.projk(pt_x2)
        pt_v = pt_blk.cross_attn.projv(pt_x2)

    mlx_q = mlx_blk.cross_attn.q(mlx_norm_y)
    mlx_kv = mlx_blk.cross_attn.kv(mlx_x2)
    mx.eval(mlx_q, mlx_kv)

    compare("cross_attn Q", pt_q.cpu().numpy()[0], np.array(mlx_q[0]))
    compare("cross_attn K", pt_k.cpu().numpy()[0], np.array(mlx_kv[0, :, :768]))
    compare("cross_attn V", pt_v.cpu().numpy()[0], np.array(mlx_kv[0, :, 768:]))

    # Cross-attention MLX (no RoPE in our implementation)
    mlx_cross_out = mlx_blk.cross_attn(mlx_norm_y, mlx_x2)
    mx.eval(mlx_cross_out)

    # Get full block output from PyTorch (uses RoPE in cross-attn)
    with torch.no_grad():
        # Call full block
        pt_blk_out_tuple = pt_blk(pt_x1, pt_x2, pt_pos1, pt_pos2)
        pt_block_out = pt_blk_out_tuple[0] if isinstance(pt_blk_out_tuple, tuple) else pt_blk_out_tuple

    # MLX full block
    mlx_block_out = mlx_blk(mlx_x1, mlx_x2)
    mx.eval(mlx_block_out)

    compare("FULL BLOCK OUTPUT", pt_block_out.cpu().numpy()[0], np.array(mlx_block_out[0]))

    # The key finding: Cross-attention in PyTorch uses RoPE 2D, but MLX doesn't!
    # Let's verify by comparing attention outputs without RoPE
    print("\n[KEY FINDING]")
    print("  PyTorch cross-attention uses RoPE 2D (position encoding on Q and K)")
    print("  MLX cross-attention does NOT use RoPE (no position encoding)")
    print("  This explains the divergence in block outputs!")

    print("\n✓ Debug complete!")


if __name__ == "__main__":
    main()
