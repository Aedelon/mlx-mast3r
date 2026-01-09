#!/usr/bin/env python3
"""Debug decoder blocks comparison between PyTorch and MLX.

Compares each decoder block output to identify where divergence occurs.
Uses hooks to capture intermediate outputs without modifying model code.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd() / "src"))
sys.path.insert(0, str(Path.home() / "Workspace/mast3r"))

import mlx.core as mx

SAFETENSORS_DIR = Path.home() / ".cache/mlx-mast3r"


def compare(name: str, pt: np.ndarray, mlx: np.ndarray, threshold: float = 0.98) -> float:
    """Compare two arrays and print correlation."""
    pt_flat = pt.flatten().astype(np.float64)
    mlx_flat = mlx.flatten().astype(np.float64)

    if pt_flat.shape != mlx_flat.shape:
        print(f"  {name}: SHAPE MISMATCH PT={pt.shape} vs MLX={mlx.shape}")
        return 0.0

    corr = np.corrcoef(pt_flat, mlx_flat)[0, 1]
    status = "✓" if corr > threshold else "✗" if corr < 0.9 else "~"
    print(
        f"  {status} {name}: corr={corr:.6f} | PT mean={pt.mean():.4f} | MLX mean={mlx.mean():.4f}"
    )
    return corr


def main():
    print("=" * 70)
    print("DECODER BLOCK-BY-BLOCK COMPARISON")
    print("=" * 70)

    # Test images
    np.random.seed(42)
    img_shape = (512, 672, 3)
    img1 = np.random.randint(0, 256, img_shape, dtype=np.uint8)
    img2 = np.random.randint(0, 256, img_shape, dtype=np.uint8)

    H, W = 32, 42  # Patch grid for 512x672 with patch_size=16

    print("\n[1] Loading PyTorch MASt3R...")
    from mast3r.model import AsymmetricMASt3R

    pt_model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    )
    pt_model = pt_model.to("mps").eval()

    print("\n[2] Loading MLX MASt3R...")
    from mlx_mast3r.decoders.mast3r import Mast3rDecoderEngine

    mlx_engine = Mast3rDecoderEngine(resolution=512, precision="fp32", compile=False)
    mlx_engine.load(SAFETENSORS_DIR / "mast3r_vit_large" / "unified.safetensors")

    print("\n[3] Preparing inputs...")
    # PyTorch input
    img1_pt = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img1_pt = (img1_pt - 0.5) / 0.5
    img1_pt = img1_pt.to("mps")
    img2_pt = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2_pt = (img2_pt - 0.5) / 0.5
    img2_pt = img2_pt.to("mps")

    view1 = {"img": img1_pt, "true_shape": torch.tensor([[512, 672]]), "idx": 0, "instance": "0"}
    view2 = {"img": img2_pt, "true_shape": torch.tensor([[512, 672]]), "idx": 1, "instance": "1"}

    # MLX input
    x1_mlx = img1.astype(np.float32) / 255.0
    x1_mlx = (x1_mlx - 0.5) / 0.5
    x2_mlx = img2.astype(np.float32) / 255.0
    x2_mlx = (x2_mlx - 0.5) / 0.5
    x1_mlx = mx.array(x1_mlx[None, :, :, :])
    x2_mlx = mx.array(x2_mlx[None, :, :, :])

    print("\n[4] Capturing PyTorch decoder block outputs...")

    # Hook to capture decoder block outputs
    pt_decoder_outputs = {}

    def capture_decoder_output(name):
        def hook(module, args, output):
            # Output is (x1, x2, pos1, pos2)
            if isinstance(output, tuple) and len(output) >= 2:
                pt_decoder_outputs[name] = (
                    output[0].detach().cpu().numpy(),
                    output[1].detach().cpu().numpy(),
                )

        return hook

    # Register hooks on all decoder blocks
    hooks = []
    for i in range(12):
        h1 = pt_model.dec_blocks[i].register_forward_hook(capture_decoder_output(f"dec_blocks.{i}"))
        h2 = pt_model.dec_blocks2[i].register_forward_hook(
            capture_decoder_output(f"dec_blocks2.{i}")
        )
        hooks.append(h1)
        hooks.append(h2)

    # Also capture decoder_embed output
    def capture_decoder_embed(module, args, output):
        pt_decoder_outputs["decoder_embed"] = output.detach().cpu().numpy()

    h_embed = pt_model.decoder_embed.register_forward_hook(capture_decoder_embed)
    hooks.append(h_embed)

    # Run PyTorch model
    with torch.no_grad():
        pt_out = pt_model(view1, view2)

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"  Captured {len(pt_decoder_outputs)} outputs")

    print("\n[5] Running MLX decoder block by block...")

    # Run encoder
    mlx_enc_out1 = mlx_engine.encoder(x1_mlx)
    mlx_enc_out2 = mlx_engine.encoder(x2_mlx)
    mx.eval(mlx_enc_out1, mlx_enc_out2)

    decoder = mlx_engine.decoder

    # Normalize encoder outputs
    mlx_enc_norm1 = mx.fast.layer_norm(
        mlx_enc_out1, decoder.enc_norm_weight, decoder.enc_norm_bias, eps=1e-6
    )
    mlx_enc_norm2 = mx.fast.layer_norm(
        mlx_enc_out2, decoder.enc_norm_weight, decoder.enc_norm_bias, eps=1e-6
    )

    # Project to decoder dim
    mlx_x1 = decoder.decoder_embed(mlx_enc_norm1)
    mlx_x2 = decoder.decoder_embed(mlx_enc_norm2)
    mx.eval(mlx_x1, mlx_x2)

    # Compare decoder_embed output
    if "decoder_embed" in pt_decoder_outputs:
        pt_embed = pt_decoder_outputs["decoder_embed"]
        # Note: PT applies decoder_embed to both views, we need to check order
        print("\n  Comparing decoder_embed:")
        # The PyTorch model processes [cat(enc1, enc2)] then splits
        # So we need to compare appropriately
        print(f"    PT decoder_embed output shape: {pt_embed.shape}")
        print(f"    MLX x1 shape: {mlx_x1.shape}, x2 shape: {mlx_x2.shape}")

    # Initialize RoPE
    if decoder._rope_cos is None:
        decoder._init_rope(H, W)

    print("\n" + "=" * 70)
    print("BLOCK-BY-BLOCK COMPARISON")
    print("=" * 70)

    # Run through decoder blocks
    for i in range(12):
        # MLX: Use OLD x1/x2 for BOTH blocks (PyTorch pattern)
        mlx_x1_old, mlx_x2_old = mlx_x1, mlx_x2
        mlx_x1 = decoder.dec_blocks[i](mlx_x1_old, mlx_x2_old)
        mlx_x2 = decoder.dec_blocks2[i](mlx_x2_old, mlx_x1_old)
        mx.eval(mlx_x1, mlx_x2)

        # Compare with PyTorch
        key1 = f"dec_blocks.{i}"
        key2 = f"dec_blocks2.{i}"

        if key1 in pt_decoder_outputs:
            pt_x1, pt_x2 = pt_decoder_outputs[key1][0], pt_decoder_outputs[key2][0]

            # Reshape for comparison (PT is [B, N, C], MLX is [B, N, C])
            mlx_np1 = np.array(mlx_x1[0])
            mlx_np2 = np.array(mlx_x2[0])

            corr1 = compare(f"Block {i} view1", pt_x1, mlx_np1)
            corr2 = compare(f"Block {i} view2", pt_x2, mlx_np2)

            if corr1 < 0.9 or corr2 < 0.9:
                print(f"\n    --> DIVERGENCE at block {i}!")

                # Get previous block outputs for detailed debugging
                if i > 0:
                    prev_key1 = f"dec_blocks.{i - 1}"
                    prev_key2 = f"dec_blocks2.{i - 1}"
                    pt_prev_x1 = pt_decoder_outputs[prev_key1][0]
                    pt_prev_x2 = pt_decoder_outputs[prev_key2][0]

                    # Use PT previous outputs as input to MLX block for testing
                    pt_x1_mx = mx.array(pt_prev_x1[None])
                    pt_x2_mx = mx.array(pt_prev_x2[None])

                    # Run MLX block with PT inputs
                    mlx_test_x1 = decoder.dec_blocks[i](pt_x1_mx, pt_x2_mx)
                    mx.eval(mlx_test_x1)

                    print(f"\n    Testing MLX block {i} with PyTorch inputs:")
                    compare(
                        f"    MLX block with PT input",
                        pt_decoder_outputs[key1][0],
                        np.array(mlx_test_x1[0]),
                    )

                break

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    # Get final outputs
    pred1, pred2 = pt_out
    pt_pts3d = pred1["pts3d"].cpu().numpy()[0]
    pt_conf = pred1["conf"].cpu().numpy()[0]

    out1, out2, ms = mlx_engine.infer(img1, img2)
    mlx_pts3d = out1["pts3d"]
    mlx_conf = out1["conf"]

    compare("pts3d", pt_pts3d, mlx_pts3d)
    compare("conf", pt_conf, mlx_conf.squeeze(-1))

    print(f"\n  MLX inference time: {ms:.1f}ms")
    print("\n✓ Debug complete!")


if __name__ == "__main__":
    main()
