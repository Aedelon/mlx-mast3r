#!/usr/bin/env python3
"""Debug first decoder block in detail."""

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
    print("FIRST BLOCK DETAILED DEBUG")
    print("=" * 70)

    # Test images
    np.random.seed(42)
    img_shape = (512, 672, 3)
    img1 = np.random.randint(0, 256, img_shape, dtype=np.uint8)
    img2 = np.random.randint(0, 256, img_shape, dtype=np.uint8)

    H, W = 32, 42

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
    img1_pt = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img1_pt = (img1_pt - 0.5) / 0.5
    img1_pt = img1_pt.to("mps")
    img2_pt = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2_pt = (img2_pt - 0.5) / 0.5
    img2_pt = img2_pt.to("mps")

    x1_mlx = mx.array((img1.astype(np.float32) / 255.0 - 0.5) / 0.5)[None]
    x2_mlx = mx.array((img2.astype(np.float32) / 255.0 - 0.5) / 0.5)[None]

    print("\n[4] Capturing PyTorch intermediate outputs...")

    # Capture all intermediate outputs
    captured = {}

    def make_hook(name):
        def hook(module, args, output):
            if isinstance(output, tuple):
                captured[name] = tuple(
                    o.detach().cpu().numpy() if hasattr(o, "cpu") else o for o in output
                )
            else:
                captured[name] = output.detach().cpu().numpy()

        return hook

    # Register hooks
    hooks = []
    hooks.append(pt_model.enc_blocks[-1].register_forward_hook(make_hook("encoder_last")))
    hooks.append(pt_model.enc_norm.register_forward_hook(make_hook("enc_norm")))
    hooks.append(pt_model.decoder_embed.register_forward_hook(make_hook("decoder_embed")))
    hooks.append(pt_model.dec_blocks[0].norm1.register_forward_hook(make_hook("blk0_norm1")))
    hooks.append(pt_model.dec_blocks[0].attn.register_forward_hook(make_hook("blk0_self_attn")))
    hooks.append(pt_model.dec_blocks[0].norm2.register_forward_hook(make_hook("blk0_norm2")))
    hooks.append(pt_model.dec_blocks[0].norm_y.register_forward_hook(make_hook("blk0_norm_y")))
    hooks.append(
        pt_model.dec_blocks[0].cross_attn.register_forward_hook(make_hook("blk0_cross_attn"))
    )
    hooks.append(pt_model.dec_blocks[0].norm3.register_forward_hook(make_hook("blk0_norm3")))
    hooks.append(pt_model.dec_blocks[0].mlp.register_forward_hook(make_hook("blk0_mlp")))
    hooks.append(pt_model.dec_blocks[0].register_forward_hook(make_hook("blk0_output")))

    view1 = {"img": img1_pt, "true_shape": torch.tensor([[512, 672]]), "idx": 0, "instance": "0"}
    view2 = {"img": img2_pt, "true_shape": torch.tensor([[512, 672]]), "idx": 1, "instance": "1"}

    with torch.no_grad():
        pt_out = pt_model(view1, view2)

    for h in hooks:
        h.remove()

    print(f"  Captured {len(captured)} outputs")
    for k, v in captured.items():
        if isinstance(v, tuple):
            print(f"    {k}: tuple of {len(v)} items")
        else:
            print(f"    {k}: shape={v.shape}, mean={v.mean():.4f}")

    print("\n[5] Running MLX encoder...")
    mlx_enc1 = mlx_engine.encoder(x1_mlx)
    mlx_enc2 = mlx_engine.encoder(x2_mlx)
    mx.eval(mlx_enc1, mlx_enc2)

    # Compare encoder output
    if "encoder_last" in captured:
        pt_enc1, pt_enc2 = captured["encoder_last"][0], captured["encoder_last"][1]
        print("\n  Comparing encoder outputs:")
        compare("encoder view1", pt_enc1, np.array(mlx_enc1[0]))
        compare("encoder view2", pt_enc2, np.array(mlx_enc2[0]))

    print("\n[6] Running MLX decoder first block step by step...")
    decoder = mlx_engine.decoder

    # enc_norm
    mlx_enc_norm1 = mx.fast.layer_norm(
        mlx_enc1, decoder.enc_norm_weight, decoder.enc_norm_bias, eps=1e-6
    )
    mlx_enc_norm2 = mx.fast.layer_norm(
        mlx_enc2, decoder.enc_norm_weight, decoder.enc_norm_bias, eps=1e-6
    )
    mx.eval(mlx_enc_norm1, mlx_enc_norm2)

    if "enc_norm" in captured:
        # PT enc_norm is applied to concatenated inputs, need to split
        pt_enc_norm = captured["enc_norm"]
        print("\n  Comparing enc_norm:")
        print(f"    PT enc_norm shape: {pt_enc_norm.shape}")
        # The PT model processes view1 first in the hook
        compare("enc_norm view1", pt_enc_norm[0], np.array(mlx_enc_norm1[0]))

    # decoder_embed
    mlx_x1 = decoder.decoder_embed(mlx_enc_norm1)
    mlx_x2 = decoder.decoder_embed(mlx_enc_norm2)
    mx.eval(mlx_x1, mlx_x2)

    if "decoder_embed" in captured:
        pt_embed = captured["decoder_embed"]
        print("\n  Comparing decoder_embed:")
        print(f"    PT decoder_embed shape: {pt_embed.shape}")
        # PT processes view1 first
        compare("decoder_embed view1", pt_embed[0], np.array(mlx_x1[0]))

    # Initialize RoPE
    if decoder._rope_cos is None:
        decoder._init_rope(H, W)

    # Get first block
    mlx_blk = decoder.dec_blocks[0]

    print("\n[7] Step-by-step first block comparison:")

    # Step 1: norm1
    mlx_norm1 = mx.fast.layer_norm(mlx_x1, mlx_blk.norm1_weight, mlx_blk.norm1_bias, eps=1e-6)
    mx.eval(mlx_norm1)

    if "blk0_norm1" in captured:
        print("\n  Step 1: norm1")
        compare("norm1 output", captured["blk0_norm1"][0], np.array(mlx_norm1[0]))

    # Step 2: self_attn
    mlx_attn = mlx_blk.self_attn(mlx_norm1)
    mx.eval(mlx_attn)

    if "blk0_self_attn" in captured:
        print("\n  Step 2: self_attn (with RoPE)")
        compare("self_attn output", captured["blk0_self_attn"][0], np.array(mlx_attn[0]))

    # Step 3: residual after self_attn
    mlx_after_self = mlx_x1 + mlx_attn
    mx.eval(mlx_after_self)

    # Step 4: norm2 (query)
    mlx_norm2 = mx.fast.layer_norm(
        mlx_after_self, mlx_blk.norm2_weight, mlx_blk.norm2_bias, eps=1e-6
    )
    mx.eval(mlx_norm2)

    if "blk0_norm2" in captured:
        print("\n  Step 4: norm2 (query)")
        compare("norm2 output", captured["blk0_norm2"][0], np.array(mlx_norm2[0]))

    # Step 5: norm_y (context)
    mlx_norm_y = mx.fast.layer_norm(mlx_x2, mlx_blk.norm_y_weight, mlx_blk.norm_y_bias, eps=1e-6)
    mx.eval(mlx_norm_y)

    if "blk0_norm_y" in captured:
        print("\n  Step 5: norm_y (context)")
        compare("norm_y output", captured["blk0_norm_y"][0], np.array(mlx_norm_y[0]))

    # Step 6: cross_attn
    mlx_cross = mlx_blk.cross_attn(mlx_norm2, mlx_norm_y)
    mx.eval(mlx_cross)

    if "blk0_cross_attn" in captured:
        print("\n  Step 6: cross_attn")
        compare("cross_attn output", captured["blk0_cross_attn"][0], np.array(mlx_cross[0]))

    # Step 7: residual after cross_attn
    mlx_after_cross = mlx_after_self + mlx_cross
    mx.eval(mlx_after_cross)

    # Step 8: norm3
    mlx_norm3 = mx.fast.layer_norm(
        mlx_after_cross, mlx_blk.norm3_weight, mlx_blk.norm3_bias, eps=1e-6
    )
    mx.eval(mlx_norm3)

    if "blk0_norm3" in captured:
        print("\n  Step 8: norm3")
        compare("norm3 output", captured["blk0_norm3"][0], np.array(mlx_norm3[0]))

    # Step 9: mlp
    mlx_mlp = mlx_blk.mlp(mlx_norm3)
    mx.eval(mlx_mlp)

    if "blk0_mlp" in captured:
        print("\n  Step 9: mlp")
        compare("mlp output", captured["blk0_mlp"][0], np.array(mlx_mlp[0]))

    # Step 10: final output
    mlx_final = mlx_after_cross + mlx_mlp
    mx.eval(mlx_final)

    if "blk0_output" in captured:
        print("\n  Step 10: block 0 final output")
        pt_blk0_out = captured["blk0_output"]
        if isinstance(pt_blk0_out, tuple):
            compare("block0 output", pt_blk0_out[0][0], np.array(mlx_final[0]))
        else:
            compare("block0 output", pt_blk0_out[0], np.array(mlx_final[0]))

    print("\n✓ Debug complete!")


if __name__ == "__main__":
    main()
