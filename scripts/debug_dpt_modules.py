#!/usr/bin/env python3
"""Debug DPT module by module comparison between PyTorch and MLX.

Compares each intermediate output:
1. act_postprocess layers (0-3)
2. layer_rn projections (1-4)
3. refinenet fusion blocks (1-4)
4. head convolutions
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

    # Handle shape mismatch
    if pt_flat.shape != mlx_flat.shape:
        print(f"  {name}: SHAPE MISMATCH PT={pt.shape} vs MLX={mlx.shape}")
        return 0.0

    corr = np.corrcoef(pt_flat, mlx_flat)[0, 1]
    status = "✓" if corr > threshold else "✗" if corr < 0.9 else "~"
    print(
        f"  {status} {name}: corr={corr:.6f} | PT shape={pt.shape} mean={pt.mean():.4f} | MLX shape={mlx.shape} mean={mlx.mean():.4f}"
    )
    return corr


def main():
    print("=" * 70)
    print("DPT MODULE-BY-MODULE COMPARISON")
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

    print("\n[4] Running PyTorch encoder + decoder to get DPT input features...")

    # Hook to capture DPT inputs
    pt_dpt_inputs = {}

    def capture_dpt_input(name):
        def hook(module, args):
            if len(args) > 0:
                encoder_tokens = args[0]
                if isinstance(encoder_tokens, list):
                    pt_dpt_inputs[name] = [t.detach().cpu().numpy() for t in encoder_tokens]

        return hook

    # Register hook on DPT
    hook_handle = pt_model.downstream_head1.dpt.register_forward_pre_hook(capture_dpt_input("dpt1"))

    with torch.no_grad():
        pt_out = pt_model(view1, view2)

    hook_handle.remove()

    # Get PyTorch DPT features
    pt_features = pt_dpt_inputs["dpt1"]
    print(f"  PyTorch DPT received {len(pt_features)} features")
    for i, f in enumerate(pt_features):
        print(f"    features[{i}]: shape={f.shape}, mean={f.mean():.4f}")

    print("\n[5] Running MLX encoder + decoder to get DPT input features...")

    # Run encoder
    feat1_mlx = mlx_engine.encoder(x1_mlx)
    feat2_mlx = mlx_engine.encoder(x2_mlx)
    mx.eval(feat1_mlx, feat2_mlx)

    # Run decoder manually to collect features
    decoder = mlx_engine.decoder
    config = decoder.config

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

    # Initialize RoPE
    if decoder._rope_cos is None:
        decoder._init_rope(H, W)

    # Hooks: [0, 6, 9, 12]
    hooks = [0, 6, 9, 12]
    mlx_features = [np.array(feat1_mlx[0])]  # Hook 0: encoder features

    # Run decoder blocks (PyTorch uses OLD x1/x2 for BOTH blocks)
    for i, (blk1, blk2) in enumerate(zip(decoder.dec_blocks, decoder.dec_blocks2)):
        x1_old, x2_old = x1, x2
        x1 = blk1(x1_old, x2_old)
        x2 = blk2(x2_old, x1_old)  # Use OLD x1, not new!

        layer_idx = i + 1
        if layer_idx in hooks[1:]:
            mlx_features.append(np.array(x1[0]))

    # Final norm for last hook
    x1_norm = mx.fast.layer_norm(x1, decoder.dec_norm_weight, decoder.dec_norm_bias, eps=1e-6)
    mlx_features[-1] = np.array(x1_norm[0])

    print(f"  MLX DPT will receive {len(mlx_features)} features")
    for i, f in enumerate(mlx_features):
        print(f"    features[{i}]: shape={f.shape}, mean={f.mean():.4f}")

    print("\n" + "=" * 70)
    print("COMPARING DPT INPUT FEATURES")
    print("=" * 70)

    # PyTorch hooks might be different - let's check what indices match
    # MLX hooks: [0, 6, 9, 12] -> features[0,1,2,3]
    # We need to find corresponding PyTorch features

    print("\n  Checking PyTorch hooks configuration...")
    pt_dpt = pt_model.downstream_head1.dpt
    print(f"  PyTorch DPT hooks: {pt_dpt.hooks}")

    # Map MLX features to PyTorch features based on hooks
    # MLX features[i] corresponds to PT features[hooks[i]]
    pt_hooks = pt_dpt.hooks  # [0, 6, 9, 12]
    mlx_hooks = [0, 6, 9, 12]

    for i, hook_idx in enumerate(mlx_hooks):
        if hook_idx < len(pt_features) and i < len(mlx_features):
            pt_f = pt_features[hook_idx]  # Use hook index, not sequential!
            mlx_f = mlx_features[i]

            # Reshape if needed (PT is [B,N,C], MLX is [N,C])
            if len(pt_f.shape) == 3:
                pt_f = pt_f[0]

            compare(f"feature[{i}] (PT[{hook_idx}])", pt_f, mlx_f)

    print("\n" + "=" * 70)
    print("COMPARING DPT INTERNAL MODULES")
    print("=" * 70)

    # Get DPT modules
    pt_dpt = pt_model.downstream_head1.dpt
    mlx_dpt = mlx_engine.decoder.head1

    # Prepare features in spatial format for DPT
    # PyTorch: [B, N, C] -> [B, C, H, W]
    # MLX: [N, C] -> [B, H, W, C]

    B = 1
    pt_layers = []
    mlx_layers_spatial = []

    # Use features at hook indices [0, 6, 9, 12]
    hook_indices = [0, 6, 9, 12]
    for i, hook_idx in enumerate(hook_indices):
        if i >= len(mlx_features):
            break
        pt_f = torch.from_numpy(pt_features[hook_idx]).to("mps")
        mlx_f = mx.array(mlx_features[i][None, :, :])  # Add batch dim

        # Reshape to spatial
        C = pt_f.shape[-1]
        pt_spatial = pt_f.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        mlx_spatial = mlx_f.reshape(B, H, W, C)  # [B, H, W, C]

        pt_layers.append(pt_spatial)
        mlx_layers_spatial.append(mlx_spatial)

    print("\n--- act_postprocess layers ---")

    # Layer 0: Conv + ConvTranspose (4x upsample)
    with torch.no_grad():
        pt_l0 = pt_dpt.act_postprocess[0](pt_layers[0])
    mlx_l0 = mlx_dpt.act_postprocess_0_conv(mlx_layers_spatial[0])
    mlx_l0 = mlx_dpt.act_postprocess_0_up(mlx_l0)
    mx.eval(mlx_l0)

    compare("act_postprocess[0]", pt_l0.permute(0, 2, 3, 1).cpu().numpy(), np.array(mlx_l0))

    # Layer 1: Conv + ConvTranspose (2x upsample)
    with torch.no_grad():
        pt_l1 = pt_dpt.act_postprocess[1](pt_layers[1])
    mlx_l1 = mlx_dpt.act_postprocess_1_conv(mlx_layers_spatial[1])
    mlx_l1 = mlx_dpt.act_postprocess_1_up(mlx_l1)
    mx.eval(mlx_l1)

    compare("act_postprocess[1]", pt_l1.permute(0, 2, 3, 1).cpu().numpy(), np.array(mlx_l1))

    # Layer 2: Conv only
    with torch.no_grad():
        pt_l2 = pt_dpt.act_postprocess[2](pt_layers[2])
    mlx_l2 = mlx_dpt.act_postprocess_2_conv(mlx_layers_spatial[2])
    mx.eval(mlx_l2)

    compare("act_postprocess[2]", pt_l2.permute(0, 2, 3, 1).cpu().numpy(), np.array(mlx_l2))

    # Layer 3: Conv + Conv (2x downsample)
    with torch.no_grad():
        pt_l3 = pt_dpt.act_postprocess[3](pt_layers[3])
    mlx_l3 = mlx_dpt.act_postprocess_3_conv(mlx_layers_spatial[3])
    mlx_l3 = mlx_dpt.act_postprocess_3_down(mlx_l3)
    mx.eval(mlx_l3)

    compare("act_postprocess[3]", pt_l3.permute(0, 2, 3, 1).cpu().numpy(), np.array(mlx_l3))

    print("\n--- layer_rn projections ---")

    # layer1_rn through layer4_rn
    with torch.no_grad():
        pt_rn1 = pt_dpt.scratch.layer1_rn(pt_l0)
        pt_rn2 = pt_dpt.scratch.layer2_rn(pt_l1)
        pt_rn3 = pt_dpt.scratch.layer3_rn(pt_l2)
        pt_rn4 = pt_dpt.scratch.layer4_rn(pt_l3)

    mlx_rn1 = mlx_dpt.layer1_rn(mlx_l0)
    mlx_rn2 = mlx_dpt.layer2_rn(mlx_l1)
    mlx_rn3 = mlx_dpt.layer3_rn(mlx_l2)
    mlx_rn4 = mlx_dpt.layer4_rn(mlx_l3)
    mx.eval(mlx_rn1, mlx_rn2, mlx_rn3, mlx_rn4)

    compare("layer1_rn", pt_rn1.permute(0, 2, 3, 1).cpu().numpy(), np.array(mlx_rn1))
    compare("layer2_rn", pt_rn2.permute(0, 2, 3, 1).cpu().numpy(), np.array(mlx_rn2))
    compare("layer3_rn", pt_rn3.permute(0, 2, 3, 1).cpu().numpy(), np.array(mlx_rn3))
    compare("layer4_rn", pt_rn4.permute(0, 2, 3, 1).cpu().numpy(), np.array(mlx_rn4))

    print("\n--- refinenet fusion blocks ---")

    # refinenet4 (no skip)
    with torch.no_grad():
        pt_path4 = pt_dpt.scratch.refinenet4(pt_rn4)
    mlx_path4 = mlx_dpt.refinenet4(mlx_rn4)
    mx.eval(mlx_path4)

    # Crop to match
    pt_path4_crop = pt_path4[:, :, : mlx_rn3.shape[1], : mlx_rn3.shape[2]]
    mlx_path4_crop = mlx_path4[:, : mlx_rn3.shape[1], : mlx_rn3.shape[2], :]

    compare("refinenet4", pt_path4_crop.permute(0, 2, 3, 1).cpu().numpy(), np.array(mlx_path4_crop))

    # refinenet3 (with skip from layer3_rn)
    with torch.no_grad():
        pt_path3 = pt_dpt.scratch.refinenet3(pt_path4_crop, pt_rn3)
    mlx_path3 = mlx_dpt.refinenet3(mlx_path4_crop, mlx_rn3)
    mx.eval(mlx_path3)

    compare("refinenet3", pt_path3.permute(0, 2, 3, 1).cpu().numpy(), np.array(mlx_path3))

    # refinenet2 (with skip from layer2_rn)
    with torch.no_grad():
        pt_path2 = pt_dpt.scratch.refinenet2(pt_path3, pt_rn2)
    mlx_path2 = mlx_dpt.refinenet2(mlx_path3, mlx_rn2)
    mx.eval(mlx_path2)

    compare("refinenet2", pt_path2.permute(0, 2, 3, 1).cpu().numpy(), np.array(mlx_path2))

    # refinenet1 (with skip from layer1_rn)
    with torch.no_grad():
        pt_path1 = pt_dpt.scratch.refinenet1(pt_path2, pt_rn1)
    mlx_path1 = mlx_dpt.refinenet1(mlx_path2, mlx_rn1)
    mx.eval(mlx_path1)

    compare("refinenet1", pt_path1.permute(0, 2, 3, 1).cpu().numpy(), np.array(mlx_path1))

    print("\n--- head convolutions ---")

    # head.0 (conv1)
    with torch.no_grad():
        pt_head1 = pt_dpt.head[0](pt_path1)
    mlx_head1 = mlx_dpt.head_conv1(mlx_path1)
    mx.eval(mlx_head1)

    compare("head_conv1", pt_head1.permute(0, 2, 3, 1).cpu().numpy(), np.array(mlx_head1))

    # head.1 (interpolate 2x) - PyTorch uses F.interpolate
    import torch.nn.functional as F

    with torch.no_grad():
        pt_head1_up = F.interpolate(pt_head1, scale_factor=2, mode="bilinear", align_corners=True)

    from mlx_mast3r.decoders.mast3r import bilinear_upsample_2x

    mlx_head1_up = bilinear_upsample_2x(mlx_head1, align_corners=True)
    mx.eval(mlx_head1_up)

    compare("head_upsample", pt_head1_up.permute(0, 2, 3, 1).cpu().numpy(), np.array(mlx_head1_up))

    # head.2 (conv2)
    with torch.no_grad():
        pt_head2 = pt_dpt.head[2](pt_head1_up)
    mlx_head2 = mlx_dpt.head_conv2(mlx_head1_up)
    mx.eval(mlx_head2)

    compare("head_conv2", pt_head2.permute(0, 2, 3, 1).cpu().numpy(), np.array(mlx_head2))

    # head.3 (ReLU)
    with torch.no_grad():
        pt_head2_relu = F.relu(pt_head2)
    import mlx.nn as nn

    mlx_head2_relu = nn.relu(mlx_head2)
    mx.eval(mlx_head2_relu)

    compare("head_relu", pt_head2_relu.permute(0, 2, 3, 1).cpu().numpy(), np.array(mlx_head2_relu))

    # head.4 (conv3 - final)
    with torch.no_grad():
        pt_head3 = pt_dpt.head[4](pt_head2_relu)
    mlx_head3 = mlx_dpt.head_conv3(mlx_head2_relu)
    mx.eval(mlx_head3)

    compare("head_conv3 (final)", pt_head3.permute(0, 2, 3, 1).cpu().numpy(), np.array(mlx_head3))

    print("\n" + "=" * 70)
    print("FINAL OUTPUT COMPARISON")
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
