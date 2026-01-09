#!/usr/bin/env python3
"""Debug script to compare DPT head outputs between PyTorch and MLX.

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

    # Handle shape mismatch
    min_len = min(len(pt_flat), len(mlx_flat))
    if len(pt_flat) != len(mlx_flat):
        print(f"  {name}: SHAPE MISMATCH - PT {pt.shape} vs MLX {mlx.shape}")

    corr = np.corrcoef(pt_flat[:min_len], mlx_flat[:min_len])[0, 1]

    # Stats
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
    print("DPT Debug Comparison: PyTorch vs MLX")
    print("=" * 70)

    img_shape = (512, 672, 3)
    img1 = create_test_image(img_shape, seed=42)
    img2 = create_test_image(img_shape, seed=43)

    # =========================================================================
    # PyTorch model
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

    # Hook to capture intermediate outputs
    pt_intermediates = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                pt_intermediates[name] = output[0].detach().cpu().numpy()
            else:
                pt_intermediates[name] = output.detach().cpu().numpy()

        return hook

    # Register hooks on DPT components
    head1 = pt_model.downstream_head1
    head1.dpt.act_postprocess[0].register_forward_hook(make_hook("act_post_0"))
    head1.dpt.act_postprocess[1].register_forward_hook(make_hook("act_post_1"))
    head1.dpt.act_postprocess[2].register_forward_hook(make_hook("act_post_2"))
    head1.dpt.act_postprocess[3].register_forward_hook(make_hook("act_post_3"))
    head1.dpt.scratch.layer1_rn.register_forward_hook(make_hook("layer1_rn"))
    head1.dpt.scratch.layer2_rn.register_forward_hook(make_hook("layer2_rn"))
    head1.dpt.scratch.layer3_rn.register_forward_hook(make_hook("layer3_rn"))
    head1.dpt.scratch.layer4_rn.register_forward_hook(make_hook("layer4_rn"))
    head1.dpt.scratch.refinenet4.register_forward_hook(make_hook("refinenet4"))
    head1.dpt.scratch.refinenet3.register_forward_hook(make_hook("refinenet3"))
    head1.dpt.scratch.refinenet2.register_forward_hook(make_hook("refinenet2"))
    head1.dpt.scratch.refinenet1.register_forward_hook(make_hook("refinenet1"))
    head1.dpt.head.register_forward_hook(make_hook("head"))

    print("\n[2] Running PyTorch forward pass...")
    with torch.no_grad():
        pt_out = pt_model(view1, view2)

    pred1, pred2 = pt_out
    pt_pts3d = pred1["pts3d"].cpu().numpy()[0]
    pt_conf = pred1["conf"].cpu().numpy()[0]

    print(f"  PyTorch pts3d shape: {pt_pts3d.shape}")
    print(f"  PyTorch conf shape: {pt_conf.shape}")
    print(f"  PyTorch pts3d range: [{pt_pts3d.min():.4f}, {pt_pts3d.max():.4f}]")
    print(f"  PyTorch conf range: [{pt_conf.min():.4f}, {pt_conf.max():.4f}]")

    # =========================================================================
    # MLX model
    # =========================================================================
    print("\n[3] Loading MLX MASt3R...")
    from mlx_mast3r.decoders.mast3r import Mast3rDecoderEngine

    mlx_engine = Mast3rDecoderEngine(resolution=512, precision="fp16", compile=False)
    safetensors_path = SAFETENSORS_DIR / "mast3r_vit_large" / "unified.safetensors"
    mlx_engine.load(safetensors_path)

    print("\n[4] Running MLX forward pass...")
    out1, out2, ms = mlx_engine.infer(img1, img2)

    mlx_pts3d = out1["pts3d"]
    mlx_conf = out1["conf"]

    print(f"  MLX pts3d shape: {mlx_pts3d.shape}")
    print(f"  MLX conf shape: {mlx_conf.shape}")
    print(f"  MLX pts3d range: [{mlx_pts3d.min():.4f}, {mlx_pts3d.max():.4f}]")
    print(f"  MLX conf range: [{mlx_conf.min():.4f}, {mlx_conf.max():.4f}]")

    # =========================================================================
    # Compare intermediate outputs
    # =========================================================================
    print("\n" + "=" * 70)
    print("INTERMEDIATE COMPARISONS")
    print("=" * 70)

    print("\n[PyTorch intermediate shapes]")
    for name, arr in pt_intermediates.items():
        print(f"  {name}: {arr.shape}")

    # =========================================================================
    # Compare final outputs
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL OUTPUT COMPARISON")
    print("=" * 70)

    compare_arrays("pts3d", pt_pts3d, mlx_pts3d)
    compare_arrays("conf", pt_conf, mlx_conf.squeeze(-1))

    # =========================================================================
    # Analyze raw DPT output (before post-processing)
    # =========================================================================
    print("\n" + "=" * 70)
    print("RAW DPT OUTPUT (before post-processing)")
    print("=" * 70)

    # Get raw PyTorch DPT output
    pt_head_raw = pt_intermediates.get("head")
    if pt_head_raw is not None:
        print(f"  PyTorch head raw shape: {pt_head_raw.shape}")
        # PyTorch format is NCHW, convert to NHWC
        pt_head_raw_nhwc = np.transpose(pt_head_raw[0], (1, 2, 0))
        print(f"  PyTorch head raw (NHWC): {pt_head_raw_nhwc.shape}")
        print(
            f"  PyTorch raw xyz range: [{pt_head_raw_nhwc[..., :3].min():.4f}, {pt_head_raw_nhwc[..., :3].max():.4f}]"
        )
        print(
            f"  PyTorch raw conf range: [{pt_head_raw_nhwc[..., 3].min():.4f}, {pt_head_raw_nhwc[..., 3].max():.4f}]"
        )

    # =========================================================================
    # Check weight loading
    # =========================================================================
    print("\n" + "=" * 70)
    print("WEIGHT COMPARISON (sample)")
    print("=" * 70)

    # Compare act_postprocess_0_conv weights
    pt_w = head1.dpt.act_postprocess[0][0].weight.detach().cpu().numpy()  # PyTorch: (O, I, H, W)
    mlx_w = np.array(mlx_engine.decoder.head1.act_postprocess_0_conv.weight)  # MLX: (O, H, W, I)

    # Convert MLX to PyTorch format for comparison
    mlx_w_pt_fmt = np.transpose(mlx_w, (0, 3, 1, 2))  # (O, H, W, I) -> (O, I, H, W)

    print(f"  act_postprocess_0_conv weight:")
    print(f"    PyTorch shape: {pt_w.shape}")
    print(f"    MLX shape: {mlx_w.shape}")
    weight_corr = np.corrcoef(pt_w.flatten(), mlx_w_pt_fmt.flatten())[0, 1]
    print(f"    Weight correlation: {weight_corr:.6f}")

    # Compare layer1_rn weights
    pt_w = head1.dpt.scratch.layer1_rn.weight.detach().cpu().numpy()
    mlx_w = np.array(mlx_engine.decoder.head1.layer1_rn.weight)
    mlx_w_pt_fmt = np.transpose(mlx_w, (0, 3, 1, 2))

    print(f"  layer1_rn weight:")
    print(f"    PyTorch shape: {pt_w.shape}")
    print(f"    MLX shape: {mlx_w.shape}")
    weight_corr = np.corrcoef(pt_w.flatten(), mlx_w_pt_fmt.flatten())[0, 1]
    print(f"    Weight correlation: {weight_corr:.6f}")

    # =========================================================================
    # Detailed hook comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("INTERMEDIATE STATS FROM PYTORCH")
    print("=" * 70)

    for name in [
        "act_post_0",
        "act_post_1",
        "act_post_2",
        "act_post_3",
        "layer1_rn",
        "layer2_rn",
        "layer3_rn",
        "layer4_rn",
        "refinenet4",
        "refinenet3",
        "refinenet2",
        "refinenet1",
    ]:
        if name in pt_intermediates:
            arr = pt_intermediates[name]
            print(
                f"  {name}: shape={arr.shape}, mean={arr.mean():.4f}, std={arr.std():.4f}, range=[{arr.min():.4f}, {arr.max():.4f}]"
            )

    # =========================================================================
    # Direct layer-by-layer comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("LAYER-BY-LAYER DPT COMPARISON")
    print("=" * 70)

    # We need to run MLX DPT step by step to compare
    # First, get the encoder features
    H, W = 32, 42  # patch grid for 512x672 with patch_size=16

    # Get encoder outputs from PyTorch hooks
    # We need the features that go into the DPT head

    # Get PyTorch's input to the DPT (the 4 hooked layers)
    # In PyTorch, hooks = [0, 6, 9, 12] for encoder + 3 decoder layers

    # Let's compare the ConvTranspose weights for act_postprocess_0
    print("\n[ConvTranspose2d weights comparison]")
    pt_ct_w = head1.dpt.act_postprocess[0][1].weight.detach().cpu().numpy()  # ConvTranspose: (I, O, H, W)
    mlx_ct_w = np.array(mlx_engine.decoder.head1.act_postprocess_0_up.weight)  # MLX: (O, H, W, I)

    print(f"  PyTorch ConvTranspose shape: {pt_ct_w.shape} (I,O,H,W)")
    print(f"  MLX ConvTranspose shape: {mlx_ct_w.shape} (O,H,W,I)")

    # For ConvTranspose2d in PyTorch: weight is (in_channels, out_channels, kH, kW)
    # For MLX ConvTranspose2d: weight should be (out_channels, kH, kW, in_channels)
    # So conversion is: (I, O, H, W) -> (O, H, W, I) which is (1, 2, 3, 0)
    mlx_ct_w_pt_fmt = np.transpose(mlx_ct_w, (3, 0, 1, 2))  # (O,H,W,I) -> (I,O,H,W)
    ct_corr = np.corrcoef(pt_ct_w.flatten(), mlx_ct_w_pt_fmt.flatten())[0, 1]
    print(f"  Weight correlation: {ct_corr:.6f}")

    # Compare head convs
    print("\n[Head convolutions weights]")
    for idx, name in [("0", "head_conv1"), ("2", "head_conv2"), ("4", "head_conv3")]:
        pt_head_w = head1.dpt.head[int(idx)].weight.detach().cpu().numpy()
        mlx_head_w = np.array(getattr(mlx_engine.decoder.head1, name).weight)
        mlx_head_w_pt = np.transpose(mlx_head_w, (0, 3, 1, 2))
        corr = np.corrcoef(pt_head_w.flatten(), mlx_head_w_pt.flatten())[0, 1]
        print(f"  head.{idx} ({name}): PT {pt_head_w.shape}, MLX {mlx_head_w.shape}, corr={corr:.6f}")

    # Compare refinenet weights
    print("\n[Refinenet weights comparison]")
    for i in range(1, 5):
        refine_pt = getattr(head1.dpt.scratch, f"refinenet{i}")
        refine_mlx = getattr(mlx_engine.decoder.head1, f"refinenet{i}")

        # out_conv
        pt_oc_w = refine_pt.out_conv.weight.detach().cpu().numpy()
        mlx_oc_w = np.array(refine_mlx.out_conv.weight)
        mlx_oc_w_pt = np.transpose(mlx_oc_w, (0, 3, 1, 2))
        corr = np.corrcoef(pt_oc_w.flatten(), mlx_oc_w_pt.flatten())[0, 1]
        print(f"  refinenet{i}.out_conv: corr={corr:.6f}")

        # resConfUnit1.conv1
        pt_c1 = refine_pt.resConfUnit1.conv1.weight.detach().cpu().numpy()
        mlx_c1 = np.array(refine_mlx.resConfUnit1.conv1.weight)
        mlx_c1_pt = np.transpose(mlx_c1, (0, 3, 1, 2))
        corr1 = np.corrcoef(pt_c1.flatten(), mlx_c1_pt.flatten())[0, 1]
        print(f"  refinenet{i}.resConfUnit1.conv1: corr={corr1:.6f}")

    print("\n✓ Debug complete!")


if __name__ == "__main__":
    main()
