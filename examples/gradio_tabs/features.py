# Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""DUNE Features tab for Gradio demo."""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from mlx_mast3r.utils import load_image

from gradio_utils.models import get_model


def visualize_features_pca(
    features: np.ndarray, img_shape: tuple[int, int], patch_size: int = 14
) -> np.ndarray:
    """Visualize features using PCA projection to RGB.

    Args:
        features: [N, D] feature vectors
        img_shape: (H, W) original image shape
        patch_size: Patch size used by encoder

    Returns:
        [H, W, 3] RGB visualization
    """
    n_patches = features.shape[0]

    # Calculate patch grid dimensions from image shape
    H, W = img_shape
    patch_h = H // patch_size
    patch_w = W // patch_size

    # Verify dimensions match
    if patch_h * patch_w != n_patches:
        # Fallback to square assumption
        patch_h = patch_w = int(np.sqrt(n_patches))

    features_centered = features - features.mean(axis=0)

    try:
        _, _, vh = np.linalg.svd(features_centered, full_matrices=False)
        features_3d = features_centered @ vh[:3].T
    except Exception:
        features_3d = features_centered[:, :3]

    for i in range(3):
        f = features_3d[:, i]
        f_min, f_max = f.min(), f.max()
        if f_max > f_min:
            features_3d[:, i] = (f - f_min) / (f_max - f_min) * 255
        else:
            features_3d[:, i] = 128

    features_img = features_3d.reshape(patch_h, patch_w, 3).astype(np.uint8)
    features_pil = Image.fromarray(features_img)
    features_pil = features_pil.resize(img_shape[::-1], Image.Resampling.NEAREST)

    return np.array(features_pil)


def extract_features(
    img_path: str | None,
    variant: str,
) -> tuple[np.ndarray | None, list, str]:
    """Extract features from image.

    Args:
        img_path: Path to input image
        variant: DUNE variant ("small" or "base")

    Returns:
        Tuple of (feature_viz, gallery, status_markdown)
    """
    if img_path is None:
        return None, [], "Please upload an image."

    model_name = f"DUNE {variant.capitalize()}"
    model = get_model(model_name)

    img = load_image(img_path, resolution=336)

    t0 = time.perf_counter()
    features = model.encode(img)
    inference_time = (time.perf_counter() - t0) * 1000

    feature_viz = visualize_features_pca(features, img.shape[:2])

    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)

    gallery = [
        (img_uint8, "Original Image"),
        (feature_viz, "PCA Features"),
    ]

    status = f"""### Statistics

**Model:** {model_name}
**Inference time:** {inference_time:.1f}ms ({1000 / inference_time:.1f} FPS)

**Features:**
- Shape: {features.shape}
- Embed dim: {model.embed_dim}
- Patches: {model.num_patches}
- Min: {features.min():.4f}
- Max: {features.max():.4f}
- Mean: {features.mean():.4f}
"""
    return feature_viz, gallery, status


def create_features_tab() -> None:
    """Create the DUNE Features tab content."""
    gr.Markdown(
        """
        ### Visual Feature Extraction
        Visualize DUNE features via PCA projection to RGB.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            feat_img_input = gr.Image(
                label="Input Image",
                type="filepath",
                height=300,
            )
            feat_variant = gr.Radio(
                choices=["small", "base"],
                value="base",
                label="DUNE Variant",
                info="small=11ms, base=32ms",
            )
            feat_run_btn = gr.Button("Extract Features", variant="primary")

        with gr.Column(scale=1):
            feat_viz = gr.Image(label="PCA Visualization", height=300)

    feat_stats = gr.Markdown(label="Statistics")
    feat_gallery = gr.Gallery(label="Comparison", columns=2, height="auto")

    # Examples for feature extraction
    examples_dir = Path(__file__).parent.parent.parent / "assets" / "NLE_tower"
    if examples_dir.exists():
        example_images = sorted(examples_dir.glob("*.jpg"))[:2]
        if example_images:
            gr.Examples(
                examples=[[str(img)] for img in example_images],
                inputs=[feat_img_input],
                label="Examples (NLE Tower)",
            )

    feat_run_btn.click(
        fn=extract_features,
        inputs=[feat_img_input, feat_variant],
        outputs=[feat_viz, feat_gallery, feat_stats],
    )
