# Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""Stereo Reconstruction tab for Gradio demo."""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr
import numpy as np

from mlx_mast3r.utils import load_image

from gradio_utils.export import convert_to_glb
from gradio_utils.models import (
    get_default_conf_threshold,
    get_model,
    get_model_params,
    get_resolution,
)
from gradio_utils.viz import conf_to_colormap, depth_to_colormap


def reconstruct_stereo(
    img1_path: str | None,
    img2_path: str | None,
    model_name: str,
    min_conf_thr: float,
    as_pointcloud: bool,
    show_cameras: bool,
    cam_size: float,
) -> tuple[str | None, list, str, str | None]:
    """Run 3D reconstruction on stereo pair.

    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        model_name: Model to use
        min_conf_thr: Minimum confidence threshold
        as_pointcloud: Export as point cloud
        show_cameras: Show camera frustums
        cam_size: Camera frustum size

    Returns:
        Tuple of (glb_file, gallery, status, download_file)
    """
    if img1_path is None or img2_path is None:
        return None, [], "Please upload two images.", None

    model = get_model(model_name)
    resolution = get_resolution(model_name)
    params = get_model_params(model_name)

    img1 = load_image(img1_path, resolution=resolution, **params)
    img2 = load_image(img2_path, resolution=resolution, **params)

    t0 = time.perf_counter()
    out1, out2 = model.reconstruct(img1, img2)
    inference_time = (time.perf_counter() - t0) * 1000

    # Convert MLX arrays to numpy for visualization
    pts3d_1 = np.array(out1["pts3d"])
    pts3d_2 = np.array(out2["pts3d"])
    conf_1 = np.array(out1["conf"])
    conf_2 = np.array(out2["conf"])

    img1_uint8 = (img1 * 255).astype(np.uint8) if img1.max() <= 1 else img1.astype(np.uint8)
    img2_uint8 = (img2 * 255).astype(np.uint8) if img2.max() <= 1 else img2.astype(np.uint8)

    t1 = time.perf_counter()
    glb_file = convert_to_glb(
        imgs=[img1_uint8, img2_uint8],
        pts3d=[pts3d_1, pts3d_2],
        confs=[conf_1, conf_2],
        min_conf_thr=min_conf_thr,
        as_pointcloud=as_pointcloud,
    )
    export_time = (time.perf_counter() - t1) * 1000

    total_time = inference_time + export_time

    depth1 = depth_to_colormap(pts3d_1[:, :, 2])
    depth2 = depth_to_colormap(pts3d_2[:, :, 2])
    conf1_viz = conf_to_colormap(conf_1)
    conf2_viz = conf_to_colormap(conf_2)

    gallery = [
        (img1_uint8, "Image 1"),
        (depth1, "Depth 1"),
        (conf1_viz, "Confidence 1"),
        (img2_uint8, "Image 2"),
        (depth2, "Depth 2"),
        (conf2_viz, "Confidence 2"),
    ]

    n_valid_1 = ((conf_1.squeeze() > min_conf_thr) & np.isfinite(pts3d_1.sum(axis=-1))).sum()
    n_valid_2 = ((conf_2.squeeze() > min_conf_thr) & np.isfinite(pts3d_2.sum(axis=-1))).sum()

    status = f"""### Reconstruction Complete

**Model:** {model_name}
**Resolution:** {img1.shape[1]}x{img1.shape[0]}

**Timing:**
- Inference: {inference_time:.1f}ms
- GLB Export: {export_time:.1f}ms
- **Total: {total_time:.1f}ms** ({1000 / total_time:.1f} FPS)

**Depth range:**
- View 1: {pts3d_1[:, :, 2].min():.2f} - {pts3d_1[:, :, 2].max():.2f}
- View 2: {pts3d_2[:, :, 2].min():.2f} - {pts3d_2[:, :, 2].max():.2f}

**Valid 3D points:**
- View 1: {n_valid_1:,}
- View 2: {n_valid_2:,}
"""

    return glb_file, gallery, status, glb_file


def create_stereo_tab() -> None:
    """Create the Stereo Reconstruction tab content."""
    gr.Markdown(
        """
        ### Stereo 3D Reconstruction
        Upload two images of the same scene to get a 3D model.
        """
    )

    with gr.Row():
        recon_img1 = gr.Image(label="Image 1", type="filepath", height=250)
        recon_img2 = gr.Image(label="Image 2", type="filepath", height=250)

    with gr.Row():
        recon_model = gr.Dropdown(
            choices=["MASt3R Full", "DuneMASt3R Base", "DuneMASt3R Small"],
            value="MASt3R Full",
            label="Model",
            scale=2,
        )
        recon_run_btn = gr.Button("Reconstruct", variant="primary", scale=1)

    with gr.Accordion("Advanced Options", open=False):
        with gr.Row():
            recon_min_conf = gr.Slider(
                label="Confidence threshold",
                value=1.0,
                minimum=0.0,
                maximum=10.0,
                step=0.1,
            )
            recon_cam_size = gr.Slider(
                label="Camera size",
                value=0.05,
                minimum=0.01,
                maximum=0.2,
                step=0.01,
            )
        with gr.Row():
            recon_pointcloud = gr.Checkbox(value=True, label="Point cloud")
            recon_show_cams = gr.Checkbox(value=True, label="Show cameras")

    recon_status = gr.Markdown()

    with gr.Row():
        with gr.Column(scale=2):
            recon_model3d = gr.Model3D(label="3D Model", height=400)
        with gr.Column(scale=1):
            recon_download = gr.File(label="Download GLB")

    recon_gallery = gr.Gallery(label="RGB | Depth | Confidence", columns=3, rows=2)

    # Examples for stereo reconstruction
    examples_dir = Path(__file__).parent.parent.parent / "assets" / "NLE_tower"
    if examples_dir.exists():
        example_images = sorted(examples_dir.glob("*.jpg"))[:4]
        if len(example_images) >= 2:
            gr.Examples(
                examples=[
                    [str(example_images[0]), str(example_images[1])],
                    [str(example_images[2]), str(example_images[3])]
                    if len(example_images) >= 4
                    else [str(example_images[0]), str(example_images[1])],
                ],
                inputs=[recon_img1, recon_img2],
                label="Examples (NLE Tower)",
            )

    recon_run_btn.click(
        fn=reconstruct_stereo,
        inputs=[
            recon_img1,
            recon_img2,
            recon_model,
            recon_min_conf,
            recon_pointcloud,
            recon_show_cams,
            recon_cam_size,
        ],
        outputs=[recon_model3d, recon_gallery, recon_status, recon_download],
    )

    # Update confidence threshold when model changes
    recon_model.change(
        fn=get_default_conf_threshold,
        inputs=[recon_model],
        outputs=[recon_min_conf],
    )
