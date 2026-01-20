# Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""Multi-View Reconstruction tab for Gradio demo."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import mlx.core as mx
import numpy as np
import trimesh

from mlx_mast3r.cloud_opt import SparseGAResult, sparse_global_alignment
from mlx_mast3r.cloud_opt.tsdf import TSDFPostProcess
from mlx_mast3r.image_pairs import make_pairs
from mlx_mast3r.utils import load_image

from gradio_utils.export import export_multiview_glb, get_temp_dir
from gradio_utils.models import (
    get_default_conf_threshold,
    get_model,
    get_model_params,
    get_resolution,
    get_retrieval_model,
)


@dataclass
class SceneState:
    """State container for multi-view reconstruction."""

    sparse_ga: SparseGAResult | None = None
    cache_dir: str | None = None
    outfile: str | None = None


def get_scene_graph_type(
    sg_type: str, winsize: int, refid: int, cyclic: bool, na: int = 20, k: int = 10
) -> str:
    """Build scene graph string from UI parameters."""
    if sg_type == "complete":
        return "complete"
    elif sg_type == "swin":
        suffix = "" if cyclic else "-noncyclic"
        return f"swin-{winsize}{suffix}"
    elif sg_type == "logwin":
        suffix = "" if cyclic else "-noncyclic"
        return f"logwin-{winsize}{suffix}"
    elif sg_type == "oneref":
        return f"oneref-{refid}"
    elif sg_type == "retrieval":
        return f"retrieval-{na}-{k}"
    return "complete"


def run_multiview_reconstruction(
    files: list[str] | None,
    model_name: str,
    scenegraph_type: str,
    winsize: int,
    refid: int,
    cyclic: bool,
    retrieval_na: int,
    retrieval_k: int,
    lr1: float,
    niter1: int,
    lr2: float,
    niter2: int,
    min_conf_thr: float,
    matching_conf_thr: float,
    shared_intrinsics: bool,
    as_pointcloud: bool,
    mask_sky: bool,
    clean_depth: bool,
    transparent_cams: bool,
    cam_size: float,
    tsdf_thresh: float,
) -> tuple[str | None, str, Any]:
    """Run multi-view reconstruction with sparse global alignment."""
    if files is None or len(files) < 2:
        return None, "Please upload at least 2 images.", None

    temp_dir = get_temp_dir()

    # Build cache path
    cache_dir = f"{temp_dir}/cache_{time.time():.0f}"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    model = get_model(model_name)
    resolution = get_resolution(model_name)
    params = get_model_params(model_name)

    # Load all images
    print(f"Loading {len(files)} images...")
    imgs_data = []
    imgs_np = []  # For retrieval
    for idx, file_input in enumerate(files):
        # Handle both Gradio 6.x FileData objects and plain paths
        filepath = file_input.name if hasattr(file_input, "name") else str(file_input)
        img = load_image(filepath, resolution=resolution, **params)
        imgs_np.append(img)
        imgs_data.append(
            {
                "img": mx.array(img).transpose(2, 0, 1)[None],  # [1, C, H, W]
                "true_shape": np.array(img.shape[:2]),
                "idx": idx,
                "instance": filepath,
            }
        )

    # Build scene graph / pairs
    if scenegraph_type == "retrieval":
        # Retrieval requires MASt3R Full (embed_dim=1024)
        if model_name != "MASt3R Full":
            return (
                None,
                f"**Error**: Retrieval mode requires 'MASt3R Full' (embed_dim=1024). "
                f"Model '{model_name}' is not compatible. "
                f"Please select 'MASt3R Full' or use another scene graph type.",
                None,
            )

        from mlx_mast3r import make_pairs_retrieval

        # Use retrieval-based pair selection
        print(f"Loading retrieval model for pair selection (Na={retrieval_na}, k={retrieval_k})...")
        retrieval_model = get_retrieval_model()

        try:
            pairs_indices = make_pairs_retrieval(
                retrieval=retrieval_model,
                backbone=model,
                images=imgs_np,
                na=retrieval_na,
                k=retrieval_k,
            )
        except ValueError as e:
            return None, f"**Retrieval error**: {e}", None

        # Convert to format expected by sparse_global_alignment
        pairs = []
        for i, j in pairs_indices:
            pairs.append((imgs_data[i], imgs_data[j]))
            pairs.append((imgs_data[j], imgs_data[i]))  # Symmetrize

        scene_graph = f"retrieval-{retrieval_na}-{retrieval_k}"
        print(f"Retrieval selected {len(pairs_indices)} unique pairs ({len(pairs)} with symmetry)")
    else:
        scene_graph = get_scene_graph_type(scenegraph_type, winsize, refid, cyclic)
        print(f"Scene graph: {scene_graph}")
        pairs = make_pairs(imgs_data, scene_graph=scene_graph, symmetrize=True)
        print(f"Generated {len(pairs)} pairs")

    # Convert files to paths for sparse_global_alignment
    file_paths = [f.name if hasattr(f, "name") else str(f) for f in files]

    # Run sparse global alignment
    t0 = time.perf_counter()
    try:
        result = sparse_global_alignment(
            imgs=file_paths,
            pairs_in=pairs,
            cache_path=cache_dir,
            model=model,
            subsample=8,
            lr1=lr1,
            niter1=niter1,
            lr2=lr2,
            niter2=niter2,
            matching_conf_thr=matching_conf_thr,
            shared_intrinsics=shared_intrinsics,
            verbose=True,
        )
    except Exception as e:
        return None, f"Error: {e!s}", None

    optimization_time = time.perf_counter() - t0

    # Apply TSDF cleaning if requested
    if tsdf_thresh > 0 and clean_depth:
        processor = TSDFPostProcess(result, tsdf_thresh=tsdf_thresh)
        pts3d_list, depth_list, conf_list = processor.get_dense_pts3d(clean_depth=True)
    else:
        pts3d_list, depth_list, conf_list = result.get_dense_pts3d()

    # Export to GLB
    t1 = time.perf_counter()
    try:
        glb_file = export_multiview_glb(
            result=result,
            pts3d_list=pts3d_list,
            conf_list=conf_list,
            min_conf_thr=min_conf_thr,
            as_pointcloud=as_pointcloud,
            mask_sky=mask_sky,
            cam_size=cam_size,
            transparent_cams=transparent_cams,
        )
    except Exception as e:
        glb_file = f"{temp_dir}/fallback_{time.time():.0f}.glb"
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.PointCloud([[0, 0, 0]]))
        scene.export(glb_file)
        print(f"GLB export error: {e}")

    export_time = time.perf_counter() - t1

    # Build status
    status = f"""### Multi-View Reconstruction Complete

**Configuration:**
- Model: {model_name}
- Images: {len(files)}
- Pairs: {len(pairs)}
- Scene graph: {scene_graph}

**Optimization:**
- Phase 1 (coarse): lr={lr1}, {niter1} iterations
- Phase 2 (fine): lr={lr2}, {niter2} iterations
- Total time: {optimization_time:.1f}s

**Export:**
- Time: {export_time:.1f}s
- Format: {"Pointcloud" if as_pointcloud else "Mesh"}

**Results:**
- Estimated cameras: {result.n_imgs}
- Focals: {[f"{float(f):.1f}" for f in result.focals]}
"""

    # Store state for parameter updates
    scene_state = SceneState(
        sparse_ga=result,
        cache_dir=cache_dir,
        outfile=glb_file,
    )

    return glb_file, status, scene_state


def update_multiview_visualization(
    min_conf_thr: float,
    as_pointcloud: bool,
    mask_sky: bool,
    clean_depth: bool,
    transparent_cams: bool,
    cam_size: float,
    tsdf_thresh: float,
    scene_state: SceneState | None,
) -> tuple[str | None, str]:
    """Update visualization without re-running optimization."""
    if scene_state is None or scene_state.sparse_ga is None:
        return None, "No scene to update."

    result = scene_state.sparse_ga

    # Re-apply TSDF if needed
    if tsdf_thresh > 0 and clean_depth:
        processor = TSDFPostProcess(result, tsdf_thresh=tsdf_thresh)
        pts3d_list, depth_list, conf_list = processor.get_dense_pts3d(clean_depth=True)
    else:
        pts3d_list, depth_list, conf_list = result.get_dense_pts3d()

    # Re-export
    glb_file = export_multiview_glb(
        result=result,
        pts3d_list=pts3d_list,
        conf_list=conf_list,
        min_conf_thr=min_conf_thr,
        as_pointcloud=as_pointcloud,
        mask_sky=mask_sky,
        cam_size=cam_size,
        transparent_cams=transparent_cams,
    )

    return glb_file, "Visualization updated."


def create_multiview_tab() -> None:
    """Create the Multi-View Reconstruction tab content."""
    gr.Markdown(
        """
        ### Multi-View Reconstruction with Global Alignment
        Upload multiple images (3+) for complete reconstruction with camera pose optimization.
        """
    )

    # State for scene persistence
    mv_scene_state = gr.State(None)

    # Input section
    with gr.Row():
        mv_files = gr.File(
            label="Images (drag-drop or select)",
            file_count="multiple",
            file_types=["image"],
        )

    with gr.Row():
        mv_model = gr.Dropdown(
            choices=["MASt3R Full", "DuneMASt3R Base", "DuneMASt3R Small"],
            value="MASt3R Full",
            label="Model",
            scale=2,
        )
        mv_run_btn = gr.Button("Reconstruct", variant="primary", scale=1)

    # Scene Graph options
    with gr.Accordion("Scene Graph", open=True):
        with gr.Row():
            mv_scenegraph = gr.Dropdown(
                choices=["complete", "swin", "logwin", "oneref", "retrieval"],
                value="complete",
                label="Type",
                info="complete=all pairs, retrieval=auto selection by similarity",
            )
            mv_winsize = gr.Slider(
                label="Window size",
                value=3,
                minimum=1,
                maximum=10,
                step=1,
                visible=False,
            )
            mv_refid = gr.Slider(
                label="Reference ID",
                value=0,
                minimum=0,
                maximum=20,
                step=1,
                visible=False,
            )
            mv_cyclic = gr.Checkbox(value=True, label="Cyclic", visible=False)
        with gr.Row():
            mv_retrieval_na = gr.Slider(
                label="Na (adjacents)",
                value=20,
                minimum=5,
                maximum=50,
                step=5,
                visible=False,
                info="Number of adjacent images to consider",
            )
            mv_retrieval_k = gr.Slider(
                label="k (pairs per image)",
                value=10,
                minimum=2,
                maximum=30,
                step=1,
                visible=False,
                info="Number of pairs per image",
            )

    # Optimization parameters
    with gr.Accordion("Optimization Parameters", open=False):
        with gr.Row():
            mv_lr1 = gr.Slider(
                label="LR Phase 1 (coarse)",
                value=0.07,
                minimum=0.001,
                maximum=0.2,
                step=0.01,
            )
            mv_niter1 = gr.Slider(
                label="Iterations Phase 1",
                value=300,
                minimum=0,
                maximum=1000,
                step=50,
            )
        with gr.Row():
            mv_lr2 = gr.Slider(
                label="LR Phase 2 (fine)",
                value=0.01,
                minimum=0.001,
                maximum=0.1,
                step=0.005,
            )
            mv_niter2 = gr.Slider(
                label="Iterations Phase 2",
                value=300,
                minimum=0,
                maximum=1000,
                step=50,
            )
        with gr.Row():
            mv_matching_conf = gr.Slider(
                label="Matching threshold",
                value=5.0,
                minimum=0.0,
                maximum=20.0,
                step=0.5,
            )
            mv_shared_intrinsics = gr.Checkbox(value=False, label="Shared intrinsics")

    # Post-processing / visualization
    with gr.Accordion("Visualization", open=True):
        with gr.Row():
            mv_min_conf = gr.Slider(
                label="Confidence threshold",
                value=1.0,
                minimum=0.0,
                maximum=20.0,
                step=0.5,
            )
            mv_cam_size = gr.Slider(
                label="Camera size",
                value=0.05,
                minimum=0.0,
                maximum=0.3,
                step=0.01,
            )
        with gr.Row():
            mv_tsdf = gr.Slider(
                label="TSDF threshold",
                value=0.0,
                minimum=0.0,
                maximum=0.1,
                step=0.01,
                info="0=disabled, >0=depth cleaning",
            )
        with gr.Row():
            mv_pointcloud = gr.Checkbox(value=True, label="Point cloud")
            mv_mask_sky = gr.Checkbox(value=False, label="Mask sky")
            mv_clean_depth = gr.Checkbox(value=True, label="Clean depth")
            mv_transparent_cams = gr.Checkbox(value=False, label="Transparent cameras")

        mv_update_viz_btn = gr.Button("Update visualization")

    # Output section
    mv_status = gr.Markdown()

    with gr.Row():
        mv_model3d = gr.Model3D(label="3D Model", height=500)

    # Examples for multi-view reconstruction
    examples_dir = Path(__file__).parent.parent.parent / "assets" / "NLE_tower"
    if examples_dir.exists():
        example_images = sorted(examples_dir.glob("*.jpg"))
        if len(example_images) >= 3:
            # Create example with all images as a list
            gr.Markdown("### Examples")
            gr.Markdown(
                f"**NLE Tower**: {len(example_images)} images available in `assets/NLE_tower/`\n\n"
                "Click 'Select' and choose images from the `assets/NLE_tower/` folder"
            )

    # Dynamic visibility for scene graph options
    def update_sg_visibility(sg_type):
        show_win = sg_type in ["swin", "logwin"]
        show_ref = sg_type == "oneref"
        show_retrieval = sg_type == "retrieval"
        return (
            gr.update(visible=show_win),
            gr.update(visible=show_ref),
            gr.update(visible=show_win),
            gr.update(visible=show_retrieval),
            gr.update(visible=show_retrieval),
        )

    mv_scenegraph.change(
        fn=update_sg_visibility,
        inputs=[mv_scenegraph],
        outputs=[mv_winsize, mv_refid, mv_cyclic, mv_retrieval_na, mv_retrieval_k],
    )

    # Update confidence threshold when model changes
    mv_model.change(
        fn=get_default_conf_threshold,
        inputs=[mv_model],
        outputs=[mv_min_conf],
    )

    # Main reconstruction button
    mv_run_btn.click(
        fn=run_multiview_reconstruction,
        inputs=[
            mv_files,
            mv_model,
            mv_scenegraph,
            mv_winsize,
            mv_refid,
            mv_cyclic,
            mv_retrieval_na,
            mv_retrieval_k,
            mv_lr1,
            mv_niter1,
            mv_lr2,
            mv_niter2,
            mv_min_conf,
            mv_matching_conf,
            mv_shared_intrinsics,
            mv_pointcloud,
            mv_mask_sky,
            mv_clean_depth,
            mv_transparent_cams,
            mv_cam_size,
            mv_tsdf,
        ],
        outputs=[mv_model3d, mv_status, mv_scene_state],
    )

    # Update visualization button
    mv_update_viz_btn.click(
        fn=update_multiview_visualization,
        inputs=[
            mv_min_conf,
            mv_pointcloud,
            mv_mask_sky,
            mv_clean_depth,
            mv_transparent_cams,
            mv_cam_size,
            mv_tsdf,
            mv_scene_state,
        ],
        outputs=[mv_model3d, mv_status],
    )
