#!/usr/bin/env python3
"""Gradio demo: Full MLX-MASt3R demonstration with multi-view reconstruction.

Includes:
- DUNE feature extraction
- Stereo reconstruction (2 views)
- Multi-view reconstruction with sparse global alignment (N views)

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.

Usage:
    uv sync --extra demo
    uv run python examples/gradio_demo.py
"""

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
from PIL import Image

from mlx_mast3r import DUNE, DuneMast3r, Mast3rFull, RetrievalModel, make_pairs_retrieval
from mlx_mast3r.cloud_opt import SparseGAResult, sparse_global_alignment
from mlx_mast3r.cloud_opt.tsdf import TSDFPostProcess
from mlx_mast3r.image_pairs import make_pairs
from mlx_mast3r.utils import load_image
from mlx_mast3r.viz import convert_scene_to_glb, segment_sky

# =============================================================================
# Global state
# =============================================================================
_models: dict[str, Any] = {}
_temp_dir = tempfile.mkdtemp(prefix="mlx_mast3r_demo_")


@dataclass
class SceneState:
    """State container for multi-view reconstruction."""

    sparse_ga: SparseGAResult | None = None
    cache_dir: str | None = None
    outfile: str | None = None


def get_model(model_name: str):
    """Get or load model (cached)."""
    if model_name not in _models:
        print(f"Loading {model_name}...")
        if model_name == "DUNE Small":
            _models[model_name] = DUNE.from_pretrained(variant="small", resolution=336)
        elif model_name == "DUNE Base":
            _models[model_name] = DUNE.from_pretrained(variant="base", resolution=336)
        elif model_name == "DuneMASt3R Small":
            _models[model_name] = DuneMast3r.from_pretrained(
                encoder_variant="small", resolution=336
            )
        elif model_name == "DuneMASt3R Base":
            _models[model_name] = DuneMast3r.from_pretrained(encoder_variant="base", resolution=448)
        elif model_name == "MASt3R Full":
            _models[model_name] = Mast3rFull.from_pretrained(resolution=512)
        elif model_name == "Retrieval":
            _models[model_name] = RetrievalModel.from_pretrained()
        print(f"{model_name} loaded!")
    return _models[model_name]


def get_retrieval_model():
    """Get or load retrieval model (lazy loading)."""
    return get_model("Retrieval")


def get_resolution(model_name: str) -> int:
    """Get resolution for model (long edge size)."""
    if model_name == "MASt3R Full":
        return 512
    elif model_name == "DuneMASt3R Base":
        return 448  # Must be multiple of 14 (patch_size)
    elif model_name == "DuneMASt3R Small":
        return 336  # Must be multiple of 14
    return 336  # Default for DUNE variants


def get_model_params(model_name: str) -> dict:
    """Get preprocessing params for model."""
    if "DUNE" in model_name or "DuneMASt3R" in model_name:
        return {"square_ok": True, "patch_size": 14}
    else:
        # MASt3R uses patch_size=16 and 4:3 aspect ratio
        return {"square_ok": False, "patch_size": 16}


# =============================================================================
# Feature extraction (DUNE)
# =============================================================================
def visualize_features_pca(features: np.ndarray, img_shape: tuple[int, int], patch_size: int = 14) -> np.ndarray:
    """Visualize features using PCA projection to RGB."""
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
    """Extract features from image."""
    if img_path is None:
        return None, [], "Veuillez charger une image."

    model_name = f"DUNE {variant.capitalize()}"
    model = get_model(model_name)

    img = load_image(img_path, resolution=336)

    t0 = time.perf_counter()
    features = model.encode(img)
    inference_time = (time.perf_counter() - t0) * 1000

    feature_viz = visualize_features_pca(features, img.shape[:2])

    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)

    gallery = [
        (img_uint8, "Image originale"),
        (feature_viz, "Features PCA"),
    ]

    status = f"""### Statistiques

**Modele:** {model_name}
**Temps inference:** {inference_time:.1f}ms ({1000 / inference_time:.1f} FPS)

**Features:**
- Shape: {features.shape}
- Embed dim: {model.embed_dim}
- Patches: {model.num_patches}
- Min: {features.min():.4f}
- Max: {features.max():.4f}
- Mean: {features.mean():.4f}
"""
    return feature_viz, gallery, status


# =============================================================================
# Stereo reconstruction (2 views)
# =============================================================================
def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    """Convert depth map to turbo colormap with enhanced contrast."""
    import matplotlib

    d = depth.copy().astype(np.float32)
    valid = np.isfinite(d) & (d > 0.1)

    if valid.sum() == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    d_log = np.zeros_like(d)
    d_log[valid] = np.log1p(d[valid])

    valid_vals = d_log[valid]
    hist, bin_edges = np.histogram(valid_vals, bins=256)
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]

    d_norm = np.zeros_like(d)
    bin_indices = np.clip(np.digitize(d_log[valid], bin_edges[:-1]) - 1, 0, len(cdf_normalized) - 1)
    d_norm[valid] = cdf_normalized[bin_indices]

    cmap = matplotlib.colormaps.get_cmap("turbo")
    rgb = cmap(d_norm)[:, :, :3]
    rgb[~valid] = 0

    return (rgb * 255).astype(np.uint8)


def conf_to_colormap(conf: np.ndarray) -> np.ndarray:
    """Convert confidence map to colormap."""
    c = conf.squeeze()
    c_min, c_max = c.min(), c.max()
    if c_max > c_min:
        c = (c - c_min) / (c_max - c_min)
    else:
        c = np.ones_like(c)

    r = np.clip(1.5 - np.abs(4 * c - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * c - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * c - 1), 0, 1)

    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def pts3d_to_trimesh_simple(
    img: np.ndarray,
    pts3d: np.ndarray,
    valid: np.ndarray,
) -> dict:
    """Convert 3D points to trimesh format."""
    H, W = img.shape[:2]
    vertices = pts3d.reshape(-1, 3)

    idx = np.arange(len(vertices)).reshape(H, W)
    idx1 = idx[:-1, :-1].ravel()
    idx2 = idx[:-1, +1:].ravel()
    idx3 = idx[+1:, :-1].ravel()
    idx4 = idx[+1:, +1:].ravel()

    faces = np.concatenate(
        [
            np.c_[idx1, idx2, idx3],
            np.c_[idx3, idx2, idx1],
            np.c_[idx2, idx3, idx4],
            np.c_[idx4, idx3, idx2],
        ],
        axis=0,
    )

    face_colors = np.concatenate(
        [
            img[:-1, :-1].reshape(-1, 3),
            img[:-1, :-1].reshape(-1, 3),
            img[+1:, +1:].reshape(-1, 3),
            img[+1:, +1:].reshape(-1, 3),
        ],
        axis=0,
    )

    valid_idxs = valid.ravel()
    valid_faces = valid_idxs[faces].all(axis=-1)
    faces = faces[valid_faces]
    face_colors = face_colors[valid_faces]

    return dict(vertices=vertices, faces=faces, face_colors=face_colors)


def convert_to_glb(
    imgs: list[np.ndarray],
    pts3d: list[np.ndarray],
    confs: list[np.ndarray],
    min_conf_thr: float = 1.5,
    as_pointcloud: bool = True,
) -> str:
    """Convert reconstruction output to GLB file."""
    scene = trimesh.Scene()

    def flip_points(pts: np.ndarray) -> np.ndarray:
        flipped = pts.copy()
        flipped[..., 1] *= -1
        return flipped

    if as_pointcloud:
        all_pts = []
        all_colors = []
        for img, pts, conf in zip(imgs, pts3d, confs):
            # Resize image if it doesn't match pts3d shape
            if img.shape[:2] != pts.shape[:2]:
                from PIL import Image as PILImage
                img_pil = PILImage.fromarray(img)
                img_pil = img_pil.resize((pts.shape[1], pts.shape[0]), PILImage.Resampling.LANCZOS)
                img = np.array(img_pil)

            mask = (conf.squeeze() > min_conf_thr) & np.isfinite(pts.sum(axis=-1))
            all_pts.append(flip_points(pts[mask]))
            all_colors.append(img[mask])

        if all_pts:
            pts = np.concatenate(all_pts).reshape(-1, 3)
            colors = np.concatenate(all_colors).reshape(-1, 3)

            if len(pts) > 0:
                colors_rgba = np.c_[colors, np.full(len(colors), 255, dtype=np.uint8)]
                pct = trimesh.PointCloud(pts, colors=colors_rgba)
                scene.add_geometry(pct)
    else:
        all_verts = []
        all_faces = []
        all_colors = []
        vert_offset = 0

        for img, pts, conf in zip(imgs, pts3d, confs):
            # Resize image if it doesn't match pts3d shape
            if img.shape[:2] != pts.shape[:2]:
                from PIL import Image as PILImage
                img_pil = PILImage.fromarray(img)
                img_pil = img_pil.resize((pts.shape[1], pts.shape[0]), PILImage.Resampling.LANCZOS)
                img = np.array(img_pil)

            mask = (conf.squeeze() > min_conf_thr) & np.isfinite(pts.sum(axis=-1))
            mesh_data = pts3d_to_trimesh_simple(img, flip_points(pts), mask)

            all_verts.append(mesh_data["vertices"])
            all_faces.append(mesh_data["faces"] + vert_offset)
            all_colors.append(mesh_data["face_colors"])
            vert_offset += len(mesh_data["vertices"])

        if all_verts:
            vertices = np.concatenate(all_verts)
            faces = np.concatenate(all_faces)
            face_colors = np.concatenate(all_colors)

            face_colors_rgba = np.c_[face_colors, np.full(len(face_colors), 255, dtype=np.uint8)]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=face_colors_rgba)
            scene.add_geometry(mesh)

    outfile = f"{_temp_dir}/scene_{time.time():.0f}.glb"

    if len(scene.geometry) == 0:
        pct = trimesh.PointCloud([[0, 0, 0]], colors=[[128, 128, 128, 255]])
        scene.add_geometry(pct)

    scene.export(file_obj=outfile)
    return outfile


def reconstruct_stereo(
    img1_path: str | None,
    img2_path: str | None,
    model_name: str,
    min_conf_thr: float,
    as_pointcloud: bool,
    show_cameras: bool,
    cam_size: float,
) -> tuple[str | None, list, str, str | None]:
    """Run 3D reconstruction on stereo pair."""
    if img1_path is None or img2_path is None:
        return None, [], "Veuillez charger deux images.", None

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

    status = f"""### Reconstruction terminee

**Modele:** {model_name}
**Resolution:** {img1.shape[1]}x{img1.shape[0]}

**Temps:**
- Inference: {inference_time:.1f}ms
- Export GLB: {export_time:.1f}ms
- **Total: {total_time:.1f}ms** ({1000 / total_time:.1f} FPS)

**Depth range:**
- View 1: {pts3d_1[:, :, 2].min():.2f} - {pts3d_1[:, :, 2].max():.2f}
- View 2: {pts3d_2[:, :, 2].min():.2f} - {pts3d_2[:, :, 2].max():.2f}

**Points 3D valides:**
- View 1: {n_valid_1:,}
- View 2: {n_valid_2:,}
"""

    return glb_file, gallery, status, glb_file


# =============================================================================
# Multi-view reconstruction (N views with sparse GA)
# =============================================================================
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
        return None, "Veuillez charger au moins 2 images.", None

    # Build cache path
    cache_dir = f"{_temp_dir}/cache_{time.time():.0f}"
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
        filepath = file_input.name if hasattr(file_input, 'name') else str(file_input)
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
                f"**Erreur**: Le mode retrieval necessite 'MASt3R Full' (embed_dim=1024). "
                f"Le modele '{model_name}' n'est pas compatible. "
                f"Veuillez selectionner 'MASt3R Full' ou utiliser un autre type de scene graph.",
                None,
            )

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
            return None, f"**Erreur retrieval**: {e}", None

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
    file_paths = [f.name if hasattr(f, 'name') else str(f) for f in files]

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
        return None, f"Erreur: {e!s}", None

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
        glb_file = f"{_temp_dir}/fallback_{time.time():.0f}.glb"
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.PointCloud([[0, 0, 0]]))
        scene.export(glb_file)
        print(f"GLB export error: {e}")

    export_time = time.perf_counter() - t1

    # Build status
    status = f"""### Reconstruction Multi-Vues terminee

**Configuration:**
- Modele: {model_name}
- Images: {len(files)}
- Paires: {len(pairs)}
- Scene graph: {scene_graph}

**Optimisation:**
- Phase 1 (coarse): lr={lr1}, {niter1} iterations
- Phase 2 (fine): lr={lr2}, {niter2} iterations
- Temps total: {optimization_time:.1f}s

**Export:**
- Temps: {export_time:.1f}s
- Format: {"Pointcloud" if as_pointcloud else "Mesh"}

**Resultats:**
- Cameras estimees: {result.n_imgs}
- Focales: {[f"{float(f):.1f}" for f in result.focals]}
"""

    # Store state for parameter updates
    scene_state = SceneState(
        sparse_ga=result,
        cache_dir=cache_dir,
        outfile=glb_file,
    )

    return glb_file, status, scene_state


def export_multiview_glb(
    result: SparseGAResult,
    pts3d_list: list,
    conf_list: list,
    min_conf_thr: float,
    as_pointcloud: bool,
    mask_sky: bool,
    cam_size: float,
    transparent_cams: bool,
) -> str:
    """Export multi-view result to GLB."""
    scene = trimesh.Scene()

    # Collect all valid points
    all_pts = []
    all_colors = []

    for i in range(result.n_imgs):
        pts = np.array(pts3d_list[i])
        conf = np.array(conf_list[i])
        img = result.imgs[i]

        # Flatten
        H, W = pts.shape[:2] if pts.ndim > 2 else (1, len(pts))
        pts_flat = pts.reshape(-1, 3)
        conf_flat = conf.reshape(-1)

        # Get colors
        if img.ndim == 3:
            colors = img.reshape(-1, 3)
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
        else:
            colors = np.ones((len(pts_flat), 3), dtype=np.uint8) * 128

        # Apply masks
        valid_conf = conf_flat > min_conf_thr
        valid_finite = np.isfinite(pts_flat.sum(axis=-1))
        valid = valid_conf & valid_finite

        if mask_sky and len(colors) == len(pts_flat):
            sky_mask = segment_sky(colors.reshape(H, W, 3) if H > 1 else colors)
            valid = valid & ~sky_mask.reshape(-1)

        pts_valid = pts_flat[valid]
        colors_valid = colors[valid] if len(colors) == len(pts_flat) else colors[: len(pts_valid)]

        # Flip Y for GLB
        pts_valid[:, 1] *= -1

        all_pts.append(pts_valid)
        all_colors.append(colors_valid)

    if all_pts:
        pts_combined = np.concatenate(all_pts, axis=0)
        colors_combined = np.concatenate(all_colors, axis=0)

        if len(pts_combined) > 0:
            if as_pointcloud:
                colors_rgba = np.c_[
                    colors_combined, np.full(len(colors_combined), 255, dtype=np.uint8)
                ]
                pct = trimesh.PointCloud(pts_combined, colors=colors_rgba)
                scene.add_geometry(pct, node_name="pointcloud")
            else:
                # Simple point cloud as mesh vertices
                colors_rgba = np.c_[
                    colors_combined, np.full(len(colors_combined), 255, dtype=np.uint8)
                ]
                pct = trimesh.PointCloud(pts_combined, colors=colors_rgba)
                scene.add_geometry(pct, node_name="mesh")

    # Add camera frustums
    if cam_size > 0:
        cam2w = np.array(result.cam2w)
        focals = np.array(result.focals)

        for i in range(result.n_imgs):
            pose = cam2w[i]
            focal = float(focals[i])

            # Create simple camera frustum
            frustum = create_camera_frustum(
                pose=pose,
                focal=focal,
                size=cam_size,
                color=get_camera_color(i),
                alpha=0.5 if transparent_cams else 1.0,
            )
            if frustum is not None:
                scene.add_geometry(frustum, node_name=f"camera_{i}")

    if len(scene.geometry) == 0:
        scene.add_geometry(trimesh.PointCloud([[0, 0, 0]]))

    outfile = f"{_temp_dir}/multiview_{time.time():.0f}.glb"
    scene.export(outfile)
    return outfile


def create_camera_frustum(
    pose: np.ndarray,
    focal: float,
    size: float = 0.1,
    color: tuple = (255, 0, 0),
    alpha: float = 1.0,
) -> trimesh.Trimesh | None:
    """Create a simple camera frustum mesh."""
    try:
        # Camera frustum vertices (in camera space)
        # Near plane corners
        near = size * 0.1
        far = size
        hw = size * 0.5  # Half width

        vertices = np.array(
            [
                [0, 0, 0],  # Camera center
                [-hw, -hw, far],  # Far plane corners
                [hw, -hw, far],
                [hw, hw, far],
                [-hw, hw, far],
            ]
        )

        # Transform to world space
        R = pose[:3, :3]
        t = pose[:3, 3]
        vertices_world = vertices @ R.T + t

        # Flip Y for GLB
        vertices_world[:, 1] *= -1

        # Faces (triangles)
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 1],
                [1, 2, 3],
                [1, 3, 4],
            ]
        )

        # Colors with alpha
        face_colors = np.array([list(color) + [int(alpha * 255)]] * len(faces), dtype=np.uint8)

        mesh = trimesh.Trimesh(vertices=vertices_world, faces=faces, face_colors=face_colors)
        return mesh
    except Exception:
        return None


def get_camera_color(idx: int) -> tuple:
    """Get color for camera index."""
    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
    ]
    return colors[idx % len(colors)]


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
        return None, "Aucune scene a mettre a jour."

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

    return glb_file, "Visualisation mise a jour."


# =============================================================================
# Gradio interface
# =============================================================================
def create_demo():
    """Create unified Gradio demo."""
    with gr.Blocks(title="MLX-MASt3R Demo") as demo:
        gr.HTML(
            """
            <div style="text-align: center; padding: 20px;">
                <h1>MLX-MASt3R</h1>
                <p style="font-size: 1.2em;">Reconstruction 3D ultra-rapide sur Apple Silicon</p>
            </div>
            """
        )

        with gr.Tabs():
            # =================================================================
            # Tab 1: Feature Extraction
            # =================================================================
            with gr.TabItem("Features DUNE"):
                gr.Markdown(
                    """
                    ### Extraction de features visuelles
                    Visualisation des features DUNE via projection PCA en RGB.
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        feat_img_input = gr.Image(
                            label="Image d'entree",
                            type="filepath",
                            height=300,
                        )
                        feat_variant = gr.Radio(
                            choices=["small", "base"],
                            value="base",
                            label="Variante DUNE",
                            info="small=11ms, base=32ms",
                        )
                        feat_run_btn = gr.Button("Extraire Features", variant="primary")

                    with gr.Column(scale=1):
                        feat_viz = gr.Image(label="Visualisation PCA", height=300)

                feat_stats = gr.Markdown(label="Statistiques")
                feat_gallery = gr.Gallery(label="Comparaison", columns=2, height="auto")

                # Examples for feature extraction
                examples_dir = Path(__file__).parent.parent / "assets" / "NLE_tower"
                if examples_dir.exists():
                    example_images = sorted(examples_dir.glob("*.jpg"))[:2]
                    if example_images:
                        gr.Examples(
                            examples=[[str(img)] for img in example_images],
                            inputs=[feat_img_input],
                            label="Exemples (NLE Tower)",
                        )

                feat_run_btn.click(
                    fn=extract_features,
                    inputs=[feat_img_input, feat_variant],
                    outputs=[feat_viz, feat_gallery, feat_stats],
                )

            # =================================================================
            # Tab 2: Stereo Reconstruction
            # =================================================================
            with gr.TabItem("Stereo (2 vues)"):
                gr.Markdown(
                    """
                    ### Reconstruction 3D stereo
                    Chargez deux images d'une meme scene pour obtenir un modele 3D.
                    """
                )

                with gr.Row():
                    recon_img1 = gr.Image(label="Image 1", type="filepath", height=250)
                    recon_img2 = gr.Image(label="Image 2", type="filepath", height=250)

                with gr.Row():
                    recon_model = gr.Dropdown(
                        choices=["MASt3R Full", "DuneMASt3R Base", "DuneMASt3R Small"],
                        value="MASt3R Full",
                        label="Modele",
                        scale=2,
                    )
                    recon_run_btn = gr.Button("Reconstruire", variant="primary", scale=1)

                with gr.Accordion("Options avancees", open=False):
                    with gr.Row():
                        recon_min_conf = gr.Slider(
                            label="Seuil confiance",
                            value=0.5,  # Lower default for DuneMASt3R compatibility
                            minimum=0.0,
                            maximum=10.0,
                            step=0.1,
                        )
                        recon_cam_size = gr.Slider(
                            label="Taille cameras",
                            value=0.05,
                            minimum=0.01,
                            maximum=0.2,
                            step=0.01,
                        )
                    with gr.Row():
                        recon_pointcloud = gr.Checkbox(value=True, label="Nuage de points")
                        recon_show_cams = gr.Checkbox(value=True, label="Afficher cameras")

                recon_status = gr.Markdown()

                with gr.Row():
                    with gr.Column(scale=2):
                        recon_model3d = gr.Model3D(label="Modele 3D", height=400)
                    with gr.Column(scale=1):
                        recon_download = gr.File(label="Telecharger GLB")

                recon_gallery = gr.Gallery(label="RGB | Depth | Confidence", columns=3, rows=2)

                # Examples for stereo reconstruction
                examples_dir = Path(__file__).parent.parent / "assets" / "NLE_tower"
                if examples_dir.exists():
                    example_images = sorted(examples_dir.glob("*.jpg"))[:4]
                    if len(example_images) >= 2:
                        gr.Examples(
                            examples=[
                                [str(example_images[0]), str(example_images[1])],
                                [str(example_images[2]), str(example_images[3])] if len(example_images) >= 4 else [str(example_images[0]), str(example_images[1])],
                            ],
                            inputs=[recon_img1, recon_img2],
                            label="Exemples (NLE Tower)",
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

            # =================================================================
            # Tab 3: Multi-view Reconstruction (NEW!)
            # =================================================================
            with gr.TabItem("Multi-Vues (N images)"):
                gr.Markdown(
                    """
                    ### Reconstruction Multi-Vues avec Alignement Global
                    Chargez plusieurs images (3+) pour une reconstruction complete avec optimisation des poses cameras.
                    """
                )

                # State for scene persistence
                mv_scene_state = gr.State(None)

                # Input section
                with gr.Row():
                    mv_files = gr.File(
                        label="Images (glisser-deposer ou selectionner)",
                        file_count="multiple",
                        file_types=["image"],
                    )

                with gr.Row():
                    mv_model = gr.Dropdown(
                        choices=["MASt3R Full", "DuneMASt3R Base", "DuneMASt3R Small"],
                        value="MASt3R Full",
                        label="Modele",
                        scale=2,
                    )
                    mv_run_btn = gr.Button("Reconstruire", variant="primary", scale=1)

                # Scene Graph options
                with gr.Accordion("Scene Graph", open=True):
                    with gr.Row():
                        mv_scenegraph = gr.Dropdown(
                            choices=["complete", "swin", "logwin", "oneref", "retrieval"],
                            value="complete",
                            label="Type",
                            info="complete=toutes paires, retrieval=selection auto par similarite",
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
                            info="Nombre d'images adjacentes a considerer",
                        )
                        mv_retrieval_k = gr.Slider(
                            label="k (pairs per image)",
                            value=10,
                            minimum=2,
                            maximum=30,
                            step=1,
                            visible=False,
                            info="Nombre de paires par image",
                        )

                # Optimization parameters
                with gr.Accordion("Parametres d'optimisation", open=False):
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
                            label="Seuil matching",
                            value=5.0,
                            minimum=0.0,
                            maximum=20.0,
                            step=0.5,
                        )
                        mv_shared_intrinsics = gr.Checkbox(
                            value=False, label="Intrinsics partagees"
                        )

                # Post-processing / visualization
                with gr.Accordion("Visualisation", open=True):
                    with gr.Row():
                        mv_min_conf = gr.Slider(
                            label="Seuil confiance",
                            value=0.5,  # Lower default for DuneMASt3R compatibility
                            minimum=0.0,
                            maximum=20.0,
                            step=0.5,
                        )
                        mv_cam_size = gr.Slider(
                            label="Taille cameras",
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
                            info="0=desactive, >0=nettoyage profondeur",
                        )
                    with gr.Row():
                        mv_pointcloud = gr.Checkbox(value=True, label="Nuage de points")
                        mv_mask_sky = gr.Checkbox(value=False, label="Masquer ciel")
                        mv_clean_depth = gr.Checkbox(value=True, label="Nettoyer depth")
                        mv_transparent_cams = gr.Checkbox(
                            value=False, label="Cameras transparentes"
                        )

                    mv_update_viz_btn = gr.Button("Mettre a jour visualisation")

                # Output section
                mv_status = gr.Markdown()

                with gr.Row():
                    mv_model3d = gr.Model3D(label="Modele 3D", height=500)

                # Examples for multi-view reconstruction
                examples_dir = Path(__file__).parent.parent / "assets" / "NLE_tower"
                if examples_dir.exists():
                    example_images = sorted(examples_dir.glob("*.jpg"))
                    if len(example_images) >= 3:
                        # Create example with all images as a list
                        gr.Markdown("### Exemples")
                        gr.Markdown(
                            f"**NLE Tower** : {len(example_images)} images disponibles dans `assets/NLE_tower/`\n\n"
                            "Cliquez sur 'Selectionner' et choisissez les images du dossier `assets/NLE_tower/`"
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

            # =================================================================
            # Tab 4: About
            # =================================================================
            with gr.TabItem("A propos"):
                gr.Markdown(
                    """
                    ## MLX-MASt3R

                    Implementation MLX optimisee pour Apple Silicon des modeles:

                    ### Modeles disponibles

                    | Modele | Encodeur | Resolution | Temps | Usage |
                    |--------|----------|------------|-------|-------|
                    | **DUNE Small** | ViT-S | 336 | ~11ms | Features rapides |
                    | **DUNE Base** | ViT-B | 336 | ~32ms | Features de qualite |
                    | **DuneMASt3R Small** | DUNE-S + MASt3R | 336 | ~50ms | Reconstruction rapide |
                    | **DuneMASt3R Base** | DUNE-B + MASt3R | 448 | ~90ms | Reconstruction equilibree |
                    | **MASt3R Full** | ViT-L | 512 | ~200ms | Meilleure qualite |

                    ### Modes de reconstruction

                    | Mode | Images | Description |
                    |------|--------|-------------|
                    | **Stereo** | 2 | Reconstruction rapide entre deux vues |
                    | **Multi-Vues** | 3+ | Alignement global avec optimisation des poses |

                    ### Scene Graph

                    | Type | Description |
                    |------|-------------|
                    | **complete** | Toutes les paires possibles (N*(N-1)/2) |
                    | **swin** | Fenetre glissante de taille K |
                    | **logwin** | Fenetre logarithmique (puissances de 2) |
                    | **oneref** | Une image de reference avec toutes les autres |
                    | **retrieval** | Selection automatique par similarite visuelle |

                    #### Mode Retrieval
                    Utilise un modele de retrieval pre-entraine pour calculer la similarite
                    entre images et selectionner automatiquement les meilleures paires.
                    Ideal pour les grandes collections d'images non ordonnees.

                    ### Credits

                    - **MASt3R/DUSt3R**: [Naver Labs](https://github.com/naver/mast3r) (CC BY-NC-SA 4.0)
                    - **DUNE**: [Facebook Research](https://github.com/facebookresearch/dune) (Apache 2.0)
                    - **MLX**: [Apple](https://github.com/ml-explore/mlx)

                    ### Auteur

                    Copyright (c) 2025 Delanoe Pirard / Aedelon - Apache 2.0 License
                    """
                )

    return demo


if __name__ == "__main__":
    import gradio as gr
    demo = create_demo()
    demo.launch(share=False, server_port=7860)
