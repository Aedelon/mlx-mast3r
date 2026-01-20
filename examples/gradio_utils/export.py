# Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""GLB export utilities for Gradio demo."""

from __future__ import annotations

import tempfile
import time
from typing import TYPE_CHECKING

import numpy as np
import trimesh

if TYPE_CHECKING:
    from mlx_mast3r.cloud_opt import SparseGAResult

# Temp directory for exports
_temp_dir = tempfile.mkdtemp(prefix="mlx_mast3r_demo_")


def get_temp_dir() -> str:
    """Get temporary directory for exports."""
    return _temp_dir


def pts3d_to_trimesh_simple(
    img: np.ndarray,
    pts3d: np.ndarray,
    valid: np.ndarray,
) -> dict:
    """Convert 3D points to trimesh format.

    Args:
        img: [H, W, 3] RGB image
        pts3d: [H, W, 3] 3D points
        valid: [H, W] boolean mask

    Returns:
        Dict with vertices, faces, face_colors
    """
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
    """Convert stereo reconstruction output to GLB file.

    Args:
        imgs: List of [H, W, 3] RGB images
        pts3d: List of [H, W, 3] 3D point maps
        confs: List of [H, W] confidence maps
        min_conf_thr: Minimum confidence threshold
        as_pointcloud: If True, export as point cloud; else as mesh

    Returns:
        Path to exported GLB file
    """
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


def create_camera_frustum(
    pose: np.ndarray,
    focal: float,
    size: float = 0.1,
    color: tuple = (255, 0, 0),
    alpha: float = 1.0,
) -> trimesh.Trimesh | None:
    """Create a simple camera frustum mesh.

    Args:
        pose: [4, 4] camera-to-world pose matrix
        focal: Focal length
        size: Frustum size
        color: RGB color tuple
        alpha: Transparency (0-1)

    Returns:
        trimesh.Trimesh or None if creation fails
    """
    try:
        # Camera frustum vertices (in camera space)
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
    """Get color for camera index.

    Args:
        idx: Camera index

    Returns:
        RGB color tuple
    """
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


def export_multiview_glb(
    result: "SparseGAResult",
    pts3d_list: list,
    conf_list: list,
    min_conf_thr: float,
    as_pointcloud: bool,
    mask_sky: bool,
    cam_size: float,
    transparent_cams: bool,
) -> str:
    """Export multi-view result to GLB.

    Args:
        result: SparseGAResult from multi-view reconstruction
        pts3d_list: List of 3D point arrays per view
        conf_list: List of confidence arrays per view
        min_conf_thr: Minimum confidence threshold
        as_pointcloud: Export as point cloud if True
        mask_sky: Mask sky regions if True
        cam_size: Camera frustum size (0 = no cameras)
        transparent_cams: Use transparent camera frustums

    Returns:
        Path to exported GLB file
    """
    from mlx_mast3r.viz import segment_sky

    scene = trimesh.Scene()

    # Collect all valid points
    all_pts = []
    all_colors = []

    for i in range(result.n_imgs):
        pts = np.array(pts3d_list[i])
        conf = np.array(conf_list[i])
        img = result.imgs[i]

        # Flatten pts3d
        H, W = pts.shape[:2] if pts.ndim > 2 else (1, len(pts))
        pts_flat = pts.reshape(-1, 3)
        conf_flat = conf.reshape(-1)

        # Get colors - subsample image to match pts3d resolution
        if img.ndim == 3:
            img_H, img_W = img.shape[:2]
            # Calculate subsample factor
            subsample = max(1, img_H // H) if H > 1 else 1
            if subsample > 1:
                # Subsample image to match pts3d grid
                colors = img[::subsample, ::subsample, :].reshape(-1, 3)
            else:
                colors = img.reshape(-1, 3)
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
        else:
            colors = np.ones((len(pts_flat), 3), dtype=np.uint8) * 128

        # Ensure colors match pts length
        if len(colors) != len(pts_flat):
            # Fallback: use gray
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
