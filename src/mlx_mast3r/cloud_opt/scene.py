"""Scene class for 3D reconstruction visualization and export.

Adapted from dust3r/viz.py for trimesh export.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
Original dust3r code: Copyright (C) 2024-present Naver Corporation. CC BY-NC-SA 4.0.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import trimesh

    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

from scipy.spatial.transform import Rotation

from .pair_viewer import PairViewerResult


# OpenGL convention matrix
OPENGL = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)

# Camera colors for visualization
CAM_COLORS = [
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 255),
    (255, 204, 0),
    (0, 204, 204),
    (128, 255, 255),
    (255, 128, 255),
]


def pts3d_to_trimesh(
    img: np.ndarray,
    pts3d: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict:
    """Convert 3D points to trimesh format.

    Creates a mesh where each pixel becomes 2 triangles, textured with image colors.

    Args:
        img: RGB image (H, W, 3) with values in [0, 255].
        pts3d: 3D points (H, W, 3).
        valid: Optional validity mask (H, W).

    Returns:
        Dict with vertices, faces, and face_colors for trimesh.Trimesh.
    """
    H, W = img.shape[:2]
    assert img.shape == pts3d.shape == (H, W, 3)

    vertices = pts3d.reshape(-1, 3)

    # Make squares: each pixel == 2 triangles
    idx = np.arange(len(vertices)).reshape(H, W)
    idx1 = idx[:-1, :-1].ravel()  # top-left
    idx2 = idx[:-1, +1:].ravel()  # top-right
    idx3 = idx[+1:, :-1].ravel()  # bottom-left
    idx4 = idx[+1:, +1:].ravel()  # bottom-right

    faces = np.concatenate(
        [
            np.c_[idx1, idx2, idx3],
            np.c_[idx3, idx2, idx1],  # backward for face culling
            np.c_[idx2, idx3, idx4],
            np.c_[idx4, idx3, idx2],  # backward for face culling
        ],
        axis=0,
    )

    # Triangle colors from image
    face_colors = np.concatenate(
        [
            img[:-1, :-1].reshape(-1, 3),
            img[:-1, :-1].reshape(-1, 3),
            img[+1:, +1:].reshape(-1, 3),
            img[+1:, +1:].reshape(-1, 3),
        ],
        axis=0,
    )

    # Remove invalid faces
    if valid is not None:
        assert valid.shape == (H, W)
        valid_idxs = valid.ravel()
        valid_faces = valid_idxs[faces].all(axis=-1)
        faces = faces[valid_faces]
        face_colors = face_colors[valid_faces]

    return dict(vertices=vertices, faces=faces, face_colors=face_colors)


def cat_meshes(meshes: list[dict]) -> dict:
    """Concatenate multiple meshes into one.

    Args:
        meshes: List of mesh dicts with vertices, faces, face_colors.

    Returns:
        Combined mesh dict.
    """
    vertices, faces, colors = zip(
        *[(m["vertices"], m["faces"].copy(), m["face_colors"]) for m in meshes]
    )

    n_vertices = np.cumsum([0] + [len(v) for v in vertices])

    for i in range(len(faces)):
        faces[i][:] += n_vertices[i]

    return dict(
        vertices=np.concatenate(vertices),
        faces=np.concatenate(faces),
        face_colors=np.concatenate(colors),
    )


def add_camera_to_scene(
    scene: "trimesh.Scene",
    pose_c2w: np.ndarray,
    color: tuple[int, int, int],
    focal: float,
    imsize: tuple[int, int],
    screen_width: float = 0.03,
) -> None:
    """Add a camera frustum to the scene.

    Args:
        scene: Trimesh scene.
        pose_c2w: Camera-to-world pose (4x4).
        color: RGB color for the camera.
        focal: Focal length.
        imsize: Image size (W, H).
        screen_width: Width of the camera frustum.
    """
    W, H = imsize

    # Create camera frustum as a cone
    height = max(screen_width / 10, focal * screen_width / H)
    width = screen_width * 0.5**0.5

    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler("z", np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height

    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = W / H

    transform = pose_c2w @ OPENGL @ aspect_ratio @ rot45

    cam = trimesh.creation.cone(width, height, sections=4)
    cam.apply_transform(transform)
    cam.visual.face_colors = [*color, 255]

    scene.add_geometry(cam)


class Scene:
    """Scene for 3D reconstruction visualization and export."""

    def __init__(
        self,
        result: PairViewerResult,
        min_conf_thr: float = 1.5,
    ):
        """Initialize Scene.

        Args:
            result: PairViewerResult from reconstruction.
            min_conf_thr: Minimum confidence threshold.
        """
        if not HAS_TRIMESH:
            raise ImportError("trimesh is required for Scene. Install with: pip install trimesh")

        self.result = result
        self.min_conf_thr = min_conf_thr

    def get_masks(self) -> list[np.ndarray]:
        """Get validity masks based on confidence threshold."""
        masks = []
        for conf, pts in zip(self.result.confs, self.result.pts3d):
            mask = (conf > self.min_conf_thr) & np.isfinite(pts).all(axis=-1)
            masks.append(mask)
        return masks

    def export_glb(
        self,
        path: str | Path | None = None,
        as_pointcloud: bool = False,
        cam_size: float = 0.05,
        show_cameras: bool = True,
    ) -> str:
        """Export scene to GLB file.

        Args:
            path: Output path. If None, creates temp file.
            as_pointcloud: Export as point cloud instead of mesh.
            cam_size: Camera frustum size.
            show_cameras: Whether to show camera frustums.

        Returns:
            Path to the exported GLB file.
        """
        scene = trimesh.Scene()
        masks = self.get_masks()

        if as_pointcloud:
            # Point cloud mode
            all_pts = []
            all_colors = []
            for img, pts, mask in zip(self.result.imgs, self.result.pts3d, masks):
                all_pts.append(pts[mask])
                all_colors.append(img[mask])

            pts = np.concatenate(all_pts).reshape(-1, 3)
            colors = np.concatenate(all_colors).reshape(-1, 3)

            if len(pts) > 0:
                pct = trimesh.PointCloud(pts, colors=colors)
                scene.add_geometry(pct)
        else:
            # Mesh mode
            meshes = []
            for img, pts, mask in zip(self.result.imgs, self.result.pts3d, masks):
                meshes.append(pts3d_to_trimesh(img, pts, mask))

            if meshes:
                mesh = trimesh.Trimesh(**cat_meshes(meshes))
                scene.add_geometry(mesh)

        # Add cameras
        if show_cameras:
            for i, (pose, focal, pp) in enumerate(
                zip(
                    self.result.im_poses,
                    self.result.focals,
                    self.result.pp,
                )
            ):
                H, W = self.result.pts3d[i].shape[:2]
                color = CAM_COLORS[i % len(CAM_COLORS)]
                add_camera_to_scene(scene, pose, color, focal, (W, H), cam_size)

        # Apply initial transform to align with first camera
        rot = np.eye(4)
        rot[:3, :3] = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()
        scene.apply_transform(np.linalg.inv(self.result.im_poses[0] @ OPENGL @ rot))

        # Export
        if path is None:
            path = f"{tempfile.gettempdir()}/scene_{time.time():.0f}.glb"
        else:
            path = str(path)

        scene.export(file_obj=path)
        return path

    def get_pts3d(self) -> list[np.ndarray]:
        """Get 3D points for each view."""
        return self.result.pts3d

    def get_depths(self) -> list[np.ndarray]:
        """Get depth maps for each view."""
        return self.result.depths

    def get_focals(self) -> list[float]:
        """Get focal lengths."""
        return self.result.focals

    def get_im_poses(self) -> list[np.ndarray]:
        """Get camera poses (camera-to-world)."""
        return self.result.im_poses

    def get_intrinsics(self) -> list[np.ndarray]:
        """Get intrinsic matrices."""
        return self.result.intrinsics
