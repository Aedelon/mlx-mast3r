"""Visualization utilities for 3D reconstruction.

Adapted from dust3r/viz.py for MLX (no torch dependency).

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import PIL.Image
from scipy.spatial.transform import Rotation

try:
    import trimesh
except ImportError:
    trimesh = None
    print("/!\\ module trimesh is not installed, cannot visualize results /!\\")


# =============================================================================
# Constants
# =============================================================================

OPENGL = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)

CAM_COLORS = [
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 255),
    (255, 204, 0),
    (0, 204, 204),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (0, 0, 0),
    (128, 128, 128),
]


# =============================================================================
# Utility functions
# =============================================================================


def to_numpy(x: Any) -> Any:
    """Convert to numpy array (handles various input types)."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return type(x)(to_numpy(v) for v in x)
    if hasattr(x, "numpy"):  # mlx.array or torch.Tensor
        return np.array(x)
    return np.asarray(x)


def uint8(colors: np.ndarray | list | tuple) -> np.ndarray:
    """Convert colors to uint8."""
    if not isinstance(colors, np.ndarray):
        colors = np.array(colors)
    if np.issubdtype(colors.dtype, np.floating):
        colors = colors * 255
    assert 0 <= colors.min() and colors.max() < 256
    return np.uint8(colors)


def cat_3d(vecs: list | np.ndarray) -> np.ndarray:
    """Concatenate 3D vectors."""
    if isinstance(vecs, np.ndarray):
        vecs = [vecs]
    return np.concatenate([p.reshape(-1, 3) for p in to_numpy(vecs)])


def geotrf(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply geometric transformation to points.

    Args:
        T: 4x4 transformation matrix
        pts: Nx3 points

    Returns:
        Transformed Nx3 points
    """
    pts = np.asarray(pts)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)

    # Homogeneous coordinates
    ones = np.ones((len(pts), 1), dtype=pts.dtype)
    pts_h = np.concatenate([pts, ones], axis=1)

    # Transform
    pts_t = pts_h @ T.T

    # Back to 3D
    return pts_t[:, :3] / pts_t[:, 3:4]


# =============================================================================
# Mesh utilities
# =============================================================================


def pts3d_to_trimesh(
    img: np.ndarray,
    pts3d: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict:
    """Convert 3D points to trimesh format.

    Creates a mesh where each pixel becomes 2 triangles (a square).

    Args:
        img: HxWx3 RGB image
        pts3d: HxWx3 3D points
        valid: HxW boolean mask for valid points

    Returns:
        Dict with vertices, faces, face_colors
    """
    H, W, THREE = img.shape
    assert THREE == 3
    assert img.shape == pts3d.shape

    vertices = pts3d.reshape(-1, 3)

    # Make squares: each pixel == 2 triangles
    idx = np.arange(len(vertices)).reshape(H, W)
    idx1 = idx[:-1, :-1].ravel()  # top-left
    idx2 = idx[:-1, +1:].ravel()  # top-right
    idx3 = idx[+1:, :-1].ravel()  # bottom-left
    idx4 = idx[+1:, +1:].ravel()  # bottom-right

    # Create faces (both directions to disable face culling)
    faces = np.concatenate(
        [
            np.c_[idx1, idx2, idx3],
            np.c_[idx3, idx2, idx1],
            np.c_[idx2, idx3, idx4],
            np.c_[idx4, idx3, idx2],
        ],
        axis=0,
    )

    # Face colors
    face_colors = np.concatenate(
        [
            img[:-1, :-1].reshape(-1, 3),
            img[:-1, :-1].reshape(-1, 3),
            img[+1:, +1:].reshape(-1, 3),
            img[+1:, +1:].reshape(-1, 3),
        ],
        axis=0,
    )

    # Filter invalid faces
    if valid is not None:
        assert valid.shape == (H, W)
        valid_idxs = valid.ravel()
        valid_faces = valid_idxs[faces].all(axis=-1)
        faces = faces[valid_faces]
        face_colors = face_colors[valid_faces]

    return dict(vertices=vertices, faces=faces, face_colors=face_colors)


def cat_meshes(meshes: list[dict]) -> dict:
    """Concatenate multiple meshes.

    Args:
        meshes: List of mesh dicts with vertices, faces, face_colors

    Returns:
        Combined mesh dict
    """
    vertices_list = []
    faces_list = []
    colors_list = []

    n_vertices = 0
    for m in meshes:
        vertices_list.append(m["vertices"])
        faces_list.append(m["faces"] + n_vertices)
        colors_list.append(m["face_colors"])
        n_vertices += len(m["vertices"])

    return dict(
        vertices=np.concatenate(vertices_list),
        faces=np.concatenate(faces_list),
        face_colors=np.concatenate(colors_list),
    )


# =============================================================================
# Camera visualization
# =============================================================================


def add_scene_cam(
    scene: "trimesh.Scene",
    pose_c2w: np.ndarray,
    edge_color: tuple[int, int, int],
    image: np.ndarray | None = None,
    focal: float | None = None,
    imsize: tuple[int, int] | None = None,
    screen_width: float = 0.03,
    marker: str | None = None,
) -> None:
    """Add a camera frustum to a trimesh scene.

    Args:
        scene: trimesh Scene to add camera to
        pose_c2w: 4x4 camera-to-world transformation
        edge_color: RGB color for camera wireframe
        image: Optional HxWx3 image to texture the frustum
        focal: Camera focal length
        imsize: (W, H) image size if no image provided
        screen_width: Size of camera visualization
        marker: Optional marker type ('o' for sphere)
    """
    if trimesh is None:
        raise ImportError("trimesh is required for visualization")

    # Determine image dimensions
    if image is not None:
        image = np.asarray(image)
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255 * image)
    elif imsize is not None:
        W, H = imsize
    elif focal is not None:
        H = W = focal / 1.1
    else:
        H = W = 1

    # Handle focal
    if isinstance(focal, np.ndarray):
        focal = float(focal.flat[0])
    if not focal:
        focal = min(H, W) * 1.1

    # Create camera cone geometry
    height = max(screen_width / 10, focal * screen_width / H)
    width = screen_width * 0.5**0.5

    # Rotation matrices
    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler("z", np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height  # tip at optical center

    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = W / H

    transform = pose_c2w @ OPENGL @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4)

    # Add image texture
    if image is not None:
        vertices = geotrf(transform, cam.vertices[[4, 5, 1, 3]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
        img_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        img_mesh.visual = trimesh.visual.TextureVisuals(uv_coords, image=PIL.Image.fromarray(image))
        scene.add_geometry(img_mesh)

    # Create wireframe edges
    rot2 = np.eye(4)
    rot2[:3, :3] = Rotation.from_euler("z", np.deg2rad(2)).as_matrix()

    vertices = np.concatenate(
        [
            cam.vertices,
            0.95 * cam.vertices,
            geotrf(rot2, cam.vertices),
        ]
    )
    vertices = geotrf(transform, vertices)

    faces = []
    for face in cam.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam.vertices)
        a3, b3, c3 = face + 2 * len(cam.vertices)

        # Pseudo-edges
        faces.extend(
            [
                (a, b, b2),
                (a, a2, c),
                (c2, b, c),
                (a, b, b3),
                (a, a3, c),
                (c3, b, c),
            ]
        )

    # No culling - add reversed faces
    faces += [(c, b, a) for a, b, c in faces]

    cam_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam_mesh.visual.face_colors[:, :3] = edge_color
    scene.add_geometry(cam_mesh)

    # Optional marker
    if marker == "o":
        sphere = trimesh.creation.icosphere(3, radius=screen_width / 4)
        sphere.vertices += pose_c2w[:3, 3]
        sphere.visual.face_colors[:, :3] = edge_color
        scene.add_geometry(sphere)


def auto_cam_size(im_poses: np.ndarray) -> float:
    """Compute automatic camera size based on pose distances.

    Args:
        im_poses: Nx4x4 camera poses

    Returns:
        Recommended camera visualization size
    """
    im_poses = to_numpy(im_poses)
    centers = im_poses[:, :3, 3]

    # Compute pairwise distances
    n = len(centers)
    if n < 2:
        return 0.1

    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(centers[i] - centers[j]))

    return 0.1 * np.median(dists)


# =============================================================================
# SceneViz class
# =============================================================================


class SceneViz:
    """Helper class for building trimesh scenes."""

    def __init__(self):
        if trimesh is None:
            raise ImportError("trimesh is required for visualization")
        self.scene = trimesh.Scene()

    def add_pointcloud(
        self,
        pts3d: np.ndarray | list,
        color: np.ndarray | tuple = (0, 0, 0),
        mask: np.ndarray | list | None = None,
        denoise: bool = False,
    ) -> "SceneViz":
        """Add a point cloud to the scene.

        Args:
            pts3d: Points (HxWx3 or list of HxWx3)
            color: Colors (HxWx3 or single RGB tuple)
            mask: Optional validity mask
            denoise: Remove outlier points

        Returns:
            self for chaining
        """
        pts3d = to_numpy(pts3d)
        mask = to_numpy(mask)

        if not isinstance(pts3d, list):
            pts3d = [pts3d.reshape(-1, 3)]
            if mask is not None:
                mask = [mask.ravel()]

        if not isinstance(color, (tuple, list)) or (
            isinstance(color, (list, np.ndarray)) and np.asarray(color).ndim > 1
        ):
            color = to_numpy(color)
            if not isinstance(color, list):
                color = [color.reshape(-1, 3)]

        if mask is None:
            mask = [slice(None)] * len(pts3d)

        # Concatenate points
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        pct = trimesh.PointCloud(pts)

        # Set colors
        if isinstance(color, (list, np.ndarray)):
            col = np.concatenate([c[m] for c, m in zip(color, mask)])
            pct.visual.vertex_colors = uint8(col.reshape(-1, 3))
        else:
            pct.visual.vertex_colors = np.broadcast_to(uint8(color), pts.shape)

        # Denoise
        if denoise:
            centroid = np.median(pct.vertices, axis=0)
            dist = np.linalg.norm(pct.vertices - centroid, axis=-1)
            dist_thr = np.quantile(dist, 0.99)
            valid = dist < dist_thr
            pct = trimesh.PointCloud(
                pct.vertices[valid],
                colors=pct.visual.vertex_colors[valid],
            )

        self.scene.add_geometry(pct)
        return self

    def add_camera(
        self,
        pose_c2w: np.ndarray,
        focal: float | None = None,
        color: tuple = (0, 0, 0),
        image: np.ndarray | None = None,
        imsize: tuple[int, int] | None = None,
        cam_size: float = 0.03,
    ) -> "SceneViz":
        """Add a camera to the scene."""
        pose_c2w = to_numpy(pose_c2w)
        focal = to_numpy(focal)
        image = to_numpy(image)

        if isinstance(focal, np.ndarray) and focal.shape == (3, 3):
            intrinsics = focal
            focal = float((intrinsics[0, 0] * intrinsics[1, 1]) ** 0.5)
            if imsize is None:
                imsize = (2 * intrinsics[0, 2], 2 * intrinsics[1, 2])

        add_scene_cam(
            self.scene,
            pose_c2w,
            color,
            image,
            focal,
            imsize=imsize,
            screen_width=cam_size,
        )
        return self

    def add_cameras(
        self,
        poses: np.ndarray,
        focals: np.ndarray | None = None,
        images: list | None = None,
        imsizes: list | None = None,
        colors: list | None = None,
        **kw,
    ) -> "SceneViz":
        """Add multiple cameras to the scene."""

        def get(arr, idx):
            return None if arr is None else arr[idx]

        for i, pose_c2w in enumerate(poses):
            self.add_camera(
                pose_c2w,
                get(focals, i),
                image=get(images, i),
                color=get(colors, i) or CAM_COLORS[i % len(CAM_COLORS)],
                imsize=get(imsizes, i),
                **kw,
            )
        return self

    def export(self, path: str) -> str:
        """Export scene to file (GLB/GLTF/PLY)."""
        self.scene.export(file_obj=path)
        return path

    def show(self, point_size: int = 2) -> None:
        """Display the scene."""
        self.scene.show(line_settings={"point_size": point_size})


# =============================================================================
# Sky segmentation
# =============================================================================


def segment_sky(image: np.ndarray) -> np.ndarray:
    """Segment sky regions in an image using HSV analysis.

    Args:
        image: HxWx3 RGB image

    Returns:
        HxW boolean mask where True = sky
    """
    try:
        import cv2
        from scipy import ndimage
    except ImportError:
        raise ImportError("cv2 and scipy are required for sky segmentation")

    image = to_numpy(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.uint8(255 * image.clip(min=0, max=1))

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Blue sky detection
    lower_blue = np.array([0, 0, 100])
    upper_blue = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue).astype(bool)

    # Add luminous gray (overcast sky)
    mask |= (hsv[:, :, 1] < 10) & (hsv[:, :, 2] > 150)
    mask |= (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 180)
    mask |= (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 220)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask2 = ndimage.binary_opening(mask, structure=kernel)

    # Keep only largest connected components
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask2.astype(np.uint8), connectivity=8)
    cc_sizes = stats[1:, cv2.CC_STAT_AREA]

    if len(cc_sizes) == 0:
        return mask2

    order = cc_sizes.argsort()[::-1]
    selection = []
    i = 0
    while i < len(order) and cc_sizes[order[i]] > cc_sizes[order[0]] / 2:
        selection.append(1 + order[i])
        i += 1

    mask3 = np.isin(labels, selection)
    return mask3


# =============================================================================
# High-level functions
# =============================================================================


def convert_scene_to_glb(
    outfile: str,
    imgs: list[np.ndarray],
    pts3d: list[np.ndarray],
    masks: list[np.ndarray],
    focals: np.ndarray,
    cams2world: np.ndarray,
    cam_size: float = 0.05,
    cam_color: list | tuple | None = None,
    as_pointcloud: bool = False,
    transparent_cams: bool = False,
    silent: bool = False,
) -> str:
    """Convert scene data to GLB file with cameras.

    Args:
        outfile: Output path
        imgs: List of HxWx3 images
        pts3d: List of HxWx3 3D points
        masks: List of HxW validity masks
        focals: N focal lengths
        cams2world: Nx4x4 camera poses
        cam_size: Camera frustum size
        cam_color: Camera colors (list or single)
        as_pointcloud: Use point cloud instead of mesh
        transparent_cams: Don't show images on cameras
        silent: Suppress output

    Returns:
        Output file path
    """
    if trimesh is None:
        raise ImportError("trimesh is required")

    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)
    masks = to_numpy(masks)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # Add point cloud or mesh
    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, masks)]).reshape(-1, 3)
        col = np.concatenate([im[m] for im, m in zip(imgs, masks)]).reshape(-1, 3)
        valid = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid], colors=col[valid])
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = masks[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # Add cameras
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            color = cam_color[i]
        else:
            color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]

        add_scene_cam(
            scene,
            pose_c2w,
            color,
            None if transparent_cams else (imgs[i] if i < len(imgs) else None),
            focals[i],
            imsize=imgs[i].shape[1::-1] if i < len(imgs) else None,
            screen_width=cam_size,
        )

    # Apply transform to center on first camera
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))

    if not silent:
        print(f"(exporting 3D scene to {outfile})")

    scene.export(file_obj=outfile)
    return outfile
