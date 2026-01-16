"""Image preprocessing utilities.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose


def _resize_pil_image(img: Image.Image, long_edge_size: int) -> Image.Image:
    """Resize image so long edge equals long_edge_size, preserving aspect ratio."""
    S = max(img.size)
    if S > long_edge_size:
        interp = Image.Resampling.LANCZOS
    else:
        interp = Image.Resampling.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def load_image(
    path: str | Path,
    resolution: int | tuple[int, int] | None = None,
    square_ok: bool = False,
    patch_size: int = 16,
) -> np.ndarray:
    """Load image from path with DUSt3R-compatible preprocessing.

    Matches the original DUSt3R/MASt3R preprocessing:
    - Resize long edge to `resolution`
    - Crop to dimensions divisible by `patch_size`
    - Force 4:3 aspect ratio unless `square_ok=True`

    Args:
        path: Path to image file
        resolution: Target resolution. Can be:
            - int: long edge size (default DUSt3R behavior)
            - tuple (H, W): specific height and width (direct resize)
            - None: keep original size
        square_ok: If False, force 4:3 aspect ratio (default DUSt3R behavior)
        patch_size: Patch size for encoder (16 for ViT)

    Returns:
        [H, W, 3] uint8 numpy array
    """
    img = exif_transpose(Image.open(path)).convert("RGB")

    if resolution is not None:
        if isinstance(resolution, tuple):
            # Direct resize to specific (H, W)
            size = (resolution[1], resolution[0])  # PIL uses (W, H)
            img = img.resize(size, Image.Resampling.LANCZOS)
        else:
            # DUSt3R-style preprocessing
            # 1. Resize long edge to resolution
            img = _resize_pil_image(img, resolution)

            # 2. Crop to patch_size-aligned dimensions
            W, H = img.size
            cx, cy = W // 2, H // 2

            halfw = ((2 * cx) // patch_size) * patch_size / 2
            halfh = ((2 * cy) // patch_size) * patch_size / 2

            # 3. Force 4:3 ratio unless square_ok
            if not square_ok and W == H:
                halfh = 3 * halfw / 4

            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    return np.array(img)


def resize_image(image: np.ndarray, resolution: int) -> np.ndarray:
    """Resize image to target resolution.

    Args:
        image: [H, W, 3] uint8 numpy array
        resolution: Target resolution (square)

    Returns:
        [resolution, resolution, 3] uint8 numpy array
    """
    img = Image.fromarray(image)
    img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
    return np.array(img)


def load_images(paths: list[str | Path], resolution: int | None = None) -> list[np.ndarray]:
    """Load multiple images.

    Args:
        paths: List of image paths
        resolution: Target resolution (square). If None, keep original size.

    Returns:
        List of [H, W, 3] uint8 numpy arrays
    """
    return [load_image(p, resolution) for p in paths]
