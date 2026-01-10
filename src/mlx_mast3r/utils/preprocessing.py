"""Image preprocessing utilities.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(
    path: str | Path,
    resolution: int | tuple[int, int] | None = None,
) -> np.ndarray:
    """Load image from path and optionally resize.

    Args:
        path: Path to image file
        resolution: Target resolution. Can be:
            - int: square resolution (H=W=resolution)
            - tuple (H, W): specific height and width
            - None: keep original size

    Returns:
        [H, W, 3] uint8 numpy array
    """
    img = Image.open(path).convert("RGB")

    if resolution is not None:
        if isinstance(resolution, int):
            size = (resolution, resolution)
        else:
            # PIL resize takes (W, H), but we use (H, W) convention
            size = (resolution[1], resolution[0])
        img = img.resize(size, Image.Resampling.LANCZOS)

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
