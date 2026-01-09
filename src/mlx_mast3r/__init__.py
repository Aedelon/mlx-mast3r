"""MLX-MASt3R: Ultra-optimized MLX implementation for 3D reconstruction.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.

Supported models:
- MASt3R ViT-Large encoder + decoder (full pipeline)
- DUNE ViT-Small/Base encoders
- DuneMASt3R decoder (DUNE encoder + MASt3R decoder)
"""

__version__ = "0.1.0"
__author__ = "Delanoe Pirard"

from mlx_mast3r.encoders import DuneEncoder, Mast3rEncoder
from mlx_mast3r.decoders import DuneMast3rDecoder, Mast3rDecoder
from mlx_mast3r.models import DUNE, DuneMast3r, Mast3r, Mast3rFull

__all__ = [
    "DuneEncoder",
    "Mast3rEncoder",
    "DuneMast3rDecoder",
    "Mast3rDecoder",
    "DUNE",
    "DuneMast3r",
    "Mast3r",
    "Mast3rFull",
]
