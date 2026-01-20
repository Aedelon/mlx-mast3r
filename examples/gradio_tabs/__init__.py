# Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""Gradio tab modules for MLX-MASt3R demo."""

from .about import create_about_tab
from .features import create_features_tab
from .multiview import create_multiview_tab
from .stereo import create_stereo_tab

__all__ = [
    "create_about_tab",
    "create_features_tab",
    "create_multiview_tab",
    "create_stereo_tab",
]
