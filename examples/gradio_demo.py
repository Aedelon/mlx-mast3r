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

import gradio as gr

from gradio_tabs import (
    create_about_tab,
    create_features_tab,
    create_multiview_tab,
    create_stereo_tab,
)


def create_demo() -> gr.Blocks:
    """Create unified Gradio demo.

    Returns:
        Gradio Blocks application
    """
    with gr.Blocks(title="MLX-MASt3R Demo") as demo:
        gr.HTML(
            """
            <div style="text-align: center; padding: 20px;">
                <h1>MLX-MASt3R</h1>
                <p style="font-size: 1.2em;">Ultra-fast 3D reconstruction on Apple Silicon</p>
            </div>
            """
        )

        with gr.Tabs():
            # Tab 1: Feature Extraction
            with gr.TabItem("DUNE Features"):
                create_features_tab()

            # Tab 2: Stereo Reconstruction
            with gr.TabItem("Stereo (2 views)"):
                create_stereo_tab()

            # Tab 3: Multi-view Reconstruction
            with gr.TabItem("Multi-View (N images)"):
                create_multiview_tab()

            # Tab 4: About
            with gr.TabItem("About"):
                create_about_tab()

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False, server_port=7860)
