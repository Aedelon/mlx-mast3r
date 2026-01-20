# Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""About tab for Gradio demo."""

from __future__ import annotations

import gradio as gr


def create_about_tab() -> None:
    """Create the About tab content."""
    gr.Markdown(
        """
        ## MLX-MASt3R

        MLX implementation optimized for Apple Silicon of the following models:

        ### Available Models

        | Model | Encoder | Resolution | Time | Use Case |
        |-------|---------|------------|------|----------|
        | **DUNE Small** | ViT-S | 336 | ~11ms | Fast features |
        | **DUNE Base** | ViT-B | 336 | ~32ms | Quality features |
        | **DuneMASt3R Small** | DUNE-S + MASt3R | 336 | ~50ms | Fast reconstruction |
        | **DuneMASt3R Base** | DUNE-B + MASt3R | 448 | ~90ms | Balanced reconstruction |
        | **MASt3R Full** | ViT-L | 512 | ~200ms | Best quality |

        ### Reconstruction Modes

        | Mode | Images | Description |
        |------|--------|-------------|
        | **Stereo** | 2 | Fast reconstruction between two views |
        | **Multi-View** | 3+ | Global alignment with pose optimization |

        ### Scene Graph

        | Type | Description |
        |------|-------------|
        | **complete** | All possible pairs (N*(N-1)/2) |
        | **swin** | Sliding window of size K |
        | **logwin** | Logarithmic window (powers of 2) |
        | **oneref** | One reference image with all others |
        | **retrieval** | Automatic selection by visual similarity |

        #### Retrieval Mode
        Uses a pre-trained retrieval model to compute similarity
        between images and automatically select the best pairs.
        Ideal for large unordered image collections.

        ### Credits

        - **MASt3R/DUSt3R**: [Naver Labs](https://github.com/naver/mast3r) (CC BY-NC-SA 4.0)
        - **DUNE**: [Facebook Research](https://github.com/facebookresearch/dune) (Apache 2.0)
        - **MLX**: [Apple](https://github.com/ml-explore/mlx)

        ### Author

        Copyright (c) 2025 Delanoe Pirard / Aedelon - Apache 2.0 License
        """
    )
