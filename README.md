# MLX-MASt3R

Ultra-optimized MLX implementation of MASt3R and DuneMASt3R for 3D reconstruction on Apple Silicon.

## Features

- **Native Apple Silicon**: Optimized for M1/M2/M3/M4 chips using MLX
- **Real-time Performance**: Up to 1.87× faster than PyTorch MPS (avg 1.59×)
- **Multiple Models**: MASt3R ViT-L, DUNE Small/Base, DuneMASt3R
- **Custom Metal Kernels**: Fused RoPE 2D, bilinear upsample, grid sample
- **FP16/BF16 Support**: Reduced memory footprint with minimal quality loss

## Performance (M4 Max)

### MLX vs PyTorch MPS Benchmark

| Model | Resolution | PyTorch MPS | MLX FP16 | Speedup | FPS |
|-------|------------|-------------|----------|---------|-----|
| **DUNE Small** | 336×336 | 13.1ms | 8.6ms | **1.54×** | 117 |
| **DUNE Small** | 448×448 | 28.9ms | 15.4ms | **1.87×** | 65 |
| **DUNE Base** | 336×336 | 36.0ms | 24.8ms | **1.45×** | 40 |
| **DUNE Base** | 448×448 | 75.1ms | 43.4ms | **1.73×** | 23 |
| **MASt3R Encoder** | 512×672 | 331.7ms | 184.4ms | **1.80×** | 5.4 |
| **MASt3R Full** | 512×672 | 1203.7ms | 767.1ms | **1.57×** | 1.3 |
| **DuneMASt3R Small** | 336×336 | 229.7ms | 145.4ms | **1.58×** | 6.9 |
| **DuneMASt3R Small** | 448×448 | 432.0ms | 257.1ms | **1.68×** | 3.9 |
| **DuneMASt3R Base** | 336×336 | 283.0ms | 183.5ms | **1.54×** | 5.5 |
| **DuneMASt3R Base** | 448×448 | 566.7ms | 504.5ms | **1.12×** | 2.0 |

**Average speedup: 1.59×** faster than PyTorch MPS

> Benchmarked on MacBook Pro M4 Max, 10 iterations after 10 warmup runs.
> Run `uv run python scripts/benchmark_complete.py` to reproduce.

## Installation

```bash
# With uv (recommended)
uv add mlx-mast3r

# With pip
pip install mlx-mast3r
```

### From Source

```bash
git clone https://github.com/aedelon/mlx-mast3r.git
cd mlx-mast3r
uv sync
```

### With Benchmarks (PyTorch comparison)

```bash
# Clone with submodules for PyTorch reference implementations
git clone --recurse-submodules https://github.com/aedelon/mlx-mast3r.git
cd mlx-mast3r

# Install with benchmark dependencies (torch, timm, etc.)
uv sync --extra benchmark

# Run benchmarks
uv run python scripts/benchmark_complete.py
```

## Quick Start

### DuneMASt3R (Recommended for Real-time)

```python
from mlx_mast3r import DuneMast3r

# Load model (downloads weights automatically)
model = DuneMast3r.from_pretrained("base", resolution=336)

# Reconstruct 3D from stereo pair
out1, out2 = model.reconstruct(img1, img2)

# Access outputs
pts3d = out1["pts3d"]      # [H, W, 3] - 3D points
conf = out1["conf"]        # [H, W] - confidence map
desc = out1["desc"]        # [H, W, 24] - descriptors
```

### DUNE Encoder (Fast Feature Extraction)

```python
from mlx_mast3r import DUNE

# Load encoder
encoder = DUNE.from_pretrained("base", resolution=336)

# Extract features
features = encoder.encode(image)  # [N, 768]
```

### MASt3R Full (Best Quality)

```python
from mlx_mast3r import Mast3rFull

# Load full MASt3R pipeline
model = Mast3rFull.from_pretrained(resolution=512)

# Reconstruct 3D
out1, out2 = model.reconstruct(img1, img2)
```

### Multi-View Reconstruction (3+ Images)

For complete scene reconstruction from multiple images with camera pose optimization:

```python
from mlx_mast3r import DuneMast3r
from mlx_mast3r.cloud_opt import sparse_global_alignment
from mlx_mast3r.image_pairs import make_pairs
from mlx_mast3r.utils import load_image
import mlx.core as mx

# Load model
model = DuneMast3r.from_pretrained(encoder_variant="base", resolution=336)

# Load images
images = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
resolution = 336

imgs_data = []
for idx, path in enumerate(images):
    img = load_image(path, resolution=resolution)
    imgs_data.append({
        "img": mx.array(img).transpose(2, 0, 1)[None],
        "true_shape": img.shape[:2],
        "idx": idx,
        "instance": path,
    })

# Build pairs (complete graph for small sets)
pairs = make_pairs(imgs_data, scene_graph="complete", symmetrize=True)

# Run global alignment
result = sparse_global_alignment(
    imgs=images,
    pairs_in=pairs,
    cache_path="/tmp/cache",
    model=model,
    lr1=0.07,      # Coarse phase learning rate
    niter1=300,    # Coarse phase iterations
    lr2=0.01,      # Fine phase learning rate
    niter2=300,    # Fine phase iterations
)

# Access results
poses = result.get_im_poses()      # [N, 4, 4] camera-to-world matrices
focals = result.get_focals()       # [N] focal lengths
pts3d, depths, confs = result.get_dense_pts3d()  # Dense reconstruction
```

### Retrieval-Based Pair Selection

For large image sets, use retrieval to automatically select the best pairs:

```python
from mlx_mast3r import Mast3rFull, make_pairs_retrieval, RetrievalModel

# Load models
model = Mast3rFull.from_pretrained(resolution=512)
retrieval = RetrievalModel.from_pretrained()

# Select pairs using visual similarity
pairs_indices = make_pairs_retrieval(
    retrieval=retrieval,
    backbone=model,
    images=images_np,  # List of numpy images
    na=20,             # Number of adjacent candidates
    k=10,              # Pairs per image
)
# Returns list of (i, j) tuples
```

## Gradio Demo

Interactive web interface with stereo and multi-view reconstruction:

```bash
# Install demo dependencies
uv sync --extra demo

# Launch demo
uv run python examples/gradio_demo.py
```

Open http://localhost:7860 in your browser. Features:
- **DUNE Features**: Extract and visualize feature maps
- **Stereo (2 views)**: Quick reconstruction from image pairs
- **Multi-View (N images)**: Full scene reconstruction with optimization

## Examples

Command-line demos are available in `examples/`:

```bash
# DUNE feature extraction
uv run python examples/demo_dune.py

# DuneMASt3R stereo reconstruction
uv run python examples/demo_dunemast3r.py

# MASt3R full pipeline
uv run python examples/demo_mast3r.py
```

## API Reference

### Models Overview

| Class | Use Case | Speed (MLX) | Quality |
|-------|----------|-------------|---------|
| `DUNE` | Feature extraction | 9-43ms | Good |
| `Mast3r` | Feature extraction | 184ms | Best |
| `DuneMast3r` | 3D reconstruction | 145-257ms | Good |
| `Mast3rFull` | 3D reconstruction | 767ms | Best |

### DUNE

Fast feature encoder based on DINOv2.

```python
from mlx_mast3r import DUNE

# Load model
model = DUNE.from_pretrained(
    variant="base",      # "small" (384d) or "base" (768d)
    resolution=336,      # 336 or 448
    precision="fp16",    # "fp32", "fp16", "bf16"
)

# Encode image
features = model.encode(image)           # [N, D] numpy array
features = model.encode_batch([img1, img2])  # List of arrays

# Properties
model.embed_dim    # 384 (small) or 768 (base)
model.num_patches  # 576 (336) or 1024 (448)
```

### Mast3r

MASt3R ViT-Large encoder for highest quality features.

```python
from mlx_mast3r import Mast3r

model = Mast3r.from_pretrained(
    resolution=512,      # Image height (width = 4:3 ratio)
    precision="fp16",
)

features = model.encode(image)  # [N, 1024] numpy array
```

### DuneMast3r

DUNE encoder + MASt3R decoder for fast 3D reconstruction.

```python
from mlx_mast3r import DuneMast3r

model = DuneMast3r.from_pretrained(
    encoder_variant="base",  # "small" or "base"
    resolution=336,
    precision="fp16",
)

# Stereo 3D reconstruction
out1, out2 = model.reconstruct(img1, img2)

# Output format
out1["pts3d"]  # [H, W, 3] - 3D points in camera space
out1["conf"]   # [H, W] - confidence map (0-1)
out1["desc"]   # [H, W, 24] - dense descriptors

# Single image encoding (for retrieval)
features = model.encode(image)  # [N, D]
```

### Mast3rFull

Full MASt3R pipeline for highest quality 3D reconstruction.

```python
from mlx_mast3r import Mast3rFull

model = Mast3rFull.from_pretrained(
    resolution=512,
    precision="fp16",
)

# Stereo 3D reconstruction
out1, out2 = model.reconstruct(img1, img2)

# Same output format as DuneMast3r
pts3d = out1["pts3d"]  # [H, W, 3]
```

### Input Format

All models expect:
- **Images**: `np.ndarray` of shape `[H, W, 3]`, dtype `uint8`, RGB order
- **Resolution**: Images are automatically resized to model resolution

### Output Format

Reconstruction outputs (`reconstruct()`):
- `pts3d`: 3D points in camera 1 coordinate system
- `conf`: Per-pixel confidence (higher = more reliable)
- `desc`: Dense descriptors for matching between views

### Scene Graph Types

Control which image pairs are processed in multi-view reconstruction:

| Type | Description | Use Case |
|------|-------------|----------|
| `complete` | All pairs (N×N) | Small sets (<10 images) |
| `swin-{k}` | Sliding window of size k | Sequential captures |
| `logwin-{k}` | Logarithmic window | Long sequences |
| `oneref-{id}` | One reference image | Panoramas |
| `retrieval-{na}-{k}` | Similarity-based selection | Large unordered sets |

```python
from mlx_mast3r.image_pairs import make_pairs

# Complete graph (default)
pairs = make_pairs(imgs_data, scene_graph="complete")

# Sliding window of 5
pairs = make_pairs(imgs_data, scene_graph="swin-5")

# Logarithmic window (good for video)
pairs = make_pairs(imgs_data, scene_graph="logwin-5-noncyclic")
```

### TSDF Post-Processing

Clean depth maps using Truncated Signed Distance Function:

```python
from mlx_mast3r.cloud_opt import TSDFPostProcess

# After sparse_global_alignment
processor = TSDFPostProcess(result, tsdf_thresh=0.05)
pts3d, depths, confs = processor.get_dense_pts3d(clean_depth=True)
```

## Architecture

```
mlx-mast3r/
├── src/mlx_mast3r/
│   ├── encoders/          # Vision encoders
│   │   ├── dune.py        # DUNE ViT-Small/Base
│   │   └── mast3r.py      # MASt3R ViT-Large
│   ├── decoders/          # 3D reconstruction decoders
│   │   ├── mast3r.py      # MASt3R decoder + DPT head
│   │   └── dunemast3r.py  # DUNE + MASt3R decoder
│   ├── cloud_opt/         # Multi-view optimization
│   │   ├── sparse_ga.py   # Sparse global alignment
│   │   ├── optimizer.py   # Scene optimizer (MST, poses, depths)
│   │   ├── geometry.py    # 3D geometry utilities
│   │   ├── losses.py      # Optimization losses
│   │   └── tsdf.py        # TSDF post-processing
│   ├── kernels/           # Custom Metal kernels
│   │   ├── rope2d.py      # Fused 2D RoPE
│   │   ├── bilinear.py    # Fused bilinear upsample
│   │   └── grid_sample.py # Grid sampling
│   ├── models.py          # High-level API
│   ├── retrieval.py       # Image pair retrieval
│   └── image_pairs.py     # Scene graph construction
├── examples/
│   ├── gradio_demo.py     # Interactive web demo
│   ├── demo_dune.py       # DUNE feature extraction
│   ├── demo_dunemast3r.py # DuneMASt3R stereo
│   └── demo_mast3r.py     # MASt3R full pipeline
├── scripts/
│   └── benchmark_complete.py  # MLX vs PyTorch benchmarks
└── docs/
    └── BENCHMARKS.md      # Detailed benchmarks
```

## Optimizations

### MLX Fast Operations

- `mx.fast.scaled_dot_product_attention` - Fused SDPA
- `mx.fast.layer_norm` - Fused LayerNorm
- `nn.gelu_fast_approx` - Fast GELU approximation
- `mx.compile()` - Graph compilation

### Custom Metal Kernels

| Kernel | Operation | Speedup |
|--------|-----------|---------|
| `rope2d_fused` | 2D Rotary Position Embedding | 2x |
| `bilinear_upsample_2x` | Bilinear upsampling | 1.5x |
| `grid_sample` | Differentiable grid sampling | 1.3x |

### Memory Optimizations

- FP16/BF16 precision (50% memory reduction)
- Lazy evaluation with strategic `mx.eval()` calls
- LRU cache for bilinear interpolation parameters

## Model Weights

Weights are **automatically downloaded** from HuggingFace Hub when calling `from_pretrained()`.

### HuggingFace Repositories (Recommended)

Pre-converted safetensors, ready to use:

| Model | Repository | Files |
|-------|------------|-------|
| MASt3R ViT-L | [Aedelon/mast3r-vit-large-fp16](https://huggingface.co/Aedelon/mast3r-vit-large-fp16) | `unified.safetensors` |
| DUNE/DuneMASt3R | [Aedelon/dunemast3r-models-fp16](https://huggingface.co/Aedelon/dunemast3r-models-fp16) | `encoder.safetensors`, `decoder.safetensors` |

### Manual Download (Python API)

```python
from mlx_mast3r.utils.download import (
    download_mast3r,
    download_dune,
    download_dunemast3r,
)

# Download MASt3R (safetensors)
mast3r_path = download_mast3r()

# Download DUNE encoder only
encoder_path = download_dune(variant="base", resolution=336)

# Download DuneMASt3R (encoder + decoder)
encoder_path, decoder_path = download_dunemast3r(variant="base", resolution=336)
```

### PTH Checkpoints (For Manual Conversion)

Original Naver checkpoints for custom conversion:

```python
from mlx_mast3r.utils.download import (
    download_dune_pth,
    download_dunemast3r_pth,
    download_mast3r_pth,
)

# Download DUNE PTH from Naver
dune_pth = download_dune_pth(variant="base", resolution=336)

# Download DuneMASt3R decoder PTH
decoder_pth = download_dunemast3r_pth(variant="base")

# Download MASt3R from Naver HF repo
mast3r_pth = download_mast3r_pth()
```

### Cache Location

All weights are cached in `~/.cache/mlx-mast3r/`.

## Benchmarking

Run the complete MLX vs PyTorch MPS benchmark:

```bash
uv run python scripts/benchmark_complete.py
```

This will test all models (DUNE, MASt3R, DuneMASt3R) with warmup and correlation validation.

## Requirements

- macOS 13.0+ (Ventura or later)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- MLX 0.22+

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Lint
uv run ruff check src/

# Format
uv run ruff format src/
```

## Troubleshooting

### MLX requires warmup

MLX compiles computation graphs on first execution. Always warm up models before benchmarking:

```python
# Warmup (10 iterations recommended)
for _ in range(10):
    _ = model.reconstruct(img1, img2)

# Now benchmark
```

### Memory issues on large images

Reduce resolution or use FP16 precision:

```python
model = DuneMast3r.from_pretrained(
    resolution=336,    # Lower resolution
    precision="fp16",  # Half precision
)
```

### Multi-view reconstruction fails

1. Ensure at least 2 images with overlap
2. Check `matching_conf_thr` (lower = more matches)
3. Use `complete` scene graph for small sets
4. Enable `verbose=True` to debug

### Weights not downloading

Manually download from HuggingFace:

```bash
# MASt3R
huggingface-cli download Aedelon/mast3r-vit-large-fp16 --local-dir ~/.cache/mlx-mast3r/mast3r

# DuneMASt3R
huggingface-cli download Aedelon/dunemast3r-models-fp16 --local-dir ~/.cache/mlx-mast3r/dunemast3r
```

### PyTorch benchmarks not running

Install benchmark dependencies:

```bash
uv sync --extra benchmark
```

## Citation

If you use MLX-MASt3R in your research, please cite:

```bibtex
@software{mlx_mast3r,
  author = {Pirard, Delanoe},
  title = {MLX-MASt3R: Ultra-optimized MLX implementation for 3D reconstruction},
  year = {2025},
  url = {https://github.com/aedelon/mlx-mast3r}
}
```

And the original papers:

```bibtex
@inproceedings{mast3r,
  title={MASt3R: Matching And Stereo 3D Reconstruction},
  author={Leroy, Vincent and Cabon, Yohann and Revaud, Jerome},
  booktitle={CVPR},
  year={2024}
}

@inproceedings{dune,
  title={DUNE: Dataset for Unified Novel View Estimation},
  author={...},
  booktitle={CVPR},
  year={2025}
}
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Credits

- [MASt3R](https://github.com/naver/mast3r) - Original PyTorch implementation
- [DUNE](https://github.com/naver/dune) - DUNE encoder
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
