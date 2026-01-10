# MLX-MASt3R

Ultra-optimized MLX implementation of MASt3R and DuneMASt3R for 3D reconstruction on Apple Silicon.

## Features

- **Native Apple Silicon**: Optimized for M1/M2/M3/M4 chips using MLX
- **Real-time Performance**: Up to 4.4x faster than PyTorch MPS
- **Multiple Models**: MASt3R ViT-L, DUNE Small/Base, DuneMASt3R
- **Custom Metal Kernels**: Fused RoPE 2D, bilinear upsample, grid sample
- **FP16/BF16 Support**: Reduced memory footprint with minimal quality loss

## Performance (M4 Max)

| Model | Resolution | Latency | FPS | Speedup vs PyTorch |
|-------|------------|---------|-----|-------------------|
| **DUNE Small** | 336x336 | 11ms | 90 | 1.8x |
| **DUNE Base** | 336x336 | 32ms | 31 | 1.6x |
| **DuneMASt3R Small** | 336x336 | 184ms | 5.4 | 4.4x |
| **DuneMASt3R Base** | 336x336 | 207ms | 4.8 | 3.9x |
| **MASt3R Full** | 512x672 | 805ms | 1.2 | 1.5x |

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for detailed benchmarks.

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

## API Reference

### Models Overview

| Class | Use Case | Speed | Quality |
|-------|----------|-------|---------|
| `DUNE` | Feature extraction | 11-32ms | Good |
| `Mast3r` | Feature extraction | 183ms | Best |
| `DuneMast3r` | 3D reconstruction | 184-207ms | Good |
| `Mast3rFull` | 3D reconstruction | 805ms | Best |

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
│   ├── kernels/           # Custom Metal kernels
│   │   ├── rope2d.py      # Fused 2D RoPE
│   │   ├── bilinear.py    # Fused bilinear upsample
│   │   └── grid_sample.py # Grid sampling
│   └── models.py          # High-level API
├── scripts/
│   ├── benchmark_complete.py  # MLX vs PyTorch benchmarks
│   └── profile_gpu.py         # Component profiling
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

Weights are automatically downloaded from HuggingFace Hub:

| Model | Hub Path | Size |
|-------|----------|------|
| DUNE Small | `aedelon/dune-vit-small` | 85MB |
| DUNE Base | `aedelon/dune-vit-base` | 330MB |
| MASt3R ViT-L | `aedelon/mast3r-vit-large` | 1.2GB |

### Manual Download

```bash
# Download to cache directory
python -m mlx_mast3r.download --model dune-base --resolution 336
```

## Benchmarking

### Run Complete Benchmark

```bash
uv run python scripts/benchmark_complete.py
```

### Profile GPU Components

```bash
uv run python scripts/profile_gpu.py
```

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
