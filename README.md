# MLX-MASt3R

Ultra-optimized MLX implementation for 3D reconstruction on Apple Silicon.

## Models

| Model | Encoder | Decoder | Speed (M4 Max) | Use Case |
|-------|---------|---------|----------------|----------|
| **DUNE Small** | DINOv2-S (384d) | - | 11ms (90 FPS) | Fast encoding |
| **DUNE Base** | DINOv2-B (768d) | - | 32ms (31 FPS) | Quality encoding |
| **DuneMASt3R** | DUNE | MASt3R | ~50ms | Real-time 3D |
| **MASt3R Full** | ViT-L (1024d) | MASt3R | ~200ms | Best quality |

## Installation

```bash
uv add mlx-mast3r
```

## Usage

### DUNE Encoder (Fast)

```python
from mlx_mast3r import DUNE

# Load model
model = DUNE.from_pretrained("base", resolution=336)

# Encode image
features = model.encode(image)  # [N, 768]
```

### DuneMASt3R (Real-time 3D)

```python
from mlx_mast3r import DuneMast3r

# Load model
model = DuneMast3r.from_pretrained("base", resolution=336)

# Reconstruct 3D from stereo pair
out1, out2 = model.reconstruct(img1, img2)
pts3d = out1["pts3d"]  # [H, W, 3]
```

### Full MASt3R (Best Quality)

```python
from mlx_mast3r import Mast3rFull

# Load model
model = Mast3rFull.from_pretrained(resolution=512)

# Reconstruct 3D
out1, out2 = model.reconstruct(img1, img2)
```

## Optimizations

- `mx.fast.scaled_dot_product_attention` (fused SDPA)
- `mx.fast.layer_norm` (fused LayerNorm)
- `mx.compile()` graph compilation
- FP16/BF16 precision support
- 2D RoPE for positional encoding

## License

Apache 2.0

## Credits

- MASt3R: [naver/mast3r](https://github.com/naver/mast3r)
- DUNE: [naver/dune](https://github.com/naver/dune)