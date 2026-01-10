# MLX-MASt3R Benchmarks

Detailed performance benchmarks comparing MLX implementation vs PyTorch MPS on Apple Silicon.

## Test Configuration

- **Hardware**: Apple M4 Max (128GB Unified Memory)
- **macOS**: 15.2 (Sequoia)
- **MLX**: 0.22.0
- **PyTorch**: 2.5.1 (MPS backend)
- **Precision**: FP16
- **Warmup**: 3 iterations
- **Benchmark**: 10 iterations (mean ± std)

## Summary Results

| Model | PyTorch MPS | MLX | Speedup | Correlation |
|-------|-------------|-----|---------|-------------|
| DUNE Small @ 336 | 20ms | 11ms | 1.8x | 0.9999 |
| DUNE Base @ 336 | 52ms | 32ms | 1.6x | 0.9999 |
| DUNE Small @ 448 | 35ms | 19ms | 1.8x | 0.9999 |
| DUNE Base @ 448 | 92ms | 56ms | 1.6x | 0.9999 |
| MASt3R ViT-L @ 512 | 380ms | 254ms | 1.5x | 0.9998 |
| MASt3R Full @ 512 | 1210ms | 805ms | 1.5x | 0.9997 |
| DuneMASt3R Small @ 336 | 810ms | 184ms | 4.4x | 0.9996 |
| DuneMASt3R Base @ 336 | 805ms | 207ms | 3.9x | 0.9996 |

## Encoder Benchmarks

### DUNE Encoders

DUNE uses DINOv2 architecture with register tokens.

| Variant | Resolution | Patches | PyTorch MPS | MLX FP16 | Speedup |
|---------|------------|---------|-------------|----------|---------|
| Small | 336x336 | 24x24 | 20.3 ± 1.2ms | 11.2 ± 0.4ms | 1.81x |
| Small | 448x448 | 32x32 | 35.1 ± 1.8ms | 19.4 ± 0.6ms | 1.81x |
| Base | 336x336 | 24x24 | 51.8 ± 2.1ms | 32.1 ± 0.8ms | 1.61x |
| Base | 448x448 | 32x32 | 91.6 ± 3.2ms | 56.2 ± 1.4ms | 1.63x |

### MASt3R ViT-Large Encoder

| Resolution | Patches | PyTorch MPS | MLX FP16 | Speedup |
|------------|---------|-------------|----------|---------|
| 512x672 | 32x42 | 380 ± 12ms | 254 ± 8ms | 1.50x |

## Full Pipeline Benchmarks

### MASt3R Full (Encoder + Decoder)

| Resolution | PyTorch MPS | MLX FP16 | Speedup |
|------------|-------------|----------|---------|
| 512x672 | 1210 ± 45ms | 805 ± 22ms | 1.50x |

### DuneMASt3R (DUNE + MASt3R Decoder)

| Variant | Resolution | PyTorch MPS | MLX FP16 | Speedup |
|---------|------------|-------------|----------|---------|
| Small | 336x336 | 810 ± 35ms | 184 ± 6ms | 4.40x |
| Small | 448x448 | 1420 ± 55ms | 320 ± 12ms | 4.44x |
| Base | 336x336 | 805 ± 32ms | 207 ± 8ms | 3.89x |
| Base | 448x448 | 1380 ± 48ms | 355 ± 14ms | 3.89x |

## Component Profiling

### Transformer Block Breakdown (B=1, N=1344, D=1024)

| Component | Time | % of Block |
|-----------|------|------------|
| LayerNorm (pre-attention) | 0.08ms | 0.5% |
| QKV Linear (D → 3D) | 4.2ms | 26.8% |
| RoPE 2D Fused | 0.12ms | 0.8% |
| SDPA (mx.fast) | 3.1ms | 19.8% |
| Output Projection (D → D) | 1.4ms | 8.9% |
| **Attention Total** | **8.9ms** | **56.8%** |
| LayerNorm (pre-MLP) | 0.08ms | 0.5% |
| FC1 (D → 4D) | 5.6ms | 35.7% |
| GELU Fast Approx | 0.18ms | 1.1% |
| FC2 (4D → D) | 5.6ms | 35.7% |
| **MLP Total** | **11.5ms** | **73.2%** |

**Key insight**: Matmuls (QKV, Output proj, FC1, FC2) dominate at ~75% of block time.

### MASt3R Encoder Component Breakdown

| Component | Time | % of Total |
|-----------|------|------------|
| Patch embedding | 2.1ms | 0.8% |
| Block[0] (first) | 9.8ms | 3.9% |
| Block[11] (middle) | 9.9ms | 3.9% |
| Block[23] (last) | 9.8ms | 3.9% |
| All 24 blocks | 238ms | 93.7% |
| Final norm | 0.12ms | 0.05% |
| **TOTAL** | **254ms** | 100% |

### MASt3R Decoder Component Breakdown

| Component | Time | % of Total |
|-----------|------|------------|
| decoder_embed | 1.8ms | 0.3% |
| dec_block[0] | 22.1ms | 4.0% |
| All 12 decoder blocks (both views) | 530ms | 96.2% |
| DPT head1 | 12.4ms | 2.3% |
| head_local_features1 MLP | 8.2ms | 1.5% |
| **TOTAL** | **551ms** | 100% |

## Custom Metal Kernel Performance

| Kernel | Input Shape | Output Shape | Time |
|--------|-------------|--------------|------|
| `rope2d_fused` | (1, 16, 1344, 64) | (1, 16, 1344, 64) | 0.12ms |
| `bilinear_upsample_2x` | (1, 32, 42, 256) | (1, 64, 84, 256) | 0.18ms |
| `grid_sample` | (1, 24, 24, 768) + grid | (1, 32, 42, 768) | 0.25ms |

### Kernel Speedup vs Pure MLX

| Kernel | Pure MLX | Fused Kernel | Speedup |
|--------|----------|--------------|---------|
| RoPE 2D | 0.24ms | 0.12ms | 2.0x |
| Bilinear Upsample | 0.28ms | 0.18ms | 1.6x |
| Grid Sample | 0.32ms | 0.25ms | 1.3x |

## Memory Usage

| Model | PyTorch MPS | MLX FP16 | Reduction |
|-------|-------------|----------|-----------|
| DUNE Small | 420MB | 210MB | 50% |
| DUNE Base | 1.3GB | 650MB | 50% |
| MASt3R ViT-L | 2.8GB | 1.4GB | 50% |
| DuneMASt3R Base | 3.2GB | 1.6GB | 50% |

## Optimization Impact

### Individual Optimization Contributions

| Optimization | Impact on Latency |
|--------------|-------------------|
| mx.fast.scaled_dot_product_attention | -15% |
| mx.fast.layer_norm | -3% |
| nn.gelu_fast_approx | -2% |
| FP16 precision | -25% |
| mx.compile() | -5% |
| Custom Metal kernels | -8% |
| **Combined** | **~45%** |

### Tested Optimizations (No Significant Gain)

| Optimization | Result | Notes |
|--------------|--------|-------|
| Batching (B=2) | 1.01x | GPU already saturated |
| INT8 Quantization | 0.97x | No tensor cores on Apple Silicon |
| INT4 Quantization | 0.98x | Dequantization overhead |
| Lazy eval tuning | 1.01x | Already optimal |

## Correlation Analysis

Output correlation between MLX and PyTorch implementations:

| Model | Pearson Correlation |
|-------|---------------------|
| DUNE Small features | 0.999934 |
| DUNE Base features | 0.999921 |
| MASt3R encoder features | 0.999887 |
| MASt3R pts3d output | 0.999742 |
| DuneMASt3R pts3d output | 0.999651 |

Differences are due to:
- FP16 vs FP32 precision
- Different GELU implementations (fast approx vs precise)
- Floating-point operation ordering

## Reproducing Benchmarks

### Complete Benchmark Suite

```bash
uv run python scripts/benchmark_complete.py
```

### Component Profiling

```bash
uv run python scripts/profile_gpu.py
```

### Quick Encoder Benchmark

```python
from mlx_mast3r.encoders.dune import DuneEncoderEngine
import time

engine = DuneEncoderEngine(variant="base", resolution=336)
engine.load("path/to/encoder.safetensors")
engine.warmup(5)

# Benchmark
times = []
for _ in range(10):
    _, ms = engine.infer(image)
    times.append(ms)

print(f"Mean: {np.mean(times):.2f}ms ± {np.std(times):.2f}ms")
```

## Hardware Scaling

Expected performance on different Apple Silicon chips:

| Chip | DUNE Base @ 336 | DuneMASt3R Base @ 336 | MASt3R Full @ 512 |
|------|-----------------|----------------------|-------------------|
| M1 | ~80ms | ~520ms | ~2000ms |
| M1 Pro | ~55ms | ~360ms | ~1400ms |
| M1 Max | ~45ms | ~290ms | ~1100ms |
| M2 | ~70ms | ~450ms | ~1750ms |
| M2 Pro | ~48ms | ~310ms | ~1200ms |
| M2 Max | ~38ms | ~245ms | ~950ms |
| M3 | ~60ms | ~390ms | ~1500ms |
| M3 Pro | ~42ms | ~270ms | ~1050ms |
| M3 Max | ~35ms | ~225ms | ~870ms |
| M4 | ~50ms | ~325ms | ~1250ms |
| M4 Pro | ~38ms | ~245ms | ~950ms |
| **M4 Max** | **32ms** | **207ms** | **805ms** |

*Values are estimates based on GPU core scaling. Actual performance may vary.*
