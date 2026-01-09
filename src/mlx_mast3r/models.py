"""High-level model APIs.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np

from mlx_mast3r.encoders import DuneEncoder, DuneConfig, Mast3rEncoder, Mast3rEncoderConfig
from mlx_mast3r.encoders.dune import DuneEncoderEngine
from mlx_mast3r.encoders.mast3r import Mast3rEncoderEngine
from mlx_mast3r.decoders.dunemast3r import DuneMast3rEngine
from mlx_mast3r.decoders.mast3r import Mast3rDecoderEngine


class DUNE:
    """DUNE encoder for fast feature extraction.

    Variants:
    - small: 384 dim, 11ms @ 336 (90 FPS)
    - base: 768 dim, 32ms @ 336 (31 FPS)

    Example:
        >>> model = DUNE.from_pretrained("base", resolution=336)
        >>> features = model.encode(image)
    """

    def __init__(
        self,
        variant: Literal["small", "base"] = "base",
        resolution: int = 336,
        precision: Literal["fp32", "fp16", "bf16"] = "fp16",
    ):
        self.engine = DuneEncoderEngine(
            variant=variant,
            resolution=resolution,
            precision=precision,
            compile=True,
        )
        self.variant = variant
        self.resolution = resolution

    @classmethod
    def from_pretrained(
        cls,
        variant: Literal["small", "base"] = "base",
        resolution: int = 336,
        precision: str = "fp16",
        cache_dir: str | Path | None = None,
    ) -> "DUNE":
        """Load pretrained DUNE model."""
        model = cls(variant=variant, resolution=resolution, precision=precision)

        if cache_dir is None:
            cache_dir = Path.home() / ".cache/mlx-mast3r"
        cache_dir = Path(cache_dir)

        weights_path = cache_dir / f"dune_vit_{variant}_{resolution}" / "encoder.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found at {weights_path}. "
                f"Download from: https://download.europe.naverlabs.com/dune/"
            )

        model.engine.load(weights_path)
        return model

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode image to features.

        Args:
            image: [H, W, 3] uint8 image

        Returns:
            [N, D] features
        """
        features, _ = self.engine.infer(image)
        return features

    def encode_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Encode batch of images."""
        return [self.encode(img) for img in images]

    @property
    def embed_dim(self) -> int:
        return self.engine.config.embed_dim

    @property
    def num_patches(self) -> int:
        return self.engine.config.num_patches


class Mast3r:
    """MASt3R encoder for high-quality feature extraction.

    Specs:
    - ViT-Large: 1024 dim, 183ms @ 512 (5.5 FPS)
    - Best 3D reconstruction quality

    Example:
        >>> model = Mast3r.from_pretrained(resolution=512)
        >>> features = model.encode(image)
    """

    def __init__(
        self,
        resolution: int = 512,
        precision: Literal["fp32", "fp16", "bf16"] = "fp16",
    ):
        self.engine = Mast3rEncoderEngine(
            resolution=resolution,
            precision=precision,
            compile=True,
        )
        self.resolution = resolution

    @classmethod
    def from_pretrained(
        cls,
        resolution: int = 512,
        precision: str = "fp16",
        cache_dir: str | Path | None = None,
    ) -> "Mast3r":
        """Load pretrained MASt3R model."""
        model = cls(resolution=resolution, precision=precision)

        if cache_dir is None:
            cache_dir = Path.home() / ".cache/mlx-mast3r"
        cache_dir = Path(cache_dir)

        weights_path = cache_dir / "mast3r_vit_large" / "unified.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found at {weights_path}. "
                f"Download from HuggingFace: naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
            )

        model.engine.load(weights_path)
        return model

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode image to features."""
        features, _ = self.engine.infer(image)
        return features

    @property
    def embed_dim(self) -> int:
        return self.engine.config.embed_dim

    @property
    def num_patches(self) -> int:
        return self.engine.config.num_patches


class DuneMast3r:
    """DuneMASt3R: DUNE encoder + MASt3R decoder for fast 3D reconstruction.

    Pipeline:
    1. DUNE encoder extracts features (11-32ms)
    2. MASt3R decoder predicts 3D points + descriptors

    Example:
        >>> model = DuneMast3r.from_pretrained("base", resolution=336)
        >>> out1, out2 = model.reconstruct(img1, img2)
        >>> pts3d = out1["pts3d"]  # [H, W, 3]
    """

    def __init__(
        self,
        encoder_variant: Literal["small", "base"] = "base",
        resolution: int = 336,
        precision: Literal["fp32", "fp16", "bf16"] = "fp16",
    ):
        self.engine = DuneMast3rEngine(
            encoder_variant=encoder_variant,
            resolution=resolution,
            precision=precision,
            compile=True,
        )
        self.variant = encoder_variant
        self.resolution = resolution

    @classmethod
    def from_pretrained(
        cls,
        encoder_variant: Literal["small", "base"] = "base",
        resolution: int = 336,
        precision: str = "fp16",
        cache_dir: str | Path | None = None,
    ) -> "DuneMast3r":
        """Load pretrained DuneMASt3R model."""
        model = cls(
            encoder_variant=encoder_variant,
            resolution=resolution,
            precision=precision,
        )

        if cache_dir is None:
            cache_dir = Path.home() / ".cache/mlx-mast3r"
        cache_dir = Path(cache_dir)

        encoder_path = (
            cache_dir / f"dune_vit_{encoder_variant}_{resolution}" / "encoder.safetensors"
        )
        decoder_path = cache_dir / f"dunemast3r_cvpr25_vit{encoder_variant}.pth"

        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder weights not found at {encoder_path}")
        if not decoder_path.exists():
            raise FileNotFoundError(f"Decoder weights not found at {decoder_path}")

        model.engine.load(encoder_path, decoder_path)
        return model

    def reconstruct(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
    ) -> tuple[dict, dict]:
        """Reconstruct 3D from stereo pair.

        Args:
            img1, img2: [H, W, 3] uint8 images

        Returns:
            (output1, output2) dicts with:
            - pts3d: [H, W, 3] 3D points
            - conf: [H, W, 1] confidence
            - desc: [H, W, 24] descriptors
        """
        out1, out2, _ = self.engine.infer(img1, img2)
        return out1, out2

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode single image (for retrieval)."""
        x = image.astype(np.float32) / 127.5 - 1.0
        x = mx.array(x[None, :, :, :])

        feat = self.engine.encoder(x)
        mx.eval(feat)
        return np.array(feat[0])


class Mast3rFull:
    """Full MASt3R pipeline: encoder + decoder for highest quality 3D.

    Pipeline:
    1. MASt3R ViT-Large encoder extracts features (183ms)
    2. MASt3R decoder predicts 3D points + descriptors

    Best quality but slower than DuneMASt3R.

    Example:
        >>> model = Mast3rFull.from_pretrained(resolution=512)
        >>> out1, out2 = model.reconstruct(img1, img2)
        >>> pts3d = out1["pts3d"]  # [H, W, 3]
    """

    def __init__(
        self,
        resolution: int = 512,
        precision: Literal["fp32", "fp16", "bf16"] = "fp16",
    ):
        self.engine = Mast3rDecoderEngine(
            resolution=resolution,
            precision=precision,
            compile=True,
        )
        self.resolution = resolution

    @classmethod
    def from_pretrained(
        cls,
        resolution: int = 512,
        precision: str = "fp16",
        cache_dir: str | Path | None = None,
    ) -> "Mast3rFull":
        """Load pretrained full MASt3R model."""
        model = cls(resolution=resolution, precision=precision)

        if cache_dir is None:
            cache_dir = Path.home() / ".cache/mlx-mast3r"
        cache_dir = Path(cache_dir)

        weights_path = cache_dir / "mast3r_vit_large" / "unified.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found at {weights_path}. "
                f"Download from HuggingFace: naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
            )

        model.engine.load(weights_path)
        return model

    def reconstruct(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
    ) -> tuple[dict, dict]:
        """Reconstruct 3D from stereo pair.

        Args:
            img1, img2: [H, W, 3] uint8 images

        Returns:
            (output1, output2) dicts with:
            - pts3d: [H, W, 3] 3D points
            - conf: [H, W, 1] confidence
            - desc: [H, W, 24] descriptors
        """
        out1, out2, _ = self.engine.infer(img1, img2)
        return out1, out2

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode single image to features."""
        x = image.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = mx.array(x[None, :, :, :])

        feat = self.engine.encoder(x)
        mx.eval(feat)
        return np.array(feat[0])

    @property
    def embed_dim(self) -> int:
        return self.engine.encoder_config.embed_dim

    @property
    def num_patches(self) -> int:
        return self.engine.encoder_config.num_patches
