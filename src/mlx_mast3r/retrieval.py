"""Image retrieval model for automatic pair selection.

Implements the retrieval model from MASt3R for computing image similarity
and selecting optimal image pairs for 3D reconstruction.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
Original dust3r code: Copyright (C) 2024-present Naver Corporation. CC BY-NC-SA 4.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download

if TYPE_CHECKING:
    from .models import Dune, DuneMast3r, Mast3r

# HuggingFace repo for retrieval weights
HF_REPO_ID = "Aedelon/mast3r-vit-large-fp16"
RETRIEVAL_FILENAME = "retrieval.safetensors"


class Whitener(nn.Module):
    """PCA whitening with centering.

    Applies: output = (x - m) @ p

    If l2norm is True, applies L2 normalization after PCA.
    """

    def __init__(self, dim: int, l2norm: bool = False):
        """Initialize whitener.

        Args:
            dim: Feature dimension
            l2norm: Whether to apply L2 normalization
        """
        super().__init__()
        self.m = mx.zeros((1, dim))  # Mean for centering
        self.p = mx.eye(dim)  # PCA transformation matrix
        self.l2norm = l2norm

    def __call__(self, x: mx.array) -> mx.array:
        """Apply whitening transformation.

        Args:
            x: Input features [..., dim]

        Returns:
            Whitened features [..., dim]
        """
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])

        # Center and transform
        x_centered = x_flat - self.m
        x_pca = x_centered @ self.p

        # Reshape back
        x_out = x_pca.reshape(shape)

        # Optional L2 normalization
        if self.l2norm:
            x_out = x_out / (mx.linalg.norm(x_out, axis=-1, keepdims=True) + 1e-8)

        return x_out


class RetrievalModel(nn.Module):
    """Image retrieval model for computing similarity signatures.

    Architecture:
        Backbone features → PreWhiten → Projector → PostWhiten → Weighted SPoC

    The model computes a global signature for each image that can be used
    to compute similarity between images.

    Note: This model does NOT include the backbone. Features must be
    pre-computed using a backbone model (DUNE, MASt3R, etc.).
    """

    def __init__(
        self,
        backbone_dim: int = 1024,
        proj_dims: list[int] | None = None,
    ):
        """Initialize retrieval model.

        Args:
            backbone_dim: Dimension of backbone features (1024 for ViT-Large)
            proj_dims: Projector hidden dimensions. Default: [1024] (single layer)
        """
        super().__init__()

        if proj_dims is None:
            proj_dims = [1024]

        self.backbone_dim = backbone_dim
        self.proj_dims = proj_dims
        self.output_dim = proj_dims[-1] if proj_dims else backbone_dim

        # Pre-whitening (before projector)
        self.prewhiten = Whitener(backbone_dim, l2norm=False)

        # Projector MLP
        self.projector = self._build_projector(backbone_dim, proj_dims)

        # Post-whitening (after projector, with L2 norm)
        self.postwhiten = Whitener(self.output_dim, l2norm=True)

    def _build_projector(self, input_dim: int, hidden_dims: list[int]) -> nn.Sequential | None:
        """Build projector MLP.

        Architecture: [Linear → LayerNorm → GELU] × (n-1) → Linear

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden dimensions

        Returns:
            Sequential module or None if no projection
        """
        if not hidden_dims:
            return None

        layers = []
        d = input_dim

        # Intermediate layers with LayerNorm + GELU
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(d, hidden_dims[i]))
            layers.append(nn.LayerNorm(hidden_dims[i]))
            layers.append(nn.GELU())
            d = hidden_dims[i]

        # Final linear layer
        layers.append(nn.Linear(d, hidden_dims[-1]))

        return nn.Sequential(*layers)

    @classmethod
    def from_pretrained(
        cls,
        cache_dir: str | Path | None = None,
        backbone_dim: int = 1024,
    ) -> RetrievalModel:
        """Load pretrained retrieval model from HuggingFace.

        Args:
            cache_dir: Cache directory for weights
            backbone_dim: Backbone feature dimension

        Returns:
            Loaded RetrievalModel
        """
        from safetensors import safe_open

        # Download weights
        weights_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=RETRIEVAL_FILENAME,
            cache_dir=cache_dir,
        )

        # Load weights to inspect structure
        weights = {}
        with safe_open(weights_path, framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)

        # Detect projector structure from weights
        proj_layers = []
        i = 0
        while f"projector.{i}.weight" in weights:
            w = weights[f"projector.{i}.weight"]
            proj_layers.append(w.shape[0])
            # Skip LayerNorm (i+1) and GELU (i+2)
            if f"projector.{i + 1}.weight" in weights:
                i += 3  # Linear + LayerNorm + GELU
            else:
                break

        # Handle case where last layer is just Linear
        if not proj_layers:
            # Fallback: single layer
            if "projector.0.weight" in weights:
                proj_layers = [weights["projector.0.weight"].shape[0]]
            else:
                proj_layers = [backbone_dim]

        # Create model
        model = cls(backbone_dim=backbone_dim, proj_dims=proj_layers)

        # Load weights
        model._load_weights(weights)

        return model

    def _load_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Load weights from dictionary.

        Args:
            weights: Dictionary of weight arrays
        """
        # Pre-whitening
        if "prewhiten.m" in weights:
            self.prewhiten.m = mx.array(weights["prewhiten.m"])
        if "prewhiten.p" in weights:
            self.prewhiten.p = mx.array(weights["prewhiten.p"])

        # Post-whitening
        if "postwhiten.m" in weights:
            self.postwhiten.m = mx.array(weights["postwhiten.m"])
        if "postwhiten.p" in weights:
            self.postwhiten.p = mx.array(weights["postwhiten.p"])

        # Projector
        if self.projector is not None:
            proj_params = {}
            for key, value in weights.items():
                if key.startswith("projector."):
                    # Parse layer index and param name
                    parts = key.split(".")
                    layer_idx = int(parts[1])
                    param_name = parts[2]

                    # Map to sequential layers
                    if param_name == "weight":
                        proj_params[f"layers.{layer_idx}.weight"] = mx.array(value)
                    elif param_name == "bias":
                        proj_params[f"layers.{layer_idx}.bias"] = mx.array(value)

            # Load projector weights
            if proj_params:
                self.projector.load_weights(list(proj_params.items()))

    def forward_features(self, features: mx.array) -> tuple[mx.array, mx.array]:
        """Process backbone features through retrieval head.

        Args:
            features: Backbone features [B, N, D] or [N, D]

        Returns:
            Tuple of (whitened_features, attention_weights)
        """
        # Add batch dimension if needed
        if features.ndim == 2:
            features = features[None, :, :]

        # Pre-whitening
        x = self.prewhiten(features)

        # Projector
        if self.projector is not None:
            x = self.projector(x)

        # Compute attention (L2 norm of features)
        attention = mx.linalg.norm(x, axis=-1)

        # Post-whitening (includes L2 norm)
        x = self.postwhiten(x)

        return x, attention

    def forward_global(self, features: mx.array) -> mx.array:
        """Compute global signature from backbone features.

        Uses weighted SPoC (Sum Pooling of Convolutions) with L2-weighted pooling.

        Args:
            features: Backbone features [B, N, D] or [N, D]

        Returns:
            Global signature [B, output_dim] or [output_dim]
        """
        squeeze = features.ndim == 2
        if squeeze:
            features = features[None, :, :]

        # Get whitened features and attention
        feat, attn = self.forward_features(features)

        # Weighted SPoC: L2-weighted sum pooling
        # signature = sum(feat * attn) / sum(attn), then L2 normalize
        weighted_sum = mx.sum(feat * attn[:, :, None], axis=1)
        attn_sum = mx.sum(attn, axis=1, keepdims=True) + 1e-8
        pooled = weighted_sum / attn_sum

        # L2 normalize
        signature = pooled / (mx.linalg.norm(pooled, axis=-1, keepdims=True) + 1e-8)

        if squeeze:
            signature = signature[0]

        return signature

    def __call__(self, features: mx.array) -> mx.array:
        """Forward pass (computes global signature)."""
        return self.forward_global(features)


def compute_similarity_matrix(
    retrieval: RetrievalModel,
    features_list: list[mx.array],
) -> np.ndarray:
    """Compute pairwise similarity matrix from pre-computed features.

    Args:
        retrieval: RetrievalModel instance
        features_list: List of backbone features, one per image [N_i, D]

    Returns:
        Similarity matrix [num_images, num_images]
    """
    # Compute global signatures
    signatures = []
    for feat in features_list:
        sig = retrieval.forward_global(feat)
        signatures.append(sig)

    # Stack signatures
    sigs = mx.stack(signatures, axis=0)  # [num_images, output_dim]

    # Normalize (should already be normalized, but ensure)
    sigs = sigs / (mx.linalg.norm(sigs, axis=-1, keepdims=True) + 1e-8)

    # Compute similarity matrix (cosine similarity)
    sim_matrix = sigs @ sigs.T

    return np.array(sim_matrix)


def select_pairs_from_retrieval(
    sim_matrix: np.ndarray,
    na: int = 20,
    k: int = 10,
) -> list[tuple[int, int]]:
    """Select image pairs using retrieval-based strategy.

    For each image, selects up to `k` most similar images.
    Uses a greedy approach to ensure coverage.

    Args:
        sim_matrix: Similarity matrix [N, N]
        na: Not used directly (kept for API compatibility)
        k: Number of pairs per image

    Returns:
        List of (i, j) pairs
    """
    n = sim_matrix.shape[0]
    pairs = set()

    for i in range(n):
        # Get similarities for image i (exclude self)
        sims = sim_matrix[i].copy()
        sims[i] = -1  # Exclude self

        # Get top-k most similar images
        top_k_indices = np.argsort(sims)[::-1][:k]

        for j in top_k_indices:
            if sims[j] > 0:  # Valid similarity
                # Add pair (sorted to avoid duplicates)
                pair = (min(i, j), max(i, j))
                pairs.add(pair)

    return sorted(list(pairs))


def make_pairs_retrieval(
    retrieval: RetrievalModel,
    backbone: Dune | Mast3r | DuneMast3r,
    images: list[np.ndarray],
    na: int = 20,
    k: int = 10,
) -> list[tuple[int, int]]:
    """Create image pairs using retrieval-based selection.

    This is the main entry point for retrieval-based pair selection.

    IMPORTANT: The retrieval model requires backbone features of dimension 1024
    (MASt3R ViT-Large). Use Mast3r, Mast3rFull, or any backbone with embed_dim=1024.
    DUNE Small (384) and DUNE Base (768) are NOT compatible.

    Args:
        retrieval: RetrievalModel instance
        backbone: Backbone model with encode() method (must have embed_dim=1024)
        images: List of input images [H, W, 3] uint8
        na: Not used directly (kept for API compatibility)
        k: Number of pairs per image

    Returns:
        List of (i, j) pairs sorted by index

    Raises:
        ValueError: If backbone embed_dim != retrieval backbone_dim
    """
    # Encode all images
    features_list = []
    for i, img in enumerate(images):
        feat = backbone.encode(img)
        feat_array = mx.array(feat)

        # Check backbone compatibility on first image
        if i == 0:
            actual_dim = feat_array.shape[-1]
            if actual_dim != retrieval.backbone_dim:
                raise ValueError(
                    f"Backbone feature dimension ({actual_dim}) doesn't match retrieval model "
                    f"backbone_dim ({retrieval.backbone_dim}). "
                    f"Use Mast3r or Mast3rFull (embed_dim=1024) for retrieval."
                )

        features_list.append(feat_array)

    # Compute similarity matrix
    sim_matrix = compute_similarity_matrix(retrieval, features_list)

    # Select pairs
    pairs = select_pairs_from_retrieval(sim_matrix, na=na, k=k)

    return pairs
