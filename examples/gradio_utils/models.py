# Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""Model loading and configuration utilities for Gradio demo."""

from __future__ import annotations

from typing import Any

# Global model cache
_models: dict[str, Any] = {}


def get_model(model_name: str) -> Any:
    """Get or load model (cached).

    Args:
        model_name: One of "DUNE Small", "DUNE Base", "DuneMASt3R Small",
                   "DuneMASt3R Base", "MASt3R Full", "Retrieval"

    Returns:
        Loaded model instance
    """
    if model_name not in _models:
        print(f"Loading {model_name}...")

        from mlx_mast3r import DUNE, DuneMast3r, Mast3rFull, RetrievalModel

        if model_name == "DUNE Small":
            _models[model_name] = DUNE.from_pretrained(variant="small", resolution=336)
        elif model_name == "DUNE Base":
            _models[model_name] = DUNE.from_pretrained(variant="base", resolution=336)
        elif model_name == "DuneMASt3R Small":
            _models[model_name] = DuneMast3r.from_pretrained(
                encoder_variant="small", resolution=336
            )
        elif model_name == "DuneMASt3R Base":
            _models[model_name] = DuneMast3r.from_pretrained(encoder_variant="base", resolution=448)
        elif model_name == "MASt3R Full":
            _models[model_name] = Mast3rFull.from_pretrained(resolution=512)
        elif model_name == "Retrieval":
            _models[model_name] = RetrievalModel.from_pretrained()

        print(f"{model_name} loaded!")

    return _models[model_name]


def get_retrieval_model() -> Any:
    """Get or load retrieval model (lazy loading)."""
    return get_model("Retrieval")


def get_resolution(model_name: str) -> int:
    """Get resolution for model (long edge size).

    Args:
        model_name: Model name

    Returns:
        Resolution in pixels
    """
    if model_name == "MASt3R Full":
        return 512
    elif model_name == "DuneMASt3R Base":
        return 448  # Must be multiple of 14 (patch_size)
    elif model_name == "DuneMASt3R Small":
        return 336  # Must be multiple of 14
    return 336  # Default for DUNE variants


def get_model_params(model_name: str) -> dict:
    """Get preprocessing params for model.

    Args:
        model_name: Model name

    Returns:
        Dict with square_ok and patch_size parameters
    """
    if "DUNE" in model_name or "DuneMASt3R" in model_name:
        return {"square_ok": True, "patch_size": 14}
    else:
        # MASt3R uses patch_size=16 and 4:3 aspect ratio
        return {"square_ok": False, "patch_size": 16}


def get_default_conf_threshold(model_name: str) -> float:
    """Get recommended confidence threshold for model.

    DuneMASt3R produces higher confidence scores than MASt3R Full,
    so we use different default thresholds for optimal visualization.

    Args:
        model_name: Model name

    Returns:
        Recommended confidence threshold
    """
    if "MASt3R Full" in model_name:
        return 1.0  # MASt3R Full has lower confidence scores
    elif "DuneMASt3R" in model_name:
        return 1.5  # DuneMASt3R has higher confidence scores
    else:
        return 0.5  # DUNE features only, low threshold
