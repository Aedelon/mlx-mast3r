"""Weight download utilities.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.

Two download modes:
1. Safetensors from HuggingFace (recommended, ready to use)
2. PTH from Naver (for manual conversion)
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# HuggingFace repositories (safetensors, ready to use)
# =============================================================================

HF_MAST3R_REPO = "Aedelon/mast3r-vit-large-fp16"
HF_DUNEMAST3R_REPO = "Aedelon/dunemast3r-models-fp16"

# =============================================================================
# Naver URLs (PTH checkpoints, for manual conversion)
# =============================================================================

DUNE_PTH_URLS = {
    "small_336": "https://download.europe.naverlabs.com/dune/dune_vitsmall14_336.pth",
    "small_448": "https://download.europe.naverlabs.com/dune/dune_vitsmall14_448.pth",
    "base_336": "https://download.europe.naverlabs.com/dune/dune_vitbase14_336.pth",
    "base_448": "https://download.europe.naverlabs.com/dune/dune_vitbase14_448.pth",
}

DUNEMAST3R_PTH_URLS = {
    "small": "https://download.europe.naverlabs.com/dune/dunemast3r_cvpr25_vitsmall.pth",
    "base": "https://download.europe.naverlabs.com/dune/dunemast3r_cvpr25_vitbase.pth",
}

MAST3R_HF_REPO = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

# Model configurations
DUNE_VARIANTS = ["small", "base"]
DUNE_RESOLUTIONS = [336, 448]


def get_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Get or create cache directory."""
    if cache_dir is None:
        cache_dir = Path.home() / ".cache/mlx-mast3r"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# =============================================================================
# Safetensors downloads (HuggingFace) - Ready to use
# =============================================================================


def download_mast3r(
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download MASt3R ViT-Large safetensors from HuggingFace.

    Args:
        cache_dir: Cache directory (default: ~/.cache/mlx-mast3r)
        force: Force re-download even if exists

    Returns:
        Path to unified.safetensors
    """
    from huggingface_hub import hf_hub_download

    cache_dir = get_cache_dir(cache_dir)
    local_dir = cache_dir / "mast3r_vit_large"
    local_dir.mkdir(parents=True, exist_ok=True)

    output_path = local_dir / "unified.safetensors"
    if output_path.exists() and not force:
        return output_path

    print(f"Downloading MASt3R ViT-Large from {HF_MAST3R_REPO}...")
    hf_hub_download(
        repo_id=HF_MAST3R_REPO,
        filename="unified.safetensors",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Saved to: {output_path}")

    return output_path


def download_dune(
    variant: str = "base",
    resolution: int = 336,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download DUNE encoder safetensors from HuggingFace.

    Args:
        variant: "small" or "base"
        resolution: 336 or 448
        cache_dir: Cache directory
        force: Force re-download

    Returns:
        Path to encoder.safetensors
    """
    from huggingface_hub import hf_hub_download

    if variant not in DUNE_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {DUNE_VARIANTS}")
    if resolution not in DUNE_RESOLUTIONS:
        raise ValueError(f"Unknown resolution: {resolution}. Choose from {DUNE_RESOLUTIONS}")

    cache_dir = get_cache_dir(cache_dir)
    model_name = f"dune_vit_{variant}_{resolution}"
    local_dir = cache_dir / model_name
    local_dir.mkdir(parents=True, exist_ok=True)

    output_path = local_dir / "encoder.safetensors"
    if output_path.exists() and not force:
        return output_path

    print(f"Downloading DUNE {variant} @ {resolution} from {HF_DUNEMAST3R_REPO}...")
    hf_hub_download(
        repo_id=HF_DUNEMAST3R_REPO,
        filename=f"{model_name}/encoder.safetensors",
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Saved to: {output_path}")

    return output_path


def download_dunemast3r(
    variant: str = "base",
    resolution: int = 336,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """Download DuneMASt3R safetensors (encoder + decoder) from HuggingFace.

    Args:
        variant: "small" or "base"
        resolution: 336 or 448
        cache_dir: Cache directory
        force: Force re-download

    Returns:
        (encoder_path, decoder_path)
    """
    from huggingface_hub import hf_hub_download

    if variant not in DUNE_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {DUNE_VARIANTS}")
    if resolution not in DUNE_RESOLUTIONS:
        raise ValueError(f"Unknown resolution: {resolution}. Choose from {DUNE_RESOLUTIONS}")

    cache_dir = get_cache_dir(cache_dir)
    model_name = f"dune_vit_{variant}_{resolution}"
    local_dir = cache_dir / model_name
    local_dir.mkdir(parents=True, exist_ok=True)

    encoder_path = local_dir / "encoder.safetensors"
    decoder_path = local_dir / "decoder.safetensors"

    if not encoder_path.exists() or force:
        print(f"Downloading DuneMASt3R {variant} @ {resolution} encoder...")
        hf_hub_download(
            repo_id=HF_DUNEMAST3R_REPO,
            filename=f"{model_name}/encoder.safetensors",
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
        )

    if not decoder_path.exists() or force:
        print(f"Downloading DuneMASt3R {variant} @ {resolution} decoder...")
        hf_hub_download(
            repo_id=HF_DUNEMAST3R_REPO,
            filename=f"{model_name}/decoder.safetensors",
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
        )

    print(f"Saved to: {local_dir}")
    return encoder_path, decoder_path


# =============================================================================
# PTH downloads (Naver) - For manual conversion
# =============================================================================


def download_dune_pth(
    variant: str = "base",
    resolution: int = 336,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download DUNE PTH checkpoint from Naver (for conversion).

    Args:
        variant: "small" or "base"
        resolution: 336 or 448
        cache_dir: Cache directory
        force: Force re-download

    Returns:
        Path to .pth file
    """
    import urllib.request

    key = f"{variant}_{resolution}"
    if key not in DUNE_PTH_URLS:
        raise ValueError(f"Unknown DUNE model: {key}. Available: {list(DUNE_PTH_URLS.keys())}")

    cache_dir = get_cache_dir(cache_dir)
    pth_dir = cache_dir / "checkpoints"
    pth_dir.mkdir(parents=True, exist_ok=True)

    url = DUNE_PTH_URLS[key]
    filename = Path(url).name
    output_path = pth_dir / filename

    if output_path.exists() and not force:
        print(f"Using cached: {output_path}")
        return output_path

    print(f"Downloading DUNE {variant} @ {resolution} PTH from Naver...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to: {output_path}")

    return output_path


def download_dunemast3r_pth(
    variant: str = "base",
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download DuneMASt3R decoder PTH checkpoint from Naver (for conversion).

    Args:
        variant: "small" or "base"
        cache_dir: Cache directory
        force: Force re-download

    Returns:
        Path to .pth file
    """
    import urllib.request

    if variant not in DUNEMAST3R_PTH_URLS:
        raise ValueError(
            f"Unknown variant: {variant}. Available: {list(DUNEMAST3R_PTH_URLS.keys())}"
        )

    cache_dir = get_cache_dir(cache_dir)
    pth_dir = cache_dir / "checkpoints"
    pth_dir.mkdir(parents=True, exist_ok=True)

    url = DUNEMAST3R_PTH_URLS[variant]
    filename = Path(url).name
    output_path = pth_dir / filename

    if output_path.exists() and not force:
        print(f"Using cached: {output_path}")
        return output_path

    print(f"Downloading DuneMASt3R {variant} decoder PTH from Naver...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to: {output_path}")

    return output_path


def download_mast3r_pth(
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download MASt3R PTH checkpoint from HuggingFace (for conversion).

    Args:
        cache_dir: Cache directory
        force: Force re-download

    Returns:
        Path to model.safetensors (original Naver format)
    """
    from huggingface_hub import hf_hub_download

    cache_dir = get_cache_dir(cache_dir)
    pth_dir = cache_dir / "checkpoints"
    pth_dir.mkdir(parents=True, exist_ok=True)

    output_path = pth_dir / "MASt3R_ViTLarge_BaseDecoder_512.safetensors"
    if output_path.exists() and not force:
        print(f"Using cached: {output_path}")
        return output_path

    print(f"Downloading MASt3R from {MAST3R_HF_REPO}...")
    downloaded = hf_hub_download(
        repo_id=MAST3R_HF_REPO,
        filename="model.safetensors",
        local_dir=pth_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Saved to: {downloaded}")

    return Path(downloaded)


# =============================================================================
# Utilities
# =============================================================================


def list_available_models() -> dict:
    """List all available models."""
    dune_models = [f"{v}_{r}" for v in DUNE_VARIANTS for r in DUNE_RESOLUTIONS]
    return {
        "mast3r": ["vit_large"],
        "dune": dune_models,
        "dunemast3r": dune_models,
    }


def download_all(cache_dir: str | Path | None = None, force: bool = False) -> None:
    """Download all available safetensors models."""
    print("Downloading all models from HuggingFace...")

    # MASt3R
    download_mast3r(cache_dir, force)

    # DUNE/DuneMASt3R variants
    for variant in DUNE_VARIANTS:
        for resolution in DUNE_RESOLUTIONS:
            download_dunemast3r(variant, resolution, cache_dir, force)

    print("\nAll models downloaded!")
