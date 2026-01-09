"""Weight download utilities.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""

from __future__ import annotations

from pathlib import Path


DUNE_URLS = {
    "small_336": "https://download.europe.naverlabs.com/dune/dune_vitsmall14_336.pth",
    "small_448": "https://download.europe.naverlabs.com/dune/dune_vitsmall14_448.pth",
    "base_336": "https://download.europe.naverlabs.com/dune/dune_vitbase14_336.pth",
    "base_448": "https://download.europe.naverlabs.com/dune/dune_vitbase14_448.pth",
}

DUNEMAST3R_URLS = {
    "small": "https://download.europe.naverlabs.com/dune/dunemast3r_cvpr25_vitsmall.pth",
    "base": "https://download.europe.naverlabs.com/dune/dunemast3r_cvpr25_vitbase.pth",
}


def download_weights(
    model: str = "dune_base_336",
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download model weights.

    Args:
        model: Model name (e.g., "dune_base_336", "dunemast3r_base")
        cache_dir: Cache directory (default: ~/.cache/mlx-mast3r)
        force: Force re-download even if exists

    Returns:
        Path to downloaded weights
    """
    from huggingface_hub import hf_hub_download
    import urllib.request

    if cache_dir is None:
        cache_dir = Path.home() / ".cache/mlx-mast3r"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine URL
    if model.startswith("dune_"):
        key = model.replace("dune_", "")
        if key not in DUNE_URLS:
            raise ValueError(f"Unknown DUNE model: {model}. Available: {list(DUNE_URLS.keys())}")
        url = DUNE_URLS[key]
        filename = Path(url).name
    elif model.startswith("dunemast3r_"):
        key = model.replace("dunemast3r_", "")
        if key not in DUNEMAST3R_URLS:
            raise ValueError(f"Unknown DuneMASt3R model: {model}")
        url = DUNEMAST3R_URLS[key]
        filename = Path(url).name
    elif model == "mast3r":
        # Download from HuggingFace
        local_path = hf_hub_download(
            repo_id="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
            filename="model.safetensors",
            cache_dir=cache_dir,
        )
        return Path(local_path)
    else:
        raise ValueError(f"Unknown model: {model}")

    # Download
    output_path = cache_dir / filename
    if output_path.exists() and not force:
        print(f"Using cached weights: {output_path}")
        return output_path

    print(f"Downloading {model} from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to: {output_path}")

    return output_path


def list_available_models() -> dict:
    """List all available models."""
    return {
        "dune": list(DUNE_URLS.keys()),
        "dunemast3r": list(DUNEMAST3R_URLS.keys()),
        "mast3r": ["vit_large_512"],
    }
