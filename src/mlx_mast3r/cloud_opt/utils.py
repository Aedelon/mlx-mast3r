# Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
"""Utility functions for cloud optimization."""

from __future__ import annotations

import hashlib
from pathlib import Path


def hash_md5(s: str) -> str:
    """Create MD5 hash of string.

    Args:
        s: Input string

    Returns:
        First 16 characters of MD5 hash
    """
    return hashlib.md5(s.encode()).hexdigest()[:16]


def mkdir_for(path: str) -> str:
    """Create directory for file path.

    Args:
        path: File path

    Returns:
        The same path (for chaining)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path
