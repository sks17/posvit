"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
"""

from __future__ import annotations
import hashlib
from pathlib import Path


def sha256_file(path: "Path | str", chunk: int = 1 << 20) -> str:
    """
    Computes the SHA-256 hash of a file.
    - Pre: `path` points to a readable file.
        `chunk` is a positive integer.
    - Post: Returns the SHA-256 digest as a lowercase hex string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()