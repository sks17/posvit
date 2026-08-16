"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
"""

from __future__ import annotations
import json
import os
from typing import Any

import numpy as np
from scipy import ndimage
from PIL import Image

from .. import paths
from . import CANONICAL_CLASSES
from .loaders import parse_fg_id

_IMG_EXT = (".jpeg", ".jpg", ".png")
_THRESH = 8
_GRID = 14 
_COVERAGE_GATE = 0.99


def pixel_mask_from_only_fg(path, thresh: int = _THRESH) -> np.ndarray:
    """
    Recovers the foreground pixel mask from an ONLY-FG image.
    - Pre: `path` is a path to an ONLY-FG image (object on a black canvas).
    - Post: Returns a boolean array of shape (H, W) where True indicates foreground pixels. 
        Pixels with intensity greater than `thresh` are considered foreground.
    """
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    return arr.max(axis=2) > thresh


def clean_mask(fg: np.ndarray) -> np.ndarray:
    """
    Cleans a foreground mask.
    - Pre: `fg` is a boolean array of shape (H, W).
    - Post: Returns a boolean array of shape (H, W).
        The method fills small gaps with binary closing.
        Then it keeps only the largest connected component.
        If no component exists, it returns an all-False mask.
    """
    fg = ndimage.binary_closing(fg, structure=np.ones((3, 3), dtype=bool))
    labeled, n = ndimage.label(fg)
    if n == 0:
        return np.zeros_like(fg, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = int(sizes.argmax())
    return labeled == largest


def patch_mask(fg: np.ndarray, grid: int = _GRID) -> np.ndarray:
    """
    Converts a pixel mask to a patch mask.
    - Pre: `fg` is a boolean array of shape (H, W).
        `grid` is a positive integer for the output grid size.
    - Post: Returns a boolean array of shape (`grid`, `grid`).
        A patch is foreground when at least 50 percent of its pixels are foreground.
    """
    h, w = fg.shape
    ph, pw = h // grid, w // grid
    out = np.zeros((grid, grid), dtype=bool)
    for i in range(grid):
        for j in range(grid):
            cell = fg[i * ph : (i + 1) * ph, j * pw : (j + 1) * pw]
            out[i, j] = cell.mean() >= 0.5
    return out


def _center_offset(mask: np.ndarray) -> float:
    """
    Computes the normalized center offset of foreground pixels.
    - Pre: `mask` is a boolean array of shape (H, W) and has foreground pixels.
    - Post: Returns a float in [0, 1].
        The value is the distance from the foreground centroid to the image center,
        normalized by the center-to-corner distance.
    """
    cy, cx = ndimage.center_of_mass(mask)
    h, w = mask.shape
    center = np.array([(h - 1) / 2.0, (w - 1) / 2.0])
    return float(np.linalg.norm(np.array([cy, cx]) - center) / np.linalg.norm(center))


def derive_masks(cfg, variant: "str | None" = None) -> dict[str, Any]:
    """
    Generates masks for all foreground items in the ONLY-FG variant.
    - Pre: `cfg` has valid mask source settings.
        `variant` is None or a valid variant name.
    - Post: Returns a manifest dictionary with coverage and quality-gate fields.
        The method writes mask files to disk and saves `_manifest.json`.
    """
    variant = variant or cfg.mask_source_variant
    src_root = paths.in9_bg_root() / variant / "val"
    masks_dir = paths.results_raw() / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    total, written, empty = 0, 0, []
    for cls in CANONICAL_CLASSES:
        d = src_root / cls
        if not d.is_dir():
            continue
        out_dir = masks_dir / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith(_IMG_EXT):
                continue
            fg_id = parse_fg_id(name)
            if not fg_id:
                continue
            total += 1
            raw = pixel_mask_from_only_fg(d / name)
            cleaned = clean_mask(raw)
            if not cleaned.any():
                empty.append({"class": cls, "fg_id": fg_id, "fname": name})
                continue
            np.savez_compressed(
                out_dir / f"{fg_id}.npz",
                pixel_mask=cleaned,
                patch_mask=patch_mask(cleaned),
                fg_area_frac=float(cleaned.mean()),
                fg_area_frac_raw=float(raw.mean()),
                center_offset=_center_offset(cleaned),
                fg_id=fg_id,
                source_fname=name,
            )
            written += 1

    coverage = written / total if total else 0.0
    manifest = {
        "source_variant": variant,
        "threshold": _THRESH,
        "patch_grid": _GRID,
        "total_fg_ids": total,
        "written": written,
        "empty_count": len(empty),
        "empty": empty[:20],
        "coverage": round(coverage, 4),
        "coverage_gate": _COVERAGE_GATE,
        "passed": coverage >= _COVERAGE_GATE,
    }
    (masks_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    from posvit.config import load_config
    print(json.dumps(derive_masks(load_config()), indent=2))  # Print the manifest so you can check run results in the terminal.