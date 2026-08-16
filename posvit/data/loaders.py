"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
"""

from __future__ import annotations

import os
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .. import paths
from . import CANONICAL_CLASSES

_FG_RE = re.compile(r"^fg_(n\d+_\d+)_bg_")
_IMG_EXT = (".jpeg", ".jpg", ".png")


def parse_fg_id(filename: str) -> "str | None":
    """
    Extracts the foreground id from an image filename.
    - Pre: `filename` is an image filename string.
    - Post: Returns the foreground id string when detected.
        Returns None when the foreground id cannot be extracted.
    """
    m = _FG_RE.match(filename)
    if m:
        return m.group(1)
    stem, ext = os.path.splitext(filename)
    if ext.lower() in _IMG_EXT and stem.startswith("n"):
        return stem
    return None


def pairing_key(filename: str) -> str:
    """
    Builds a stable join key for a filename.
    - Pre: `filename` is an image filename string.
    - Post: Returns the parsed foreground id when available.
        Returns the original filename when parsing fails.
    """
    return parse_fg_id(filename) or filename


def build_transform(cfg) -> transforms.Compose:
    """
    Builds the image transform pipeline.
    - Pre: `cfg` has `resize_policy`, `norm_mean`, and `norm_std` fields.
    - Post: Returns a torchvision Compose transform.
        The transform applies optional resize/crop, then tensor conversion and normalization.
    """
    steps: list = []
    if cfg.resize_policy == "resize256_crop224":
        steps += [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=list(cfg.norm_mean), std=list(cfg.norm_std)),
    ]
    return transforms.Compose(steps)


class IndexedImageFolder(ImageFolder):
    """
    Extends ImageFolder to also return sample index.
    - Pre: Dataset is initialized as a valid ImageFolder.
    - Post: Each item returns `(image, target, index)`.
    """

    def __getitem__(self, index: int):
        """
        Loads one indexed sample.
        - Pre: `index` is a valid integer sample index.
        - Post: Returns `(image, target, index)`.
        """
        img, target = super().__getitem__(index)
        return img, target, index


def build_loader(cfg, variant: str):
    """
    Builds a deterministic data loader for one variant.
    - Pre: `cfg` contains loader settings.
        `variant` is a valid dataset variant name.
    - Post: Returns `(loader, dataset)` for the variant.
        Raises ValueError when class index mapping is not canonical.
    """
    root = paths.in9_bg_root() / variant / "val"
    ds = IndexedImageFolder(root=str(root), transform=build_transform(cfg))
    expected = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
    if ds.class_to_idx != expected:
        raise ValueError(
            f"class_to_idx mismatch for {variant!r}:\n  got      {ds.class_to_idx}\n"
            f"  expected {expected}\nThe data may be extracted wrong or the root is off."
        )
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, ds


def preload_uint8(cfg, variant: str):
    """
    Loads one dataset variant into RAM as uint8 tensors.
    - Pre: `cfg` contains valid dataset settings.
        `variant` is a valid dataset variant name.
    - Post: Returns `(images, labels, fg_ids)`.
        `images` is NCHW uint8 tensor.
        `labels` is integer numpy array.
        `fg_ids` is list of pairing keys.
    """
    root = paths.in9_bg_root() / variant / "val"
    imgs: list = []
    labels: list = []
    fg_ids: list = []
    for ci, cls in enumerate(CANONICAL_CLASSES):
        d = root / cls
        if not d.is_dir():
            continue
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith(_IMG_EXT):
                continue
            arr = np.asarray(Image.open(d / name).convert("RGB"), dtype=np.uint8)
            imgs.append(np.transpose(arr, (2, 0, 1)))
            labels.append(ci)
            fg_ids.append(pairing_key(name))
    return torch.from_numpy(np.stack(imgs)), np.array(labels), fg_ids
