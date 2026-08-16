"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
"""

from __future__ import annotations

import numpy as np


def stratified_sample(labels, per_class: int, *, fg_ids=None, seed: int = 0) -> np.ndarray:
    """
    Samples indices per class with reproducible stratification.
    - Pre: `labels` is an array-like class label sequence.
        `per_class` is a positive integer.
        `fg_ids` is None or an array-like sequence aligned with `labels`.
        `seed` is an integer.
    - Post: Returns sorted integer index array.
        Includes up to `per_class` samples per class.
        When `fg_ids` is provided, each foreground identity is used at most once.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    seen: set = set()
    picked: list[int] = []

    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        taken = 0
        for j in rng.permutation(len(idx)):
            if taken >= per_class:
                break
            i = int(idx[j])
            if fg_ids is not None:
                fid = fg_ids[i]
                if fid in seen:
                    continue
                seen.add(fid)
            picked.append(i)
            taken += 1
    # Sorted so downstream slicing preserves dataset order and stays comparable
    # across calls with the same seed.
    return np.array(sorted(picked), dtype=int)