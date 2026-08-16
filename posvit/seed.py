"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
~ Files styled after ""The Role of Image Backgrounds in Object Recognition" GitHub
"""

from __future__ import annotations
import random
from typing import Any
import numpy as np
import torch

def seed_everything(seed: int = 0) -> dict[str, Any]:
    """
    Seeds Python, NumPy, and PyTorch random generators.
    - Pre: `seed` is an integer.
    - Post: Returns a dictionary with seed and runtime details.
        The method sets deterministic CUDA/CuDNN behavior when available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        try:
            torch.backends.mkldnn.enabled = False
        except AttributeError:
            pass
    return {
        "seed": seed,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_deterministic": True,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }