"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


def blocks_of(model):
    """
    Resolves transformer blocks from a model.
    - Pre: `model` is a model object with `blocks` or nested `.model.blocks`.
    - Post: Returns the transformer block list.
        Raises AttributeError when blocks cannot be found.
    """
    for obj in (model, getattr(model, "model", None)):
        blocks = getattr(obj, "blocks", None)
        if blocks is not None:
            return blocks
    raise AttributeError(
        f"cannot find `.blocks` on {type(model).__name__} or its `.model` attribute - "
        "this does not look like a ViT from the registry."
    )


def forward_features_of(model):
    """
    Resolves forward_features callable from a model.
    - Pre: `model` is a model object with `forward_features` or nested `.model.forward_features`.
    - Post: Returns the `forward_features` callable.
        Raises AttributeError when callable cannot be found.
    """
    for obj in (model, getattr(model, "model", None)):
        ff = getattr(obj, "forward_features", None)
        if ff is not None:
            return ff
    raise AttributeError(f"cannot find `.forward_features` on {type(model).__name__}")


def _cls_to_patch(attn: torch.Tensor, grid: int = 14) -> torch.Tensor:
    """
    Reduces attention to CLS-to-patch map.
    - Pre: `attn` has shape `(B, H, N, N)`.
        `grid` is a positive integer patch-grid size.
    - Post: Returns tensor with shape `(B, H, grid, grid)`.
    """
    a = attn[:, :, 0, 1:]
    b, h, n = a.shape
    return a.reshape(b, h, grid, grid)


_REDUCERS = {"cls_to_patch": _cls_to_patch, "full": lambda a, grid=14: a}


@contextmanager
def capture_attention(model, *, layers=(-1,), reduce: str = "cls_to_patch", grid: int = 14):
    """
    Captures attention activations for selected layers.
    - Pre: `model` has transformer blocks with attention dropout hook points.
        `layers` is an iterable of layer indices.
        `reduce` is one of supported reduction modes.
        `grid` is a positive integer patch-grid size.
    - Post: Yields dictionary mapping layer index to captured tensor.
        Always removes all registered hooks on context exit.
        Raises ValueError for unknown reduction mode.
    """
    if reduce not in _REDUCERS:
        raise ValueError(f"unknown reduce {reduce!r}; expected one of {sorted(_REDUCERS)}")
    fn = _REDUCERS[reduce]
    blocks = blocks_of(model)
    idxs = [li % len(blocks) for li in layers]
    store: dict[int, torch.Tensor] = {}
    handles = []

    def make(li):
        """
        Builds one forward hook callback.
        - Pre: `li` is an integer layer index.
        - Post: Returns hook function that stores reduced attention tensor.
        """
        def hook(_mod, inp, _out):
            store[li] = fn(inp[0], grid).detach().cpu()
        return hook

    try:
        for li in idxs:
            handles.append(blocks[li].attn.attn_drop.register_forward_hook(make(li)))
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def capture_tokens(model, *, layers=(-1,), patch_only: bool = True):
    """
    Captures token activations for selected layers.
    - Pre: `model` has transformer blocks.
        `layers` is an iterable of layer indices.
        `patch_only` is True or False.
    - Post: Yields dictionary mapping layer index to captured token tensor.
        Always removes all registered hooks on context exit.
    """
    blocks = blocks_of(model)
    idxs = [li % len(blocks) for li in layers]
    store: dict[int, torch.Tensor] = {}
    handles = []

    def make(li):
        """
        Builds one token-capture hook callback.
        - Pre: `li` is an integer layer index.
        - Post: Returns hook function that stores token tensor.
        """
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            store[li] = (t[:, 1:, :] if patch_only else t).detach().cpu()
        return hook

    try:
        for li in idxs:
            handles.append(blocks[li].register_forward_hook(make(li)))
        yield store
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def penultimate_features(model, x: torch.Tensor) -> torch.Tensor:
    """
    Computes penultimate CLS features.
    - Pre: `model` provides `forward_features` callable.
        `x` is input tensor with batch dimension.
    - Post: Returns penultimate CLS features tensor with shape `(B, embed_dim)`.
    """
    return forward_features_of(model)(x)