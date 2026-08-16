"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from . import paths
from .hashing import sha256_file
from .registry import ModelSpec, all_models

CKPT_FILENAME = "pytorch_model.bin"
_RESOLUTION_BUFFERS = ("freqs_t_x", "freqs_t_y")


class IntegrityError(RuntimeError):
    """Error that occurs when the bytes on disk are not the bytes configs/models.yaml declares."""


def _import_constructor(spec: ModelSpec):
    """
    Resolves the constructor name in the model registry.
    - Pre: `spec` is a ModelSpec object.
    - Post: Returns the constructor function for the model.
    """
    paths.ensure_on_syspath(paths.rope_vit_root())
    try:
        mod = importlib.import_module(spec.module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"{spec.model_id}: cannot import {spec.module!r}. The vendored RoPE-ViT repo must "
            f"provide it at {paths.rope_vit_root()} (see lessons/SETUP_TRACKER.md §1)."
        ) from exc
    if not hasattr(mod, spec.constructor):
        raise AttributeError(
            f"{spec.model_id}: {spec.module!r} has no constructor {spec.constructor!r}. "
            "The vendored repo may be a different revision than the registry expects."
        )
    return getattr(mod, spec.constructor)


def _expected_missing(spec: ModelSpec) -> tuple[str, ...]:
    """
    Returns root keys that can be missing in the state dict.
    - Pre: `spec` is a ModelSpec object.
    - Post: Returns a tuple of allowed missing root keys.
        It always includes `head`.
        It includes rotary frequency keys only when `spec.has_rope` is True.
    """
    allowed = ["head"]
    if spec.has_rope:
        allowed += ["freqs_t_x", "freqs_t_y", "freqs_cis"]
    return tuple(allowed)


def download_checkpoint(spec: ModelSpec) -> Path:
    """
    Downloads the checkpoint file to the Hugging Face cache.
    - Pre: `spec` is a ModelSpec object with a valid `hf_repo`.
    - Post: Returns the local path to the checkpoint file.
        If the file is already cached, it reuses the cached file.
    """
    return Path(hf_hub_download(repo_id=spec.hf_repo, filename=CKPT_FILENAME))


def verify_checkpoint(spec: ModelSpec, path: Path, allow_unpinned: bool = False) -> str:
    """
    Verifies the checkpoint hash.
    - Pre: `spec` is a ModelSpec object and `path` points to a checkpoint file.
        `allow_unpinned` is True or False.
    - Post: Returns the SHA-256 digest as a string.
        Raises `IntegrityError` when the digest does not match the expected value,
        or when the spec is unpinned and `allow_unpinned` is False.
    """
    digest = sha256_file(path)

    if spec.expected_sha256 is None:
        if not allow_unpinned:
            raise IntegrityError(
                f"{spec.model_id}: configs/models.yaml has expected_sha256: null, so these "
                f"weights are UNVERIFIED.\n"
                f"  observed {digest}\n  file     {path}\n"
                "Pass allow_unpinned=True to load anyway (results will be flagged), or run\n"
                "  python -m posvit.checkpoints --record\n"
                "and paste the digest into configs/models.yaml deliberately."
            )
        return digest

    if digest != spec.expected_sha256:
        raise IntegrityError(
            f"{spec.model_id}: checkpoint SHA-256 MISMATCH.\n"
            f"  got      {digest}\n  expected {spec.expected_sha256}\n  file     {path}\n"
            "Delete the cached file and re-download. If upstream genuinely republished the "
            "weights, update configs/models.yaml deliberately - never silently."
        )
    return digest


def _torch_load(path: Path):
    """
    Loads a checkpoint object from disk.
    - Pre: `path` points to a verified checkpoint file.
    - Post: Returns the loaded checkpoint object.
        It tries `weights_only=True` first.
        If that fails, it falls back to `weights_only=False`.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - restricted unpickler raises several types
        print(
            f"  weights_only=True rejected this checkpoint ({type(exc).__name__}); "
            "falling back to the full unpickler - safe because the SHA-256 is pinned."
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def load_checkpoint(
    spec: ModelSpec,
    img_size: int = 224,
    device: str = "cpu",
    allow_unpinned: bool = False,
) -> "tuple[torch.nn.Module, dict[str, Any]]":
    """
    Builds, verifies, and loads one model from the registry.
    - Pre: `spec` is a ModelSpec object.
        `img_size` is a positive integer.
        `device` is a valid torch device string.
        `allow_unpinned` is True or False.
    - Post: Returns a tuple `(model, receipt)`.
        `model` is in eval mode on `device`.
        `receipt` contains load details and hash status.
    """
    build = _import_constructor(spec)

    # Use pretrained=False to avoid a known wrapper-dict load path issue.
    model = build(pretrained=False, img_size=img_size)

    ckpt_path = download_checkpoint(spec)
    digest = verify_checkpoint(spec, ckpt_path, allow_unpinned=allow_unpinned)

    obj = _torch_load(ckpt_path)
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    for k in _RESOLUTION_BUFFERS:
        sd.pop(k, None)

    missing, unexpected = model.load_state_dict(sd, strict=False)

    # Check missing keys by root name against the spec allowlist.
    allowed = _expected_missing(spec)
    bad = [k for k in missing if k.split(".", 1)[0] not in allowed]
    if bad:
        raise RuntimeError(
            f"{spec.model_id}: {len(bad)} unexpected missing key(s): {bad[:10]}\n"
            f"Allowed roots for this spec (has_rope={spec.has_rope}): {list(allowed)}"
        )

    model = model.to(device).eval()

    return model, {
        "model_id": spec.model_id,
        "safe_id": spec.safe_id,
        "hf_repo": spec.hf_repo,
        "constructor": f"{spec.module}.{spec.constructor}",
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": digest,
        # This field shows if the hash is pinned or only observed.
        "sha256_pinned": spec.expected_sha256 is not None,
        "img_size": img_size,
        "device": str(device),
        "n_missing_keys": len(missing),
        "n_unexpected_keys": len(unexpected),
    }


def record_hashes(specs: "tuple[ModelSpec, ...] | None" = None) -> dict[str, str]:
    """
    Records observed checkpoint hashes for model specs.
    - Pre: `specs` is None or a tuple of ModelSpec objects.
    - Post: Returns a dict that maps `model_id` to observed SHA-256 digest.
        Prints a YAML-ready block for unpinned models.
        Does not write changes to config files.
    """
    specs = specs or all_models()
    observed: dict[str, str] = {}
    for spec in specs:
        digest = sha256_file(download_checkpoint(spec))
        observed[spec.model_id] = digest
        state = "pinned" if spec.expected_sha256 else "UNPINNED"
        print(f"{spec.model_id:24s} {digest}  ({state})")

    print("\n# Paste into configs/models.yaml under the matching entry:")
    for spec in specs:
        if spec.expected_sha256 is None:
            print(f"#   {spec.model_id}\n    expected_sha256: {observed[spec.model_id]}")
    return observed


if __name__ == "__main__":
    import sys

    if "--record" in sys.argv:
        record_hashes()
    else:
        failed = []
        for s in all_models():
            try:
                verify_checkpoint(s, download_checkpoint(s))
                print(f"OK       {s.model_id}")
            except IntegrityError as e:
                failed.append(s.model_id)
                print(f"FAIL     {s.model_id}: {str(e).splitlines()[0]}")
        sys.exit(1 if failed else 0)