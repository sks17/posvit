"""
Registry of all PosViT models, loaded from configs/models.yaml.
The registry is cached in memory for fast repeated access, and validated on load.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import yaml

from . import paths

_VALID_SCALES = ("vit_s", "vit_b")


@dataclass(frozen=True)
class ModelSpec:
    scale: str
    key: str
    constructor: str
    module: str
    hf_repo: str
    pe_type: str
    use_ape: bool
    has_rope: bool
    expected_sha256: "str | None" = None

    @property
    def model_id(self) -> str:
        """
        Returns the model identifier used by CLI and logs.
        - Pre: `self.scale` and `self.key` are valid strings.
        - Post: Returns a string in `scale/key` format.
        """
        return f"{self.scale}/{self.key}"

    @property
    def safe_id(self) -> str:
        """
        Returns a filename-safe model identifier.
        - Pre: `self.scale` and `self.key` are valid strings.
        - Post: Returns a string in `scale__key` format.
        """
        return f"{self.scale}__{self.key}"


@lru_cache(maxsize=1)
def load_registry() -> "tuple[ModelSpec, ...]":
    """
    Loads model specs from the registry file.
    - Pre: `configs/models.yaml` exists and is valid YAML.
    - Post: Returns a validated tuple of ModelSpec objects.
        The result is cached for repeated calls.
    """
    path = paths.configs_dir() / "models.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    specs = tuple(ModelSpec(**entry) for entry in data["models"])
    _validate(specs)
    return specs


def _validate(specs: "tuple[ModelSpec, ...]") -> None:
    """
    Validates registry entries.
    - Pre: `specs` is a tuple of ModelSpec objects.
    - Post: Returns None when all entries are valid.
        Raises ValueError on invalid scale, duplicate id, or empty registry.
    """
    if not specs:
        raise ValueError("The model registry is empty. Please check configs/models.yaml.")
    seen: set[str] = set()
    for s in specs:
        if s.scale not in _VALID_SCALES:
            raise ValueError(f"{s.key!r}: unknown scale {s.scale!r} (expected one of {_VALID_SCALES})")
        if s.model_id in seen:
            raise ValueError(f"duplicate model id: {s.model_id}")
        seen.add(s.model_id)


def all_models() -> "tuple[ModelSpec, ...]":
    """
    Returns all registered model specs.
    - Pre: No input is required.
    - Post: Returns the cached tuple from the registry loader.
    """
    return load_registry()


def by_index(index: int) -> ModelSpec:
    """
    Resolves a model spec by numeric index.
    - Pre: `index` is an integer.
    - Post: Returns the ModelSpec at `index`.
        Raises IndexError if the index is out of range.
    """
    reg = load_registry()
    if not (0 <= index < len(reg)):
        raise IndexError(f"model index {index} out of range 0..{len(reg) - 1}")
    return reg[index]


def by_id(model_id: str) -> ModelSpec:
    """
    Resolves a model spec by model id.
    - Pre: `model_id` is a string in `scale/key` format.
    - Post: Returns the matching ModelSpec.
        Raises KeyError if no match exists.
    """
    reg = load_registry()
    for s in reg:
        if s.model_id == model_id:
            return s
    valid = ", ".join(s.model_id for s in reg)
    raise KeyError(f"no model {model_id!r}; valid ids: {valid}")