"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, fields
from typing import Any
import yaml
from . import paths

_TUPLE_FIELDS = (
    "norm_mean",
    "norm_std",
    "plan_a",
    "plan_b_extra",
    "tost_margins_sensitivity",
    "strat_bins_sensitivity",
    "ape_lowpass_rings",
)

_RESIZE_POLICIES = ("none", "resize256_crop224")
_CORRECTIONS = ("bonferroni", "bh")
_NEGATIVE_CLASSES = ("all", "discordant")

@dataclass(frozen=True)
class Config:
    """
    Stores configuration parameters.

    Every field here is a value that changes a reported number, so none of them is
    stored as a function default or a module constant elsewhere in the package.
    - Pre: The dataclass fields use valid values.
    - Post: Creates an immutable config object.
    """
    seed: int = 0
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    norm_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    norm_std: tuple[float, ...] = (0.229, 0.224, 0.225)
    resize_policy: str = "none"
    bootstrap_n: int = 10000
    bootstrap_seed: int = 0
    ci_alpha: float = 0.05
    plan_a: tuple[str, ...] = ("original", "mixed_same", "mixed_rand", "only_fg", "only_bg_t")
    plan_b_extra: tuple[str, ...] = ("mixed_next",)
    mask_source_variant: str = "only_fg"
    # Interpretability guard. BG-Gap only measures background reliance while the model
    # still works; below this ORIGINAL accuracy the gap turns non-monotone.
    acc_floor: float = 0.80
    # Equivalence testing. `tost_margin` is the pre-declared definition of "equivalent";
    # the sensitivity list is a robustness check and not the test itself.
    tost_margin: float = 0.020
    tost_margins_sensitivity: tuple[float, ...] = (0.010, 0.015, 0.020)
    power_alpha: float = 0.05
    power_target: float = 0.80
    # Stratification. Bin count and within-class ranking both change the reported trend.
    strat_n_bins: int = 4
    strat_within_class: bool = False
    strat_min_bin_n: int = 100
    strat_bins_sensitivity: tuple[int, ...] = (3, 4, 5, 10)
    per_class_correction: str = "bonferroni"
    # Waterbirds. The floor differs from `acc_floor` because the chance rate differs:
    # IN-9 is nine-class (chance 0.11), Waterbirds is two-class (chance 0.50).
    wb_acc_floor: float = 0.70
    wb_head_seeds: int = 5
    wb_min_baseline_gap: float = 0.02
    # Mechanism probes.
    probe_steps: int = 300
    probe_lr: float = 0.05
    probe_weight_decay: float = 0.001
    probe_test_frac: float = 0.5
    probe_seeds: int = 3
    auc_boot_n: int = 2000
    reliance_negative_class: str = "all"
    artifact_k: float = 3.0
    probe_auc_support: float = 0.55
    probe_excess_support_pp: float = 1.0
    rope_band_frac: float = 0.25
    ape_lowpass_rings: tuple[int, ...] = (1, 2, 3, 5, 7)

    def __post_init__(self) -> None:
        """
        Validates config field values after initialization.
        - Pre: Dataclass fields are initialized.
        - Post: Returns None when all values are valid.
            Raises ValueError for invalid configuration values.
        """
        if len(self.norm_mean) != 3 or len(self.norm_std) != 3:
            raise ValueError("norm_mean and norm_std must each have exactly 3 values")
        if self.img_size <= 0 or self.batch_size <= 0:
            raise ValueError("img_size and batch_size must be positive")
        if self.bootstrap_n <= 0:
            raise ValueError("bootstrap_n must be positive")
        if not (0.0 < self.ci_alpha < 1.0):
            raise ValueError("ci_alpha must be in the open interval (0, 1)")
        if self.resize_policy not in _RESIZE_POLICIES:
            raise ValueError(f"unknown resize_policy: {self.resize_policy!r}")
        if not self.plan_a:
            raise ValueError("plan_a must list at least one variant")
        if not (0.0 < self.acc_floor < 1.0):
            raise ValueError("acc_floor must be in the open interval (0, 1)")
        if not (0.0 < self.tost_margin < 1.0):
            raise ValueError("tost_margin must be in the open interval (0, 1)")
        if self.tost_margin not in self.tost_margins_sensitivity:
            raise ValueError("tost_margin must appear in tost_margins_sensitivity")
        if not (0.0 < self.power_alpha < 1.0) or not (0.0 < self.power_target < 1.0):
            raise ValueError("power_alpha and power_target must be in the open interval (0, 1)")
        if self.strat_n_bins < 2:
            raise ValueError("strat_n_bins must be at least 2")
        if self.strat_n_bins not in self.strat_bins_sensitivity:
            raise ValueError("strat_n_bins must appear in strat_bins_sensitivity")
        if self.strat_min_bin_n < 1:
            raise ValueError("strat_min_bin_n must be at least 1")
        if self.per_class_correction not in _CORRECTIONS:
            raise ValueError(f"unknown per_class_correction: {self.per_class_correction!r}")
        if not (0.0 < self.wb_acc_floor < 1.0):
            raise ValueError("wb_acc_floor must be in the open interval (0, 1)")
        if self.wb_head_seeds < 1:
            raise ValueError("wb_head_seeds must be at least 1")
        if not (0.0 <= self.wb_min_baseline_gap < 1.0):
            raise ValueError("wb_min_baseline_gap must be in the half-open interval [0, 1)")
        if self.probe_steps < 1 or self.probe_seeds < 1:
            raise ValueError("probe_steps and probe_seeds must be at least 1")
        if not (0.0 < self.probe_test_frac < 1.0):
            raise ValueError("probe_test_frac must be in the open interval (0, 1)")
        if self.auc_boot_n < 1:
            raise ValueError("auc_boot_n must be at least 1")
        if self.reliance_negative_class not in _NEGATIVE_CLASSES:
            raise ValueError(
                f"unknown reliance_negative_class: {self.reliance_negative_class!r}"
            )
        if self.artifact_k <= 0.0:
            raise ValueError("artifact_k must be positive")
        if not (0.5 < self.probe_auc_support < 1.0):
            raise ValueError("probe_auc_support must be in the open interval (0.5, 1)")
        if not (0.0 < self.rope_band_frac < 1.0):
            raise ValueError("rope_band_frac must be in the open interval (0, 1)")
        if not self.ape_lowpass_rings:
            raise ValueError("ape_lowpass_rings must list at least one ring count")

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the config object to a dictionary.
        - Pre: `self` is a valid Config object.
        - Post: Returns a dictionary with all config fields.
        """
        return asdict(self)


def _read_yaml(path) -> dict[str, Any]:
    """
    Reads a YAML file.
    - Pre: `path` points to a readable YAML file.
    - Post: Returns the parsed data as a dictionary.
        Returns an empty dictionary if the file is empty.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(overrides: "dict[str, Any] | None" = None) -> Config:
    """
    Loads the project configuration.
    - Pre: `overrides` is None or a dictionary of config key-value pairs.
    - Post: Returns a validated Config object.
        Values from `overrides` replace values from YAML files.
        Raises ValueError for unknown keys or invalid values.
    """
    cfg_dir = paths.configs_dir()
    data = _read_yaml(cfg_dir / "base.yaml")
    data.update(_read_yaml(cfg_dir / "variants.yaml"))
    if overrides:
        data.update(overrides)
    for k in _TUPLE_FIELDS:
        if k in data and isinstance(data[k], list):
            data[k] = tuple(data[k])
    known = {f.name for f in fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}; known: {sorted(known)}")
    return Config(**data)
