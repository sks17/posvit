"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
Structural contracts: the registry ordering, artifact naming, and configuration.

These need no dataset, no network and no GPU, so they run on every push.
"""

from __future__ import annotations

import dataclasses

import pytest

from posvit.cli import COMMANDS, array_spec
from posvit.config import load_config
from posvit.paths import vendored
from posvit.registry import all_models, by_index, load_registry

# The position of each model in configs/models.yaml is the SLURM array contract: task k
# must always resolve to the same checkpoint, or historical results stop lining up. This
# is pinned against load_registry(), the function every consumer resolves through, rather
# than against any module that keeps its own copy of the ordering.
EXPECTED_ORDER = [
    ("vit_s", "ape"),
    ("vit_s", "axial"),
    ("vit_s", "axial_ape"),
    ("vit_s", "mixed"),
    ("vit_s", "mixed_ape"),
    ("vit_b", "ape"),
    ("vit_b", "mixed"),
    ("vit_b", "mixed_ape"),
]


def test_index_contract_holds():
    """
    Checks the registry ordering matches the array contract.
    """
    assert [(s.scale, s.key) for s in load_registry()] == EXPECTED_ORDER


def test_by_index_resolves_through_the_registry():
    """
    Checks index lookup returns the registry entry at that position.
    """
    for i, spec in enumerate(load_registry()):
        assert by_index(i) is spec
    with pytest.raises(IndexError):
        by_index(len(load_registry()))


def test_safe_id_is_scale_qualified_and_unique():
    """
    Checks every artifact id carries its scale and no two models collide.
    """
    for spec in all_models():
        assert spec.safe_id == f"{spec.scale}__{spec.key}"
        assert "/" not in spec.safe_id
    assert len({s.safe_id for s in all_models()}) == len(all_models())


def test_unknown_config_key_is_rejected():
    """
    Checks a misspelled configuration key fails at load rather than later.
    """
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config({"btch_size": 8})


@pytest.mark.parametrize(
    "override, match",
    [
        ({"ci_alpha": 2.0}, "ci_alpha"),
        ({"acc_floor": 1.5}, "acc_floor"),
        ({"tost_margin": 0.05}, "tost_margins_sensitivity"),
        ({"per_class_correction": "holm"}, "per_class_correction"),
        ({"reliance_negative_class": "some"}, "reliance_negative_class"),
    ],
)
def test_invalid_config_value_is_rejected(override, match):
    """
    Checks each validated threshold refuses an out-of-range value.
    """
    with pytest.raises(ValueError, match=match):
        load_config(override)


def test_config_is_immutable(cfg):
    """
    Checks a run cannot change its own settings midway.
    """
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.batch_size = 1


def test_every_declared_threshold_is_serialized(cfg):
    """
    Checks the values that decide a reported number travel with every artifact.
    """
    declared = cfg.to_dict()
    for field in ("acc_floor", "tost_margin", "strat_n_bins", "wb_acc_floor", "artifact_k"):
        assert field in declared


def test_missing_vendored_path_fails_loudly():
    """
    Checks a missing vendored repository raises with the locations searched.
    """
    with pytest.raises(FileNotFoundError) as excinfo:
        vendored("definitely-not-a-real-repository")
    assert "Looked in" in str(excinfo.value)


@pytest.mark.parametrize(
    "command, expected",
    [
        ("evaluate", "0-7"),
        ("mechanisms", "3-4,6-7"),
        ("waterbirds", "0-4"),
    ],
)
def test_array_specs_are_derived_from_the_registry(command, expected):
    """
    Checks each SLURM array range is computed rather than transcribed.
    """
    assert array_spec(COMMANDS[command]) == expected


def test_every_command_requirement_names_a_real_command():
    """
    Checks the declared pipeline ordering has no dangling dependency.
    """
    for command in COMMANDS.values():
        for required in command.requires:
            assert required in COMMANDS, f"{command.name} requires unknown {required!r}"
