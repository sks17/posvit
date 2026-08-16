"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
Refusals and reproducibility.

Most of this package's value is in what it declines to do: an unpinned checkpoint, a
lossy join, an equivalence test without a declared margin. Those paths are asserted here
rather than assumed.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from posvit.equivalence import tost
from posvit.metrics import join_variants
from posvit.probes import CHANCE, decodability, fg_disjoint_split, reliance_label
from posvit.seed import seed_everything


def test_seeding_is_reproducible_across_every_generator():
    """
    Checks the three random number generators are all seeded.
    """

    def draw():
        seed_everything(0)
        return random.random(), float(np.random.rand()), torch.randn(1).item()

    assert draw() == draw()


def test_seed_receipt_is_serializable():
    """
    Checks the provenance receipt can be embedded in a JSON artifact.
    """
    import json

    json.dumps(seed_everything(0))


def test_lossy_join_is_refused(prediction_records):
    """
    Checks a foreground present in one variant only stops the reduction.
    """
    ms, mr = prediction_records
    with pytest.raises(ValueError, match="lossy"):
        join_variants(ms, mr[:-5])


def test_join_reports_its_losses_when_not_strict(prediction_records):
    """
    Checks the loss report is populated rather than silently absorbed.
    """
    ms, mr = prediction_records
    join = join_variants(ms, mr[:-5], strict=False)
    assert join.losses["n_only_ms"] == 5
    assert not join.losses["lossless"]


def test_duplicate_foreground_ids_are_refused(prediction_records):
    """
    Checks a repeated key cannot silently overwrite its predecessor.
    """
    ms, _ = prediction_records
    duplicated = ms + [dict(ms[0])]
    with pytest.raises(ValueError, match="lossy"):
        join_variants(duplicated, duplicated)


def test_join_carries_labels(prediction_records):
    """
    Checks stratified analyses can subset the join instead of re-deriving one.
    """
    ms, mr = prediction_records
    join = join_variants(ms, mr)
    assert join.labels is not None
    assert len(join.labels) == join.n


def test_equivalence_requires_an_explicit_margin():
    """
    Checks the margin cannot be inherited from a signature default.

    The margin is the definition of "equivalent" for the study, so a caller that omits
    it must fail rather than silently receive a different claim.
    """
    a = np.ones(10, dtype=bool)
    with pytest.raises(TypeError, match="margin"):
        tost(a, ~a, a, ~a)


def test_equivalence_uses_the_two_alpha_interval():
    """
    Checks TOST reads the 90 percent interval at alpha = 0.05, not the 95 percent one.
    """
    rng = np.random.default_rng(0)
    correct = rng.random(400) < 0.9
    result = tost(correct, ~correct, correct, ~correct, margin=0.02, n=200, seed=0)
    assert result["ci_level"] == pytest.approx(0.90)


def test_equivalence_refuses_a_real_difference():
    """
    Checks the test can return a negative verdict.

    An equivalence test that never declines is not evidence of anything.
    """
    rng = np.random.default_rng(0)
    n = 2000
    ms_a, mr_a = rng.random(n) < 0.90, rng.random(n) < 0.85
    ms_b, mr_b = rng.random(n) < 0.90, rng.random(n) < 0.75
    result = tost(ms_a, mr_a, ms_b, mr_b, margin=0.02, n=400, seed=0)
    assert not result["equivalent"]


def test_reliance_label_matches_the_difference_vector(paired_join):
    """
    Checks the per-image reliance label is the +1 entries of the difference vector.
    """
    keep_all, label_all = reliance_label(paired_join, negative_class="all")
    keep_disc, label_disc = reliance_label(paired_join, negative_class="discordant")
    assert keep_all.sum() == paired_join.n
    assert label_all.sum() == label_disc.sum() == 2
    assert keep_disc.sum() == 3


def test_probe_split_shares_no_foreground_object():
    """
    Checks the probe split is on identities so no object leaks across halves.
    """
    fg_ids = [f"fg{i // 3}" for i in range(90)]
    train, test = fg_disjoint_split(fg_ids, test_frac=0.5, seed=0)
    train_ids = {fg_ids[i] for i in np.flatnonzero(train)}
    test_ids = {fg_ids[i] for i in np.flatnonzero(test)}
    assert train.any() and test.any()
    assert not (train_ids & test_ids)


def test_probe_separates_signal_from_noise(cfg):
    """
    Checks the probe recovers a class signal and finds none in noise.

    A positive and a negative control together: an instrument that reports signal in
    noise cannot support a null result elsewhere.
    """
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(9), 40)
    fg_ids = [f"fg{i}" for i in range(len(labels))]
    centers = rng.normal(size=(9, 32)) * 4.0
    separable = (centers[labels] + rng.normal(size=(len(labels), 32))).astype(np.float32)
    noise = rng.normal(size=(len(labels), 32)).astype(np.float32)

    assert decodability(separable, labels, fg_ids, cfg)["accuracy"] > 0.6
    assert decodability(noise, labels, fg_ids, cfg)["accuracy"] < 0.45
    assert CHANCE == pytest.approx(1 / 9)
