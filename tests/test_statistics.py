"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
Closed-form statistical identities.

Each of these has an exact answer that can be checked by hand, so a regression in the
statistics surfaces immediately rather than as a slightly different published number.
"""

from __future__ import annotations

import numpy as np
import pytest

from posvit.doseresponse import bh_fdr
from posvit.metrics import bg_gap, mcnemar_test
from posvit.probes import auc
from posvit.stratify import quantile_bins


def test_bggap_equals_the_mcnemar_cells(paired_join):
    """
    Checks BG-Gap equals (b - c) / N exactly.
    """
    cells = mcnemar_test(paired_join)
    expected = (cells["b_only_ms"] - cells["c_only_mr"]) / paired_join.n
    assert bg_gap(paired_join) == pytest.approx(expected, abs=1e-12)


def test_bggap_equals_the_paired_accuracy_difference(paired_join):
    """
    Checks BG-Gap equals acc(mixed-same) minus acc(mixed-rand).
    """
    difference = float(paired_join.ms.mean() - paired_join.mr.mean())
    assert bg_gap(paired_join) == pytest.approx(difference, abs=1e-12)


def test_mcnemar_uses_the_continuity_corrected_statistic():
    """
    Checks the asymptotic statistic is (|b - c| - 1)^2 / (b + c).

    Worked against the published ViT-S/APE cells: b = 328, c = 120 gives
    207^2 / 448 = 95.64508928571429.
    """
    b, c = 328, 120
    assert (abs(b - c) - 1) ** 2 / (b + c) == pytest.approx(95.64508928571429, abs=1e-9)
    assert (b - c) / 4050 == pytest.approx(0.051358, abs=1e-6)


def test_bh_fdr_is_a_step_up_procedure():
    """
    Checks rejection extends to the largest passing index, not only per-value passes.

    With m = 5 and q = 0.05, p = 0.025 exceeds its own threshold of 0.02, yet the
    step-up rejects it because a later index passes.
    """
    pvalues = [0.001, 0.025, 0.03, 0.039, 0.04]
    naive = np.asarray(pvalues) <= 0.05 * np.arange(1, 6) / 5
    assert not naive[1]
    assert bh_fdr(pvalues).all()


def test_bh_fdr_rejects_nothing_when_every_pvalue_is_large():
    """
    Checks the correction can decline to reject.
    """
    assert not bh_fdr([0.9, 0.8, 0.7]).any()


@pytest.mark.parametrize(
    "score, expected",
    [
        ([9, 8, 7, 3, 2, 1], 1.0),
        ([1, 2, 3, 7, 8, 9], 0.0),
    ],
)
def test_auc_reaches_both_endpoints(score, expected):
    """
    Checks a perfectly anti-predictive score scores 0, not 0.5.
    """
    label = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    assert auc(score, label) == pytest.approx(expected)


def test_quantile_bins_report_the_actual_count():
    """
    Checks tied values collapse edges and the report says so.
    """
    values = np.array([0.1] * 500 + [0.5] * 500)
    binning = quantile_bins(values, 4)
    assert binning.n_requested == 4
    assert binning.n_actual == 2
