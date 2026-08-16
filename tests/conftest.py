"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
Shared fixtures and tier markers for the test suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from posvit.config import load_config


def pytest_configure(config):
    """
    Registers the tier markers used to keep the default run data-free.
    - Pre: `config` is the pytest configuration object.
    - Post: Returns None after registering the `smoke` and `gpu` markers.
    """
    config.addinivalue_line("markers", "smoke: needs the IN-9 dataset (CPU, minutes)")
    config.addinivalue_line("markers", "gpu: needs a GPU and downloaded checkpoints")


@pytest.fixture(scope="session")
def cfg():
    """
    Provides the project configuration.
    - Pre: `configs/base.yaml` and `configs/variants.yaml` are readable.
    - Post: Returns a validated Config object shared across the session.
    """
    return load_config()


@pytest.fixture
def paired_join():
    """
    Builds a small PairedJoin with a known difference vector.

    Six foregrounds giving d = [+1, +1, 0, 0, -1, 0], so b = 2, c = 1 and N = 6.
    - Pre: No input is required.
    - Post: Returns a PairedJoin object.
    """
    from posvit.metrics import PairedJoin

    ms = np.array([1, 1, 1, 0, 0, 1], dtype=bool)
    mr = np.array([0, 0, 1, 0, 1, 1], dtype=bool)
    return PairedJoin(
        keys=tuple(f"fg{i}" for i in range(6)),
        ms=ms,
        mr=mr,
        losses={"lossless": True},
        labels=np.array([0, 0, 1, 1, 2, 2]),
    )


@pytest.fixture
def prediction_records():
    """
    Builds a minimal pair of prediction record lists that join losslessly.
    - Pre: No input is required.
    - Post: Returns `(ms_records, mr_records)` over the same 20 foreground ids.
    """
    ms = [{"fg_id": f"fg{i}", "pred9": i % 9, "label9": i % 9} for i in range(20)]
    mr = [{"fg_id": f"fg{i}", "pred9": (i + 1) % 9, "label9": i % 9} for i in range(20)]
    return ms, mr
