"""Test pcorp."""
import numpy as np
import pytest
from moirais.fn.pcorp import partial_correlation


def test_pcorp_basic():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 4))
    r = partial_correlation(X, i=0, j=1, controls=[2, 3])
    assert -1.0 <= r.value <= 1.0
    assert r.name == "pcorp"


def test_pcorp_no_controls():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((50, 2))
    r = partial_correlation(X, i=0, j=1, controls=[])
    assert -1.0 <= r.value <= 1.0
