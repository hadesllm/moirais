"""Tests for morie.fn.cliffs — alias for Cliff's delta."""
import numpy as np

from morie.fn.cliffs import cliffs


def test_cliffs_is_callable():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(30)
    y = rng.standard_normal(30)
    result = cliffs(x, y)
    assert hasattr(result, "estimate")


def test_cliffs_same_as_cliff():
    from morie.fn.cliff import cliffs_delta
    rng = np.random.default_rng(42)
    x = rng.standard_normal(20)
    y = rng.standard_normal(20)
    assert cliffs(x, y).estimate == cliffs_delta(x, y).estimate
