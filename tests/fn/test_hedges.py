"""Tests for morie.fn.hedges — alias for Hedges' g."""
import numpy as np

from morie.fn.hedges import hedges


def test_hedges_is_callable():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    y = rng.standard_normal(50) + 1
    result = hedges(x, y)
    assert isinstance(result, float)


def test_hedges_same_as_g():
    from morie.fn.g import hedges_g
    rng = np.random.default_rng(42)
    x = rng.standard_normal(30)
    y = rng.standard_normal(30)
    assert hedges(x, y) == hedges_g(x, y)
