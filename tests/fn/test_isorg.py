"""Tests for moirais.fn.isorg -- isotonic regression."""

import numpy as np
from moirais.fn.isorg import isotonic_regression, isorg


def test_isorg_monotone():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    r = isorg(x)
    assert r.name == "isotonic_regression"
    assert np.allclose(r.value, x)


def test_isorg_violation():
    x = np.array([1.0, 3.0, 2.0, 4.0])
    r = isorg(x)
    assert r.value[1] <= r.value[2] or np.isclose(r.value[1], r.value[2])


def test_isorg_alias():
    assert isorg is isotonic_regression
