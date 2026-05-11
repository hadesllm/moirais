"""Tests for morie.fn.difev -- Differential evolution."""

import numpy as np

from morie.fn.difev import difev


def test_sphere():
    """DE should minimize f(x) = sum(x^2) close to 0."""
    f = lambda x: np.sum(x ** 2)
    result = difev(f, [(-5, 5), (-5, 5)], seed=42)
    assert isinstance(result, np.ndarray)
    assert np.sum(result ** 2) < 1.0


def test_returns_ndarray():
    f = lambda x: (x[0] - 1.0) ** 2
    result = difev(f, [(-5, 5)], seed=0)
    assert isinstance(result, np.ndarray)


def test_full_output():
    f = lambda x: np.sum(x ** 2)
    x, info = difev(f, [(-5, 5), (-5, 5)], full_output=True, seed=1)
    assert isinstance(x, np.ndarray)
    assert "generations" in info
