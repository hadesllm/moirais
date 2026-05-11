"""Tests for fractal dimension."""
import numpy as np
import pytest
from morie.fn.frcdm import fractal_dimension, frcdm


def test_line():
    x = np.arange(100, dtype=float)
    r = fractal_dimension(x)
    assert 0.8 < r.estimate < 1.5


def test_random():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 500)
    r = fractal_dimension(x)
    assert r.estimate > 1.0


def test_alias():
    assert frcdm is fractal_dimension


def test_too_short():
    with pytest.raises(ValueError):
        fractal_dimension([1, 2, 3])
