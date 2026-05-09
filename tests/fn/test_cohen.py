"""Tests for moirais.fn.cohen — alias for Cohen's d."""
import numpy as np

from moirais.fn.cohen import cohen


def test_cohen_is_callable():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    y = rng.standard_normal(50) + 1
    result = cohen(x, y)
    assert isinstance(result, float)


def test_cohen_same_as_d():
    from moirais.fn.d import cohens_d
    rng = np.random.default_rng(42)
    x = rng.standard_normal(30)
    y = rng.standard_normal(30)
    assert cohen(x, y) == cohens_d(x, y)
