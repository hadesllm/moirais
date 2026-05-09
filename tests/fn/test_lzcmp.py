"""Tests for Lempel-Ziv complexity."""
import numpy as np
from moirais.fn.lzcmp import lempel_ziv_complexity, lzcmp


def test_constant():
    x = np.ones(100)
    r = lempel_ziv_complexity(x)
    assert r.estimate >= 1


def test_random_higher():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r = lempel_ziv_complexity(x)
    assert r.estimate > 1


def test_alias():
    assert lzcmp is lempel_ziv_complexity
