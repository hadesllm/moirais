"""Tests for normalized redundancy."""
import numpy as np
import pytest
from moirais.fn.normr import normalized_redundancy, normr


def test_identical():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 5000)
    r = normalized_redundancy(x, x, bins=10)
    assert r.estimate == pytest.approx(1.0, abs=0.15)


def test_independent():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 5000)
    y = rng.uniform(0, 1, 5000)
    r = normalized_redundancy(x, y, bins=10)
    assert r.estimate < 0.3


def test_alias():
    assert normr is normalized_redundancy
