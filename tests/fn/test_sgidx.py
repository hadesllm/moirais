"""Tests for index of dispersion."""
import numpy as np
from morie.fn.sgidx import sgidx


def test_sgidx_smoke():
    counts = np.array([5, 3, 7, 4, 6, 5, 4, 8, 3, 5])
    r = sgidx(counts)
    assert r.name == "index_of_dispersion"
    assert "VMR" in r.extra
    assert "pattern" in r.extra


def test_sgidx_poisson():
    rng = np.random.default_rng(42)
    counts = rng.poisson(10, 100)
    r = sgidx(counts)
    assert 0.5 < r.extra["VMR"] < 2.0
