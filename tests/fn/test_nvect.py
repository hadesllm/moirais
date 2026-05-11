"""Tests for normal vector projection."""
import numpy as np
from morie.fn.nvect import nvect


def test_nvect_smoke():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 2))
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + rng.standard_normal(50) * 0.1
    r = nvect(X, y)
    assert r.name == "normal_vector_projection"
    assert r.extra["r_squared"] > 0.5
    assert "normal_vector" in r.extra


def test_cheatsheet():
    from morie.fn.nvect import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
