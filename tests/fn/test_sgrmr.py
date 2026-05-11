"""Tests for Moran residual test."""
import numpy as np
from morie.fn.sgrmr import sgrmr


def test_sgrmr_smoke():
    rng = np.random.default_rng(15)
    n = 20
    resid = rng.normal(0, 1, n)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i-1), min(n, i+2)):
            if i != j:
                W[i, j] = 0.5
    r = sgrmr(resid, W)
    assert r.name == "moran_residual_test"
    assert r.p_value is not None
    assert 0 <= r.p_value <= 1


def test_cheatsheet():
    from morie.fn.sgrmr import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
