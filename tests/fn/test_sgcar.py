"""Tests for CAR model."""
import numpy as np
from morie.fn.sgcar import sgcar


def test_sgcar_smoke():
    rng = np.random.default_rng(17)
    n = 12
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i-1), min(n, i+2)):
            if i != j:
                W[i, j] = 1.0
    Z = rng.normal(0, 1, n)
    r = sgcar(Z, W)
    assert r.name == "conditional_autoregressive"
    assert 0 < r.statistic < 1
    assert "beta" in r.extra
    assert "tau2" in r.extra


def test_cheatsheet():
    from morie.fn.sgcar import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
