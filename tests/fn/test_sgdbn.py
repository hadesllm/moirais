"""Tests for spatial Durbin model."""
import numpy as np
from moirais.fn.sgdbn import sgdbn


def test_sgdbn_smoke():
    rng = np.random.default_rng(16)
    n = 15
    X = np.column_stack([np.ones(n), rng.uniform(0, 5, n)])
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i-2), min(n, i+3)):
            if i != j:
                W[i, j] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    W = W / rs
    Z = X @ np.array([1.0, 0.5]) + rng.normal(0, 1, n)
    r = sgdbn(Z, X, W)
    assert r.name == "spatial_durbin_model"
    assert "beta" in r.extra
    assert "theta" in r.extra


def test_cheatsheet():
    from moirais.fn.sgdbn import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
