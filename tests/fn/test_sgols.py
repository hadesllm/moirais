"""Tests for OLS spatial diagnostics."""
import numpy as np
from morie.fn.sgols import sgols


def test_sgols_smoke():
    rng = np.random.default_rng(10)
    n = 20
    coords = rng.uniform(0, 10, (n, 2))
    X = np.column_stack([np.ones(n), coords[:, 0]])
    Z = X @ np.array([1.0, 0.5]) + rng.normal(0, 0.5, n)
    r = sgols(Z, X, coords)
    assert r.name == "ols_spatial_diagnostics"
    assert "beta" in r.extra
    assert "r_squared" in r.extra


def test_cheatsheet():
    from morie.fn.sgols import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
