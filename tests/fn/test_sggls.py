"""Tests for GLS spatial."""
import numpy as np
from moirais.fn.sggls import sggls


def test_sggls_smoke():
    n = 10
    rng = np.random.default_rng(11)
    X = np.column_stack([np.ones(n), rng.uniform(0, 5, n)])
    Z = X @ np.array([2.0, 1.5]) + rng.normal(0, 0.5, n)
    V = np.eye(n) + 0.2 * np.ones((n, n))
    r = sggls(Z, X, V)
    assert r.name == "gls_spatial"
    assert "beta" in r.extra
    assert len(r.extra["beta"]) == 2


def test_cheatsheet():
    from moirais.fn.sggls import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
