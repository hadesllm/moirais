"""Tests for BLUP spatial."""
import numpy as np
from moirais.fn.sgblp import sgblp


def test_sgblp_smoke():
    n = 10
    rng = np.random.default_rng(6)
    coords = rng.uniform(0, 5, (n, 2))
    X = np.column_stack([np.ones(n), coords[:, 0]])
    Z = X @ np.array([1.0, 2.0]) + rng.normal(0, 0.5, n)
    C = np.eye(n) + 0.3 * np.exp(-np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)))
    r = sgblp(Z, coords, X, C)
    assert r.name == "blup_spatial"
    assert "beta_gls" in r.extra


def test_cheatsheet():
    from moirais.fn.sgblp import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
