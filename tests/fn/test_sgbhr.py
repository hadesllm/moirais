"""Tests for Bayesian hierarchical spatial model."""
import numpy as np
from moirais.fn.sgbhr import sgbhr


def test_sgbhr_smoke():
    rng = np.random.default_rng(21)
    n = 10
    coords = rng.uniform(0, 5, (n, 2))
    X = np.column_stack([np.ones(n), coords[:, 0]])
    Z = X @ np.array([1.0, 0.5]) + rng.normal(0, 0.5, n)
    r = sgbhr(Z, X, coords, n_samples=100)
    assert r.name == "bayesian_hierarchical_spatial"
    assert "beta_samples" in r.extra
    assert r.extra["beta_samples"].shape == (100, 2)


def test_cheatsheet():
    from moirais.fn.sgbhr import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
