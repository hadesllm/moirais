"""Tests for moirais.fn.bayid — Bayesian ideal points."""
import numpy as np
from moirais.fn.bayid import bayid


def test_bayid_basic():
    rng = np.random.default_rng(42)
    V = rng.integers(0, 2, (4, 3)).astype(float)
    r = bayid(V, n_dims=1, n_iter=50, burnin=10)
    assert r.value["posterior_mean"].shape == (4, 1)


def test_bayid_ci():
    V = np.array([[1, 1, 0], [0, 0, 1]], dtype=float)
    r = bayid(V, n_iter=50, burnin=10)
    assert r.value["ci_lower"].shape == r.value["ci_upper"].shape


def test_bayid_acceptance():
    rng = np.random.default_rng(42)
    V = rng.integers(0, 2, (3, 2)).astype(float)
    r = bayid(V, n_iter=30, burnin=5)
    assert 0 <= r.extra["acceptance_rate"] <= 1
