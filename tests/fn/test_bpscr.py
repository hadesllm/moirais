"""Tests for moirais.fn.bpscr -- Bayesian propensity score."""

import numpy as np
from moirais.fn.bpscr import bayesian_propensity


def test_returns_dict():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((80, 2))
    t = (rng.uniform(size=80) < 0.5).astype(float)
    result = bayesian_propensity(X, t, n_samples=100)
    assert isinstance(result, dict)
    assert "propensity_mean" in result


def test_probs_in_range():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 2))
    t = (rng.uniform(size=50) < 0.5).astype(float)
    result = bayesian_propensity(X, t, n_samples=100)
    assert all(0 <= p <= 1 for p in result["propensity_mean"])


def test_propensity_length():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((40, 2))
    t = (rng.uniform(size=40) < 0.5).astype(float)
    result = bayesian_propensity(X, t, n_samples=50)
    assert len(result["propensity_mean"]) == 40
