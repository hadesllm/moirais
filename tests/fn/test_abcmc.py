"""Tests for moirais.fn.abcmc -- ABC-MCMC."""

import numpy as np
from moirais.fn.abcmc import abc_mcmc


def _simulator(theta):
    rng = np.random.default_rng(int(abs(theta[0] * 1000)) % (2**31))
    return np.array([rng.normal(theta[0], 1.0)])


def _prior_log(theta):
    if np.all(np.abs(theta) < 10):
        return 0.0
    return -np.inf


def test_returns_dict():
    result = abc_mcmc(_simulator, _prior_log, [0.0], [0.0], epsilon=3.0, n_iter=200)
    assert isinstance(result, dict)
    assert "samples" in result


def test_samples_shape():
    result = abc_mcmc(_simulator, _prior_log, [0.0], [0.0], epsilon=3.0, n_iter=100)
    assert result["samples"].shape == (100, 1)


def test_acceptance_positive():
    result = abc_mcmc(_simulator, _prior_log, [0.0], [0.0], epsilon=5.0, n_iter=500)
    assert result["acceptance_rate"] >= 0
