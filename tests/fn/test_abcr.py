"""Tests for moirais.fn.abcr -- ABC rejection."""

import numpy as np
from moirais.fn.abcr import abc_rejection


def _simulator(theta):
    rng = np.random.default_rng(int(abs(theta[0] * 1000)) % (2**31))
    return np.array([rng.normal(theta[0], 1.0)])


def _prior(rng):
    return rng.uniform(-5, 5, size=1)


def test_returns_dict():
    result = abc_rejection(_simulator, _prior, [0.0], epsilon=2.0, n_particles=500, n_accepted=10)
    assert isinstance(result, dict)
    assert "accepted_params" in result


def test_acceptance_count():
    result = abc_rejection(_simulator, _prior, [0.0], epsilon=5.0, n_particles=1000, n_accepted=20)
    assert result["n_accepted"] >= 1


def test_distances_within_epsilon():
    result = abc_rejection(_simulator, _prior, [0.0], epsilon=3.0, n_particles=500, n_accepted=10)
    assert all(d <= 3.0 for d in result["distances"])
