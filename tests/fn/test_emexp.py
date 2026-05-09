"""Tests for moirais.fn.emexp -- EM expectation step."""
import numpy as np
from moirais.fn.emexp import em_expectation_step, emexp


def test_alias():
    assert emexp is em_expectation_step


def test_smoke():
    votes = np.array([[1, 0, 1], [0, 1, 0]], dtype=float)
    theta = np.array([0.5, -0.5])
    alpha = np.array([1.0, 1.0, 1.0])
    beta = np.array([0.0, 0.0, 0.0])
    r = em_expectation_step(theta, alpha, beta, votes)
    assert r.name == "em_expectation_step"
    assert r.extra["Q"] < 0  # log-lik is negative
    assert r.extra["n_legislators"] == 2
