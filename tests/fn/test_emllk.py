"""Tests for moirais.fn.emllk -- EM log-likelihood."""
import numpy as np
from moirais.fn.emllk import em_log_likelihood, emllk


def test_alias():
    assert emllk is em_log_likelihood


def test_smoke():
    votes = np.array([[1, 0], [0, 1]], dtype=float)
    theta = np.array([1.0, -1.0])
    alpha = np.array([1.0, 1.0])
    beta = np.array([0.0, 0.0])
    r = em_log_likelihood(votes, theta, alpha, beta)
    assert r.name == "em_log_likelihood"
    assert r.value < 0  # log-lik is negative


def test_perfect_fit():
    votes = np.array([[1, 0]], dtype=float)
    theta = np.array([5.0])
    alpha = np.array([2.0, 2.0])
    beta = np.array([0.0, 0.0])
    r = em_log_likelihood(votes, theta, alpha, beta)
    assert r.value < 0
