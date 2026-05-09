"""Tests for moirais.fn.nusnc — nuisance parameter profiling."""

import numpy as np
import pytest

from moirais.fn.nusnc import nusnc


def test_normal_mean_variance():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200) + 2.0

    def ll(data, theta):
        mu, log_sig = theta[0], theta[1]
        sig = np.exp(log_sig)
        return float(np.sum(-0.5 * np.log(2 * np.pi * sig ** 2) - (data - mu) ** 2 / (2 * sig ** 2)))

    result = nusnc(x, ll, n_interest=1, theta0=np.array([0.0, 0.0]))
    assert result["psi_mle"][0] == pytest.approx(2.0, abs=0.3)


def test_profile_ll_shape():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(100)

    def ll(data, theta):
        return float(-np.sum((data - theta[0]) ** 2))

    result = nusnc(x, ll, n_interest=1, theta0=np.array([0.0, 0.0]), grid_points=30)
    assert len(result["profile_ll"]) == 30


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        nusnc(np.array([]), lambda d, t: 0.0)
