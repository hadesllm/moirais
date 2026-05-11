"""Tests for morie.fn.plest — profile likelihood estimation."""

import numpy as np
import pytest

from morie.fn.plest import plest


def test_normal_mean():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200) + 2.0

    def ll(data, theta):
        mu, log_sig = theta[0], theta[1]
        sig = np.exp(log_sig)
        return float(np.sum(-0.5 * np.log(2 * np.pi * sig ** 2) - (data - mu) ** 2 / (2 * sig ** 2)))

    result = plest(x, ll, n_params=2)
    assert result["mle"][0] == pytest.approx(2.0, abs=0.3)


def test_profile_ci_contains_mle():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(100) + 1.0

    def ll(data, theta):
        return float(-np.sum((data - theta[0]) ** 2))

    result = plest(x, ll, n_params=1)
    ci_lo, ci_hi = result["profile_ci"]
    assert ci_lo <= result["mle"][0] <= ci_hi


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        plest(np.array([]), lambda d, t: 0.0)
