"""Tests for moirais.fn.npmle — Nonparametric MLE."""

import numpy as np
import pytest

from moirais.fn.npmle import npmle


@pytest.fixture()
def continuous_data():
    rng = np.random.default_rng(42)
    return rng.standard_normal(200)


@pytest.fixture()
def interval_data():
    rng = np.random.default_rng(42)
    n = 100
    true_vals = rng.exponential(2.0, size=n)
    left = np.floor(true_vals)
    right = left + 1.0
    return left, right


def test_returns_dict(continuous_data):
    result = npmle(continuous_data)
    assert isinstance(result, dict)
    for key in ("support", "weights", "cdf_values", "log_likelihood", "n_iter", "converged"):
        assert key in result


def test_weights_sum_to_one(continuous_data):
    result = npmle(continuous_data)
    np.testing.assert_allclose(result["weights"].sum(), 1.0, atol=1e-10)


def test_cdf_monotone(continuous_data):
    result = npmle(continuous_data)
    assert np.all(np.diff(result["cdf_values"]) >= -1e-12)


def test_cdf_ends_at_one(continuous_data):
    result = npmle(continuous_data)
    np.testing.assert_allclose(result["cdf_values"][-1], 1.0, atol=1e-10)


def test_converged(continuous_data):
    result = npmle(continuous_data)
    assert result["converged"] is True


def test_interval_censored(interval_data):
    left, right = interval_data
    result = npmle(left, left=left, right=right)
    assert result["converged"] is True
    np.testing.assert_allclose(result["weights"].sum(), 1.0, atol=1e-6)


def test_interval_cdf_monotone(interval_data):
    left, right = interval_data
    result = npmle(left, left=left, right=right)
    assert np.all(np.diff(result["cdf_values"]) >= -1e-10)


def test_support_matches_data(continuous_data):
    result = npmle(continuous_data)
    assert len(result["support"]) == len(np.unique(continuous_data))
