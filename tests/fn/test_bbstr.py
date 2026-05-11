"""Tests for bbstr: Bayesian bootstrap."""

import numpy as np
import pytest

from morie.fn.bbstr import bayesian_bootstrap


def test_bbstr_basic():
    """Test Bayesian bootstrap on simple data."""
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)

    result = bayesian_bootstrap(x, n_boot=100)

    assert "statistic_samples" in result
    assert "posterior_mean" in result
    assert "posterior_std" in result
    assert len(result["statistic_samples"]) == 100


def test_bbstr_credible_interval():
    """Test credible interval computation."""
    x = np.random.normal(5, 1, 50)
    result = bayesian_bootstrap(x, n_boot=500)

    # For normal data with mean 5, credible interval should contain 5
    assert result["ci_lower"] < 5 < result["ci_upper"]


def test_bbstr_custom_statistic():
    """Test with custom statistic function."""
    x = np.array([1, 2, 3, 4, 5])

    def median_stat(data):
        return float(np.median(data))

    result = bayesian_bootstrap(x, statistic=median_stat, n_boot=100)

    assert "statistic_samples" in result
    # Posterior mean should be near the empirical median
    assert 2 < result["posterior_mean"] < 4


def test_bbstr_reproducibility():
    """Test reproducibility with seed."""
    x = np.random.normal(0, 1, 50)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    result1 = bayesian_bootstrap(x, n_boot=50, rng=rng1)
    result2 = bayesian_bootstrap(x, n_boot=50, rng=rng2)

    np.testing.assert_almost_equal(result1["posterior_mean"], result2["posterior_mean"])


def test_bbstr_empty_data():
    """Test error handling for empty data."""
    with pytest.raises(ValueError):
        bayesian_bootstrap(np.array([]))
