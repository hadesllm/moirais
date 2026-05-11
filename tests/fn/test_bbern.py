"""Tests for morie.fn.bbern -- Bayesian Bernoulli model."""

import numpy as np
from morie.fn.bbern import bayesian_bernoulli


def test_returns_dict():
    result = bayesian_bernoulli(5, 10)
    assert isinstance(result, dict)
    assert "posterior_mean" in result


def test_conjugate_update():
    result = bayesian_bernoulli(7, 10, prior_a=1, prior_b=1)
    np.testing.assert_allclose(result["posterior_a"], 8.0)
    np.testing.assert_allclose(result["posterior_b"], 4.0)
    np.testing.assert_allclose(result["posterior_mean"], 8.0 / 12.0, atol=1e-10)


def test_uniform_prior_zero_data():
    result = bayesian_bernoulli(0, 0, prior_a=1, prior_b=1)
    np.testing.assert_allclose(result["posterior_mean"], 0.5)


def test_mode_exists():
    result = bayesian_bernoulli(5, 10, prior_a=2, prior_b=2)
    assert not np.isnan(result["posterior_mode"])


def test_mode_nan_for_small_params():
    result = bayesian_bernoulli(0, 0, prior_a=0.5, prior_b=0.5)
    assert np.isnan(result["posterior_mode"])


def test_ci_in_01():
    result = bayesian_bernoulli(5, 10)
    assert 0 <= result["ci_lower"] <= result["ci_upper"] <= 1


def test_invalid_successes():
    try:
        bayesian_bernoulli(11, 10)
        assert False
    except ValueError:
        pass


def test_negative_trials():
    try:
        bayesian_bernoulli(0, -1)
        assert False
    except ValueError:
        pass
