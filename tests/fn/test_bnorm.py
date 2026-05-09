"""Tests for moirais.fn.bnorm -- Bayesian normal model."""

import numpy as np
from moirais.fn.bnorm import bayesian_normal


def test_returns_dict():
    result = bayesian_normal([1, 2, 3])
    assert isinstance(result, dict)
    assert "posterior_mean" in result
    assert "posterior_sd" in result


def test_large_sample_converges_to_data_mean():
    rng = np.random.default_rng(42)
    data = rng.normal(5.0, 1.0, 10000)
    result = bayesian_normal(data, prior_mu=0, prior_sigma=10, sigma_known=1)
    assert abs(result["posterior_mean"] - 5.0) < 0.1


def test_small_sample_shrinks_to_prior():
    result = bayesian_normal(
        [10.0], prior_mu=0.0, prior_sigma=1.0, sigma_known=1.0
    )
    assert result["posterior_mean"] < 10.0
    assert result["posterior_mean"] > 0.0


def test_weights_sum_to_one():
    result = bayesian_normal([1, 2, 3], prior_mu=0, prior_sigma=1, sigma_known=1)
    np.testing.assert_allclose(
        result["prior_weight"] + result["data_weight"], 1.0, atol=1e-10
    )


def test_ci_contains_mean():
    result = bayesian_normal([1, 2, 3])
    assert result["ci_lower"] < result["posterior_mean"] < result["ci_upper"]


def test_empty_data():
    try:
        bayesian_normal([])
        assert False
    except ValueError:
        pass


def test_known_posterior():
    data = [0.0]
    result = bayesian_normal(data, prior_mu=0, prior_sigma=1, sigma_known=1)
    expected_post_var = 1.0 / (1.0 + 1.0)
    np.testing.assert_allclose(result["posterior_sd"] ** 2, expected_post_var, atol=1e-10)
