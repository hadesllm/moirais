"""Tests for moirais.fn.bpred -- Bayesian predictive distribution."""

import numpy as np
from moirais.fn.bpred import bayesian_predictive


def test_returns_dict():
    result = bayesian_predictive([1, 2, 3, 4, 5])
    assert isinstance(result, dict)
    assert "predictive_mean" in result
    assert "predictive_samples" in result


def test_predictive_interval_contains_mean():
    result = bayesian_predictive([1, 2, 3, 4, 5])
    assert result["pi_lower"] < result["predictive_mean"] < result["pi_upper"]


def test_predictive_wider_than_posterior():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 100)
    result = bayesian_predictive(data)
    pi_width = result["pi_upper"] - result["pi_lower"]
    assert pi_width > 2.0


def test_n_draws():
    result = bayesian_predictive([1, 2, 3], n_draws=1000)
    assert len(result["predictive_samples"]) == 1000


def test_large_sample_mean_near_data():
    rng = np.random.default_rng(42)
    data = rng.normal(5.0, 1.0, 1000)
    result = bayesian_predictive(data, prior_mu=0, prior_kappa=0.01)
    assert abs(result["predictive_mean"] - 5.0) < 0.3


def test_empty_data():
    try:
        bayesian_predictive([])
        assert False
    except ValueError:
        pass
