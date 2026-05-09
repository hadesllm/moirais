"""Tests for moirais.fn.bpchk -- Posterior predictive check."""

import numpy as np
from moirais.fn.bpchk import posterior_predictive_check


def test_returns_dict():
    y_obs = np.random.default_rng(42).standard_normal(50)
    y_rep = np.random.default_rng(43).standard_normal((100, 50))
    result = posterior_predictive_check(y_obs, y_rep)
    assert isinstance(result, dict)
    assert "p_value" in result


def test_p_value_in_range():
    y_obs = np.random.default_rng(42).standard_normal(50)
    y_rep = np.random.default_rng(43).standard_normal((100, 50))
    result = posterior_predictive_check(y_obs, y_rep)
    assert 0 <= result["p_value"] <= 1


def test_extreme_observed():
    y_obs = np.ones(50) * 100
    y_rep = np.random.default_rng(42).standard_normal((200, 50))
    result = posterior_predictive_check(y_obs, y_rep)
    assert result["p_value"] < 0.05
