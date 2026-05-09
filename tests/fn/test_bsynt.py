"""Tests for moirais.fn.bsynt -- Bayesian synthetic control."""

import numpy as np
from moirais.fn.bsynt import bayesian_synthetic_control


def test_returns_dict():
    rng = np.random.default_rng(42)
    T, J = 40, 5
    Y_donors = rng.standard_normal((T, J))
    y_treat = Y_donors @ rng.uniform(0, 1, J) + rng.standard_normal(T) * 0.1
    y_treat[30:] += 3
    result = bayesian_synthetic_control(y_treat, Y_donors, t0=30, n_iter=200)
    assert isinstance(result, dict)
    assert "effect_mean" in result


def test_effect_length():
    rng = np.random.default_rng(42)
    Y_donors = rng.standard_normal((20, 3))
    y_treat = rng.standard_normal(20)
    result = bayesian_synthetic_control(y_treat, Y_donors, t0=15, n_iter=100)
    assert len(result["effect_mean"]) == 5


def test_weight_length():
    rng = np.random.default_rng(42)
    Y_donors = rng.standard_normal((20, 4))
    y_treat = rng.standard_normal(20)
    result = bayesian_synthetic_control(y_treat, Y_donors, t0=15, n_iter=100)
    assert len(result["weight_mean"]) == 4
