"""Tests for morie.fn.dpmmx -- DP mixture model."""

import numpy as np
from morie.fn.dpmmx import dp_mixture_model


def test_returns_dict():
    data = np.random.default_rng(42).standard_normal(40)
    result = dp_mixture_model(data, n_iter=30)
    assert isinstance(result, dict)
    assert "n_active" in result


def test_n_active_positive():
    data = np.random.default_rng(42).standard_normal(40)
    result = dp_mixture_model(data, n_iter=30)
    assert result["n_active"] >= 1


def test_weights_sum_to_one():
    data = np.random.default_rng(42).standard_normal(40)
    result = dp_mixture_model(data, n_iter=30)
    assert abs(sum(result["weights"]) - 1.0) < 0.05
