"""Tests for matau2pi.ma_tau2_predict_interval."""
import numpy as np
import pytest
from moirais.fn.matau2pi import ma_tau2_predict_interval


def test_matau2pi_basic():
    """Test basic functionality."""
    theta = 0.0
    se = np.random.default_rng(42).normal(0, 1, 100)
    tau2 = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    level = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_tau2_predict_interval(theta, se, tau2, k, level)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_matau2pi_edge():
    """Test edge cases."""
    theta = 0.0
    se = np.random.default_rng(42).normal(0, 1, 100)
    tau2 = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    level = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_tau2_predict_interval(theta, se, tau2, k, level)
    assert isinstance(result, dict)
