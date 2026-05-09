"""Tests for hrzycp.horowitz_conditional_prediction."""
import numpy as np
import pytest
from moirais.fn.hrzycp import horowitz_conditional_prediction


def test_hrzycp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y_threshold = np.random.default_rng(42).normal(0, 1, 100)
    T_hat = np.random.default_rng(42).normal(0, 1, 100)
    F_hat = np.random.default_rng(42).normal(0, 1, 100)
    beta_hat = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_conditional_prediction(x, y_threshold, T_hat, F_hat, beta_hat)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzycp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y_threshold = np.random.default_rng(42).normal(0, 1, 100)
    T_hat = np.random.default_rng(42).normal(0, 1, 100)
    F_hat = np.random.default_rng(42).normal(0, 1, 100)
    beta_hat = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_conditional_prediction(x, y_threshold, T_hat, F_hat, beta_hat)
    assert isinstance(result, dict)
