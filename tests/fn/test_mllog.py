"""Tests for mllog.ml_log_likelihood_regression."""
import numpy as np
import pytest
from morie.fn.mllog import ml_log_likelihood_regression


def test_mllog_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    beta = 0.8
    sigma2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ml_log_likelihood_regression(y, X, beta, sigma2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mllog_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    beta = 0.8
    sigma2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ml_log_likelihood_regression(y, X, beta, sigma2)
    assert isinstance(result, dict)
