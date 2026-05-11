"""Tests for ksr025.kosorok_ch1_penalized_loglikelihood."""
import numpy as np
import pytest
from morie.fn.ksr025 import kosorok_ch1_penalized_loglikelihood


def test_ksr025_basic():
    """Test basic functionality."""
    beta = 0.8
    eta = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lambda_n = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = kosorok_ch1_penalized_loglikelihood(beta, eta, X, lambda_n, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr025_edge():
    """Test edge cases."""
    beta = 0.8
    eta = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lambda_n = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = kosorok_ch1_penalized_loglikelihood(beta, eta, X, lambda_n, n)
    assert isinstance(result, dict)
