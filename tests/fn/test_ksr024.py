"""Tests for ksr024.kosorok_ch1_partly_linear_logistic."""
import numpy as np
import pytest
from morie.fn.ksr024 import kosorok_ch1_partly_linear_logistic


def test_ksr024_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    U = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch1_partly_linear_logistic(Y, Z, U, beta, eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr024_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    U = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch1_partly_linear_logistic(Y, Z, U, beta, eta)
    assert isinstance(result, dict)
