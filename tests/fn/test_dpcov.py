"""Tests for dpcov.dp_covariance."""
import numpy as np
import pytest
from moirais.fn.dpcov import dp_covariance


def test_dpcov_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    C = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_covariance(X, C, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpcov_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    C = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_covariance(X, C, epsilon)
    assert isinstance(result, dict)
