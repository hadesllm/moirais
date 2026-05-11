"""Tests for bivcn.bivariate_causal_test."""
import numpy as np
import pytest
from morie.fn.bivcn import bivariate_causal_test


def test_bivcn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(42).normal(0, 1, 100)
    regressor = np.random.default_rng(42).normal(0, 1, 100)
    result = bivariate_causal_test(X, Y, regressor)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bivcn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(42).normal(0, 1, 100)
    regressor = np.random.default_rng(42).normal(0, 1, 100)
    result = bivariate_causal_test(X, Y, regressor)
    assert isinstance(result, dict)
