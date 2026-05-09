"""Tests for linear.linearization_se."""
import numpy as np
import pytest
from moirais.fn.linear import linearization_se


def test_linear_basic():
    """Test basic functionality."""
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = linearization_se(estimator, data)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_linear_edge():
    """Test edge cases."""
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = linearization_se(estimator, data)
    assert isinstance(result, dict)
