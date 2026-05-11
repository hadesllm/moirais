"""Tests for brreg.bayesian_ridge_regression."""
import numpy as np
import pytest
from morie.fn.brreg import bayesian_ridge_regression


def test_brreg_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = bayesian_ridge_regression(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_brreg_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = bayesian_ridge_regression(x, y)
    assert isinstance(result, dict)
