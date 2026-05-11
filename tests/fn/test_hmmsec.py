"""Tests for hmmsec.geron_linreg_mse_cost."""
import numpy as np
import pytest
from morie.fn.hmmsec import geron_linreg_mse_cost


def test_hmmsec_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = geron_linreg_mse_cost(X, y, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmsec_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = geron_linreg_mse_cost(X, y, theta)
    assert isinstance(result, dict)
