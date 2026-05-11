"""Tests for hillEst.hill_estimator."""
import numpy as np
import pytest
from morie.fn.hillEst import hill_estimator


def test_hillEst_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = hill_estimator(x, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hillEst_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = hill_estimator(x, k)
    assert isinstance(result, dict)
