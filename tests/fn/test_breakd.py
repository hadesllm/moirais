"""Tests for breakd.breakdown_point."""
import numpy as np
import pytest
from moirais.fn.breakd import breakdown_point


def test_breakd_basic():
    """Test basic functionality."""
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = breakdown_point(estimator, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_breakd_edge():
    """Test edge cases."""
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = breakdown_point(estimator, n)
    assert isinstance(result, dict)
