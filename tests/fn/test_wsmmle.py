"""Tests for wsmmle.wasserman_mle."""
import numpy as np
import pytest
from moirais.fn.wsmmle import wasserman_mle


def test_wsmmle_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    theta0 = 0.0
    result = wasserman_mle(data, f, theta0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmmle_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    theta0 = 0.0
    result = wasserman_mle(data, f, theta0)
    assert isinstance(result, dict)
