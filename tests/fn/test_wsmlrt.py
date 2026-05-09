"""Tests for wsmlrt.wasserman_lrt."""
import numpy as np
import pytest
from moirais.fn.wsmlrt import wasserman_lrt


def test_wsmlrt_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    theta0 = 0.0
    result = wasserman_lrt(data, f, theta0)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wsmlrt_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    theta0 = 0.0
    result = wasserman_lrt(data, f, theta0)
    assert isinstance(result, dict)
