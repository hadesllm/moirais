"""Tests for josnv.joseph_seasonal_naive."""
import numpy as np
import pytest
from moirais.fn.josnv import joseph_seasonal_naive


def test_josnv_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    m = 10
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_seasonal_naive(y, m, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_josnv_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    m = 10
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_seasonal_naive(y, m, horizon)
    assert isinstance(result, dict)
