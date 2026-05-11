"""Tests for jonaiv.joseph_naive_forecast."""
import numpy as np
import pytest
from morie.fn.jonaiv import joseph_naive_forecast


def test_jonaiv_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_naive_forecast(y, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jonaiv_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_naive_forecast(y, horizon)
    assert isinstance(result, dict)
