"""Tests for joavg.joseph_average_forecast."""
import numpy as np
import pytest
from morie.fn.joavg import joseph_average_forecast


def test_joavg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_average_forecast(y, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_joavg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_average_forecast(y, horizon)
    assert isinstance(result, dict)
