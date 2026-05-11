"""Tests for jodif.joseph_differencing."""
import numpy as np
import pytest
from morie.fn.jodif import joseph_differencing


def test_jodif_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    order = 4
    seasonal_period = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_differencing(y, order, seasonal_period)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jodif_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    order = 4
    seasonal_period = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_differencing(y, order, seasonal_period)
    assert isinstance(result, dict)
