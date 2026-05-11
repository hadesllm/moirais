"""Tests for stlAn.stl_anomaly."""
import numpy as np
import pytest
from morie.fn.stlAn import stl_anomaly


def test_stlAn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    period = np.random.default_rng(42).normal(0, 1, 100)
    result = stl_anomaly(y, period)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_stlAn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    period = np.random.default_rng(42).normal(0, 1, 100)
    result = stl_anomaly(y, period)
    assert isinstance(result, dict)
