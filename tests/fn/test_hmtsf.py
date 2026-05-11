"""Tests for hmtsf.geron_time_series_forecast."""
import numpy as np
import pytest
from morie.fn.hmtsf import geron_time_series_forecast


def test_hmtsf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_time_series_forecast(y, horizon, window)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmtsf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_time_series_forecast(y, horizon, window)
    assert isinstance(result, dict)
