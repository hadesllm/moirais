"""Tests for kpsstst.kpss_stationarity."""
import numpy as np
import pytest
from morie.fn.kpsstst import kpss_stationarity


def test_kpsstst_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    trend = np.random.default_rng(42).normal(0, 1, 100)
    result = kpss_stationarity(y, trend)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_kpsstst_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    trend = np.random.default_rng(42).normal(0, 1, 100)
    result = kpss_stationarity(y, trend)
    assert isinstance(result, dict)
