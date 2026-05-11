"""Tests for kpssTst.kpss_test."""
import numpy as np
import pytest
from morie.fn.kpssTst import kpss_test


def test_kpssTst_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    trend = np.random.default_rng(42).normal(0, 1, 100)
    result = kpss_test(y, trend)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_kpssTst_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    trend = np.random.default_rng(42).normal(0, 1, 100)
    result = kpss_test(y, trend)
    assert isinstance(result, dict)
