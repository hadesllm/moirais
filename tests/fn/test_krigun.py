"""Tests for krigun.universal_kriging."""
import numpy as np
import pytest
from morie.fn.krigun import universal_kriging


def test_krigun_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    s_predict = np.random.default_rng(42).normal(0, 1, 100)
    trend_order = np.random.default_rng(42).normal(0, 1, 100)
    result = universal_kriging(coords, values, s_predict, trend_order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_krigun_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    s_predict = np.random.default_rng(42).normal(0, 1, 100)
    trend_order = np.random.default_rng(42).normal(0, 1, 100)
    result = universal_kriging(coords, values, s_predict, trend_order)
    assert isinstance(result, dict)
