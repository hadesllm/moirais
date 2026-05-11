"""Tests for wsmcdf.wasserman_empirical_cdf."""
import numpy as np
import pytest
from morie.fn.wsmcdf import wasserman_empirical_cdf


def test_wsmcdf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_empirical_cdf(x, data)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmcdf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_empirical_cdf(x, data)
    assert isinstance(result, dict)
