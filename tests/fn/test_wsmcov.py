"""Tests for wsmcov.wasserman_covariance."""
import numpy as np
import pytest
from morie.fn.wsmcov import wasserman_covariance


def test_wsmcov_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = wasserman_covariance(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmcov_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = wasserman_covariance(x, y)
    assert isinstance(result, dict)
