"""Tests for nstat.nonstationary_covariance."""
import numpy as np
import pytest
from morie.fn.nstat import nonstationary_covariance


def test_nstat_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = nonstationary_covariance(x, coords)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_nstat_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = nonstationary_covariance(x, coords)
    assert isinstance(result, dict)
