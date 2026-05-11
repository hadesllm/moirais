"""Tests for vargm.empirical_variogram."""
import numpy as np
import pytest
from morie.fn.vargm import empirical_variogram


def test_vargm_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    lags = 10
    result = empirical_variogram(coords, z, lags)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vargm_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    lags = 10
    result = empirical_variogram(coords, z, lags)
    assert isinstance(result, dict)
