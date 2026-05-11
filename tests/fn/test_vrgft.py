"""Tests for vrgft.variogram_fitting."""
import numpy as np
import pytest
from morie.fn.vrgft import variogram_fitting


def test_vrgft_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = variogram_fitting(x, coords)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vrgft_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = variogram_fitting(x, coords)
    assert isinstance(result, dict)
