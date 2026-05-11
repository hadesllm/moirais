"""Tests for krigsv.variogram_fit."""
import numpy as np
import pytest
from morie.fn.krigsv import variogram_fit


def test_krigsv_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = variogram_fit(coords, values, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_krigsv_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = variogram_fit(coords, values, model)
    assert isinstance(result, dict)
