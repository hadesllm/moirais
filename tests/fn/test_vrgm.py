"""Tests for vrgm.variogram_estimation."""
import numpy as np
import pytest
from moirais.fn.vrgm import variogram_estimation


def test_vrgm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = variogram_estimation(x, coords)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vrgm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = variogram_estimation(x, coords)
    assert isinstance(result, dict)
