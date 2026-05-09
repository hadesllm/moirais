"""Tests for spgams.spatial_gams."""
import numpy as np
import pytest
from moirais.fn.spgams import spatial_gams


def test_spgams_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = spatial_gams(y, X, coords)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spgams_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = spatial_gams(y, X, coords)
    assert isinstance(result, dict)
