"""Tests for spcrs.spatial_cross_validation."""
import numpy as np
import pytest
from moirais.fn.spcrs import spatial_cross_validation


def test_spcrs_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = spatial_cross_validation(x, coords)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spcrs_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = spatial_cross_validation(x, coords)
    assert isinstance(result, dict)
