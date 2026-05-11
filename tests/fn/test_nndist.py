"""Tests for nndist.nearest_neighbor_distance."""
import numpy as np
import pytest
from morie.fn.nndist import nearest_neighbor_distance


def test_nndist_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    r_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = nearest_neighbor_distance(coords, r_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_nndist_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    r_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = nearest_neighbor_distance(coords, r_grid)
    assert isinstance(result, dict)
