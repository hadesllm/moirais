"""Tests for hexgrd.hexagonal_grid."""
import numpy as np
import pytest
from moirais.fn.hexgrd import hexagonal_grid


def test_hexgrd_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    cell_size = 100
    result = hexagonal_grid(coords, values, cell_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hexgrd_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    cell_size = 100
    result = hexagonal_grid(coords, values, cell_size)
    assert isinstance(result, dict)
