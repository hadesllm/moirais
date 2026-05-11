"""Tests for sfcret.surface_retrieval."""
import numpy as np
import pytest
from morie.fn.sfcret import surface_retrieval


def test_sfcret_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    grid = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = surface_retrieval(coords, values, grid, method)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sfcret_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    grid = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = surface_retrieval(coords, values, grid, method)
    assert isinstance(result, dict)
