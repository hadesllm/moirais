"""Tests for empfun.empty_space_function."""
import numpy as np
import pytest
from moirais.fn.empfun import empty_space_function


def test_empfun_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    r_grid = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = empty_space_function(coords, r_grid, window)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_empfun_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    r_grid = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = empty_space_function(coords, r_grid, window)
    assert isinstance(result, dict)
