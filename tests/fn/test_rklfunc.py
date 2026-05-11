"""Tests for rklfunc.ripley_l."""
import numpy as np
import pytest
from morie.fn.rklfunc import ripley_l


def test_rklfunc_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    r_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = ripley_l(coords, r_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rklfunc_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    r_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = ripley_l(coords, r_grid)
    assert isinstance(result, dict)
