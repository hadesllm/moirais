"""Tests for krigCl.ordinary_kriging."""
import numpy as np
import pytest
from moirais.fn.krigCl import ordinary_kriging


def test_krigCl_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    new = np.random.default_rng(42).normal(0, 1, 100)
    result = ordinary_kriging(coords, values, new)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_krigCl_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    new = np.random.default_rng(42).normal(0, 1, 100)
    result = ordinary_kriging(coords, values, new)
    assert isinstance(result, dict)
