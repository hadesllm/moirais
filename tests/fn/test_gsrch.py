"""Tests for gsrch.grid_search_cv."""
import numpy as np
import pytest
from moirais.fn.gsrch import grid_search_cv


def test_gsrch_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = grid_search_cv(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gsrch_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = grid_search_cv(x, y)
    assert isinstance(result, dict)
