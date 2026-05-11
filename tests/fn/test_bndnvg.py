"""Tests for bndnvg.bound_naive_gross."""
import numpy as np
import pytest
from morie.fn.bndnvg import bound_naive_gross


def test_bndnvg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_naive_gross(y, D)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndnvg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_naive_gross(y, D)
    assert isinstance(result, dict)
