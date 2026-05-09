"""Tests for fdwarp.functional_warping."""
import numpy as np
import pytest
from moirais.fn.fdwarp import functional_warping


def test_fdwarp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    cost = np.random.default_rng(42).normal(0, 1, 100)
    result = functional_warping(x, y, cost)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fdwarp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    cost = np.random.default_rng(42).normal(0, 1, 100)
    result = functional_warping(x, y, cost)
    assert isinstance(result, dict)
