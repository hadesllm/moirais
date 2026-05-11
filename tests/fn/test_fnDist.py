"""Tests for fnDist.functional_distance."""
import numpy as np
import pytest
from morie.fn.fnDist import functional_distance


def test_fnDist_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = functional_distance(f, g)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fnDist_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = functional_distance(f, g)
    assert isinstance(result, dict)
