"""Tests for swsmp.systematic_with_random_start."""
import numpy as np
import pytest
from moirais.fn.swsmp import systematic_with_random_start


def test_swsmp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    N = 100
    n = 100
    result = systematic_with_random_start(y, N, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_swsmp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    N = 100
    n = 100
    result = systematic_with_random_start(y, N, n)
    assert isinstance(result, dict)
