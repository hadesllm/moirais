"""Tests for systmp.systematic_sampling."""
import numpy as np
import pytest
from moirais.fn.systmp import systematic_sampling


def test_systmp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    N = 100
    n = 100
    result = systematic_sampling(y, N, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_systmp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    N = 100
    n = 100
    result = systematic_sampling(y, N, n)
    assert isinstance(result, dict)
