"""Tests for brnstn.bernstein_inequality."""
import numpy as np
import pytest
from morie.fn.brnstn import bernstein_inequality


def test_brnstn_basic():
    """Test basic functionality."""
    sigma2 = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    n = 100
    t = np.linspace(0, 10, 100)
    result = bernstein_inequality(sigma2, M, n, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_brnstn_edge():
    """Test edge cases."""
    sigma2 = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    n = 100
    t = np.linspace(0, 10, 100)
    result = bernstein_inequality(sigma2, M, n, t)
    assert isinstance(result, dict)
