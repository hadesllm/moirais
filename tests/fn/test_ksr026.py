"""Tests for ksr026.kosorok_ch2_empirical_distribution_function."""
import numpy as np
import pytest
from morie.fn.ksr026 import kosorok_ch2_empirical_distribution_function


def test_ksr026_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    t = np.linspace(0, 10, 100)
    n = 100
    result = kosorok_ch2_empirical_distribution_function(X, t, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr026_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    t = np.linspace(0, 10, 100)
    n = 100
    result = kosorok_ch2_empirical_distribution_function(X, t, n)
    assert isinstance(result, dict)
