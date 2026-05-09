"""Tests for rgmp.rangayyan_matching_pursuit."""
import numpy as np
import pytest
from moirais.fn.rgmp import rangayyan_matching_pursuit


def test_rgmp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    dictionary = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = rangayyan_matching_pursuit(x, dictionary, max_iter, tol)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgmp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    dictionary = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = rangayyan_matching_pursuit(x, dictionary, max_iter, tol)
    assert isinstance(result, dict)
