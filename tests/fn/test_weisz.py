"""Tests for weisz.weiszfeld."""
import numpy as np
import pytest
from moirais.fn.weisz import weiszfeld


def test_weisz_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    tol = 1e-6
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = weiszfeld(X, tol, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_weisz_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    tol = 1e-6
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = weiszfeld(X, tol, max_iter)
    assert isinstance(result, dict)
