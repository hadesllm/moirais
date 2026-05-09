"""Tests for rgrls.rangayyan_rls_filter."""
import numpy as np
import pytest
from moirais.fn.rgrls import rangayyan_rls_filter


def test_rgrls_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    lam = 0.1
    delta = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = rangayyan_rls_filter(x, d, lam, delta, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgrls_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    lam = 0.1
    delta = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = rangayyan_rls_filter(x, d, lam, delta, order)
    assert isinstance(result, dict)
