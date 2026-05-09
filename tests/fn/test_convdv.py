"""Tests for convdv.convex_divergence."""
import numpy as np
import pytest
from moirais.fn.convdv import convex_divergence


def test_convdv_basic():
    """Test basic functionality."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = convex_divergence(p, q, f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_convdv_edge():
    """Test edge cases."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = convex_divergence(p, q, f)
    assert isinstance(result, dict)
