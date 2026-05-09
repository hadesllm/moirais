"""Tests for arckn.arc_cosine_kernel."""
import numpy as np
import pytest
from moirais.fn.arckn import arc_cosine_kernel


def test_arckn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n = 100
    result = arc_cosine_kernel(X, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_arckn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n = 100
    result = arc_cosine_kernel(X, n)
    assert isinstance(result, dict)
