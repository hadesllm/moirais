"""Tests for rappor.rappor."""
import numpy as np
import pytest
from morie.fn.rappor import rappor


def test_rappor_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = rappor(x, f, p, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rappor_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = rappor(x, f, p, q)
    assert isinstance(result, dict)
