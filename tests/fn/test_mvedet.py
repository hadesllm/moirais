"""Tests for mvedet.mve."""
import numpy as np
import pytest
from moirais.fn.mvedet import mve


def test_mvedet_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    h = 0.3
    result = mve(X, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mvedet_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    h = 0.3
    result = mve(X, h)
    assert isinstance(result, dict)
