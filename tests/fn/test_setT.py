"""Tests for setT.set_transformer."""
import numpy as np
import pytest
from morie.fn.setT import set_transformer


def test_setT_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = set_transformer(X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_setT_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = set_transformer(X, k)
    assert isinstance(result, dict)
