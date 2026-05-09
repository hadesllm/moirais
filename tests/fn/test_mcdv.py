"""Tests for mcdv.mcd."""
import numpy as np
import pytest
from moirais.fn.mcdv import mcd


def test_mcdv_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    h = 0.3
    n_starts = np.random.default_rng(42).normal(0, 1, 100)
    result = mcd(X, h, n_starts)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mcdv_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    h = 0.3
    n_starts = np.random.default_rng(42).normal(0, 1, 100)
    result = mcd(X, h, n_starts)
    assert isinstance(result, dict)
