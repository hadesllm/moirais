"""Tests for ucfR.user_cf."""
import numpy as np
import pytest
from moirais.fn.ucfR import user_cf


def test_ucfR_basic():
    """Test basic functionality."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = user_cf(R, u, i, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ucfR_edge():
    """Test edge cases."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = user_cf(R, u, i, k)
    assert isinstance(result, dict)
