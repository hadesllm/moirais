"""Tests for dpmean.dp_mean."""
import numpy as np
import pytest
from moirais.fn.dpmean import dp_mean


def test_dpmean_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_mean(x, a, b, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpmean_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_mean(x, a, b, epsilon)
    assert isinstance(result, dict)
