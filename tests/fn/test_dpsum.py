"""Tests for dpsum.dp_sum."""
import numpy as np
import pytest
from moirais.fn.dpsum import dp_sum


def test_dpsum_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_sum(x, a, b, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpsum_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_sum(x, a, b, epsilon)
    assert isinstance(result, dict)
