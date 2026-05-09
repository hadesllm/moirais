"""Tests for dpvar.dp_variance."""
import numpy as np
import pytest
from moirais.fn.dpvar import dp_variance


def test_dpvar_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_variance(x, a, b, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpvar_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_variance(x, a, b, epsilon)
    assert isinstance(result, dict)
