"""Tests for dpsyn.dp_synthetic_data."""
import numpy as np
import pytest
from morie.fn.dpsyn import dp_synthetic_data


def test_dpsyn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    epsilon = 1e-6
    result = dp_synthetic_data(X, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpsyn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    epsilon = 1e-6
    result = dp_synthetic_data(X, epsilon)
    assert isinstance(result, dict)
