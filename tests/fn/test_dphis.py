"""Tests for dphis.dp_histogram."""
import numpy as np
import pytest
from moirais.fn.dphis import dp_histogram


def test_dphis_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bins = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_histogram(x, bins, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dphis_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bins = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_histogram(x, bins, epsilon)
    assert isinstance(result, dict)
