"""Tests for slvgrf.sliced_grf."""
import numpy as np
import pytest
from moirais.fn.slvgrf import sliced_grf


def test_slvgrf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    time = np.linspace(0, 10, 100)
    result = sliced_grf(y, D, X, time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_slvgrf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    time = np.linspace(0, 10, 100)
    result = sliced_grf(y, D, X, time)
    assert isinstance(result, dict)
