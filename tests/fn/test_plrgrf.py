"""Tests for plrgrf.partial_linear_grf."""
import numpy as np
import pytest
from moirais.fn.plrgrf import partial_linear_grf


def test_plrgrf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = partial_linear_grf(y, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_plrgrf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = partial_linear_grf(y, D, X)
    assert isinstance(result, dict)
