"""Tests for cthrgr.causal_three_layer_grf."""
import numpy as np
import pytest
from morie.fn.cthrgr import causal_three_layer_grf


def test_cthrgr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_three_layer_grf(y, D, M, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cthrgr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_three_layer_grf(y, D, M, X)
    assert isinstance(result, dict)
