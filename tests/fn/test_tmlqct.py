"""Tests for tmlqct.tmle_quantile."""
import numpy as np
import pytest
from moirais.fn.tmlqct import tmle_quantile


def test_tmlqct_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    quantile = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_quantile(y, D, X, quantile)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlqct_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    quantile = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_quantile(y, D, X, quantile)
    assert isinstance(result, dict)
