"""Tests for tmlavp.tmle_average_predictiveness."""
import numpy as np
import pytest
from moirais.fn.tmlavp import tmle_average_predictiveness


def test_tmlavp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    f = np.random.default_rng(42).normal(0, 1, 100)
    loss = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_average_predictiveness(y, D, X, f, loss)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlavp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    f = np.random.default_rng(42).normal(0, 1, 100)
    loss = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_average_predictiveness(y, D, X, f, loss)
    assert isinstance(result, dict)
