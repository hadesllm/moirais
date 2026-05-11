"""Tests for ctcfl.counterfactual_notation."""
import numpy as np
import pytest
from morie.fn.ctcfl import counterfactual_notation


def test_ctcfl_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    x_val = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = counterfactual_notation(Y, X, x_val, u)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ctcfl_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    x_val = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = counterfactual_notation(Y, X, x_val, u)
    assert isinstance(result, dict)
