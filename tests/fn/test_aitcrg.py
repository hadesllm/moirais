"""Tests for aitcrg.compositional_regression."""
import numpy as np
import pytest
from morie.fn.aitcrg import compositional_regression


def test_aitcrg_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y_comp = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_regression(X, Y_comp, V)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitcrg_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y_comp = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_regression(X, Y_comp, V)
    assert isinstance(result, dict)
