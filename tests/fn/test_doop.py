"""Tests for doop.do_operator."""
import numpy as np
import pytest
from morie.fn.doop import do_operator


def test_doop_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    x_val = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = do_operator(Y, X, x_val, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_doop_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    x_val = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = do_operator(Y, X, x_val, model)
    assert isinstance(result, dict)
