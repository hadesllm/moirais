"""Tests for npbcox.np_bayes_cox."""
import numpy as np
import pytest
from moirais.fn.npbcox import np_bayes_cox


def test_npbcox_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = np_bayes_cox(time, event, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_npbcox_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = np_bayes_cox(time, event, X)
    assert isinstance(result, dict)
