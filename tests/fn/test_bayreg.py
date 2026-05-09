"""Tests for bayreg.bayes_linear."""
import numpy as np
import pytest
from moirais.fn.bayreg import bayes_linear


def test_bayreg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    prior_var = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_linear(y, X, prior_var)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayreg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    prior_var = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_linear(y, X, prior_var)
    assert isinstance(result, dict)
